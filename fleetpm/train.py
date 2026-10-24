"""CUDA-capable training. This module never reads official test data or truth."""

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import ARTIFACT_DIR, DATA_DIR, ROOT, SUBSETS, TrainConfig, file_hash, profile_config, write_json
from .features import Preprocessor, grouped_folds, load_engines, prefix_end, split_engines, window_packet
from .learning import (apply_correction, choose_device, correction_features, fit_correction,
                       model_forward, predict_packets, tensor_batch)
from .metrics import conformal_radius
from .models import HybridRUL


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision("high")


def source_code_hash():
    names=("config.py","data.py","features.py","learning.py","models.py","metrics.py","train.py")
    return hashlib.sha256("".join(file_hash(ROOT/"fleetpm"/n) for n in names).encode()).hexdigest()


def atomic_torch_save(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    torch.save(value,temporary); temporary.replace(path)


class WindowDataset(Dataset):
    def __init__(self,engines,ids,cfg,corrupt=False):
        self.engines=engines; self.cfg=cfg; self.corrupt=corrupt; self.epoch=0
        self.index=[(unit,end) for unit in ids for end in range(cfg.minimum_history-1,len(engines[unit]["raw"]))]
        counts=Counter(unit for unit,_ in self.index)
        self.weights=[1/counts[unit] for unit,_ in self.index]
        if not self.index: raise ValueError("No training windows after history filtering")

    def __len__(self): return len(self.index)

    def __getitem__(self,index):
        unit,end=self.index[index]; engine=self.engines[unit]
        seed=(self.cfg.seed+index*104729+self.epoch*1009)%(2**32)
        rng=np.random.default_rng(seed); corruption=None
        if self.corrupt and rng.random()<.5:
            kind=str(rng.choice(["points","channels","block","bias","drift"]))
            corruption={"kind":kind,"rate":float(rng.uniform(.05,.20)),"length":int(rng.integers(3,6))}
        packet=window_packet(engine,end,self.cfg.window,corruption,seed)
        previous=window_packet(engine,max(0,end-1),self.cfg.window,corruption,seed)
        raw_target=len(engine["raw"])-1-end
        return {"packet":packet,"previous":previous,"target":min(self.cfg.cap,raw_target),
                "raw_target":raw_target}


def collate(samples):
    return samples


def endpoint_packets(transformed,ids,cfg):
    packets=[]; targets=[]
    for unit in ids:
        engine=transformed[unit]
        end=prefix_end(engine,cfg.split_seed,cfg.minimum_history)
        packets.append(window_packet(engine,end,cfg.window))
        targets.append(min(cfg.cap,len(engine["raw"])-1-end))
    return packets,np.asarray(targets,dtype=np.float64)


def train_phase(model,engines,ids,selection,cfg,device,folder,phase,epochs,fingerprint,progress):
    folder=Path(folder)/phase; folder.mkdir(parents=True,exist_ok=True)
    last_path=folder/"last.pt"; best_path=folder/"best.pt"
    for name,p in model.named_parameters():
        p.requires_grad_(phase!="repair" or name.startswith("repair"))
    if phase=="clean":
        for name,p in model.named_parameters():
            if name.startswith("repair"): p.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.learning_rate*(.3 if phase=="joint" else 1),weight_decay=cfg.weight_decay)
    amp=cfg.amp and device.type=="cuda"
    dtype=torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=amp and dtype==torch.float16)
    start=0; best=math.inf; stale=0; history=[]
    if last_path.exists():
        saved=torch.load(last_path,map_location=device,weights_only=True)
        if saved["fingerprint"]!=fingerprint: raise ValueError("Checkpoint configuration/source mismatch")
        torch.set_rng_state(saved["rng"].cpu())
        if device.type=="cuda" and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        if saved.get("finished"):
            model.load_state_dict(torch.load(best_path,map_location=device,weights_only=True)["model"])
            return saved["history"]
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start=saved["epoch"]+1; best=saved["best"]; stale=saved["stale"]; history=saved["history"]
    dataset=WindowDataset(engines,ids,cfg,corrupt=phase!="clean")
    for epoch in range(start,epochs):
        dataset.epoch=epoch
        generator=torch.Generator().manual_seed(cfg.seed+epoch)
        sampler=WeightedRandomSampler(dataset.weights,len(dataset),replacement=True,generator=generator)
        loader=DataLoader(dataset,batch_size=cfg.batch_size,sampler=sampler,num_workers=cfg.workers,collate_fn=collate)
        model.train(); loss_sum=0.; batches=0; tick=time.perf_counter()
        for number,samples in enumerate(loader):
            if cfg.profile=="smoke" and number>=2: break
            batch=tensor_batch([s["packet"] for s in samples],device)
            targets=torch.as_tensor([s["target"] for s in samples],device=device,dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                if phase=="repair":
                    _,generated=model.repair_packet(batch["x"])
                    prediction=None
                else:
                    prediction,generated=model_forward(model,batch)
                reconstruction=batch["reconstruction_mask"].float()
                rec=(((generated.float()-batch["clean"])**2)*reconstruction).sum()/reconstruction.sum().clamp_min(1)
                if phase=="repair": loss=rec
                else:
                    weights=1+(targets<=30).float()
                    rul=((((prediction.float()-targets)/cfg.cap)**2)*weights).sum()/weights.sum()
                    temporal=torch.zeros((),device=device)
                    if phase=="joint" and cfg.temporal_weight>0:
                        prev=tensor_batch([s["previous"] for s in samples],device)
                        with torch.no_grad(): old,_=model_forward(model,prev)
                        degradation=(targets<cfg.cap).float()
                        temporal=(torch.relu(prediction.float()-old.float()-2).square()*degradation).mean()/cfg.cap**2
                    loss=rul+(cfg.repair_weight*rec if phase=="joint" else 0)+cfg.temporal_weight*temporal+cfg.sparse_weight*model.sparsity()
            if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite training loss in {phase}; inspect data/precision")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            loss_sum+=float(loss.detach()); batches+=1
        if not batches: raise ValueError("No batches processed")
        value=loss_sum/batches; validation=None
        if selection is not None and phase!="repair":
            packets,truth=selection
            predictions=predict_packets(model,packets,device,cfg.batch_size)
            validation=float(np.sqrt(np.mean((predictions-truth)**2)))
        criterion=validation if validation is not None else value
        # OOF fits have fixed epoch budgets and never choose a checkpoint from held-out targets.
        improved=selection is None or criterion<best-1e-6
        if improved:
            best=criterion; stale=0
            atomic_torch_save({"model":model.state_dict(),"fingerprint":fingerprint},best_path)
        else: stale+=1
        record={"phase":phase,"epoch":epoch+1,"loss":value,"selection_rmse":validation,
                "seconds":time.perf_counter()-tick}
        history.append(record)
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
               "epoch":epoch,"best":best,"stale":stale,"fingerprint":fingerprint,"history":history,
               "rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else [],"finished":False}
        stopping=selection is not None and stale>=cfg.patience
        state["finished"]=stopping or epoch+1==epochs
        atomic_torch_save(state,last_path)
        progress(record)
        print(f'{cfg.subset} {phase} epoch {epoch+1}/{epochs} loss={value:.5f} selection_RMSE={validation}',flush=True)
        if stopping: break
    model.load_state_dict(torch.load(best_path,map_location=device,weights_only=True)["model"])
    return history


def fit_model(raw,ids,cfg,device,folder,fingerprint,progress,selection_ids=None):
    seed_everything(cfg.seed)
    preprocessor=Preprocessor(cfg.subset,cfg.seed).fit(raw,ids)
    transformed={i:preprocessor.transform(raw[i]) for i in set(ids)|set(selection_ids or [])}
    selection=endpoint_packets(transformed,selection_ids,cfg) if selection_ids else None
    model=HybridRUL(cfg.window,cfg.cap).to(device)
    all_history=[]
    for phase,epochs in (("clean",cfg.clean_epochs),("repair",cfg.repair_epochs),("joint",cfg.epochs)):
        all_history.extend(train_phase(model,transformed,ids,selection,cfg,device,folder,phase,epochs,fingerprint,progress))
    return model,preprocessor,all_history


def verify_bundle(folder):
    folder=Path(folder)
    manifest=json.loads((folder/"bundle_hashes.json").read_text())
    required={"model.pt","preprocessing.joblib","anomaly.joblib","residual.json","calibration.json","config.json","split.json","folds.json"}
    if not isinstance(manifest,dict) or not required<=manifest.keys():
        raise ValueError("Bundle manifest is incomplete")
    for filename,digest in manifest.items():
        if Path(filename).name!=filename or file_hash(folder/filename)!=digest:
            raise ValueError("Bundle integrity check failed")
    return manifest


def train_subset(cfg, data_dir=DATA_DIR, artifact_dir=ARTIFACT_DIR, track_mlflow=False):
    """Train only on complete training trajectories, then freeze calibration."""
    if not isinstance(cfg,TrainConfig): cfg=TrainConfig(**cfg)
    device=choose_device(cfg.device)
    train_path=Path(data_dir)/f"train_{cfg.subset}.txt"
    if not train_path.is_file(): raise FileNotFoundError(f"Place {train_path.name} in {data_dir}; data is never downloaded automatically")
    source=file_hash(train_path)
    fingerprint=hashlib.sha256((cfg.fingerprint(source)+source_code_hash()).encode()).hexdigest()[:16]
    run_id=f"{cfg.subset}-{fingerprint}"
    folder=Path(artifact_dir)/"runs"/run_id; folder.mkdir(parents=True,exist_ok=True)
    metadata_path=folder/"metadata.json"
    if metadata_path.exists():
        old=json.loads(metadata_path.read_text())
        if old.get("training_status")=="complete":
            verify_bundle(folder)
            print(f"Reusing completed immutable run {run_id}",flush=True)
            return old
    raw=load_engines(train_path); split=split_engines(raw,cfg.split_seed,cfg.calibration_fraction)
    metadata={"run_id":run_id,"subset":cfg.subset,"configuration":cfg.to_dict(),"source_sha256":source,
              "code_sha256":source_code_hash(),"training_status":"running","device":str(device),
              "torch_version":torch.__version__,"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
              "created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"profile":cfg.profile,
              "validation_scope":"fit / development-selection / independent calibration engines; official test untouched"}
    write_json(metadata_path,metadata); write_json(folder/"config.json",cfg.to_dict()); write_json(folder/"split.json",split)
    context={}
    def progress(record):
        write_json(folder/"progress.json",context|record|{"run_id":run_id,"subset":cfg.subset})
    mlrun=None
    try:
        if track_mlflow:
            import mlflow
            mlflow.set_tracking_uri("sqlite:///"+str(Path(artifact_dir).resolve()/"mlflow.db"))
            mlflow.set_experiment("FleetPulse-CMAPSS")
            mlrun=mlflow.start_run(run_name=run_id)
            mlflow.log_params(cfg.to_dict()|{"train_sha256":source,"code_sha256":metadata["code_sha256"]})
        oof_points=[]; oof_features=[]; oof_truth=[]; fold_records=[]
        for number,fit_ids,valid_ids in grouped_folds(split["fit"],cfg.folds,cfg.seed):
            fold_dir=folder/"checkpoints"/f"fold-{number+1}"
            context.update(stage="out_of_fold",fold=number+1,folds=cfg.folds)
            progress({"phase":"out_of_fold","fold":number+1,"folds":cfg.folds})
            model,processor,_=fit_model(raw,fit_ids,cfg,device,fold_dir,fingerprint+f"/fold-{number+1}",progress)
            transformed={i:processor.transform(raw[i]) for i in valid_ids}
            packets,truth=endpoint_packets(transformed,valid_ids,cfg)
            points=predict_packets(model,packets,device,cfg.batch_size)
            oof_points.extend(points); oof_truth.extend(truth)
            oof_features.extend(correction_features(points,packets))
            fold_records.append({"fold":number+1,"fit_ids":fit_ids,"heldout_ids":valid_ids,
                                 "rmse_capped":float(np.sqrt(np.mean((points-truth)**2)))})
            del model,processor,transformed,packets; gc.collect()
            if device.type=="cuda": torch.cuda.empty_cache()
        write_json(folder/"folds.json",fold_records)
        context.clear(); context.update(stage="final_fit")
        progress({"phase":"final_fit"})
        model,processor,history=fit_model(raw,split["fit"],cfg,device,folder/"checkpoints"/"final",fingerprint+"/final",progress,split["selection"])
        transformed={i:processor.transform(engine) for i,engine in raw.items()}
        selection_packets,selection_truth=endpoint_packets(transformed,split["selection"],cfg)
        selection_points=predict_packets(model,selection_packets,device,cfg.batch_size)
        correction=fit_correction(np.asarray(oof_points),np.asarray(oof_features),np.asarray(oof_truth),
            selection_points,correction_features(selection_points,selection_packets),selection_truth,cfg.correction_bound,cfg.cap)
        write_json(folder/"residual.json",correction)
        calibrations={}
        for stratum in ("clean","supported_missing"):
            packets=[]; truth=[]; included=[]; excluded=[]
            for unit in split["calibration"]:
                engine=transformed[unit]; end=prefix_end(engine,cfg.split_seed,cfg.minimum_history)
                corruption=None
                if stratum=="supported_missing":
                    rng=np.random.default_rng(cfg.split_seed+unit)
                    corruption={"kind":str(rng.choice(["points","block"])),
                                "rate":float(rng.uniform(.05,.15)),"length":int(rng.integers(3,6))}
                packet=window_packet(engine,end,cfg.window,corruption,cfg.split_seed+unit)
                if not packet["basic_quality"]:
                    excluded.append(unit); continue
                packets.append(packet); included.append(unit)
                truth.append(min(cfg.cap,len(engine["raw"])-1-end))
            if not packets: raise ValueError(f"No supported calibration engines in {stratum}")
            base=predict_packets(model,packets,device,cfg.batch_size)
            corrected=apply_correction(base,correction_features(base,packets),correction,cfg.cap)
            calibrations[stratum]=conformal_radius(np.abs(np.asarray(truth)-corrected),cfg.alpha)
            calibrations[stratum].update(included_engine_ids=included,excluded_engine_ids=excluded)
        calibrations["note"]="Separate references drawn from the same reserved engines, not twice as many independent engines. Quality-qualified fixed prefixes only. Missing reference assumes 5-15% point masking or a two-channel block of 3-5 cycles; whole-window channel loss remains unsupported."
        write_json(folder/"calibration.json",calibrations)
        healthy=[]
        for unit in split["fit"]:
            engine=transformed[unit]
            for end in range(cfg.minimum_history-1,min(20,len(engine["raw"]))):
                packet=window_packet(engine,end,cfg.window)
                if packet["basic_quality"]: healthy.append(packet["stats"])
        if len(healthy)<20: raise ValueError("Insufficient healthy-prefix reference windows")
        detector=IsolationForest(n_estimators=200,random_state=cfg.seed,contamination="auto",n_jobs=-1).fit(np.asarray(healthy))
        reference=np.sort(-detector.score_samples(np.asarray(healthy)))
        joblib.dump({"detector":detector,"reference":reference,"scope":"initial healthy-prefix proxy"},folder/"anomaly.joblib")
        joblib.dump(processor,folder/"preprocessing.joblib")
        atomic_torch_save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                           "configuration":cfg.to_dict(),"fingerprint":fingerprint},folder/"model.pt")
        write_json(folder/"history.json",history)
        metadata.update(training_status="complete",parameters=sum(p.numel() for p in model.parameters()),
                        correction_kind=correction["kind"],selection_rmse=correction["selection_rmse"],
                        calibration_engines=len(split["calibration"]),smoke_only=cfg.profile=="smoke")
        bundle_files=("model.pt","preprocessing.joblib","anomaly.joblib","residual.json","calibration.json","config.json","split.json","folds.json")
        write_json(folder/"bundle_hashes.json",{name:file_hash(folder/name) for name in bundle_files})
        write_json(metadata_path,metadata); progress({"phase":"complete","parameters":metadata["parameters"]})
        if mlrun:
            try:
                import mlflow
                mlflow.log_metric("selection_rmse_capped",metadata["selection_rmse"])
                mlflow.log_artifact(str(folder/"metadata.json")); mlflow.log_artifact(str(folder/"bundle_hashes.json"))
                mlflow.end_run()
            except Exception as tracking_error:
                metadata["tracking_error"]=str(tracking_error); write_json(metadata_path,metadata)
                print(f"Local bundle is complete; MLflow logging failed: {tracking_error}",flush=True)
        return metadata
    except BaseException as error:
        metadata.update(training_status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error))
        write_json(metadata_path,metadata)
        if mlrun:
            try:
                import mlflow
                mlflow.end_run(status="FAILED")
            except Exception: pass
        raise
    finally:
        gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()


def main():
    def interrupted(signum,frame): raise KeyboardInterrupt("Training stopped; resume from the last completed epoch")
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets",nargs="+",choices=SUBSETS,default=list(SUBSETS))
    parser.add_argument("--profile",choices=["balanced","rigorous","smoke"],default=os.environ.get("FLEET_PROFILE","balanced"))
    parser.add_argument("--device",choices=["cuda","cpu","auto"],default=os.environ.get("FLEET_TRAIN_DEVICE","cuda"))
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--mlflow",action="store_true")
    args=parser.parse_args()
    for subset in args.subsets:
        train_subset(profile_config(args.profile,subset=subset,device=args.device,seed=args.seed),track_mlflow=args.mlflow)


if __name__=="__main__": main()
