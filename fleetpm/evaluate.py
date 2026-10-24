"""Locked official endpoint evaluation, explicitly separate from model fitting."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ARTIFACT_DIR, DATA_DIR, file_hash, write_json
from .data import parse_truth
from .features import load_engines
from .inference import Bundle
from .metrics import regression_metrics
from .store import Store


def evaluate_run(run_id,data_dir=DATA_DIR,artifact_dir=ARTIFACT_DIR,device="cpu",allow_smoke=False):
    folder=Path(artifact_dir)/"runs"/run_id
    bundle=Bundle(folder,device,allow_smoke)
    subset=bundle.configuration["subset"]
    test_path=Path(data_dir)/f"test_{subset}.txt"
    truth_path=Path(data_dir)/f"RUL_{subset}.txt"
    hashes={"test":file_hash(test_path),"truth":file_hash(truth_path)}
    existing=folder/"evaluation.json"
    if existing.exists():
        old=json.loads(existing.read_text())
        if old["sources"]!=hashes:
            raise ValueError("Final evaluation is already associated with different test files")
    engines=load_engines(test_path); ids=sorted(engines)
    if ids!=list(range(1,len(ids)+1)):
        raise ValueError("Endpoint truth requires official sequential engine IDs")
    # All predictions are frozen before endpoint truth is parsed.
    predictions=[]
    for unit in ids:
        engine=engines[unit]
        predictions.extend(bundle.engine_predictions(engine,[len(engine["raw"])-1]))
    truth=parse_truth(truth_path,len(ids))
    metrics=regression_metrics(truth,[p["rul"] for p in predictions],bundle.configuration["cap"])
    supported=[(p,y) for p,y in zip(predictions,truth) if p["lower"] is not None]
    metrics["capped_interval_coverage"]=float(np.mean([p["lower"]<=min(y,bundle.configuration["cap"])<=p["upper"] for p,y in supported])) if supported else None
    metrics["interval_mean_width"]=float(np.mean([p["upper"]-p["lower"] for p,y in supported])) if supported else None
    metrics["interval_evaluated_engines"]=len(supported)
    metrics["decision_exception_rate"]=float(np.mean([not p["quality_valid"] or p["history_length"]<30 for p in predictions]))
    metrics["short_prefix_engines"]=sum(p["history_length"]<30 for p in predictions)
    report={"run_id":run_id,"subset":subset,"sources":hashes,"metrics":metrics,
            "protocol":"one endpoint per official test engine; raw truth metrics; capped coverage separately labelled",
            "smoke_only":bundle.metadata.get("smoke_only",False)}
    write_json(existing,report)
    write_json(folder/"predictions.json",predictions)
    with open(folder/"predictions.csv","w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["engine_id","cycle","prediction","true_rul","lower_capped","upper_capped","anomaly","decision_valid"])
        writer.writeheader()
        for p,y in zip(predictions,truth):
            writer.writerow({"engine_id":p["engine_id"],"cycle":p["cycle"],"prediction":p["rul"],"true_rul":y,
                "lower_capped":p["lower"],"upper_capped":p["upper"],"anomaly":p["anomaly"],
                "decision_valid":p["quality_valid"] and p["history_length"]>=30})
    return report


def stress_run(run_id,data_dir=DATA_DIR,artifact_dir=ARTIFACT_DIR,device="cpu"):
    folder=Path(artifact_dir)/"runs"/run_id; bundle=Bundle(folder,device)
    subset=bundle.configuration["subset"]
    engines=load_engines(Path(data_dir)/f"test_{subset}.txt"); ids=sorted(engines)
    truth=parse_truth(Path(data_dir)/f"RUL_{subset}.txt",len(ids))
    scenarios={"clean":None,"missing_10":{"kind":"points","rate":.1},
        "missing_20":{"kind":"points","rate":.2},"missing_30":{"kind":"points","rate":.3},
        "channel_outage":{"kind":"channels","rate":.10},"block_5":{"kind":"block","length":5},
        "block_10":{"kind":"block","length":10},"flagged_drift":{"kind":"drift"},
        "unflagged_bias":{"kind":"unflagged_bias"},"noise_005":{"kind":"noise","rate":.05}}
    results=[]
    for name,corruption in scenarios.items():
        predictions=[]
        for unit in ids:
            engine=engines[unit]
            predictions.extend(bundle.engine_predictions(engine,[len(engine["raw"])-1],corruption=corruption))
        results.append({"scenario":name,"metrics":regression_metrics(truth,[p["rul"] for p in predictions]),
                        "exception_rate":float(np.mean([not p["quality_valid"] for p in predictions]))})
    write_json(folder/"stress.json",results)
    return results


def replay_run(run_id,store,data_dir=DATA_DIR,artifact_dir=ARTIFACT_DIR,device="cpu",max_engines=12,last_cycles=60):
    """Actual model predictions enter triage; endpoint truth never enters it."""
    bundle=Bundle(Path(artifact_dir)/"runs"/run_id,device)
    subset=bundle.configuration["subset"]
    engines=load_engines(Path(data_dir)/f"test_{subset}.txt")
    outcomes=[]
    for unit in sorted(engines)[:max_engines]:
        engine=engines[unit]
        start=max(bundle.configuration["minimum_history"]-1,len(engine["raw"])-last_cycles)
        asset=f"{subset}/test/{unit:03d}"
        with store.connect() as db:
            row=db.execute("SELECT state FROM states WHERE namespace='dataset' AND asset=? AND lifecycle='1'",(asset,)).fetchone()
            prior=db.execute("SELECT packet FROM events WHERE namespace='dataset' AND asset=? ORDER BY cycle DESC LIMIT 1",(asset,)).fetchone()
        if prior and json.loads(prior[0])["bundle_id"]!=run_id:
            raise ValueError("This historical replay already uses another bundle. Choose a separate FLEET_RUNTIME_DIR for a new model replay; decisions from different models are not mixed.")
        committed=json.loads(row[0])["last_cycle"] if row else 0
        ends=[end for end in range(start,len(engine["raw"])) if end+1>committed]
        if not ends: continue
        for prediction in bundle.engine_predictions(engine,ends):
            outcomes.append(store.process(prediction,"dataset"))
    store.dispatch_local()
    summary={"run_id":run_id,"processed_cycles":len(outcomes),"orders":len(store.orders("dataset")),
             "scope":"model-backed dataset replay, local work-order delivery only"}
    write_json(Path(artifact_dir)/"runs"/run_id/"replay.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--device",choices=["cpu","cuda","auto"],default="cpu")
    args=parser.parse_args()
    print(json.dumps(evaluate_run(args.run_id,device=args.device),indent=2))


if __name__=="__main__": main()
