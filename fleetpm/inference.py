"""Load only locally generated, integrity-checked bundles for actual inference."""

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .features import window_packet
from .learning import apply_correction, choose_device, correction_features, predict_packets
from .metrics import prediction_interval
from .models import HybridRUL
from .train import verify_bundle, source_code_hash

SENSOR_NAMES=("T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30",
              "phi","NRf","NRc","BPR","farB","htBleed","Nf-demand","NRf-demand","W31","W32")


class Bundle:
    def __init__(self,folder,device="cpu",allow_smoke=False):
        self.folder=Path(folder); verify_bundle(self.folder)
        self.metadata=json.loads((self.folder/"metadata.json").read_text())
        if self.metadata.get("training_status")!="complete":
            raise ValueError("Model training is not complete")
        if self.metadata.get("code_sha256")!=source_code_hash():
            raise ValueError("Numerical source code changed since training; use the matching project version or train a new run")
        if self.metadata.get("smoke_only") and not allow_smoke:
            raise ValueError("A smoke-profile run is a pipeline check, not a deployable trained model")
        self.configuration=json.loads((self.folder/"config.json").read_text())
        self.device=choose_device(device)
        checkpoint=torch.load(self.folder/"model.pt",map_location="cpu",weights_only=True)
        self.model=HybridRUL(self.configuration["window"],self.configuration["cap"])
        self.model.load_state_dict(checkpoint["state_dict"]); self.model.to(self.device).eval()
        # These files are created by our training pipeline, never accepted as uploads.
        self.processor=joblib.load(self.folder/"preprocessing.joblib")
        self.anomaly=joblib.load(self.folder/"anomaly.joblib")
        self.correction=json.loads((self.folder/"residual.json").read_text())
        self.calibration=json.loads((self.folder/"calibration.json").read_text())

    def predict(self,packets,partition="test"):
        points=predict_packets(self.model,packets,self.device)
        values=apply_correction(points,correction_features(points,packets),self.correction,self.configuration["cap"])
        raw_anomaly=-self.anomaly["detector"].score_samples(np.stack([p["stats"] for p in packets]))
        reference=self.anomaly["reference"]
        result=[]
        for i,(point,packet) in enumerate(zip(values,packets)):
            stratum="clean" if packet["quality"][0]>=.99999 else "supported_missing"
            cal=self.calibration[stratum]
            lower,upper=prediction_interval(float(point),cal,self.configuration["cap"])
            reason=""; exception_type=None
            if not packet["basic_quality"]:
                reason="Missing sensors, long gaps or an unsupported operating condition"
                exception_type="DATA" if packet["quality"][2]>=.5 else "DOMAIN"
            elif lower is None:
                reason="Too few independent calibration engines for a finite interval"
                exception_type="CALIBRATION"
            elif upper-lower>40:
                reason="Prediction interval exceeds the 40-cycle decision gate"
                exception_type="UNCERTAINTY"
            anomaly=float(np.searchsorted(reference,raw_anomaly[i],side="right")/len(reference))
            observed=packet["x"][:packet["length"],:21]
            trusted=packet["x"][:packet["length"],24:45].astype(bool)
            deviation=np.where(trusted[-1],np.abs(observed[-1]),0)
            top=np.argsort(deviation)[-3:][::-1]
            result.append({"asset":f'{self.configuration["subset"]}/{partition}/{packet["unit"]:03d}',
                "engine_id":packet["unit"],"subset":self.configuration["subset"],"cycle":packet["cycle"],
                "lifecycle":"1","bundle_id":self.metadata["run_id"],"rul":float(point),
                "base_rul":float(points[i]),"lower":lower,"upper":upper,"anomaly":anomaly if packet["quality"][0]>=.8 else None,
                "quality":float(packet["quality"][0]),"quality_valid":not bool(reason),
                "data_quality_valid":packet["basic_quality"],"exception_type":exception_type,
                "quality_reason":reason,"history_length":packet["length"],
                "complete_loss":bool(packet["quality"][0]==0),"horizon_censored":upper is not None and upper>=self.configuration["cap"],
                "interval_reference":stratum,"calibration_engines":cal["n_engines"],"nominal_coverage":1-cal["alpha"],
                "sensors":[{"index":int(j)+1,"name":SENSOR_NAMES[j],"deviation":float(deviation[j]),
                            "observed":bool(trusted[-1,j])} for j in top]})
        return result

    def engine_predictions(self,engine,ends,partition="test",corruption=None,seed=42):
        transformed=self.processor.transform(engine)
        packets=[window_packet(transformed,int(end),self.configuration["window"],corruption,seed+int(end)) for end in ends]
        return self.predict(packets,partition)

    def sensor_occlusion(self,engine,end):
        transformed=self.processor.transform(engine)
        clean=window_packet(transformed,end,self.configuration["window"])
        baseline=self.predict([clean])[0]["rul"]
        packets=[]
        for sensor in range(21):
            packet={key:(value.copy() if isinstance(value,np.ndarray) else value) for key,value in clean.items()}
            packet["x"][:,sensor]=0; packet["x"][:,24+sensor]=0; packet["x"][:,45+sensor]=1
            # Recompute the statistical bypass for the same observed-channel intervention.
            from .features import rolling_statistics
            n=packet["length"]; mask=packet["x"][:n,24:45].astype(bool)
            packet["stats"]=rolling_statistics(packet["x"][:n,:21],mask)
            packet["quality"][0]=float(mask.mean())
            packets.append(packet)
        values=self.predict(packets)
        return {"method":"sensor-window occlusion (not SHAP, not causal diagnosis)","baseline":baseline,
                "sensors":[{"sensor":i+1,"name":SENSOR_NAMES[i],"rul_change":p["rul"]-baseline}
                           for i,p in enumerate(values)]}
