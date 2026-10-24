"""Numerical interfaces shared by training and model-backed website inference."""

import numpy as np
import torch

from .metrics import prediction_interval


def choose_device(requested="cuda"):
    if requested=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot use it. Install the official CUDA wheel into .venv, then check torch.cuda.is_available(). Choose cpu explicitly to use CPU.")
    return torch.device("cuda" if requested!="cpu" and torch.cuda.is_available() else "cpu")


def tensor_batch(packets,device):
    fields=("x","stats","quality","valid","clean","reconstruction_mask")
    batch={key:torch.as_tensor(np.stack([p[key] for p in packets]),device=device) for key in fields}
    batch["lengths"]=torch.as_tensor([p["length"] for p in packets],device=device)
    return batch


def model_forward(model,batch):
    return model(batch["x"],batch["stats"],batch["quality"],batch["lengths"],batch["valid"])


def predict_packets(model,packets,device,batch_size=256):
    model.eval(); output=[]
    with torch.inference_mode():
        for start in range(0,len(packets),batch_size):
            batch=tensor_batch(packets[start:start+batch_size],device)
            values,_=model_forward(model,batch)
            output.extend(values.float().cpu().tolist())
    return np.asarray(output,dtype=np.float64)


def correction_features(points,packets):
    # A stable low-dimensional contract; no identity, labels or sample index.
    features=[]
    for point,packet in zip(points,packets):
        regime=np.zeros(6); regime[min(5,packet["regime"])]=1
        features.append(np.concatenate([[point/125],packet["quality"],packet["stats"][:6],regime]))
    return np.asarray(features,dtype=np.float64)


def apply_correction(points,features,correction,cap=125):
    if correction.get("kind")=="identity":
        return np.clip(points,0,cap)
    z=(features-np.asarray(correction["mean"]))/np.asarray(correction["scale"])
    design=np.column_stack([z,np.ones(len(z))])
    delta=design@np.asarray(correction["weights"])
    bound=correction["bound"]
    return np.clip(points+np.clip(delta,-bound,bound),0,cap)


def fit_correction(points,features,targets,selection_points,selection_features,selection_targets,bound=10,cap=125):
    mean=features.mean(0); scale=features.std(0); scale=np.where(scale>1e-8,scale,1)
    design=np.column_stack([(features-mean)/scale,np.ones(len(features))])
    candidates=[{"kind":"identity"}]
    penalty=.1
    for ridge in (1.,10.,100.):
        regularizer=np.eye(design.shape[1])*ridge; regularizer[-1,-1]=0
        weights=np.linalg.solve((1+penalty)*design.T@design+regularizer,design.T@(targets-points))
        candidates.append({"kind":"bounded_ridge","mean":mean.tolist(),"scale":scale.tolist(),
                           "weights":weights.tolist(),"bound":bound,"ridge":ridge,
                           "magnitude_penalty":penalty,"training_scope":"fit-engine out-of-fold predictions only"})
    scores=[]
    for candidate in candidates:
        pred=apply_correction(selection_points,selection_features,candidate,cap)
        scores.append(float(np.mean((pred-selection_targets)**2)))
    best=int(np.argmin(scores))
    # Require a development benefit; otherwise keep an explicit identity map.
    if best and scores[best]>scores[0]*.99: best=0
    selected=candidates[best]
    selected["selection_rmse"]=float(np.sqrt(scores[best]))
    selected["uncorrected_selection_rmse"]=float(np.sqrt(scores[0]))
    selected["selection_scope"]="development selection engines; not a final unbiased test"
    return selected
