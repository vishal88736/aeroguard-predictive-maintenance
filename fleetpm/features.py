"""Training-only regime normalization and causal window construction."""

import math
import random

import numpy as np
from sklearn.cluster import KMeans

from .config import COUNTS
from .data import parse_rows

def load_engines(path):
    rows=np.asarray(parse_rows(path),dtype=np.float64)
    dense_cycles=sum(int(rows[rows[:,0]==unit,1].max()) for unit in np.unique(rows[:,0]))
    if dense_cycles>2_000_000:
        raise ValueError("Dataset exceeds the two-million-cycle dense allocation limit")
    engines={}
    for unit in np.unique(rows[:,0]).astype(int):
        part=rows[rows[:,0]==unit]
        cycles=part[:,1].astype(int)
        # Explicit missing-cycle slots keep the window length in cycles.
        full=np.full((int(cycles[-1]),24),np.nan,dtype=np.float32)
        full[cycles-1]=part[:,2:]
        engines[int(unit)]={"unit":int(unit),"cycles":np.arange(1,len(full)+1),"raw":full}
    return engines


def split_engines(engines,seed=42,calibration_fraction=.2):
    ids=sorted(engines); random.Random(seed).shuffle(ids)
    n=len(ids)
    if n<15:
        raise ValueError("At least 15 complete engines are required; NASA subsets contain 100 or more")
    ncal=max(2,math.ceil(n*calibration_fraction)); nselect=max(2,math.ceil(n*.2))
    result={"calibration":sorted(ids[:ncal]),"selection":sorted(ids[ncal:ncal+nselect]),
            "fit":sorted(ids[ncal+nselect:])}
    assert len(set(result["fit"])&set(result["selection"]))==0
    return result


def prefix_end(engine,seed,minimum=15):
    # Evaluation-only pseudo-truncation; never a predictor feature.
    length=len(engine["raw"])
    rng=np.random.default_rng(seed+engine["unit"]*104729)
    return max(minimum,min(length-1,int(length*rng.uniform(.35,.90))))-1


class Preprocessor:
    def __init__(self,subset,seed=42):
        self.subset=subset; self.seed=seed

    def fit(self,engines,ids):
        self.fit_ids=sorted(map(int,ids))
        raw=np.concatenate([engines[i]["raw"] for i in ids])
        settings=raw[:,:3]; good=np.isfinite(settings).all(1)
        if not good.any(): raise ValueError("No valid operating settings")
        self.settings_mean=settings[good].mean(0)
        self.settings_std=settings[good].std(0)
        self.settings_std=np.where(self.settings_std<1e-6,1,self.settings_std)
        settings_z=(settings[good]-self.settings_mean)/self.settings_std
        nregimes=COUNTS[self.subset][2]
        self.kmeans=KMeans(n_clusters=nregimes,n_init=10,random_state=self.seed).fit(settings_z)
        self.centers=self.kmeans.cluster_centers_.astype(np.float32)
        labels=self.kmeans.predict(settings_z)
        distances=np.linalg.norm(settings_z-self.centers[labels],axis=1)
        self.domain_radius=float(max(np.quantile(distances,.999)*1.5,.25))
        sensors=raw[good,3:]
        global_mean=np.nanmean(sensors,axis=0)
        global_std=np.nanstd(sensors,axis=0)
        self.global_mean=np.nan_to_num(global_mean).astype(np.float32)
        self.global_std=np.where(np.isfinite(global_std)&(global_std>1e-6),global_std,1).astype(np.float32)
        self.sensor_mean=[]; self.sensor_std=[]
        for regime in range(nregimes):
            group=sensors[labels==regime]
            mean=np.nanmean(group,axis=0); std=np.nanstd(group,axis=0)
            self.sensor_mean.append(np.where(np.isfinite(mean),mean,self.global_mean))
            self.sensor_std.append(np.where(np.isfinite(std)&(std>1e-6),std,1))
        self.sensor_mean=np.asarray(self.sensor_mean,dtype=np.float32)
        self.sensor_std=np.asarray(self.sensor_std,dtype=np.float32)
        return self

    def transform(self,engine):
        raw=engine["raw"]; setting_valid=np.isfinite(raw[:,:3]).all(1)
        sz=np.nan_to_num((raw[:,:3]-self.settings_mean)/self.settings_std).astype(np.float32)
        distances=np.linalg.norm(sz[:,None,:]-self.centers[None,:,:],axis=2)
        regime=distances.argmin(1)
        domain=setting_valid&(distances.min(1)<=self.domain_radius)
        observed=np.isfinite(raw[:,3:])&setting_valid[:,None]
        z=(raw[:,3:]-self.sensor_mean[regime])/self.sensor_std[regime]
        z=np.nan_to_num(z,nan=0,posinf=0,neginf=0).astype(np.float32)
        gap=observation_gaps(observed)
        return engine|{"sensors":z,"settings":sz,"mask":observed,"gaps":gap,
                       "domain":domain,"regime":regime}


def observation_gaps(mask,initial=None):
    steps=np.arange(1,len(mask)+1)[:,None]
    previous=np.maximum.accumulate(np.where(mask,steps,0),axis=0)
    gaps=steps-previous
    if initial is not None:
        gaps=gaps+np.where(previous==0,np.asarray(initial)[None,:],0)
    return np.minimum(gaps,30).astype(np.float32)/30
