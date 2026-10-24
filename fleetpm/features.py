"""Training-only regime normalization and causal window construction."""

import math
from pathlib import Path
import random

import numpy as np
from sklearn.cluster import KMeans

from .config import COUNTS
from .data import parse_rows

STAT_DIM = 189  # 4*21 statistics, 4*21 availability flags, 21 normalized counts


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


def grouped_folds(ids,nfolds,seed):
    ids=list(ids); random.Random(seed).shuffle(ids)
    if len(ids)<nfolds*2:
        raise ValueError("Not enough independent engines for these folds")
    for number in range(nfolds):
        valid=sorted(ids[number::nfolds]); fit=sorted(set(ids)-set(valid))
        yield number,fit,valid


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


def rolling_statistics(sensors,mask):
    """Only the latest 15 observed cycles; no interpolation using future data."""
    x=np.asarray(sensors[-15:],dtype=np.float32)
    m=np.asarray(mask[-15:],dtype=bool)
    count=m.sum(0); enough=count>=5
    den=np.maximum(count,1)
    mean=(x*m).sum(0)/den
    std=np.sqrt((((x-mean)**2)*m).sum(0)/den)
    t=np.arange(len(x),dtype=np.float32)[:,None]
    tmean=(t*m).sum(0)/den
    slope=(((t-tmean)*(x-mean))*m).sum(0)/np.maximum((((t-tmean)**2)*m).sum(0),1e-6)
    indices=np.sort(np.where(m,np.arange(len(x))[:,None],-1),axis=0)
    change_valid=count>=2
    a,b=indices[-2] if len(x)>1 else indices[-1],indices[-1]
    columns=np.arange(21)
    change=np.where(change_valid,(x[np.maximum(b,0),columns]-x[np.maximum(a,0),columns])/np.maximum(1,b-a),0)
    valid=np.concatenate([enough,enough,enough,change_valid])
    values=np.concatenate([mean,std,slope,change]).astype(np.float32)
    values=np.where(valid,values,0)
    return np.concatenate([values,valid.astype(np.float32),count.astype(np.float32)/15])


def window_packet(engine,end,window=30,corrupt=None,seed=0):
    if not 0<=end<len(engine["raw"]): raise ValueError("Endpoint is outside the observed engine prefix")
    start=max(0,end-window+1)
    sensors=engine["sensors"][start:end+1].copy()
    settings=engine["settings"][start:end+1].copy()
    trusted=engine["mask"][start:end+1].copy()
    original=sensors.copy(); original_mask=trusted.copy()
    rng=np.random.default_rng(seed)
    if corrupt is not None:
        mode=corrupt.get("kind","points"); rate=float(corrupt.get("rate",.15))
        damaged=np.zeros_like(trusted)
        if mode=="points": damaged=rng.random(trusted.shape)<rate
        elif mode=="channels": damaged[:,rng.choice(21,max(1,round(21*rate)),replace=False)]=True
        elif mode=="block":
            length=min(len(sensors),int(corrupt.get("length",5)))
            offset=int(rng.integers(0,max(1,len(sensors)-length+1)))
            damaged[offset:offset+length,rng.choice(21,2,replace=False)]=True
        elif mode in {"bias","drift","unflagged_bias"}:
            damaged[:,rng.choice(21,2,replace=False)]=True
            bias=rng.uniform(-.5,.5,size=(1,21))
            factor=np.linspace(0,1,len(sensors))[:,None] if mode=="drift" else 1
            sensors=sensors+damaged*bias*factor
        elif mode=="noise": sensors+=rng.normal(0,rate,sensors.shape).astype(np.float32)
        else: raise ValueError("Unsupported corruption mode")
        if mode not in {"noise","unflagged_bias"}:
            trusted&=~damaged
            if mode in {"points","channels","block"}: sensors[~trusted]=0
    initial=engine["gaps"][start-1]*30 if start>0 else None
    gaps=observation_gaps(trusted,initial)
    numeric=np.concatenate([sensors,settings,trusted.astype(np.float32),gaps],axis=1).astype(np.float32)
    length=len(numeric)
    x=np.zeros((window,66),np.float32); x[:length]=numeric
    clean=np.zeros((window,21),np.float32); clean[:length]=original
    rec=np.zeros((window,21),bool); rec[:length]=original_mask&~trusted
    valid=np.zeros(window,bool); valid[:length]=True
    quality=np.array([trusted.mean(),float(gaps.max()),float(engine["domain"][start:end+1].all())],np.float32)
    channel_outages=int((~trusted.any(0)).sum())
    domain=bool(engine["domain"][start:end+1].all())
    basic=bool(quality[0]>=.8 and channel_outages<=2 and gaps.max()<=5/30 and domain)
    return {"x":x,"length":length,"valid":valid,"stats":rolling_statistics(sensors,trusted),
            "quality":quality,"clean":clean,"reconstruction_mask":rec,"basic_quality":basic,
            "channel_outages":channel_outages,"unit":int(engine["unit"]),"end":int(end),
            "cycle":int(engine["cycles"][end]),"regime":int(engine["regime"][end])}
