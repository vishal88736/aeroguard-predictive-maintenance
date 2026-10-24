"""Explicit synthetic UI demonstration. Never used for model training."""

import math
import random

from .config import SUBSETS


def demo_assets():
    output=[]
    examples=[(105,90,120,.4,1),(78,64,92,.3,1),(45,32,58,.91,1),
              (23,13,33,.91,1),(8,3,13,.97,1),(91,76,106,.997,1),
              (62,None,None,None,.62),(118,102,125,.5,1)]
    for s,subset in enumerate(SUBSETS):
        for index,(r,l,u,a,q) in enumerate(examples,1):
            output.append({"asset":f"{subset}/demo/{index:03d}","engine_id":index,"subset":subset,
                "cycle":210+index*9+s*6,"lifecycle":"1","bundle_id":f"DEMO-{subset}",
                "rul":r,"lower":l,"upper":u,"anomaly":a,"quality":q,"quality_valid":q==1,
                "quality_reason":"Illustrative intermittent sensor loss" if q<1 else "",
                "history_length":30,"complete_loss":False,"horizon_censored":u==125,
                "interval_reference":"synthetic illustration","nominal_coverage":.9,
                "calibration_engines":0,"demo":True,
                "sensors":[{"index":3,"name":"T30","deviation":2.3,"observed":True},
                           {"index":7,"name":"P30","deviation":1.8,"observed":True},
                           {"index":9,"name":"Nc","deviation":1.2,"observed":q==1}]})
    return output


def seed_demo(store):
    for asset in demo_assets():
        with store.connect() as db:
            exists=db.execute("SELECT 1 FROM states WHERE namespace='demo' AND asset=?",(asset["asset"],)).fetchone()
        if exists: continue
        for cycle in range(asset["cycle"]-2,asset["cycle"]+1):
            store.process(asset|{"cycle":cycle},"demo")
    store.dispatch_local()


def detail(asset):
    rng=random.Random(asset["engine_id"]*41+int(asset["subset"][-1]))
    points=[]; sensor=[]
    for i in range(60):
        cycle=asset["cycle"]-59+i
        value=asset["rul"]+(59-i)*.65
        points.append({"cycle":cycle,"rul":round(min(125,value),2),
                       "lower":round(min(125,max(0,value-12)),2),"upper":round(min(125,value+12),2)})
        sensor.append({"cycle":cycle,"value":round(.4+i*.018+math.sin(i/5)*.18+rng.uniform(-.08,.08),3)})
    return {"asset":asset,"history":points,"sensor_history":sensor,
            "sensor_label":"S03 / illustrative normalized signal","source":"Synthetic UI demonstration; no trained-model results","demo":True}
