"""Local FleetPulse API and website; starts without ML dependencies installed."""

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse
import uuid

from .config import (ARTIFACT_DIR,DATA_DIR,ROOT,RUNTIME_DIR,SUBSETS,
                     dependency_status,write_json)
from .data import import_zip,inventory,parse_rows
from .demo import demo_assets,detail as demo_detail,seed_demo
from .repository import active_runs,activate_run,list_runs,predictions_for_active,run_folder
from .store import Store,now

WEB_DIR=ROOT/"web"


class APIError(Exception):
    def __init__(self,message,status=400):
        super().__init__(message); self.status=status


class Application:
    def __init__(self,data_dir=DATA_DIR,artifact_dir=ARTIFACT_DIR,runtime_dir=RUNTIME_DIR):
        self.data_dir=Path(data_dir); self.artifact_dir=Path(artifact_dir); self.runtime_dir=Path(runtime_dir)
        self.store=Store(self.runtime_dir/"fleetpulse.sqlite")
        self.jobs={}; self.job_lock=threading.Lock(); self.inventory_cache=None
        self.inventory_stamp=None; self.hardware_cache=None; self.hardware_time=0
        seed_demo(self.store)
        with self.store.connect() as db:
            db.execute("UPDATE jobs SET status='interrupted',error='Server restarted; resume the matching training configuration',updated_at=? WHERE status IN ('queued','running')",(now(),))

    def data_inventory(self):
        files=sorted(self.data_dir.glob("*.txt")) if self.data_dir.exists() else []
        stamp=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in files]
        if stamp!=self.inventory_stamp:
            self.inventory_cache=inventory(self.data_dir,deep=True); self.inventory_stamp=stamp
        return self.inventory_cache or inventory(self.data_dir)

    def hardware(self):
        if self.hardware_cache is not None and time.time()-self.hardware_time<60:
            return self.hardware_cache
        info={"gpu_detected":False,"name":None,"vram_mb":None,"driver":None}
        try:
            command=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5)
            if command.returncode==0 and command.stdout.strip():
                name,memory,driver=[v.strip() for v in command.stdout.splitlines()[0].split(",")]
                info.update(gpu_detected=True,name=name,vram_mb=int(memory),driver=driver)
        except (OSError,ValueError,subprocess.TimeoutExpired): pass
        self.hardware_cache=info; self.hardware_time=time.time()
        return info

    def health(self):
        deps=dependency_status()
        return {"service":"FleetPulse","python":sys.version.split()[0],"dependencies":deps,
                "training_dependencies_ready":all(deps.values()),"hardware":self.hardware(),
                "training_started_automatically":False,"local_only":True}

    def decisions(self,namespace):
        with self.store.connect() as db:
            rows=db.execute("SELECT asset,cycle,decision,packet FROM events WHERE namespace=? ORDER BY cycle",(namespace,)).fetchall()
        return {r["asset"]:json.loads(r["decision"])|{"bundle_id":json.loads(r["packet"])["bundle_id"]} for r in rows}

    def assets(self,mode,subset=None):
        result=demo_assets() if mode=="demo" else predictions_for_active(self.artifact_dir)
        decisions=self.decisions("demo" if mode=="demo" else "dataset")
        output=[]
        for item in result:
            if subset and subset!="ALL" and item["subset"]!=subset: continue
            decision=decisions.get(item["asset"])
            if decision and decision.get("bundle_id")!=item["bundle_id"]: decision=None
            item=dict(item)
            item["state"]=decision.get("latched_priority",decision["state"]) if decision else "NOT_REPLAYED"
            item["decision"]=decision
            output.append(item)
        return output

    def overview(self,mode,subset=None):
        assets=self.assets(mode,subset)
        orders=self.store.orders("demo" if mode=="demo" else "dataset")
        if subset and subset!="ALL": orders=[o for o in orders if o["asset"].startswith(subset+"/")]
        active_orders=[o for o in orders if o["status"]!="resolved"]
        valid=[a for a in assets if a.get("quality_valid")]
        runs=list_runs(self.artifact_dir)
        return {"mode":mode,"demo":mode=="demo","assets":assets,"orders":orders,"datasets":self.data_inventory(),
            "runs":runs,"active":active_runs(self.artifact_dir),"jobs":self.job_list(),
            "summary":{"engines":len(assets),"attention":sum(a["state"] in {"HIGH","CRITICAL","MEDIUM","DATA_EXCEPTION"} for a in assets),
                "open_orders":len(active_orders),"mean_quality":sum(a["quality"] for a in assets)/len(assets) if assets else None,
                "valid_predictions":len(valid),"trained_subsets":len(active_runs(self.artifact_dir)) if mode!="demo" else 0},
            "source_label":"Synthetic demonstration data" if mode=="demo" else "Model-backed NASA dataset replay"}

    def asset_detail(self,asset_id,mode):
        assets=self.assets(mode)
        asset=next((a for a in assets if a["asset"]==asset_id),None)
        if asset is None: raise APIError("Engine not found in this data mode",404)
        if mode=="demo": return demo_detail(asset)
        missing=[name for name,ready in dependency_status().items() if not ready]
        if missing: raise APIError("Install requirements.txt to inspect actual model histories: "+", ".join(missing),409)
        from .features import load_engines
        from .inference import Bundle
        engines=load_engines(self.data_dir/f'test_{asset["subset"]}.txt')
        engine=engines[asset["engine_id"]]
        end=len(engine["raw"])-1; start=max(14,end-59)
        bundle=Bundle(run_folder(asset["bundle_id"],self.artifact_dir),os.environ.get("FLEET_INFERENCE_DEVICE","cpu"))
        history=bundle.engine_predictions(engine,list(range(start,end+1)))
        sensor=[]
        for i in range(start,end+1):
            value=float(engine["raw"][i,5])  # S03, after the three operating settings
            sensor.append({"cycle":i+1,"value":value if __import__('math').isfinite(value) else None})
        return {"asset":asset,"history":history,"sensor_history":sensor,"sensor_label":"S03 / HPC outlet temperature T30 (R)",
                "source":"Actual observed test sensor prefixes and the active trained bundle","demo":False}

    def job_list(self):
        with self.store.connect() as db:
            rows=db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 20").fetchall()
        result=[]
        for row in rows:
            item=dict(row); item["configuration"]=json.loads(item["configuration"])
            path=Path(item.pop("log_path"))
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0,path.stat().st_size-14000)); item["log_tail"]=stream.read().decode(errors="replace")
            result.append(item)
        return result

    def start_training(self,payload):
        subsets=payload.get("subsets",list(SUBSETS)); profile=payload.get("profile","balanced"); device=payload.get("device","cuda")
        if not isinstance(subsets,list) or not subsets or any(s not in SUBSETS for s in subsets):
            raise APIError("Choose one or more valid subsets")
        if len(set(subsets))!=len(subsets) or profile not in {"balanced","rigorous","smoke"} or device not in {"cuda","cpu","auto"}:
            raise APIError("Invalid training configuration")
        missing=[name for name,ready in dependency_status().items() if not ready]
        if missing: raise APIError("Training is not installed yet. Follow README manual setup. Missing: "+", ".join(missing),409)
        data={item["subset"]:item for item in self.data_inventory()}
        if any(not data[s]["ready"] for s in subsets): raise APIError("Import the official dataset files before training",409)
        with self.job_lock:
            if any(process.poll() is None for process in self.jobs.values()):
                raise APIError("A training job is already running. Four subsets execute sequentially to share GPU memory.",409)
            job_id="JOB-"+uuid.uuid4().hex[:10]
            log_path=self.runtime_dir/(job_id+".log")
            command=[sys.executable,"-m","fleetpm.train","--subsets",*subsets,"--profile",profile,"--device",device]
            environment=os.environ|{"FLEET_DATA_DIR":str(self.data_dir.resolve()),
                "FLEET_ARTIFACT_DIR":str(self.artifact_dir.resolve()),"FLEET_RUNTIME_DIR":str(self.runtime_dir.resolve())}
            with log_path.open("wb") as output:
                process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=output,stderr=subprocess.STDOUT)
            self.jobs[job_id]=process
            config={"subsets":subsets,"profile":profile,"device":device}
            with self.store.connect() as db:
                db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)",(job_id,"running",process.pid,json.dumps(config),str(log_path),now(),now(),None))
            thread=threading.Thread(target=self._monitor_job,args=(job_id,process),daemon=True); thread.start()
            return {"job_id":job_id,"status":"running"}

    def _monitor_job(self,job_id,process):
        code=process.wait()
        with self.store.connect() as db:
            previous=db.execute("SELECT status FROM jobs WHERE id=?",(job_id,)).fetchone()
            status="cancelled" if previous and previous[0]=="cancelling" else ("complete" if code==0 else "failed")
            db.execute("UPDATE jobs SET status=?,updated_at=?,error=? WHERE id=?",(status,now(),None if code==0 else f"Process exited {code}; see log",job_id))

    def stop_training(self,job_id):
        with self.job_lock:
            process=self.jobs.get(job_id)
            if not process or process.poll() is not None: raise APIError("No owned running job with this ID",404)
            with self.store.connect() as db:
                db.execute("UPDATE jobs SET status='cancelling',updated_at=? WHERE id=?",(now(),job_id))
            process.terminate()
        return {"job_id":job_id,"status":"cancelling"}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version="FleetPulse/0.1"

        def _send(self,value,status=200,content_type="application/json",filename=None):
            raw=json.dumps(value,allow_nan=False).encode() if content_type=="application/json" else value
            self.send_response(status)
            self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(len(raw)))
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Referrer-Policy","same-origin")
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'")
            self.send_header("Cache-Control","no-store" if content_type=="application/json" else "no-cache")
            if filename: self.send_header("Content-Disposition",f'attachment; filename="{filename}"')
            self.end_headers(); self.wfile.write(raw)

        def _query(self):
            host=urlparse("http://"+self.headers.get("Host","")).hostname
            if host not in {"127.0.0.1","localhost","::1",self.server.server_address[0]}:
                raise APIError("This local workspace rejects unknown Host headers",403)
            parsed=urlparse(self.path); query=parse_qs(parsed.query)
            mode=query.get("mode",["dataset"])[0]
            if mode not in {"dataset","demo"}: raise APIError("Invalid data mode")
            subset=query.get("subset",["ALL"])[0]
            if subset not in {*SUBSETS,"ALL"}: raise APIError("Invalid subset")
            return parsed.path,query,mode,subset

        def do_GET(self):
            try:
                path,query,mode,subset=self._query()
                if path=="/api/health": return self._send(app.health())
                if path=="/api/overview": return self._send(app.overview(mode,subset))
                if path=="/api/datasets": return self._send(app.data_inventory())
                if path=="/api/runs": return self._send({"runs":list_runs(app.artifact_dir),"active":active_runs(app.artifact_dir)})
                if path=="/api/jobs": return self._send(app.job_list())
                if path=="/api/run":
                    folder=run_folder(query.get("id",[""])[0],app.artifact_dir)
                    return self._send({name:json.loads((folder/(name+".json")).read_text()) for name in
                        ("metadata","history","calibration","residual","evaluation") if (folder/(name+".json")).exists()})
                if path=="/api/asset": return self._send(app.asset_detail(query.get("id",[""])[0],mode))
                downloads={"/downloads/notebook":ROOT/"notebooks/C_MAPSS_All_Subsets_CUDA.ipynb",
                    "/downloads/requirements":ROOT/"requirements.txt","/downloads/environment":ROOT/".env.example",
                    "/downloads/architecture":ROOT/"docs/RESEARCH_WORKFLOW.md"}
                if path in downloads:
                    target=downloads[path]
                    return self._send(target.read_bytes(),content_type="application/octet-stream",filename=target.name)
                if path=="/documents/model.svg": target=ROOT/"docs/legacy-model-architecture.svg"
                elif path=="/documents/system.svg": target=ROOT/"docs/architecture.svg"
                else:
                    relative=unquote(path).lstrip("/") or "index.html"
                    target=(WEB_DIR/relative).resolve()
                    if not target.is_relative_to(WEB_DIR.resolve()): raise APIError("Not found",404)
                if not target.is_file(): raise APIError("Not found",404)
                content_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                if content_type.startswith("text/") or target.suffix==".js": content_type+="; charset=utf-8"
                return self._send(target.read_bytes(),content_type=content_type)
            except (ValueError,OSError,APIError) as error:
                return self._send({"error":str(error)},getattr(error,"status",400))
            except Exception as error:
                return self._send({"error":"Operation failed: "+type(error).__name__},500)

        def do_POST(self):
            try:
                origin=self.headers.get("Origin")
                expected="http://"+self.headers.get("Host","")
                if origin and origin!=expected or self.headers.get("X-Fleet-Client")!="dashboard":
                    raise APIError("Same-origin dashboard requests are required",403)
                path,query,mode,subset=self._query()
                size=int(self.headers.get("Content-Length","0"))
                limit=int(os.environ.get("FLEET_MAX_UPLOAD_MB","32"))*1024*1024
                if size<1 or size>limit: raise APIError("Invalid or oversized request",413)
                body=self.rfile.read(size)
                if len(body)!=size: raise APIError("Incomplete request")
                if path=="/api/import":
                    if mode=="demo": raise APIError("Switch to dataset mode to import actual data")
                    if any(p.poll() is None for p in app.jobs.values()): raise APIError("Wait for the active training job before importing data",409)
                    report=import_zip(body,app.data_dir); app.inventory_stamp=None
                    return self._send({"datasets":report,"message":"Dataset imported and validated"})
                payload=json.loads(body)
                if not isinstance(payload,dict): raise APIError("Expected a JSON object")
                if path=="/api/train":
                    if mode=="demo": raise APIError("Training is available only in dataset mode")
                    return self._send(app.start_training(payload),202)
                if path=="/api/jobs/stop": return self._send(app.stop_training(payload.get("job_id","")))
                if path=="/api/activate":
                    if mode=="demo": raise APIError("Illustrative demo models cannot be activated")
                    return self._send(activate_run(payload.get("run_id",""),app.artifact_dir))
                if path=="/api/evaluate":
                    if mode=="demo": raise APIError("Demo records are not benchmark results")
                    if not all(dependency_status().values()): raise APIError("Install requirements.txt before actual evaluation",409)
                    run_id=payload.get("run_id","")
                    folder=run_folder(run_id,app.artifact_dir)
                    metadata=json.loads((folder/"metadata.json").read_text())
                    if active_runs(app.artifact_dir).get(metadata["subset"])!=run_id:
                        raise APIError("Activate the development-selected bundle before opening official endpoint truth",409)
                    from .evaluate import evaluate_run
                    return self._send(evaluate_run(run_id,app.data_dir,app.artifact_dir,
                                                   os.environ.get("FLEET_INFERENCE_DEVICE","cpu")))
                if path=="/api/replay":
                    if mode=="demo": raise APIError("Demo records already have illustrative decisions")
                    if not all(dependency_status().values()): raise APIError("Install requirements.txt before running actual inference",409)
                    run_id=payload.get("run_id",""); folder=run_folder(run_id,app.artifact_dir)
                    metadata=json.loads((folder/"metadata.json").read_text())
                    if active_runs(app.artifact_dir).get(metadata["subset"])!=run_id:
                        raise APIError("Activate this bundle before replaying its predictions",409)
                    from .evaluate import replay_run
                    return self._send(replay_run(run_id,app.store,app.data_dir,app.artifact_dir,
                                               os.environ.get("FLEET_INFERENCE_DEVICE","cpu"),max_engines=12,last_cycles=60))
                if path=="/api/orders/update":
                    return self._send(app.store.update_order(payload.get("id",""),payload.get("status",""),payload.get("resolution"),
                        "demo" if mode=="demo" else "dataset"))
                raise APIError("Unknown action",404)
            except (ValueError,OSError,APIError) as error:
                return self._send({"error":str(error)},getattr(error,"status",400))
            except Exception as error:
                return self._send({"error":"Operation failed: "+type(error).__name__},500)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",default=os.environ.get("FLEET_HOST","127.0.0.1"))
    parser.add_argument("--port",type=int,default=int(os.environ.get("FLEET_PORT","8765")))
    args=parser.parse_args()
    app=Application(); server=ThreadingHTTPServer((args.host,args.port),make_handler(app))
    print(f"FleetPulse Local: http://{args.host}:{server.server_port}",flush=True)
    print("No training or downloads are started. Guided demo uses labelled synthetic records.",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()


if __name__=="__main__": main()
