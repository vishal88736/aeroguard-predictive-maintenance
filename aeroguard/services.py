"""Application orchestration: immutable jobs, model selection, replay and UI queries."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

from fleetpm.config import ARTIFACT_DIR, DATA_DIR, RUNTIME_DIR, SUBSETS, file_hash, write_json
from fleetpm.demo import demo_assets, detail as demo_detail, seed_demo
from fleetpm.store import Store, now
from .contracts import PipelineRequest, TrainRequest
from .datasets import DatasetManager
from .repository import Conflict, Repository
from .research import ResearchCatalog

logger=logging.getLogger('aeroguard')


def health_label(state):
    if state=='CRITICAL':return 'critical'
    if state=='NORMAL':return 'healthy'
    if state in {'NOT_REPLAYED','WARMING_UP'}:return 'unknown'
    return 'attention'


@dataclass
class StageResult:
    records:int=0
    detail:str=''


class Service:
    def __init__(self,data_dir=DATA_DIR,artifact_dir=ARTIFACT_DIR,runtime_dir=RUNTIME_DIR):
        self.data_dir=Path(data_dir);self.artifact_dir=Path(artifact_dir);self.runtime_dir=Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True,exist_ok=True);self.artifact_dir.mkdir(parents=True,exist_ok=True)
        self.repo=Repository(self.runtime_dir/'aeroguard.sqlite')
        self.datasets=DatasetManager(self.data_dir,self.artifact_dir)
        self.research=ResearchCatalog(self.artifact_dir,self.data_dir)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='aeroguard-job')
        self.futures={};self.lock=threading.RLock();self.cache={};self.bundles={};self.engine_cache={}
        self.active_path=self.artifact_dir/'aeroguard/active.json'
        seed_demo(self.repo.store)

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=False)

    def active(self):
        return json.loads(self.active_path.read_text()) if self.active_path.exists() else {}

    def data_summary(self):
        stamp=tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(self.data_dir.glob('*.txt')))
        with self.lock:
            if self.cache.get('data_stamp')!=stamp:
                self.cache['data_summary']=self.datasets.summary();self.cache['data_stamp']=stamp
            return self.cache.get('data_summary') or self.datasets.summary()

    def health(self):
        dependencies={name:importlib.util.find_spec(module) is not None for name,module in
            [('numpy','numpy'),('pandas','pandas'),('scikit-learn','sklearn'),('mlflow','mlflow'),('torch (optional research)','torch')]}
        hardware={'device':'CPU baseline','gpu_detected':False,'name':None}
        if time.monotonic()-self.cache.get('hardware_tick',-100)>60:
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
                if result.returncode==0:
                    name,memory=result.stdout.splitlines()[0].split(',')
                    hardware.update(gpu_detected=True,name=name.strip(),vram_mb=int(memory))
            except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):pass
            self.cache['hardware']=hardware;self.cache['hardware_tick']=time.monotonic()
        return {'application':'AeroGuard','version':'1.0.0','status':'ready','dependencies':dependencies,
            'hardware':self.cache.get('hardware',hardware),'data_ready':sum(d['ready'] for d in self.data_summary()['datasets']),
            'active_models':len(self.active()),'model_status':'ready' if self.active() else 'training_required',
            'source':'historical simulated benchmark replay','external_dispatch':False}

    def models(self):
        from .baseline import list_models
        from fleetpm.repository import list_runs
        active=self.active();runs=list_models(self.artifact_dir)
        for run in runs:
            eligible,reason=self.promotion_gate(run,active=active)
            run.update(promotion_eligible=eligible,promotion_reason=reason,active=active.get(run['subset'])==run['run_id'])
        with self.repo.connect() as db:promotions=[dict(r)|{'details':json.loads(r['details'])} for r in db.execute('SELECT * FROM ag_promotions ORDER BY id DESC LIMIT 50')]
        return {'runs':runs,'active':active,'promotions':promotions,'research_runs':list_runs(self.artifact_dir),
                'gpu_research':self.research.summary()}

    def model_detail(self,run_id):
        from .baseline import model_folder
        folder=model_folder(run_id,self.artifact_dir)
        if not folder.is_dir():raise ValueError('Model run not found')
        record=json.loads((folder/'metadata.json').read_text())
        for key in ('split','evaluation','calibration','monitoring'):
            path=folder/(key+'.json')
            if path.exists():record[key]=json.loads(path.read_text())
        record['artifacts']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(folder.iterdir()) if p.is_file()]
        record['promotion_eligible'],record['promotion_reason']=self.promotion_gate(record)
        return record

    def promotion_gate(self,candidate,settings=None,active=None):
        settings=settings or self.repo.settings();active=active if active is not None else self.active()
        if candidate.get('training_status')!='complete':return False,'Training must finish successfully'
        current_id=active.get(candidate['subset'])
        if current_id==candidate['run_id']:return False,'Already active'
        metrics=candidate.get('validation_metrics') or {}
        if any(metrics.get(k) is None or not math.isfinite(metrics[k]) for k in ('rmse','nasa_score')):return False,'Finite development RMSE and NASA score are required'
        if not current_id:return True,'First validated model for this subset'
        from .baseline import model_folder
        current=json.loads((model_folder(current_id,self.artifact_dir)/'metadata.json').read_text())
        for field in ('source_sha256','split_seed'):
            left=candidate.get(field,candidate.get('configuration',{}).get(field))
            right=current.get(field,current.get('configuration',{}).get(field))
            if left!=right:return False,'Development cohorts or training data differ; automatic comparison is not valid'
        for field in ('minimum_history','cap','feature_version'):
            if candidate.get('configuration',{}).get(field)!=current.get('configuration',{}).get(field):
                return False,'Evaluation prefix policy or prediction horizon differs; automatic promotion is blocked'
        candidate_folder=model_folder(candidate['run_id'],self.artifact_dir)
        current_folder=model_folder(current_id,self.artifact_dir)
        if json.loads((candidate_folder/'split.json').read_text())!=json.loads((current_folder/'split.json').read_text()):
            return False,'Recorded engine cohorts differ; automatic promotion is blocked'
        def evaluation_prefixes(folder):
            values=json.loads((folder/'validation_predictions.json').read_text())
            return [(v.get('engine_id'),v.get('cycle'),v.get('true_rul')) for v in values]
        if evaluation_prefixes(candidate_folder)!=evaluation_prefixes(current_folder):
            return False,'Recorded development evaluation prefixes differ; automatic promotion is blocked'
        reference=current.get('validation_metrics') or {}
        if metrics['rmse']>reference['rmse']*(1+settings.promotion_rmse_tolerance_percent/100)+1e-10:
            return False,'Development RMSE exceeds the configured promotion limit'
        score,old=metrics.get('nasa_score'),reference.get('nasa_score')
        if score is None or old is None:return False,'Finite development NASA scores are required'
        if score>old*(1+settings.promotion_nasa_tolerance_percent/100)+1e-10:
            return False,'Development NASA score exceeds the configured promotion limit'
        return True,'Development RMSE and NASA score satisfy promotion criteria'

    def promote(self,run_id,rollback=False,settings=None):
        from .baseline import model_folder,BaselineBundle
        with self.lock:
            folder=model_folder(run_id,self.artifact_dir)
            bundle=BaselineBundle(folder)  # verifies trusted local artifact integrity
            candidate=bundle.metadata;active=self.active();previous=active.get(candidate['subset'])
            eligible,reason=self.promotion_gate(candidate,settings,active)
            if rollback:
                with self.repo.connect() as db:
                    known=db.execute('SELECT 1 FROM ag_promotions WHERE run_id=? AND subset=?',(run_id,candidate['subset'])).fetchone()
                if not known:raise Conflict('Rollback is limited to a previously promoted model')
                if previous==run_id:raise Conflict('This version is already active')
                reason='Explicit rollback to a retained previously promoted version'
            elif not eligible:raise Conflict(reason)
            active[candidate['subset']]=run_id;write_json(self.active_path,active)
            details={'reason':reason,'validation_metrics':candidate['validation_metrics'],'automatic_test_selection':False,'rollback':rollback}
            with self.repo.connect() as db:db.execute('INSERT INTO ag_promotions(subset,run_id,previous_run_id,details,created_at) VALUES(?,?,?,?,?)',(candidate['subset'],run_id,previous,json.dumps(details),now()))
            self.repo.audit('model',run_id,'rollback' if rollback else 'promoted',details)
            return {'active':active,'previous_run_id':previous,'reason':reason}

    def bundle(self,run_id):
        from .baseline import BaselineBundle,model_folder,verify_bundle
        folder=model_folder(run_id,self.artifact_dir)
        verify_bundle(folder)
        stamp=(folder/'metadata.json').stat().st_mtime_ns
        key=(run_id,stamp,file_hash(folder/'bundle_hashes.json'))
        with self.lock:
            if key not in self.bundles:self.bundles[key]=BaselineBundle(folder)
            return self.bundles[key]

    def engines_for(self,subset):
        from fleetpm.features import load_engines
        path=self.data_dir/f'test_{subset}.txt';key=(subset,path.stat().st_mtime_ns)
        with self.lock:
            if key not in self.engine_cache:self.engine_cache[key]=load_engines(path)
            return self.engine_cache[key]

    def assets(self,mode='benchmark',subset='ALL',engine='',severity='ALL',cycle_min=0,cycle_max=None):
        if mode=='demo':
            assets=demo_assets();namespaces=['demo']
        else:
            from .baseline import model_folder
            assets=[];namespaces=[]
            for part,run_id in self.active().items():
                path=model_folder(run_id,self.artifact_dir)/'predictions.json'
                if path.exists():assets.extend(json.loads(path.read_text()))
                namespaces.append('benchmark:'+run_id)
        decisions=self.repo.decisions(namespaces);result=[]
        for original in assets:
            item=dict(original)
            if subset!='ALL' and item['subset']!=subset:continue
            if engine and engine.lower() not in f"{item['asset']} {item['subset']}-{item['engine_id']:03d}".lower():continue
            if item['cycle']<cycle_min or cycle_max is not None and item['cycle']>cycle_max:continue
            namespace='demo' if mode=='demo' else 'benchmark:'+item['bundle_id']
            decision=decisions.get((namespace,item['asset']))
            if decision and decision['packet'].get('cycle')==item['cycle']:
                for key in ('rul','base_rul','lower','upper','anomaly','quality','quality_valid','data_quality_valid',
                            'quality_reason','exception_type','history_length','required_history','trend','timestamp'):
                    if key in decision['packet']:item[key]=decision['packet'][key]
            state=decision.get('latched_priority',decision['state']) if decision else 'NOT_REPLAYED'
            item.update(state=state,health=health_label(state),decision=decision,timestamp=(decision or {}).get('timestamp',item.get('timestamp')))
            item.setdefault('trend',0)
            if severity.upper()!='ALL' and severity.lower() not in {item['health'],item['state'].lower()}:continue
            result.append(item)
        return result

    def dashboard(self,mode='benchmark',**filters):
        assets=self.assets(mode,**filters);models=self.models();orders=self.repo.orders(mode)
        subset=filters.get('subset','ALL')
        if subset!='ALL':orders=[o for o in orders if o['subset']==subset]
        if mode=='benchmark':
            namespaces={'benchmark:'+run_id for run_id in self.active().values()}
            orders=[o for o in orders if o['namespace'] in namespaces]
        jobs=[j for j in self.repo.jobs() if j['configuration'].get('mode','benchmark')==mode]
        ruls=[p['rul'] for p in assets]
        summary={'total':len(assets),'healthy':sum(a['health']=='healthy' for a in assets),
            'attention':sum(a['health']=='attention' for a in assets),'critical':sum(a['health']=='critical' for a in assets),
            'unknown':sum(a['health']=='unknown' for a in assets),'open_orders':sum(o['status']!='resolved' for o in orders),
            'mean_rul':sum(ruls)/len(ruls) if ruls else None,'models_active':len(models['active'])}
        return {'mode':mode,'source_label':'Synthetic demonstration fixtures; no measured model performance' if mode=='demo' else 'NASA C-MAPSS historical benchmark replay; measured CPU baseline results',
            'summary':summary,'assets':assets,'orders':orders,'runs':models['runs'],'jobs':jobs,'active':models['active'],
            'last_successful_run':next((j for j in jobs if j['status']=='complete'),None)}

    def analytics(self,mode='benchmark',**filters):
        data=self.dashboard(mode,**filters)
        data['monitoring_notes']=['Anomaly score is an empirical percentile against fit-only early-cycle reference windows, not a failure probability.',
            'Calibration is for capped RUL and a declared historical prefix distribution; thresholds are engineering assumptions.',
            'No deployed data-drift detector is claimed. Monitoring shows measured errors, quality gates and model/version history.']
        return data

    def asset_detail(self,asset,mode='benchmark',limit=60):
        if not re.fullmatch(r'FD00[1-4]/(?:test|demo)/\d{3,6}',asset):raise ValueError('Invalid engine identifier')
        item=next((a for a in self.assets(mode) if a['asset']==asset),None)
        if not item:raise ValueError('Engine is unavailable in this data mode; evaluate and activate a model first')
        from .baseline import SENSOR_SCHEMA
        if mode=='demo':
            detail=demo_detail(item);history=detail['history'][-limit:];base=detail['sensor_history'][-limit:]
            sensors=[]
            for index in (1,2,6,7,10):
                info=dict(SENSOR_SCHEMA[index]);series=[{'cycle':p['cycle'],'value':round(p['value']*(1+index*.03)+math.sin(p['cycle']/8+index)*.08,4)} for p in base]
                sensors.append(info|{'unit':'normalized synthetic signal','value':series[-1]['value'],'series':series})
            settings=[{'name':f'Operating setting {i+1}','value':v,'unit':'synthetic example'} for i,v in enumerate((0,0,100))]
        else:
            bundle=self.bundle(item['bundle_id']);engine=self.engines_for(item['subset'])[item['engine_id']]
            ends=list(range(max(0,len(engine['raw'])-limit),len(engine['raw'])))
            history=bundle.engine_predictions(engine,ends,self.repo.settings().uncertainty_gate)
            frozen=self.artifact_dir/'aeroguard/runs'/item['bundle_id']/'evaluation_targets.json'
            evaluation=self.artifact_dir/'aeroguard/runs'/item['bundle_id']/'evaluation.json'
            if frozen.exists() and evaluation.exists():
                report=json.loads(evaluation.read_text());test_name=f'test_{item["subset"]}.txt'
                if report['source_hashes'].get(test_name)!=file_hash(self.data_dir/test_name):raise Conflict('Sensor history differs from the frozen evaluation corpus')
                truth={v['engine_id']:v['true_rul'] for v in json.loads(frozen.read_text())}
                for point,end in zip(history,ends):point['true_rul']=truth[item['engine_id']]+len(engine['raw'])-1-end
            sensors=[]
            for index in range(21):
                info=dict(SENSOR_SCHEMA[index]);series=[]
                for end in ends:
                    v=float(engine['raw'][end,3+index]);series.append({'cycle':end+1,'value':v if math.isfinite(v) else None})
                sensors.append(info|{'value':series[-1]['value'],'series':series})
            settings=[{'name':f'Operating setting {i+1}','value':float(engine['raw'][-1,i]),'unit':'raw setting'} for i in range(3)]
        namespace='demo' if mode=='demo' else 'benchmark:'+item['bundle_id']
        with self.repo.connect() as db:rows=db.execute('SELECT cycle,decision,created_at FROM events WHERE namespace=? AND asset=? ORDER BY cycle DESC LIMIT 60',(namespace,asset)).fetchall()
        decisions=[json.loads(r['decision'])|{'cycle':r['cycle'],'timestamp':r['created_at']} for r in rows]
        return {'asset':item,'history':history,'sensors':sensors,'operating_settings':settings,'decisions':decisions,
            'orders':[o for o in self.repo.orders(mode) if o['asset']==asset],
            'source_label':'Synthetic demonstration signals; not physical sensor readings' if mode=='demo' else 'Observed historical test prefixes; official endpoint truth used only for retrospective evaluation',
            'demo':mode=='demo','image':'/assets/turbofan-hero.png'}

    @contextmanager
    def stage(self,job_id,name,subset='ALL'):
        stage_id=self.repo.stage_start(job_id,name,subset);tick=time.perf_counter();result=StageResult()
        try:
            yield result
        except BaseException as error:
            self.repo.stage_finish(stage_id,time.perf_counter()-tick,result.records,result.detail,error)
            raise
        else:self.repo.stage_finish(stage_id,time.perf_counter()-tick,result.records,result.detail)

    def skip_stage(self,job_id,name,subset,reason):
        stage_id=self.repo.stage_start(job_id,name,subset)
        self.repo.stage_finish(stage_id,0,0,reason)
        with self.repo.connect() as db:db.execute("UPDATE ag_stages SET status='skipped' WHERE id=?",(stage_id,))

    def start(self,kind,payload,key=None):
        if kind not in {'train','replay','prepare'}:raise ValueError('Invalid pipeline kind')
        validated=TrainRequest.model_validate(payload) if kind=='train' else PipelineRequest.model_validate(payload)
        public=validated.model_dump();key=key or uuid.uuid4().hex
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',key):raise ValueError('Invalid idempotency key; use at most 128 letters, digits or ._:-')
        with self.lock:
            previous=self.repo.job_by_key(key)
            if previous:
                original={k:v for k,v in previous['configuration'].items() if not k.startswith('_')}
                if previous['kind']!=kind or original!=public:raise Conflict('Idempotency key was already used for different client inputs')
                return {'job_id':previous['id'],'status':previous['status'],'reused':True}
            configuration=public|{'_settings':self.repo.settings().model_dump(),'_active':self.active()}
            job_id,created=self.repo.create_job(kind,configuration,key)
            if created:self.futures[job_id]=self.pool.submit(self._execute,job_id,kind,configuration)
        return {'job_id':job_id,'status':self.repo.job(job_id)['status'],'reused':not created}

    def retry(self,job_id,key=None):
        job=self.repo.job(job_id)
        if job['status'] not in {'failed','interrupted'}:raise Conflict('Only failed or interrupted jobs can be retried')
        payload={k:v for k,v in job['configuration'].items() if not k.startswith('_')}
        return self.start(job['kind'],payload,key)

    def _execute(self,job_id,kind,configuration):
        self.repo.update_job(job_id,status='running',started_at=now())
        logger.info(json.dumps({'event':'job_started','job_id':job_id,'kind':kind}))
        try:
            result=self._train(job_id,configuration) if kind=='train' else self._prepare(job_id,configuration) if kind=='prepare' else self._pipeline(job_id,configuration)
            self.repo.update_job(job_id,status='complete',finished_at=now(),result=result)
            logger.info(json.dumps({'event':'job_complete','job_id':job_id}))
        except BaseException as error:
            self.repo.update_job(job_id,status='failed',finished_at=now(),error=f'{type(error).__name__}: {error}')
            logger.exception(json.dumps({'event':'job_failed','job_id':job_id,'error':str(error)}))

    def _prepare(self,job_id,configuration):
        results=[]
        for subset in configuration['subsets']:
            with self.stage(job_id,'Validation and cleaning',subset) as stage:
                item=next(d for d in self.data_summary()['datasets'] if d['subset']==subset)
                if not item['ready']:raise ValueError(f'{subset} data is missing or invalid. Import its train/test/RUL files.')
                stage.records=item['rows'];stage.detail='Schema, identities and truth-vector mapping validated; held-out values never fit preprocessing.'
            with self.stage(job_id,'Feature engineering',subset) as stage:
                result=self.datasets.prepare([subset],seed=configuration.get('split_seed',42),window=configuration.get('window',30),cap=configuration.get('cap',125));results.append(result)
                stage.records=item['train_engines'];stage.detail='Fit-only operating regimes and causal preprocessing audit saved.'
        return {'prepared':results}

    def _train(self,job_id,configuration):
        from .baseline import train_model,evaluate_model
        from .contracts import Settings
        settings=Settings.model_validate(configuration['_settings']);results=[]
        for subset in configuration['subsets']:
            with self.stage(job_id,'Dataset ingestion',subset) as stage:
                item=next(d for d in self.data_summary()['datasets'] if d['subset']==subset)
                if not item['ready']:raise ValueError(f'{subset} is not ready. Import valid train/test/RUL files before training.')
                stage.records=item['rows'];stage.detail='Historical benchmark files read from the configured local corpus.'
            self._prepare(job_id,{'subsets':[subset],'split_seed':configuration['split_seed'],'window':configuration['window'],'cap':configuration['cap']})
            config={k:v for k,v in configuration.items() if not k.startswith('_')}|{'subset':subset}
            with self.stage(job_id,'Model training',subset) as stage:
                result=train_model(config,self.data_dir,self.artifact_dir,lambda name,records,detail:logger.info(json.dumps({'event':'training_progress','job_id':job_id,'subset':subset,'phase':name,'records':records,'detail':detail})))
                if configuration['track_mlflow'] and result.get('tracking_status')!='complete':
                    raise RuntimeError('Model saved locally, but required MLflow tracking failed: '+result.get('tracking_error','Unknown tracking error'))
                stage.records=result.get('training_samples',0)
                stage.detail='Histogram gradient-boosting RUL and independent healthy-prefix Isolation Forest fitted; calibration engines remain reserved.'
            run_id=result['run_id']
            with self.stage(job_id,'Model selection',subset) as stage:
                eligible,reason=self.promotion_gate(result,settings)
                if configuration['promote'] and eligible:self.promote(run_id,settings=settings)
                stage.records=1;stage.detail=reason+(' · promoted before official evaluation' if self.active().get(subset)==run_id else ' · candidate retained; active model unchanged')
            with self.stage(job_id,'Anomaly detection',subset) as stage:
                bundle=self.bundle(run_id);engines=self.engines_for(subset)
                # Actual scores are computed here; evaluator freezes all RUL predictions again before truth is read.
                first=engines[sorted(engines)[0]]
                probe=bundle.engine_predictions(first,[len(first['raw'])-1],settings.uncertainty_gate)
                stage.records=len(probe);stage.detail='Actual anomaly/RUL boundary check on one held-out endpoint; full inference follows.'
            with self.stage(job_id,'RUL inference and evaluation',subset) as stage:
                report=evaluate_model(run_id,self.data_dir,self.artifact_dir)
                stage.records=report.get('metrics',{}).get('n_engines',item['test_engines']);stage.detail='Official test endpoint predictions frozen before labels; raw and capped metrics saved separately.'
            if self.active().get(subset)==run_id:
                replay=self._replay(job_id,run_id,subset,configuration['replay_cycles'],settings.model_dump());result['replay']=replay
            else:
                self.skip_stage(job_id,'Rule-based triage',subset,'Candidate is not active; no maintenance decisions are generated.')
                self.skip_stage(job_id,'Work-order creation',subset,'Candidate is not active; previous fleet work orders remain unchanged.')
            with self.stage(job_id,'Experiment tracking and monitoring',subset) as stage:
                if report.get('tracking_error'):raise RuntimeError('Official evaluation saved, but MLflow metric logging failed: '+report['tracking_error'])
                from .baseline import model_folder
                monitoring={'job_id':job_id,'run_id':run_id,'timestamp':now(),'evaluation':report.get('metrics'),
                    'development_gate':self.promotion_gate(result,settings)[1],'active':self.active().get(subset)==run_id,
                    'drift_detector':'not implemented; no operational drift claim','mlflow_run_id':result.get('mlflow_run_id')}
                write_json(model_folder(run_id,self.artifact_dir)/'monitoring.json',monitoring)
                self.repo.audit('model',run_id,'trained_and_evaluated',monitoring)
                stage.records=1;stage.detail='Actual run, artifacts, metric history and MLflow tracking reference recorded.'
            results.append({'run_id':run_id,'subset':subset,'metrics':report.get('metrics'),'active':self.active().get(subset)==run_id})
        return {'runs':results}

    def _replay(self,job_id,run_id,subset,cycles,settings):
        store=Store(self.repo.store.path,policy=settings['policy']);namespace='benchmark:'+run_id
        bundle=self.bundle(run_id);engines=self.engines_for(subset);packets=[]
        with self.stage(job_id,'Anomaly detection and RUL inference',subset) as stage:
            for unit,engine in engines.items():
                asset=f'{subset}/test/{unit:03d}'
                with self.repo.connect() as db:previous=db.execute('SELECT state FROM states WHERE namespace=? AND asset=? AND lifecycle=?',(namespace,asset,'1')).fetchone()
                committed=json.loads(previous[0])['last_cycle'] if previous else 0
                ends=[end for end in range(max(0,len(engine['raw'])-cycles),len(engine['raw'])) if end+1>committed]
                if ends:packets.extend(bundle.engine_predictions(engine,ends,settings['uncertainty_gate']))
            stage.records=len(packets);stage.detail='Each prediction uses only its observed sensor prefix. No endpoint ground truth enters triage.'
        before=sum(o['namespace']==namespace for o in self.repo.orders('benchmark'))
        with self.stage(job_id,'Rule-based triage',subset) as stage:
            for packet in packets:
                decision=store.process(packet,namespace)
                if decision.get('order_id') and not decision['replayed']:
                    self.repo.audit('order',decision['order_id'],'automatic_rule',{'cycle':packet['cycle'],'decision':decision})
            stage.records=len(packets);stage.detail='Quality-first Python rules, trend/anomaly persistence and hysteresis; every decision is immutable.'
        with self.stage(job_id,'Work-order creation',subset) as stage:
            count=sum(o['namespace']==namespace for o in self.repo.orders('benchmark'));delivered=store.dispatch_local()
            stage.records=count-before;stage.detail=f'{count-before} new simulated orders committed with triage; {delivered} revisions acknowledged by the local outbox. No external dispatch.'
        return {'processed_cycles':len(packets),'new_orders':count-before}

    def _pipeline(self,job_id,configuration):
        if configuration['mode']=='demo':
            fixtures=[a for a in demo_assets() if a['subset'] in configuration['subsets']]
            with self.stage(job_id,'Dataset ingestion','DEMO') as stage:stage.records=len(fixtures);stage.detail='Explicit synthetic fixture packets; no physical stream or trained-model claims.'
            with self.stage(job_id,'Validation and cleaning','DEMO') as stage:stage.records=len(fixtures);stage.detail='Fixture identities and finite JSON packet structure validated.'
            self.skip_stage(job_id,'Feature engineering','DEMO','Demo sensor histories are illustrative; no fitted preprocessing is claimed.')
            self.skip_stage(job_id,'Anomaly detection','DEMO','Demo scores are declared fixture inputs, not live model outputs.')
            self.skip_stage(job_id,'RUL inference','DEMO','Demo RUL values are declared illustrative inputs, not benchmark predictions.')
            with self.stage(job_id,'Rule-based triage','DEMO') as stage:
                reused=0
                for packet in fixtures:reused+=int(self.repo.store.process(packet,'demo')['replayed'])
                stage.records=len(fixtures);stage.detail=f'{reused} already committed fixture endpoints reused without duplicate orders.'
            with self.stage(job_id,'Work-order creation','DEMO') as stage:
                stage.records=0;stage.detail='Previously seeded simulated work orders retained; duplicates suppressed.'
            with self.stage(job_id,'Experiment tracking and monitoring','DEMO') as stage:
                stage.records=1;stage.detail='Synthetic execution metadata recorded in local job history; no MLflow training run fabricated.'
            return {'mode':'demo','packets':len(fixtures),'new_orders':0}
        results=[]
        for subset in configuration['subsets']:
            with self.stage(job_id,'Dataset ingestion',subset) as stage:
                item=next(d for d in self.data_summary()['datasets'] if d['subset']==subset)
                if not item['ready']:raise ValueError(f'{subset} data is missing or invalid')
                run_id=configuration['_active'].get(subset)
                if not run_id:raise ValueError(f'{subset} has no active model. Train and promote a model in Model Registry first.')
                stage.records=item['rows'];stage.detail='Using the model/version snapshot captured when the job was submitted.'
            with self.stage(job_id,'Validation and cleaning',subset) as stage:
                self.bundle(run_id);stage.records=item['test_engines'];stage.detail='Active trusted bundle integrity verified; raw test file schema validated.'
            with self.stage(job_id,'Feature engineering',subset) as stage:
                self.engines_for(subset);stage.records=item['test_engines'];stage.detail='Observed histories loaded; fit-only regime normalization and causal windows computed during inference.'
            result=self._replay(job_id,run_id,subset,configuration['cycles'],configuration['_settings'])
            with self.stage(job_id,'Experiment tracking and monitoring',subset) as stage:
                self.repo.audit('pipeline',job_id,'replayed',result|{'run_id':run_id});stage.records=1;stage.detail='Model version, decision counts and local delivery outcomes recorded.'
            results.append(result|{'run_id':run_id})
        return {'runs':results}

    def notifications(self,mode):
        assets=self.assets(mode);items=[]
        for a in sorted(assets,key=lambda a:{'critical':0,'attention':1,'unknown':2,'healthy':3}[a['health']]):
            if a['health'] not in {'critical','attention'}:continue
            decision=a.get('decision') or {}
            items.append({'id':a['asset']+'/'+str(a['cycle']),'title':f"{a['subset']}-{a['engine_id']:03d} · {a['health']}",
                'message':decision.get('reason','Review engine evidence'),'asset':a['asset'],'severity':a['health'],'timestamp':a.get('timestamp')})
        for job in self.repo.jobs():
            if job['status'] in {'failed','interrupted'}:items.append({'id':job['id'],'title':'Pipeline requires attention','message':job['error'],'severity':'attention','timestamp':job['finished_at']})
        return {'items':items[:20]}
