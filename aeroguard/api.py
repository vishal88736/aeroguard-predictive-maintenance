"""FastAPI application with strict contracts and nonblocking local job execution."""

from contextlib import asynccontextmanager
from io import BytesIO
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
from typing import Annotated, Literal
from urllib.parse import urlparse
import uuid
import zipfile

from fastapi import FastAPI, File, Form, Header, Path as PathParameter, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from fleetpm.config import ROOT,DATA_DIR,ARTIFACT_DIR,RUNTIME_DIR,write_json
from .contracts import (DashboardResponse,Mode,OrderUpdate,PipelineRequest,PromoteRequest,
                        Settings,Subset,TrainRequest,LifecycleRequest,LifecycleSettings,
                        MLOpsBootstrap,RegisterRequest,ResearchTrackRequest,ReviewResolve)
from .repository import Conflict
from .services import Service
from .research import ResearchUnavailable

FRONTEND=ROOT/'frontend'


def error_response(message,code='invalid_request',status=400,details=None):
    value={'code':code,'message':message}
    if details is not None:value['details']=details
    return JSONResponse({'error':value},status_code=status)


def create_app(data_dir=DATA_DIR,artifact_dir=ARTIFACT_DIR,runtime_dir=RUNTIME_DIR):
    service=Service(data_dir,artifact_dir,runtime_dir)

    @asynccontextmanager
    async def lifespan(app):
        service.mlops.start_scheduler()
        yield
        service.close()

    app=FastAPI(title='AeroGuard API',version='1.0.0',description='NASA C-MAPSS historical replay, actual baseline models, and simulated maintenance records.',lifespan=lifespan)
    app.state.service=service
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]','testserver'])

    @app.middleware('http')
    async def local_boundary(request:Request,call_next):
        mutation=request.method in {'POST','PUT','PATCH','DELETE'}
        if mutation:
            origin=request.headers.get('origin')
            if request.headers.get('x-aeroguard-client')!='dashboard':return error_response('Use a same-origin AeroGuard client request.','forbidden',403)
            if origin and (urlparse(origin).netloc!=request.headers.get('host') or urlparse(origin).scheme!=request.url.scheme):return error_response('Cross-origin mutations are not allowed.','forbidden',403)
            try:size=int(request.headers.get('content-length','0'))
            except ValueError:return error_response('Invalid content length')
            if size>32*1024*1024:return error_response('Upload exceeds the 32 MB request limit.','request_too_large',413)
        response=await call_next(request)
        response.headers.update({'X-Content-Type-Options':'nosniff','Referrer-Policy':'same-origin',
            'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'"})
        if request.url.path.startswith('/api/'):response.headers['Cache-Control']='no-store'
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_payload(request,error):
        details=[{'location':list(e['loc']),'message':e['msg'],'type':e['type']} for e in error.errors()]
        return error_response('Check the highlighted request values.','validation_error',422,details)

    @app.exception_handler(Conflict)
    async def conflict(request,error):return error_response(str(error),'conflict',409)

    @app.exception_handler(ResearchUnavailable)
    async def research_unavailable(request,error):return error_response(str(error),'research_dependency_unavailable',503)

    @app.exception_handler(ValueError)
    async def invalid_operation(request,error):return error_response(str(error))

    @app.exception_handler(FileNotFoundError)
    async def missing_file(request,error):return error_response('Required data or artifact file is missing. Import the dataset or train a model first.','not_found',404)

    @app.exception_handler(Exception)
    async def failed_operation(request,error):
        logging.getLogger('aeroguard').exception('Unhandled API operation: %s',request.url.path)
        return error_response('The operation failed. Inspect the server log and pipeline details.','internal_error',500)

    def filters(subset,engine,severity,cycle_min,cycle_max):
        return {'subset':subset,'engine':engine,'severity':severity,'cycle_min':cycle_min,'cycle_max':cycle_max}

    @app.get('/api/health')
    def health():return service.health()

    @app.get('/api/dashboard',response_model=DashboardResponse)
    def dashboard(mode:Mode='benchmark',subset:Literal['ALL','FD001','FD002','FD003','FD004']='ALL',engine:str='',severity:str='ALL',cycle_min:int=Query(0,ge=0),cycle_max:int|None=Query(None,ge=0)):
        return service.dashboard(mode,**filters(subset,engine,severity,cycle_min,cycle_max))

    @app.get('/api/engines')
    def engines(mode:Mode='benchmark',subset:Literal['ALL','FD001','FD002','FD003','FD004']='ALL',engine:str='',severity:str='ALL',cycle_min:int=Query(0,ge=0),cycle_max:int|None=Query(None,ge=0)):
        items=service.assets(mode,**filters(subset,engine,severity,cycle_min,cycle_max))
        return {'items':items,'total':len(items),'source_label':'Synthetic fixtures' if mode=='demo' else 'Historical benchmark inference'}

    @app.get('/api/engines/{asset:path}')
    def engine_details(asset:str,mode:Mode='benchmark',limit:int=Query(60,ge=1,le=150)):
        return service.asset_detail(asset,mode,limit)

    @app.get('/api/analytics')
    def analytics(mode:Mode='benchmark',subset:Literal['ALL','FD001','FD002','FD003','FD004']='ALL',engine:str='',severity:str='ALL',cycle_min:int=Query(0,ge=0),cycle_max:int|None=Query(None,ge=0)):
        return service.analytics(mode,**filters(subset,engine,severity,cycle_min,cycle_max))

    @app.get('/api/data')
    def data():return service.data_summary()

    @app.get('/api/data/{subset}/preview')
    def preview(subset:Subset,partition:Literal['train','test','RUL']='train',limit:int=Query(20,ge=1,le=100)):
        return service.datasets.preview(subset,partition,limit)

    @app.post('/api/data/import')
    async def import_data(file:Annotated[UploadFile,File()],subset:Annotated[Subset,Form()]='FD001',partition:Annotated[Literal['train','test','RUL'],Form()]='train'):
        if any(j['status'] in {'queued','running'} for j in service.repo.jobs()):raise Conflict('Wait for the running job before importing dataset files.')
        content=await file.read(32*1024*1024+1)
        if len(content)>32*1024*1024:return error_response('File exceeds the 32 MB limit.','request_too_large',413)
        result=service.datasets.import_file(file.filename or 'upload',content,subset,partition)
        service.cache.pop('data_stamp',None)
        service.repo.audit('dataset',subset,'imported',{'filename':Path(file.filename or 'upload').name,'partition':partition,'bytes':len(content)})
        return result

    @app.post('/api/data/prepare',status_code=202)
    def prepare(payload:PipelineRequest,key:Annotated[str|None,Header(alias='Idempotency-Key')]=None):
        if payload.mode=='demo':raise ValueError('Synthetic fixtures do not train preprocessing. Select benchmark mode.')
        return service.start('prepare',payload.model_dump(),key)

    @app.get('/api/models')
    def models():return service.models()

    @app.get('/api/mlops')
    def mlops_overview():return service.mlops.overview()

    @app.put('/api/mlops/settings',response_model=LifecycleSettings)
    def mlops_settings(payload:LifecycleSettings):return service.mlops.save_settings(payload.model_dump())

    @app.post('/api/mlops/bootstrap',status_code=202)
    def mlops_bootstrap(payload:MLOpsBootstrap,key:Annotated[str|None,Header(alias='Idempotency-Key')]=None):
        return service.start('mlops_bootstrap',payload.model_dump(),key)

    @app.post('/api/mlops/register',status_code=202)
    def mlops_register(payload:RegisterRequest,key:Annotated[str|None,Header(alias='Idempotency-Key')]=None):
        return service.start('register',payload.model_dump(),key)

    @app.post('/api/mlops/research-track',status_code=202)
    def mlops_research_track(payload:ResearchTrackRequest,key:Annotated[str|None,Header(alias='Idempotency-Key')]=None):
        return service.start('research_track',payload.model_dump(),key)

    @app.post('/api/mlops/versions')
    async def mlops_upload(file:Annotated[UploadFile,File()],subset:Annotated[Subset,Form()],
                           label_status:Annotated[Literal['complete_run_to_failure','observed_prefix'],Form()],
                           provenance:Annotated[str,Form(min_length=1,max_length=1000)],
                           cohort:Annotated[str,Form(pattern=r'^[A-Za-z0-9_-]{1,48}$')]='incoming'):
        content=await file.read(32*1024*1024+1)
        return await run_in_threadpool(service.mlops.add_version,subset,file.filename or 'incoming.txt',content,label_status,provenance,cohort)

    @app.get('/api/mlops/versions/{version_id}')
    def mlops_version(version_id:str):return service.mlops.versions.get(version_id)

    @app.post('/api/mlops/run',status_code=202)
    def mlops_run(payload:LifecycleRequest,key:Annotated[str|None,Header(alias='Idempotency-Key')]=None):
        return service.start('lifecycle',payload.model_dump(),key)

    @app.post('/api/mlops/tick')
    def mlops_tick():return service.mlops.tick()

    @app.post('/api/mlops/reviews/{review_id}/resolve')
    def mlops_review(review_id:str,payload:ReviewResolve):return service.mlops.resolve_review(review_id,payload.notes)

    @app.get('/api/research')
    def research_catalog():return service.research.summary()

    @app.post('/api/research/import')
    async def import_research(file:UploadFile=File(...)):
        content=await file.read(32*1024*1024+1)
        if len(content)>32*1024*1024:return error_response('Research upload exceeds 32 MB.','request_too_large',413)
        result=await run_in_threadpool(service.research.import_archive,content)
        service.repo.audit('research_model',result['run']['run_id'],'export_imported',{'filename':Path(file.filename or 'bundle.zip').name,'reused':result['reused']})
        return result

    @app.get('/api/research/runs/{run_id}')
    def research_detail(run_id:str):return service.research.detail(run_id)

    @app.get('/api/research/runs/{run_id}/engines/{engine_id}')
    def research_engine(run_id:str,engine_id:int=PathParameter(ge=1,le=999999)):
        return service.research.engine_detail(run_id,engine_id)

    @app.get('/api/research/runs/{run_id}/predict')
    def research_predict(run_id:str,engine_id:int=Query(ge=1,le=999999),cycle:int|None=Query(None,ge=1,le=10000)):
        return service.research.predict(run_id,engine_id,cycle)

    @app.get('/api/research/runs/{run_id}/download')
    def research_download(run_id:str):
        content=service.research.archive(run_id)
        return Response(content,media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="{run_id}.zip"'})

    @app.post('/api/models/train',status_code=202)
    def train(payload:TrainRequest,key:Annotated[str|None,Header(alias='Idempotency-Key')]=None):
        return service.start('train',payload.model_dump(),key)

    @app.post('/api/models/promote')
    def promote(payload:PromoteRequest):return service.promote(payload.run_id,payload.rollback)

    @app.get('/api/models/{run_id}/download')
    def model_download(run_id:str):
        from .baseline import model_folder
        folder=model_folder(run_id,service.artifact_dir)
        if not (folder/'bundle_hashes.json').is_file():raise ValueError('Model bundle is incomplete')
        stream=BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(folder.iterdir()):
                if file.is_file():archive.write(file,run_id+'/'+file.name)
        return Response(stream.getvalue(),media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="{run_id}.zip"'})

    @app.get('/api/models/{run_id}')
    def model_details(run_id:str):return service.model_detail(run_id)

    @app.post('/api/pipeline/run',status_code=202)
    def pipeline(payload:PipelineRequest,key:Annotated[str|None,Header(alias='Idempotency-Key')]=None):
        return service.start('replay',payload.model_dump(),key)

    @app.get('/api/jobs')
    def jobs(mode:Mode|None=None):
        items=service.repo.jobs()
        if mode is not None:items=[j for j in items if j['configuration'].get('mode','benchmark')==mode]
        return {'items':items}

    @app.get('/api/jobs/{job_id}')
    def job(job_id:str):return service.repo.job(job_id)

    @app.post('/api/jobs/{job_id}/retry',status_code=202)
    def retry(job_id:str,key:Annotated[str|None,Header(alias='Idempotency-Key')]=None):return service.retry(job_id,key)

    @app.get('/api/orders')
    def orders(mode:Mode='benchmark',q:str='',status:str='ALL',priority:str='ALL'):
        items=service.repo.orders(mode)
        if q:items=[o for o in items if q.lower() in f"{o['id']} {o['asset']} {o.get('owner','')}".lower()]
        if status.upper()!='ALL':items=[o for o in items if o['status']==status.lower()]
        if priority.upper()!='ALL':items=[o for o in items if o['priority']==priority.upper()]
        return {'items':items}

    @app.get('/api/orders/{order_id}')
    def order(order_id:str,mode:Mode='benchmark'):
        record=next((o for o in service.repo.orders(mode) if o['id']==order_id),None)
        if not record:raise ValueError('Order not found in this data mode')
        return record|{'audit':service.repo.history('order',order_id)}

    @app.patch('/api/orders/{order_id}')
    def update_order(order_id:str,payload:OrderUpdate,mode:Mode='benchmark'):
        return service.repo.update_order(order_id,mode,payload)

    @app.get('/api/settings',response_model=Settings)
    def settings():return service.repo.settings()

    @app.put('/api/settings',response_model=Settings)
    def update_settings(payload:Settings):return service.repo.save_settings(payload.model_dump())

    @app.get('/api/notifications')
    def notifications(mode:Mode='benchmark'):return service.notifications(mode)

    @app.get('/api/search')
    def search(q:str=Query('',max_length=100),mode:Mode='benchmark'):
        if not q.strip():return {'engines':[],'orders':[]}
        assets=service.assets(mode,engine=q)[:8]
        orders=[o for o in service.repo.orders(mode) if q.lower() in f"{o['asset']} {o['id']} {o.get('owner','')}".lower()][:8]
        return {'engines':assets,'orders':orders}

    @app.get('/downloads/{name}')
    def downloads(name:str):
        paths={'notebook':ROOT/'notebooks/Transformer_KAN_BiLSTM_GPU.ipynb','kan-notebook':ROOT/'notebooks/Transformer_KAN_BiLSTM_GPU.ipynb',
            'legacy-notebook':ROOT/'notebooks/C_MAPSS_All_Subsets_CUDA.ipynb','requirements':ROOT/'requirements-app.txt',
            'research-requirements':ROOT/'requirements-kan.txt','kan-requirements':ROOT/'requirements-kan.txt',
            'integrated-requirements':ROOT/'requirements.txt','environment':ROOT/'.env.example',
            'architecture':ROOT/'docs/RESEARCH_WORKFLOW.md'}
        if name not in paths:return error_response('Download not found.','not_found',404)
        return FileResponse(paths[name],filename=paths[name].name,media_type='application/octet-stream')

    @app.get('/documents/{name}')
    def diagram(name:str):
        if name not in {'model.svg','system.svg','research.svg'}:return error_response('Diagram not found.','not_found',404)
        target=(ROOT/'docs/research-architecture.svg' if name=='research.svg' else
            ROOT/'docs/legacy-model-architecture.svg' if name=='model.svg' else ROOT/'docs/architecture.svg')
        return FileResponse(target,media_type='image/svg+xml')

    @app.get('/{path:path}',include_in_schema=False)
    def frontend(path:str):
        if path.startswith('api/'):return error_response('API route not found.','not_found',404)
        target=(FRONTEND/path).resolve()
        if not target.is_relative_to(FRONTEND.resolve()):return error_response('Not found.','not_found',404)
        if target.is_file():return FileResponse(target)
        if path.startswith(('assets/','vendor/')) or '.' in Path(path).name:return error_response('Static file not found.','not_found',404)
        index=FRONTEND/'index.html'
        return FileResponse(index) if index.is_file() else error_response('Frontend is still being prepared.','not_ready',503)

    return app
