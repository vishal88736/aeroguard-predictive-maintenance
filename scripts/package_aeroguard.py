"""Package source, selected corpus, trained artifacts and consistent local evidence.

No virtualenv, caches, credentials, provided third-party papers or live tracking
database is activated in the package. Historical MLflow is a separate evidence
backup; future runs create a tracking store local to the extracted project.
"""

from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fleetpm.config import ROOT,ARTIFACT_DIR,DATA_DIR,RUNTIME_DIR,file_hash


def snapshot_sqlite(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(str(source)) as origin,sqlite3.connect(str(target)) as destination:
        origin.backup(destination)


def main():
    output=ROOT/'output/aeroguard';output.mkdir(parents=True,exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True,exist_ok=True)
    tracking=ARTIFACT_DIR/'aeroguard/mlflow.db'
    if tracking.exists():
        snapshot_sqlite(tracking,output/'mlflow-history.sqlite')
        os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
        from mlflow import MlflowClient
        client=MlflowClient(tracking_uri='sqlite:///'+str(tracking.resolve()))
        tracked=[]
        for path in sorted((ARTIFACT_DIR/'aeroguard/runs').glob('*/metadata.json')):
            metadata=json.loads(path.read_text());run_id=metadata.get('mlflow_run_id')
            if not run_id:continue
            run=client.get_run(run_id)
            tracked.append({'model_run_id':metadata['run_id'],'mlflow_run_id':run_id,'status':run.info.status,
                'parameters':run.data.params,'metrics':run.data.metrics,'original_artifact_uri':run.info.artifact_uri})
        (output/'mlflow-run-export.json').write_text(json.dumps(tracked,indent=2)+'\n')
    paths={}
    def add(file,relative=None):
        file=Path(file)
        if file.is_file():paths[str(relative or file.relative_to(ROOT))]=file
    for name in ['README.md','VALIDATION.md','requirements.txt','requirements-app.txt','requirements-kan.txt','.env.example','.python-version','.gitignore','compose.mlops.yml','package.json','package-lock.json']:
        add(ROOT/name)
    for directory in ['aeroguard','fleetpm','frontend','web','notebooks','docs','tests','scripts','dags']:
        for file in (ROOT/directory).rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts and '.ipynb_checkpoints' not in file.parts:add(file)
    for file in DATA_DIR.glob('*'):
        if file.is_file() and file.suffix in {'.txt','.json','.md'}:add(file,Path('data/cmapss')/file.name)
    model_root=ARTIFACT_DIR/'aeroguard'
    for file in model_root.rglob('*'):
        if file.is_file() and file.suffix not in {'.db','.sqlite','.db-wal','.db-shm'} and file.name!='mlflow.db' and not file.name.endswith(('-wal','-shm','.tmp')):
            add(file,Path('artifacts/aeroguard')/file.relative_to(model_root))
    for file in (ARTIFACT_DIR/'mlops').rglob('*'):
        relative=file.relative_to(ARTIFACT_DIR/'mlops')
        if file.is_file() and relative.parts[0] not in {'registry','research-tracking'} and file.suffix in {'.json','.txt'}:
            add(file,Path('artifacts/mlops')/relative)
    # Only validated portable research exports; no giant resumable training state or temporary fixtures.
    from aeroguard.research import ResearchCatalog
    research=ResearchCatalog(ARTIFACT_DIR,DATA_DIR)
    for run in research.summary()['runs']:
        record=research._find(run['run_id']);folder=record['folder']
        for name in ['research_export.json',*record['manifest'].files]:
            add(folder/name,Path('artifacts/kan_bilstm')/run['subset']/run['run_id']/name)
    for file in output.glob('*'):
        if file.is_file() and file.name!='failure.png':add(file)
    add(ROOT/'output/website/measured-baseline-summary.json')
    for name in ['FINAL_ARCHITECTURE.md','PAPER_REVIEW.md','SOURCE_MANIFEST.json','README.md']:
        add(ROOT/'output/architecture'/name)
    for file in (ROOT/'output/architecture/diagrams').glob('*'):add(file)
    add(ROOT/'output/pdf/Research_Integrated_Fleet_Architecture.pdf')
    package=ROOT/'output/AeroGuard_Project.zip'
    with tempfile.TemporaryDirectory(prefix='aeroguard-package-',dir=RUNTIME_DIR) as temporary:
        state=Path(temporary)/'aeroguard.sqlite'
        if (RUNTIME_DIR/'aeroguard.sqlite').exists():
            snapshot_sqlite(RUNTIME_DIR/'aeroguard.sqlite',state)
            add(state,'runtime/aeroguard.sqlite')
        manifest={'created_at':datetime.now(timezone.utc).isoformat(),'application':'AeroGuard',
            'includes':['Application source and local assets','Selected local benchmark corpus with provenance','Trained CPU baseline artifacts and active versions','Unexecuted GPU paper/condition-gate notebook with final SHAP/LIME workflow','Validated research exports if user training has produced any','Consistent simulated work-order/job database snapshot','Measured results, tests, browser checks and MLflow history evidence'],
            'excludes':['Virtual environment and package caches','Actual .env or credentials','Original third-party supplied papers','Live MLflow tracking database at its application path'],
            'relocation_note':'Archived metadata retains original provenance/artifact paths. AeroGuard loads bundles from its configured local registry. Historical MLflow is exported under output/aeroguard. Initialize MLOps after relocation to register verified bundles in the new local tracking store without refitting; original training run IDs remain provenance.',
            'sha256':{name:file_hash(path) for name,path in sorted(paths.items())}}
        metadata=Path(temporary)/'PACKAGE_MANIFEST.json';metadata.write_text(json.dumps(manifest,indent=2)+'\n');add(metadata,'PACKAGE_MANIFEST.json')
        pending=package.with_suffix('.zip.tmp')
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name,path in sorted(paths.items()):
                assert '.venv' not in Path(name).parts and Path(name).name!='.env'
                archive.write(path,'AeroGuard/'+name)
        with zipfile.ZipFile(pending) as archive:assert archive.testzip() is None
        pending.replace(package)
    print(json.dumps({'package':str(package),'files':len(paths),'bytes':package.stat().st_size},indent=2))


if __name__=='__main__':main()
