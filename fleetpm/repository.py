"""Artifact registry shared by the notebook and local dashboard."""

import json
from pathlib import Path
import re

from .config import ARTIFACT_DIR, SUBSETS, file_hash, write_json

RUN_PATTERN=re.compile(r"FD00[1-4]-[a-f0-9]{16}")


def run_folder(run_id,artifact_dir=ARTIFACT_DIR):
    if not RUN_PATTERN.fullmatch(run_id): raise ValueError("Invalid run ID")
    parent=(Path(artifact_dir)/"runs").resolve()
    folder=(parent/run_id).resolve()
    if folder.parent!=parent or not folder.is_dir(): raise ValueError("Run not found")
    return folder


def list_runs(artifact_dir=ARTIFACT_DIR):
    root=Path(artifact_dir)/"runs"; output=[]
    if not root.exists(): return output
    for folder in root.iterdir():
        if not folder.is_dir() or not RUN_PATTERN.fullmatch(folder.name): continue
        path=folder/"metadata.json"
        if not path.is_file(): continue
        try:
            item=json.loads(path.read_text())
            for name in ("evaluation","progress","replay"):
                extra=folder/(name+".json")
                if extra.exists(): item[name]=json.loads(extra.read_text())
            output.append(item)
        except (ValueError,OSError): continue
    return sorted(output,key=lambda x:x.get("created_at",""),reverse=True)


def active_runs(artifact_dir=ARTIFACT_DIR):
    path=Path(artifact_dir)/"active.json"
    return json.loads(path.read_text()) if path.exists() else {}


def activate_run(run_id,artifact_dir=ARTIFACT_DIR):
    folder=run_folder(run_id,artifact_dir)
    metadata=json.loads((folder/"metadata.json").read_text())
    if metadata.get("training_status")!="complete" or metadata.get("smoke_only"):
        raise ValueError("Only completed, non-smoke bundles can be used for local replay")
    manifest=json.loads((folder/"bundle_hashes.json").read_text())
    required={"model.pt","preprocessing.joblib","anomaly.joblib","residual.json","calibration.json","config.json","split.json","folds.json"}
    if not required<=manifest.keys(): raise ValueError("Bundle is incomplete")
    for filename,digest in manifest.items():
        if Path(filename).name!=filename or file_hash(folder/filename)!=digest:
            raise ValueError("Bundle integrity check failed")
    active=active_runs(artifact_dir)
    active[metadata["subset"]]=run_id
    write_json(Path(artifact_dir)/"active.json",active)
    return active


def predictions_for_active(artifact_dir=ARTIFACT_DIR):
    result=[]
    for subset,run_id in active_runs(artifact_dir).items():
        if subset not in SUBSETS: continue
        path=run_folder(run_id,artifact_dir)/"predictions.json"
        if path.exists(): result.extend(json.loads(path.read_text()))
    return result
