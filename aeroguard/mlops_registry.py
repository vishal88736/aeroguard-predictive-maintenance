"""Real MLflow model versions with the complete, verified inference contract."""

import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, Schema

from fleetpm.config import ROOT, file_hash, write_json
from fleetpm.store import now
from .baseline import BaselineBundle, model_folder, verify_bundle
from .datasets import COLUMN_NAMES
from .mlops_data import digest_json

MLFLOW_LOCK = threading.RLock()


class BaselinePythonModel(mlflow.pyfunc.PythonModel):
    """Accept raw observed C-MAPSS rows, return one endpoint per engine.

    The artifact includes normalization, RUL, independent anomaly detection and
    calibration. Neither targets nor engineered, externally normalized features
    are accepted. Full observed histories, starting at cycle one, are required.
    """

    def load_context(self, context):
        self.bundle = BaselineBundle(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame) or list(model_input.columns) != COLUMN_NAMES:
            raise ValueError("Provide the 26 documented raw sensor/settings columns in order")
        values = model_input.to_numpy(dtype=np.float64)
        if not len(values) or not np.isfinite(values[:, :5]).all() or np.isinf(values[:, 5:]).any():
            raise ValueError("Invalid raw prediction records")
        if (values[:, :2] <= 0).any() or (values[:, :2] != np.floor(values[:, :2])).any():
            raise ValueError("Engine and cycle IDs must be positive integers")
        if model_input.duplicated(["unit", "cycle"]).any():
            raise ValueError("Duplicate engine/cycle records are not allowed")
        if values[:, 1].max() > 10000 or len(values) > 2_000_000:
            raise ValueError("Prediction input exceeds the benchmark allocation limit")
        result = []
        allocated = 0
        for unit in sorted(np.unique(values[:, 0]).astype(int)):
            part = values[values[:, 0] == unit]
            part = part[np.argsort(part[:, 1])]
            if int(part[0, 1]) != 1:
                raise ValueError("Provide observed history starting at cycle one, not an isolated final window")
            length = int(part[-1, 1])
            allocated += length
            if allocated > 2_000_000:
                raise ValueError("Prediction input exceeds the dense-cycle allocation limit")
            raw = np.full((length, 24), np.nan, dtype=np.float32)
            raw[part[:, 1].astype(int) - 1] = part[:, 2:]
            engine = {"unit": unit, "raw": raw, "cycles": np.arange(1, length + 1)}
            packet = self.bundle.engine_predictions(engine, [length - 1])[0]
            result.append({"unit": unit, **{k: packet[k] for k in ("cycle", "rul", "lower", "upper", "anomaly", "quality_valid")}})
        frame = pd.DataFrame(result)
        for column in ("rul", "lower", "upper", "anomaly"):
            frame[column] = frame[column].astype(float)
        return frame


class ModelRegistry:
    def __init__(self, artifact_dir):
        self.artifact_dir = Path(artifact_dir)
        self.root = self.artifact_dir / "mlops" / "registry"
        self.tracking_root = (self.artifact_dir / "aeroguard").resolve()
        self.uri = "sqlite:///" + str(self.tracking_root / "mlflow.db")

    def client(self):
        self.tracking_root.mkdir(parents=True, exist_ok=True)
        return MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)

    def get(self, run_id):
        model_folder(run_id, self.artifact_dir)
        path = self.root / (run_id + ".json")
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        version = self.client().get_model_version(record["name"], record["version"])
        if version.tags.get("aeroguard.run_id") != run_id:
            raise ValueError("MLflow registry identity differs from the saved model record")
        return record

    def list(self):
        result = []
        if self.root.exists():
            for path in sorted(self.root.glob("AG-FD00*.json")):
                try:
                    record = self.get(path.stem)
                    model = self.client().get_registered_model(record["name"])
                    result.append(record | {"aliases": {k: str(v) for k, v in model.aliases.items()}})
                except (OSError, ValueError, MlflowException) as error:
                    result.append({"run_id": path.stem, "status": "error", "error": str(error)})
        return result

    def register(self, run_id, source_path, lineage=None):
        with MLFLOW_LOCK:
            folder = model_folder(run_id, self.artifact_dir)
            manifest = verify_bundle(folder)
            metadata = json.loads((folder / "metadata.json").read_text())
            if metadata.get("training_status") != "complete":
                raise ValueError("Only complete verified models may be registered")
            try:
                existing = self.get(run_id)
            except MlflowException as error:
                if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
                existing = None  # Explicit initialization may rebind a relocated archive to its new store.
            bundle_digest = digest_json(manifest)
            if existing:
                if existing["bundle_sha256"] != bundle_digest:
                    raise ValueError("Registered immutable model bundle changed")
                return existing | {"reused": True}
            source_path = Path(source_path)
            if file_hash(source_path) != metadata["source_sha256"]:
                raise ValueError("Registration example must come from the exact training snapshot")
            client = self.client()
            experiment = client.get_experiment_by_name("AeroGuard-CMAPSS-baseline")
            if experiment is None:
                experiment_id = client.create_experiment("AeroGuard-CMAPSS-baseline", artifact_location=(self.tracking_root / "mlflow-artifacts").as_uri())
            else:
                experiment_id = experiment.experiment_id
            tracking_id = metadata.get("mlflow_run_id")
            if tracking_id:
                try:
                    client.get_run(tracking_id)
                except MlflowException as error:
                    if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                        raise
                    tracking_id = None
            if not tracking_id:
                run = client.create_run(experiment_id, tags={"mlflow.runName": run_id, "aeroguard.run_id": run_id,
                    "registration_only": "true", "original_training_run_id": metadata.get("mlflow_run_id") or "not tracked"})
                tracking_id = run.info.run_id
            name = "AeroGuard-" + metadata["subset"] + "-RUL"
            tags = {"aeroguard.run_id": run_id, "bundle_sha256": bundle_digest, "family": metadata["family"]}
            for item in client.search_model_versions(f"name='{name}'"):
                if item.tags.get("aeroguard.run_id") == run_id and item.tags.get("bundle_sha256") == bundle_digest:
                    record = {"run_id": run_id, "name": name, "version": str(item.version), "model_uri": item.source,
                              "mlflow_run_id": tracking_id, "bundle_sha256": bundle_digest, "created_at": now(), "status": "registered"}
                    self.root.mkdir(parents=True, exist_ok=True)
                    write_json(self.root / (run_id + ".json"), record)
                    return record | {"reused": True}
            try:
                client.get_registered_model(name)
            except MlflowException as error:
                if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
                client.create_registered_model(name, tags={"subset": metadata["subset"], "application": "AeroGuard"})
            example = pd.read_csv(source_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
            first = example["unit"].min()
            example = example[example["unit"] == first].head(30).astype(float)
            signature = ModelSignature(
                inputs=Schema([ColSpec("double", column) for column in COLUMN_NAMES]),
                outputs=Schema([ColSpec("long", "unit"), ColSpec("long", "cycle"),
                                *[ColSpec("double", v) for v in ("rul", "lower", "upper", "anomaly")], ColSpec("boolean", "quality_valid")]))
            old_tracking, old_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
            try:
                mlflow.set_tracking_uri(self.uri)
                mlflow.set_registry_uri(self.uri)
                self.root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".package-", dir=self.root) as temporary:
                    temporary = Path(temporary)
                    inference = temporary / "bundle"
                    inference.mkdir()
                    for filename in {*manifest, "bundle_hashes.json", "metadata.json"}:
                        shutil.copy2(folder / filename, inference / filename)
                    code = temporary / "code"
                    code.mkdir()
                    for package in ("aeroguard", "fleetpm"):
                        shutil.copytree(ROOT / package, code / package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                    with mlflow.start_run(run_id=tracking_id):
                        info = mlflow.pyfunc.log_model(
                            name="inference", python_model=BaselinePythonModel(), artifacts={"bundle": str(inference)},
                            code_paths=[str(code / "aeroguard"), str(code / "fleetpm")], signature=signature,
                            input_example=example, pip_requirements=[f"{p}=={importlib.metadata.version(p)}" for p in
                                ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")],
                            metadata={"aeroguard.run_id": run_id, "input_contract": "raw observed 26-column histories", "external_dispatch": False})
                        if lineage:
                            mlflow.log_dict(lineage, "lineage.json")
                            mlflow.set_tags({k: str(v) for k, v in lineage.items() if isinstance(v, (str, int, float))})
                        mlflow.log_metrics({"development_" + k: float(v) for k, v in metadata["validation_metrics"].items() if isinstance(v, (float, int)) and np.isfinite(v)})
                # Load and execute the actual packaged model before registration.
                loaded = mlflow.pyfunc.load_model(info.model_uri)
                output = loaded.predict(example)
                from fleetpm.features import load_engines
                raw = load_engines(source_path)[int(first)]
                length = int(example["cycle"].max())
                prefix = dict(raw, raw=raw["raw"][:length], cycles=raw["cycles"][:length])
                expected = BaselineBundle(folder).engine_predictions(prefix, [length - 1])[0]
                if not np.isclose(output.iloc[0]["rul"], expected["rul"], rtol=1e-6, atol=1e-6):
                    raise ValueError("MLflow packaged inference differs from the original bundle")
                version = client.create_model_version(name, info.model_uri, run_id=tracking_id, tags=tags,
                                                      model_id=info.model_id, await_creation_for=60)
                record = {"run_id": run_id, "name": name, "version": str(version.version), "model_uri": info.model_uri,
                          "mlflow_run_id": tracking_id, "bundle_sha256": bundle_digest, "created_at": now(),
                          "status": "registered", "input_contract": "raw observed 26-column histories", "packaged_inference_verified": True}
                write_json(self.root / (run_id + ".json"), record)
                client.set_registered_model_alias(name, "challenger", version.version)
                return record | {"reused": False}
            except Exception:
                client.set_terminated(tracking_id, status="FAILED")
                raise
            finally:
                mlflow.set_tracking_uri(old_tracking)
                mlflow.set_registry_uri(old_registry)

    def aliases(self, run_id):
        record = self.get(run_id)
        if not record:
            return None
        return dict(self.client().get_registered_model(record["name"]).aliases)

    def restore_aliases(self, run_id, aliases):
        record = self.get(run_id)
        if not record:
            return
        client = self.client()
        current = client.get_registered_model(record["name"]).aliases
        for alias in set(current) - set(aliases):
            client.delete_registered_model_alias(record["name"], alias)
        for alias, version in aliases.items():
            client.set_registered_model_alias(record["name"], alias, str(version))

    def champion(self, run_id, previous_run_id=None):
        record = self.get(run_id)
        if not record:
            return False
        client = self.client()
        if previous_run_id and previous_run_id != run_id:
            previous = self.get(previous_run_id)
            if previous:
                client.set_registered_model_alias(record["name"], "previous", previous["version"])
        client.set_registered_model_alias(record["name"], "champion", record["version"])
        aliases = client.get_registered_model(record["name"]).aliases
        if str(aliases.get("challenger")) == record["version"]:
            client.delete_registered_model_alias(record["name"], "challenger")
        return True

    def track_research(self, catalog, run_id):
        """Track verified notebook evidence without pretending it is calibrated serving."""
        with MLFLOW_LOCK:
            run = catalog._find(run_id)
            manifest = run["manifest"]
            manifest_sha = file_hash(run["folder"] / "research_export.json")
            record_path = self.artifact_dir / "mlops" / "research-tracking" / (run_id + ".json")
            previous = json.loads(record_path.read_text()) if record_path.exists() else None
            if previous and previous["manifest_sha256"] == manifest_sha:
                self.client().get_run(previous["mlflow_run_id"])
                return previous | {"reused": True}
            client = self.client()
            name = "AeroGuard-CMAPSS-KAN"
            experiment = client.get_experiment_by_name(name)
            experiment_id = experiment.experiment_id if experiment else client.create_experiment(name, artifact_location=(self.tracking_root / "mlflow-artifacts").as_uri())
            tracking_id = previous["mlflow_run_id"] if previous else client.create_run(experiment_id, tags={"mlflow.runName": run_id,
                "architecture": manifest.architecture, "source": manifest.source, "maintenance_promotion": "blocked: independent calibration contract required"}).info.run_id
            try:
                client.log_param(tracking_id, "subset", manifest.subset)
                client.log_param(tracking_id, "variant", manifest.variant)
                client.log_param(tracking_id, "seed", manifest.seed)
                client.log_metric(tracking_id, "validation_rmse", run["record"]["best_validation_rmse"])
                for key, value in run["metrics"]["raw_target_metrics"].items():
                    if isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value):
                        client.log_metric(tracking_id, "official_test_" + key, float(value))
                prefix = "verified-exports/" + manifest_sha
                for relative in ["research_export.json", *manifest.files]:
                    parent = str(Path(relative).parent)
                    artifact_path = prefix if parent == "." else prefix + "/" + parent
                    client.log_artifact(tracking_id, str(run["folder"] / relative), artifact_path)
                client.set_tag(tracking_id, "selected_on_validation", str(manifest.selected_on_validation))
                client.set_tag(tracking_id, "explanation_status", manifest.explanation_status)
                client.set_tag(tracking_id, "manifest_sha256", manifest_sha)
                client.set_terminated(tracking_id, "FINISHED")
            except Exception:
                client.set_terminated(tracking_id, "FAILED")
                raise
            record = {"run_id": run_id, "mlflow_run_id": tracking_id, "subset": manifest.subset,
                      "manifest_sha256": manifest_sha, "source": manifest.source, "status": "tracked",
                      "created_at": now(), "explanations": len(run["explanations"]),
                      "maintenance_promotion": "blocked: independent calibration and compatible serving contract required"}
            write_json(record_path, record)
            return record | {"reused": False}
