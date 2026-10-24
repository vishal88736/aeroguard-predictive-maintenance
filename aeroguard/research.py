"""Validated notebook exports, real post-hoc explanations and optional neural inference.

No training or package installation occurs here. Research exports remain distinct
from the CPU baseline's different development/calibration protocol and triage.
"""

from io import BytesIO
import importlib.util
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import threading
import zipfile

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from fleetpm.config import file_hash
from fleetpm.metrics import regression_metrics
from .baseline import SENSOR_SCHEMA
from .repository import Conflict


RUN_PATTERN = re.compile(r"KAN-FD00[1-4]-(?:paper|conditioned_kan)-\d{1,10}-[a-f0-9]{12}")
EXPLANATION_PATTERN = re.compile(r"explanations/engine-\d{3,6}-cycle-\d{1,5}\.json")
REQUIRED_FILES = {"run.json", "config.json", "preprocessor.json", "split.json", "data_manifest.json",
                  "validation_endpoints.csv", "validation_predictions.csv", "history.csv", "environment.json",
                  "metrics.json", "test_predictions.csv", "test_predictions_sensor_only.csv",
                  "inference.ts", "inference_probes.npz", "selection_record.json"}
RAW_COLUMNS = ["engine_id", "cycle", *[f"setting_{i}" for i in range(1, 4)],
               *[f"sensor_{i}" for i in range(1, 22)]]
FEATURE_COLUMNS = RAW_COLUMNS[2:]


class ResearchUnavailable(RuntimeError):
    """A real optional dependency or local dataset is unavailable."""


class ArtifactDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(ge=0, le=64 * 1024 * 1024)


class ResearchManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    run_id: str = Field(pattern=r"^KAN-FD00[1-4]-(?:paper|conditioned_kan)-\d{1,10}-[a-f0-9]{12}$")
    subset: Literal["FD001", "FD002", "FD003", "FD004"]
    variant: Literal["paper", "conditioned_kan"]
    seed: int = Field(ge=0, le=2147483647)
    implementation: Literal["xu2025-transformer-kan-bilstm-v2"]
    architecture: Literal["Transformer-KAN-BiLSTM"]
    status: Literal["completed"]
    exported_at: str = Field(max_length=80)
    source: str = Field(max_length=200)
    selected_on_validation: bool
    selection: dict
    explanation_status: Literal["available", "not_generated"]
    explanation_files: list[str] = Field(max_length=500)
    files: dict[str, ArtifactDigest] = Field(min_length=15, max_length=515)


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"JSON artifact is too large: {path.name}")
    try:
        return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path.name}") from exc


def read_predictions(path):
    frame = pd.read_csv(path)
    required = {"engine_id", "cycle", "predicted_rul", "true_rul"}
    if not required.issubset(frame.columns) or frame.empty or len(frame) > 10000:
        raise ValueError("Prediction table must contain engine_id, cycle, predicted_rul and true_rul")
    if frame.engine_id.duplicated().any():
        raise ValueError("Official test export must contain one endpoint per engine")
    values = frame[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Prediction table contains invalid numbers")
    for column in ("engine_id", "cycle"):
        if np.any(frame[column] < 1) or np.any(frame[column] != np.floor(frame[column])):
            raise ValueError("Prediction identifiers and cycles must be positive integers")
    frame[["engine_id", "cycle"]] = frame[["engine_id", "cycle"]].astype(np.int64)
    return frame.sort_values("engine_id").reset_index(drop=True)


def sensor_frame(path):
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.empty or frame.shape[1] != 26:
        raise ValueError("Neural inference needs the exact 26-column C-MAPSS schema")
    frame.columns = RAW_COLUMNS
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Neural inference requires explicit, finite observations")
    if np.any(values[:, :2] < 1) or np.any(values[:, :2] != np.floor(values[:, :2])):
        raise ValueError("Engine IDs and cycles must be positive integers")
    frame[["engine_id", "cycle"]] = frame[["engine_id", "cycle"]].astype(np.int64)
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle observations")
    frame = frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    for _, group in frame.groupby("engine_id"):
        if not np.array_equal(group.cycle.to_numpy(), np.arange(1, len(group) + 1)) or len(group) > 10000:
            raise ValueError("Engine cycles must be consecutive from 1, up to 10000")
    return frame


def observed_window(frame, cycle, preprocessor, configuration):
    """Same fixed scaling/trailing mean as the notebook; excludes all future rows."""
    if frame.engine_id.nunique() != 1 or not 1 <= cycle <= len(frame):
        raise ValueError("Select one engine and an available observed cycle")
    features = frame.iloc[:cycle][FEATURE_COLUMNS].rolling(int(configuration["smoothing_window"]), min_periods=1).mean()
    values = (features[preprocessor["columns"]].to_numpy(dtype=np.float64)
              - np.asarray(preprocessor["center"])) / np.asarray(preprocessor["scale"])
    length = int(configuration["window"])
    window = values[-length:]
    if len(window) < length:
        window = np.concatenate([np.repeat(window[:1], length - len(window), axis=0), window])
    return np.ascontiguousarray(window[None], dtype=np.float32)


def validate_explanation(report, manifest, record, preprocessor, split, predictions):
    if (report.get("schema_version") != 1 or report.get("run_id") != manifest.run_id or
            report.get("subset") != manifest.subset or report.get("variant") != manifest.variant or
            report.get("model_sha256") != record.get("best_sha256") or
            report.get("target") != "current_cycle_predicted_rul" or report.get("units") != "cycles"):
        raise ValueError("Explanation provenance differs from the trained model")
    row = predictions[predictions.engine_id == report.get("engine_id")]
    if len(row) != 1 or int(row.iloc[0].cycle) != report.get("cycle"):
        raise ValueError("Explanation does not refer to an exported test endpoint")
    point = float(report.get("prediction", float("nan")))
    if not math.isfinite(point) or not math.isclose(point, float(row.iloc[0].predicted_rul), abs_tol=.02, rel_tol=.001):
        raise ValueError("Explanation prediction differs from the exported model prediction")
    groups = report.get("feature_groups", [])
    expected = [column for column in preprocessor["columns"] if column.startswith("sensor_")]
    if groups != expected:
        raise ValueError("Explanation groups must match the selected sensor columns")
    backgrounds = report.get("backgrounds", [])
    if (report.get("background_source") != "fit_engines_only" or not backgrounds or
            any(item.get("partition") != "fit" or item.get("engine_id") not in split["fit_engine_ids"] for item in backgrounds)):
        raise ValueError("Explanation backgrounds must come exclusively from fitting engines")
    shap, lime = report.get("shap", {}), report.get("lime", {})
    if shap.get("method") != "shap.KernelExplainer" or lime.get("method") != "lime.lime_base.LimeBase":
        raise ValueError("Only actual SHAP and LIME exports are accepted")
    for values in (shap.get("values", []), lime.get("coefficients", [])):
        if len(values) != len(groups) or not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError("Explanation vectors are invalid")
    expected_residual = point - float(shap["baseline"]) - sum(shap["values"])
    if not math.isfinite(expected_residual) or not math.isclose(expected_residual, float(shap["additivity_residual_cycles"]), abs_tol=1e-6):
        raise ValueError("SHAP additivity evidence is inconsistent")
    for key in ("local_prediction", "intercept"):
        if not math.isfinite(float(lime[key])):
            raise ValueError("LIME surrogate evidence is invalid")
    fidelity = lime.get("heldout_fidelity", {})
    if (not math.isfinite(float(fidelity.get("weighted_mae_cycles", float("nan")))) or
            fidelity["weighted_mae_cycles"] < 0 or
            fidelity.get("weighted_r2") is not None and not math.isfinite(float(fidelity["weighted_r2"]))):
        raise ValueError("LIME held-out fidelity is required")
    return report


class ResearchCatalog:
    def __init__(self, artifact_dir, data_dir):
        self.root = Path(artifact_dir) / "kan_bilstm"
        self.data_dir = Path(data_dir)
        self.lock = threading.RLock()
        self.models = {}

    def _inspect(self, folder):
        try:
            return self._inspect_checked(folder)
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError("Research export is missing or has inconsistent configuration/evaluation fields") from exc

    def _inspect_checked(self, folder):
        folder = Path(folder).resolve()
        if not folder.is_relative_to(self.root.resolve()):
            raise ValueError("Research export must be inside the configured research artifact directory")
        manifest = ResearchManifest.model_validate(read_json(folder / "research_export.json"))
        expected_id = f"KAN-{manifest.subset}-{manifest.variant}-{manifest.seed}-"
        if not manifest.run_id.startswith(expected_id) or not REQUIRED_FILES.issubset(manifest.files):
            raise ValueError("Research export identity or required artifacts are incomplete")
        explanations = {name for name in manifest.files if EXPLANATION_PATTERN.fullmatch(name)}
        if set(manifest.files) != REQUIRED_FILES | explanations or set(manifest.explanation_files) != explanations:
            raise ValueError("Unexpected or missing exported files")
        if (manifest.explanation_status == "available") != bool(explanations):
            raise ValueError("Explanation status does not match its files")
        for name, digest in manifest.files.items():
            target = folder / name
            if (target.is_symlink() or not target.resolve().is_relative_to(folder) or
                    not target.is_file() or target.stat().st_size != digest.bytes or file_hash(target) != digest.sha256):
                raise ValueError(f"Research artifact integrity failed: {name}")
        record, cfg = read_json(folder / "run.json"), read_json(folder / "config.json")
        signature = hashlib.sha256(json.dumps({"implementation": manifest.implementation, "config": cfg,
                                              "training_file_sha256": record.get("training_file_sha256")},
                                             sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if (record.get("status") != "completed" or record.get("implementation") != manifest.implementation or
                record.get("config") != cfg or cfg.get("subset") != manifest.subset or
                cfg.get("variant") != manifest.variant or cfg.get("seed") != manifest.seed or
                record.get("signature") != signature or not manifest.run_id.endswith(signature[:12])):
            raise ValueError("Training status/configuration differs from the export")
        window, smoothing, scale = cfg.get("window", 0), cfg.get("smoothing_window", 0), cfg.get("target_scale", 0)
        if (not isinstance(window, int) or not 1 <= window <= 200 or not isinstance(smoothing, int) or
                not 1 <= smoothing <= 200 or not math.isfinite(float(scale)) or scale <= 0 or
                cfg.get("normalization") not in {"minmax", "standard"}):
            raise ValueError("Invalid neural preprocessing configuration")
        preprocessor, split = read_json(folder / "preprocessor.json"), read_json(folder / "split.json")
        columns = preprocessor.get("columns", [])
        if (not columns or len(set(columns)) != len(columns) or not set(columns).issubset(FEATURE_COLUMNS) or
                len(preprocessor.get("center", [])) != len(columns) or len(preprocessor.get("scale", [])) != len(columns)):
            raise ValueError("Invalid neural feature schema")
        scaling = np.asarray([preprocessor["center"], preprocessor["scale"]], dtype=float)
        if not np.isfinite(scaling).all() or np.any(scaling[1] <= 0):
            raise ValueError("Invalid fitted feature scaling")
        if (preprocessor.get("fit_engine_ids") != split.get("fit_engine_ids") or
                not split.get("fit_engine_ids") or not split.get("validation_engine_ids") or
                set(split["fit_engine_ids"]) & set(split["validation_engine_ids"])):
            raise ValueError("Fitting and validation engine cohorts must be disjoint")
        predictions = read_predictions(folder / "test_predictions.csv")
        frozen = pd.read_csv(folder / "test_predictions_sensor_only.csv").sort_values("engine_id").reset_index(drop=True)
        if ("true_rul" in frozen.columns or
                not {"engine_id", "cycle", "predicted_rul"}.issubset(frozen.columns) or
                not np.array_equal(frozen[["engine_id", "cycle"]], predictions[["engine_id", "cycle"]]) or
                not np.allclose(frozen.predicted_rul, predictions.predicted_rul, rtol=1e-7, atol=1e-7)):
            raise ValueError("Evaluated predictions differ from the frozen sensor-only outputs")
        metrics = read_json(folder / "metrics.json")
        if metrics.get("protocol") != "official_test_last_window_per_engine":
            raise ValueError("Research report must identify the official endpoint protocol")
        computed = regression_metrics(predictions.true_rul, predictions.predicted_rul)
        reported = metrics.get("raw_target_metrics", {})
        for key in ("n_engines", "rmse", "mae", "r2", "nasa_score"):
            actual, supplied = computed.get(key), reported.get(key)
            if actual is None and supplied is None:
                continue
            if actual is None or supplied is None or not math.isclose(actual, supplied, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"Reported {key} does not match the frozen prediction table")
        validation = pd.read_csv(folder / "validation_predictions.csv")
        if (validation.empty or not {"engine_id", "cycle", "true_rul", "predicted_rul"}.issubset(validation.columns) or
                not set(validation.engine_id).issubset(split["validation_engine_ids"])):
            raise ValueError("Validation predictions must belong to the recorded held-out engines")
        endpoints = pd.read_csv(folder / "validation_endpoints.csv")
        if (validation.duplicated(["engine_id", "cycle"]).any() or
                not {"engine_id", "cycle", "true_rul"}.issubset(endpoints.columns) or
                not np.array_equal(validation[["engine_id", "cycle"]], endpoints[["engine_id", "cycle"]])):
            raise ValueError("Validation prefixes differ from the frozen selection endpoints")
        expected_validation = endpoints.true_rul.to_numpy()
        if cfg.get("rul_cap") is not None:
            expected_validation = np.minimum(expected_validation, cfg["rul_cap"])
        if not np.allclose(validation.true_rul, expected_validation, rtol=1e-6, atol=1e-6):
            raise ValueError("Validation targets differ from the configured frozen prefix labels")
        validation_metric = regression_metrics(validation.true_rul, validation.predicted_rul)
        if not math.isclose(validation_metric["rmse"], record["best_validation_rmse"], rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Best validation RMSE differs from the frozen validation predictions")
        selection = read_json(folder / "selection_record.json")
        if (selection != manifest.selection or selection.get("official_test_used_for_selection") is not False or
                selection.get("subset") != manifest.subset or
                manifest.selected_on_validation != (selection.get("selected_variant") == manifest.variant)):
            raise ValueError("Architecture selection must be frozen from validation evidence")
        means = selection.get("mean_validation_rmse", {})
        if not means or any(not math.isfinite(float(value)) or value < 0 for value in means.values()):
            raise ValueError("Finite validation comparison metrics are required")
        if "paper" in means and "conditioned_kan" in means:
            expected = "conditioned_kan" if means["conditioned_kan"] < means["paper"] - 1e-10 else "paper"
            if selection.get("selected_variant") != expected:
                raise ValueError("A tied or worse extension cannot replace the paper reference")
        local_data = "matched"
        for filename, expected in [(f"train_{manifest.subset}.txt", record["training_file_sha256"]),
                                   (f"test_{manifest.subset}.txt", metrics["test_file_sha256"]),
                                   (f"RUL_{manifest.subset}.txt", metrics["rul_file_sha256"])]:
            path = self.data_dir / filename
            if not path.is_file():
                local_data = "missing"
            elif file_hash(path) != expected:
                raise Conflict(f"Research export does not match local {filename}; retain the original corpus")
        if local_data == "matched":
            truth = np.loadtxt(self.data_dir / f"RUL_{manifest.subset}.txt", ndmin=1)
            if not np.array_equal(predictions.engine_id, np.arange(1, len(truth) + 1)) or not np.array_equal(predictions.true_rul, truth):
                raise ValueError("Frozen evaluation targets differ from the supplied official RUL vector")
            observed = sensor_frame(self.data_dir / f"test_{manifest.subset}.txt")
            endpoints = observed.groupby("engine_id").cycle.max().reindex(predictions.engine_id).to_numpy()
            if not np.array_equal(predictions.cycle, endpoints):
                raise ValueError("Frozen predictions must correspond to the last observed test cycle")
        reports = [validate_explanation(read_json(folder / name), manifest, record, preprocessor, split, predictions)
                   for name in sorted(explanations)]
        return {"folder": folder, "manifest": manifest, "record": record, "config": cfg, "preprocessor": preprocessor,
                "split": split, "predictions": predictions, "metrics": metrics, "explanations": reports,
                "local_data_status": local_data}

    def summary(self):
        runs, invalid = [], []
        if self.root.exists():
            for path in sorted(self.root.glob("*/*/research_export.json"))[:200]:
                try:
                    runs.append(self._public(self._inspect(path.parent)))
                except (ValueError, OSError, KeyError, TypeError, Conflict) as exc:
                    invalid.append({"folder": path.parent.name, "error": str(exc)})
        return {"runs": runs, "invalid_exports": invalid,
                "status": "available" if runs else "awaiting_user_training",
                "dependencies": {name: importlib.util.find_spec(name) is not None for name in ("torch", "shap", "lime")},
                "architecture": "Transformer-KAN-BiLSTM", "extension": "Operating-condition gated KAN; validation-controlled ablation",
                "notebook_url": "/downloads/kan-notebook", "requirements_url": "/downloads/kan-requirements",
                "explanation_order": ["training", "validation_selection", "official_evaluation", "SHAP", "LIME"],
                "note": "No trained GPU results or explanations are fabricated. Import notebook exports after your own training."}

    def _public(self, run):
        m, record = run["manifest"], run["record"]
        return {"run_id": m.run_id, "subset": m.subset, "variant": m.variant, "seed": m.seed,
                "architecture": m.architecture, "status": m.status, "exported_at": m.exported_at,
                "selected_on_validation": m.selected_on_validation, "selection": m.selection,
                "validation_rmse": record["best_validation_rmse"], "metrics": run["metrics"]["raw_target_metrics"],
                "configuration": run["config"], "explanation_status": m.explanation_status,
                "explanations_count": len(run["explanations"]), "local_data_status": run["local_data_status"],
                "inference_dependency_available": importlib.util.find_spec("torch") is not None,
                "source": m.source, "artifact_count": len(m.files)}

    def _find(self, run_id):
        if not RUN_PATTERN.fullmatch(run_id):
            raise ValueError("Invalid research run identifier")
        for path in self.root.glob("*/*/research_export.json"):
            try:
                value = read_json(path)
                identifier = value.get("run_id") if isinstance(value, dict) else None
            except (ValueError, OSError):
                continue
            if identifier == run_id:
                return self._inspect(path.parent)
        raise FileNotFoundError("Research run not found; train and import its notebook bundle first")

    def detail(self, run_id):
        run = self._find(run_id)
        history = pd.read_csv(run["folder"] / "history.csv").replace({np.nan: None}).to_dict(orient="records")
        return self._public(run) | {"history": history, "predictions": run["predictions"].to_dict(orient="records"),
                                    "explanation_engines": [{"engine_id": e["engine_id"], "cycle": e["cycle"]} for e in run["explanations"]],
                                    "artifacts": [{"name": name, **digest.model_dump()} for name, digest in run["manifest"].files.items()]}

    def engine_detail(self, run_id, engine_id):
        run = self._find(run_id)
        row = run["predictions"][run["predictions"].engine_id == engine_id]
        if row.empty:
            raise ValueError("Engine is not part of this research evaluation")
        point = row.iloc[0].to_dict()
        report = next((item for item in run["explanations"] if item["engine_id"] == engine_id and item["cycle"] == int(point["cycle"])), None)
        sensors, settings = [], []
        if run["local_data_status"] == "matched":
            frame = sensor_frame(self.data_dir / f"test_{run['manifest'].subset}.txt")
            frame = frame[frame.engine_id == engine_id]
            if frame.empty or int(frame.cycle.max()) != int(point["cycle"]):
                raise Conflict("Frozen prediction cycle differs from the local sensor trajectory")
            for schema in SENSOR_SCHEMA:
                recent = frame.tail(40)
                sensors.append(dict(schema) | {"value": float(recent.iloc[-1][schema["key"]]),
                                                "series": [{"cycle": int(r.cycle), "value": float(r[schema["key"]])} for _, r in recent.iterrows()]})
            settings = [{"name": f"setting_{i}", "value": float(frame.iloc[-1][f"setting_{i}"])} for i in range(1, 4)]
        return {"run_id": run_id, "subset": run["manifest"].subset, "variant": run["manifest"].variant,
                "prediction": point, "explanation": report, "sensors": sensors, "operating_settings": settings,
                "explanation_status": "available" if report else "not_generated_for_engine",
                "note": "Exported historical endpoint prediction. Ground truth is retrospective evaluation only."}

    def import_archive(self, content):
        if len(content) > 32 * 1024 * 1024:
            raise ValueError("Research upload exceeds 32 MB")
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".research-import-", dir=self.root) as temporary:
            folder = Path(temporary)
            try:
                with zipfile.ZipFile(BytesIO(content)) as archive:
                    members = [item for item in archive.infolist() if not item.is_dir()]
                    if not members or len(members) > 516 or sum(item.file_size for item in members) > 128 * 1024 * 1024:
                        raise ValueError("Research archive is empty or too large after extraction")
                    seen = set()
                    for item in members:
                        name = PurePosixPath(item.filename)
                        if (name.is_absolute() or ".." in name.parts or "\\" in item.filename or
                                item.filename in seen or stat.S_ISLNK(item.external_attr >> 16) or
                                item.file_size > 64 * 1024 * 1024 or
                                item.filename not in REQUIRED_FILES | {"research_export.json"} and not EXPLANATION_PATTERN.fullmatch(item.filename)):
                            raise ValueError("Research archive contains an unexpected, duplicate or unsafe path")
                        seen.add(item.filename)
                        target = folder / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(item))
            except zipfile.BadZipFile as exc:
                raise ValueError("Upload the ZIP produced by the completed KAN notebook") from exc
            run = self._inspect(folder)
            run_id = run["manifest"].run_id
            destination = self.root / "imported" / run_id
            with self.lock:
                try:
                    previous = self._find(run_id)
                except FileNotFoundError:
                    previous = None
                if previous is not None:
                    before, after = previous["manifest"], run["manifest"]
                    if any(before.files[name] != after.files[name] for name in REQUIRED_FILES):
                        raise Conflict("This run already exists with different model or evaluation artifacts; use a new run identity")
                    if any(name not in after.files for name in before.explanation_files):
                        raise Conflict("Updated bundles must retain previously exported explanations")
                    destination = previous["folder"]
                    # Explanations may be added/recomputed after fitting. Retain the earlier report on disk.
                    for name in after.explanation_files:
                        target = destination / name
                        if target.exists() and file_hash(target) != after.files[name].sha256:
                            backup = target.with_name(target.name + ".previous")
                            shutil.copy2(target, backup)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary_report = target.with_name(target.name + ".tmp")
                        shutil.copy2(folder / name, temporary_report)
                        temporary_report.replace(target)
                    temporary_manifest = destination / "research_export.json.tmp"
                    shutil.copy2(folder / "research_export.json", temporary_manifest)
                    temporary_manifest.replace(destination / "research_export.json")
                    return {"run": self._public(self._inspect(destination)), "reused": True}
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(folder), str(destination))
            return {"run": self._public(run), "reused": False}

    def _runtime(self, run):
        if importlib.util.find_spec("torch") is None:
            raise ResearchUnavailable("Neural inference requires PyTorch. Install the CUDA notebook environment manually; exported metrics and explanations remain viewable.")
        import torch
        key = (run["manifest"].run_id, run["manifest"].files["inference.ts"].sha256)
        with self.lock:
            if key not in self.models:
                model = torch.jit.load(str(run["folder"] / "inference.ts"), map_location="cpu").eval()
                with zipfile.ZipFile(run["folder"] / "inference_probes.npz") as archive:
                    if (set(archive.namelist()) != {"windows.npy", "predictions_cycles.npy"} or
                            sum(item.file_size for item in archive.infolist()) > 1024 * 1024):
                        raise ValueError("Inference probe archive exceeds its fixed numerical contract")
                with np.load(run["folder"] / "inference_probes.npz", allow_pickle=False) as probes:
                    windows, expected = probes["windows"], probes["predictions_cycles"]
                if (windows.ndim != 3 or not 1 <= len(windows) <= 4 or
                        windows.shape[1:] != (run["config"]["window"], len(run["preprocessor"]["columns"])) or
                        not np.isfinite(windows).all() or not np.isfinite(expected).all()):
                    raise ValueError("Inference probes have an invalid shape or values")
                with torch.inference_mode():
                    actual = model(torch.from_numpy(windows.astype(np.float32))).numpy() * run["config"]["target_scale"]
                if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-4, atol=.002):
                    raise ValueError("Neural artifact failed its exported inference probe comparison")
                self.models[key] = model
            return self.models[key]

    def predict(self, run_id, engine_id, cycle=None):
        run = self._find(run_id)
        if run["local_data_status"] != "matched":
            raise ResearchUnavailable("Import the matching benchmark corpus before neural sensor inference")
        frame = sensor_frame(self.data_dir / f"test_{run['manifest'].subset}.txt")
        frame = frame[frame.engine_id == engine_id].reset_index(drop=True)
        if frame.empty:
            raise ValueError("Engine not found in the research model's test corpus")
        cycle = cycle or len(frame)
        inputs = observed_window(frame, cycle, run["preprocessor"], run["config"])
        model = self._runtime(run)
        import torch
        with torch.inference_mode():
            forecast = model(torch.from_numpy(inputs)).numpy()[0] * run["config"]["target_scale"]
        if not np.isfinite(forecast).all():
            raise ValueError("Neural inference produced a nonfinite prediction")
        return {"run_id": run_id, "subset": run["manifest"].subset, "engine_id": engine_id, "cycle": cycle,
                "predicted_rul": float(forecast[0]), "forecast_rul": forecast.tolist(),
                "device": "CPU TorchScript", "source": "Actual saved neural model; observed sensor prefix only",
                "probe_verified": True, "ground_truth_used_for_inference": False}

    def archive(self, run_id):
        run = self._find(run_id)
        content = BytesIO()
        with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ["research_export.json", *run["manifest"].files]:
                archive.write(run["folder"] / name, name)
        return content.getvalue()
