"""Causal, CPU-sized C-MAPSS baseline and independent anomaly reference.

The retained CUDA model remains in ``fleetpm``.  This module deliberately has no
PyTorch dependency.  Training consumes only complete ``train_*.txt`` histories;
the separate evaluator opens official endpoint truth after predictions freeze.
"""

from datetime import datetime, timezone
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import time
import uuid

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from threadpoolctl import threadpool_limits

from fleetpm.config import ARTIFACT_DIR, DATA_DIR, ROOT, SUBSETS, file_hash, write_json
from fleetpm.data import parse_truth
from fleetpm.features import Preprocessor, load_engines, prefix_end, split_engines
from fleetpm.metrics import conformal_radius, prediction_interval, regression_metrics

LOGGER = logging.getLogger(__name__)
FEATURE_VERSION = "causal-rolling-v1"
RUN_PATTERN = re.compile(r"AG-FD00[1-4]-[a-f0-9]{12}")
SENSOR_SCHEMA = [
    {"key": f"sensor_{i}", "index": i, "symbol": symbol, "label": label, "unit": unit}
    for i, (symbol, label, unit) in enumerate([
        ("T2", "Fan inlet total temperature", "°R"),
        ("T24", "LPC outlet total temperature", "°R"),
        ("T30", "HPC outlet total temperature", "°R"),
        ("T50", "LPT outlet total temperature", "°R"),
        ("P2", "Fan inlet total pressure", "psia"),
        ("P15", "Bypass duct total pressure", "psia"),
        ("P30", "HPC outlet total pressure", "psia"),
        ("Nf", "Physical fan speed", "rpm"),
        ("Nc", "Physical core speed", "rpm"),
        ("epr", "Engine pressure ratio P50/P2", "ratio"),
        ("Ps30", "HPC outlet static pressure", "psia"),
        ("phi", "Fuel-flow to Ps30 ratio", "dataset ratio"),
        ("NRf", "Corrected fan speed", "rpm"),
        ("NRc", "Corrected core speed", "rpm"),
        ("BPR", "Bypass ratio", "ratio"),
        ("farB", "Burner fuel-air ratio", "ratio"),
        ("htBleed", "Bleed enthalpy", "dataset units"),
        ("Nf_dmd", "Demanded fan speed", "rpm"),
        ("PCNfR_dmd", "Demanded corrected fan speed", "dataset units"),
        ("W31", "HPT coolant bleed flow", "lbm/s"),
        ("W32", "LPT coolant bleed flow", "lbm/s"),
    ], 1)
]
FEATURE_NAMES = (
    [f"current_{s['key']}" for s in SENSOR_SCHEMA]
    + [f"rolling_mean_{s['key']}" for s in SENSOR_SCHEMA]
    + [f"rolling_std_{s['key']}" for s in SENSOR_SCHEMA]
    + [f"rolling_slope_{s['key']}" for s in SENSOR_SCHEMA]
    + ["setting_1", "setting_2", "setting_3", "observed_fraction", "maximum_gap", "observed_cycle_capped"]
)


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _rolling_sum(values, starts):
    values = np.asarray(values, dtype=np.float64)
    prefix = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)], axis=0)
    return prefix[np.arange(len(values)) + 1] - prefix[starts]


def causal_features(engine, window=30, cap=125):
    """One feature row per observed cycle, never using future trajectory length.

    ``engine`` is the output of a fitted Preprocessor.transform. Rolling moments
    use only trusted observations in [t-window+1,t].  The final age feature is an
    observed cycle, not normalized by the eventual engine lifetime. Quality
    statistics are returned separately and never use training/evaluation labels.
    """
    x = np.asarray(engine["sensors"], dtype=np.float64)
    mask = np.asarray(engine["mask"], dtype=bool)
    if x.ndim != 2 or x.shape[1] != 21 or not len(x):
        raise ValueError("Causal features require a non-empty 21-sensor trajectory")
    n = len(x)
    positions = np.arange(n, dtype=np.float64)[:, None]
    starts = np.maximum(0, np.arange(n) - int(window) + 1)
    lengths = np.arange(n) + 1 - starts
    trusted = mask.astype(np.float64)
    values = np.where(mask, x, 0)
    count = _rolling_sum(trusted, starts)
    denominator = np.maximum(count, 1)
    total = _rolling_sum(values, starts)
    mean = total / denominator
    variance = np.maximum(0, _rolling_sum(values * values, starts) / denominator - mean * mean)
    time_total = _rolling_sum(positions * trusted, starts)
    time_square_total = _rolling_sum(positions * positions * trusted, starts)
    time_value_total = _rolling_sum(positions * values, starts)
    time_variance = time_square_total - time_total * time_total / denominator
    slope = (time_value_total - time_total * total / denominator) / np.maximum(time_variance, 1e-10)
    enough = count >= 5
    mean = np.where(enough, mean, 0)
    std = np.where(enough, np.sqrt(variance), 0)
    slope = np.where(enough, slope, 0)
    fraction = count.sum(axis=1) / (lengths * 21)
    max_gap = np.max(engine["gaps"], axis=1).astype(np.float64)
    # Rolling maximum without padding with future observations.
    gap_window = max_gap.copy()
    for offset in range(1, min(int(window), n)):
        gap_window[offset:] = np.maximum(gap_window[offset:], max_gap[:-offset])
    domain_count = _rolling_sum(np.asarray(engine["domain"], dtype=np.float64)[:, None], starts).ravel()
    domain = domain_count == lengths
    quality = (fraction >= .8) & (gap_window <= (5 / 30 + 1e-7)) & domain
    age = np.minimum(np.asarray(engine["cycles"], dtype=np.float64), cap) / cap
    features = np.column_stack([values, mean, std, slope, engine["settings"], fraction, gap_window, age]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite causal features; inspect sensor magnitudes and operating settings")
    return features, {
        "quality": fraction,
        "max_gap": gap_window,
        "domain": domain,
        "basic_quality": quality,
        "lengths": lengths,
        "trend": np.mean(np.abs(slope), axis=1),
    }


def model_folder(run_id, artifact_dir=ARTIFACT_DIR):
    if not isinstance(run_id, str) or RUN_PATTERN.fullmatch(run_id) is None:
        raise ValueError("Invalid AeroGuard model run ID")
    parent = (Path(artifact_dir) / "aeroguard" / "runs").resolve()
    folder = (parent / run_id).resolve()
    if folder.parent != parent or not folder.is_dir():
        raise ValueError("Model run not found")
    return folder


def list_models(artifact_dir=ARTIFACT_DIR):
    directory = Path(artifact_dir) / "aeroguard" / "runs"
    output = []
    if not directory.exists():
        return output
    for folder in directory.iterdir():
        if RUN_PATTERN.fullmatch(folder.name) is None or not folder.is_dir():
            continue
        try:
            value = json.loads((folder / "metadata.json").read_text())
            for name in ("evaluation", "replay"):
                file = folder / (name + ".json")
                if file.exists():
                    value[name] = json.loads(file.read_text())
            output.append(value)
        except (OSError, ValueError):
            LOGGER.warning("Could not read model metadata", extra={"run_id": folder.name})
    return sorted(output, key=lambda item: item.get("created_at", ""), reverse=True)


def _config(configuration):
    from .contracts import TrainRequest
    supplied = dict(configuration)
    subset = supplied.pop("subset", None)
    request = TrainRequest.model_validate(supplied)
    if subset is None:
        if len(request.subsets) != 1:
            raise ValueError("train_model requires one subset; iterate selected subsets in the job worker")
        subset = request.subsets[0]
    if subset not in SUBSETS or subset not in request.subsets:
        raise ValueError("Training subset must be one of the selected C-MAPSS subsets")
    result = request.model_dump()
    result.update(subset=subset, feature_version=FEATURE_VERSION, minimum_history=min(15, request.window))
    return result


def _prefix_samples(raw, transformed, ids, cfg):
    rows, targets, mapping = [], [], []
    for unit in ids:
        features, quality = transformed[unit]
        engine = raw[unit]
        if len(features) < cfg["minimum_history"]:
            continue
        end = prefix_end(engine, cfg["split_seed"], cfg["minimum_history"])
        if not 0 <= end < len(features):
            continue
        rows.append(features[end])
        targets.append(float(len(features) - 1 - end))
        mapping.append({"engine_id": int(unit), "cycle": int(engine["cycles"][end]), "basic_quality": bool(quality["basic_quality"][end])})
    if not rows:
        raise ValueError("No usable engine prefixes for development/calibration")
    return np.asarray(rows, dtype=np.float32), np.asarray(targets), mapping


def _atomic_joblib(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def verify_bundle(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "bundle_hashes.json").read_text())
    required = {"model.joblib", "preprocessing.joblib", "anomaly.joblib", "config.json", "split.json", "calibration.json", "feature_names.json"}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise ValueError("Model bundle is incomplete")
    for name, digest in manifest.items():
        if Path(name).name != name or not (folder / name).is_file() or file_hash(folder / name) != digest:
            raise ValueError("Model bundle integrity check failed")
    return manifest


def train_model(config, data_dir=DATA_DIR, artifact_dir=ARTIFACT_DIR, progress=None):
    """Fit one baseline using training trajectories only, with three engine cohorts.

    Progress callback receives ``(stage_name, actual_record_count, detail)``. This
    method never reads ``test_`` or ``RUL_`` files and never promotes the result.
    """
    cfg = _config(config)
    data_dir, artifact_dir = Path(data_dir), Path(artifact_dir)
    path = data_dir / f"train_{cfg['subset']}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Import {path.name} before training")
    run_id = f"AG-{cfg['subset']}-{uuid.uuid4().hex[:12]}"
    folder = artifact_dir / "aeroguard" / "runs" / run_id
    folder.mkdir(parents=True)
    begin = time.perf_counter()
    metadata = {
        "run_id": run_id, "subset": cfg["subset"], "family": "histogram-gradient-boosting",
        "training_status": "running", "created_at": utcnow(), "finished_at": None,
        "configuration": cfg, "artifact_path": str(folder.resolve()),
        "source_hashes": {path.name: file_hash(path)}, "source_sha256": file_hash(path),
        "code_sha256": hashlib.sha256((file_hash(Path(__file__)) + file_hash(ROOT / "fleetpm/features.py")).encode()).hexdigest(),
        "dependencies": {"numpy": np.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
        "mlflow_run_id": None, "tracking_status": "disabled" if not cfg["track_mlflow"] else "pending",
        "protocol": "Engine-disjoint 60% fit / 20% fixed development / 20% independent calibration; official test untouched",
    }
    write_json(folder / "metadata.json", metadata)
    write_json(folder / "config.json", cfg)
    tracking = None
    def report(name, records, detail):
        LOGGER.info(name, extra={"run_id": run_id, "records": records, "detail": detail})
        if progress:
            progress(name, int(records), detail)
    try:
        with threadpool_limits(limits=2):
            raw = load_engines(path)
            split = split_engines(raw, cfg["split_seed"], .2)
            groups = [set(split[k]) for k in ("fit", "selection", "calibration")]
            if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
                raise ValueError("Engine cohorts overlap")
            write_json(folder / "split.json", split)
            processor = Preprocessor(cfg["subset"], cfg["split_seed"]).fit(raw, split["fit"])
            transformed = {unit: causal_features(processor.transform(engine), cfg["window"], cfg["cap"]) for unit, engine in raw.items()}
            feature_rows, targets, weights = [], [], []
            mapping = []
            for unit in split["fit"]:
                features, _ = transformed[unit]
                ends = np.arange(cfg["minimum_history"] - 1, len(features), cfg["stride"])
                if not len(ends):
                    continue
                feature_rows.append(features[ends])
                targets.extend(np.minimum(cfg["cap"], len(features) - 1 - ends))
                weights.extend(np.full(len(ends), 1 / len(ends)))
                mapping.extend({"engine_id": int(unit), "cycle": int(raw[unit]["cycles"][end])} for end in ends)
            if not feature_rows:
                raise ValueError("No training windows after minimum-history filtering")
            X = np.concatenate(feature_rows)
            y = np.asarray(targets, dtype=np.float64)
            sample_weights = np.asarray(weights, dtype=np.float64)
            sample_weights /= sample_weights.mean()
            report("features", len(X), f"{X.shape[1]} past-only features; {len(split['fit'])} fit engines; no held-out normalization")
            model = HistGradientBoostingRegressor(
                max_iter=cfg["iterations"], learning_rate=cfg["learning_rate"],
                max_leaf_nodes=cfg["max_leaf_nodes"], max_bins=127,
                min_samples_leaf=15, l2_regularization=1.,
                early_stopping=False, random_state=cfg["seed"],
            ).fit(X, y, sample_weight=sample_weights)
            validation_X, validation_y, validation_mapping = _prefix_samples(raw, transformed, split["selection"], cfg)
            validation_points = np.clip(model.predict(validation_X), 0, cfg["cap"])
            validation_metrics = regression_metrics(validation_y, validation_points, cfg["cap"])
            validation_predictions = [m | {"true_rul": float(t), "rul": float(p)} for m, t, p in zip(validation_mapping, validation_y, validation_points)]
            report("fit", len(X), f"{model.n_iter_} fixed-budget histogram trees; development RMSE {validation_metrics['rmse']:.3f} cycles")
            calibration_X, calibration_y, calibration_mapping = _prefix_samples(raw, transformed, split["calibration"], cfg)
            include = np.asarray([m["basic_quality"] for m in calibration_mapping], dtype=bool)
            if not include.any():
                raise ValueError("No input-quality-qualified calibration prefixes")
            calibration_points = np.clip(model.predict(calibration_X[include]), 0, cfg["cap"])
            calibration = conformal_radius(np.abs(np.minimum(cfg["cap"], calibration_y[include]) - calibration_points), .1)
            calibration.update(
                included_engine_ids=[m["engine_id"] for m, take in zip(calibration_mapping, include) if take],
                excluded_engine_ids=[m["engine_id"] for m, take in zip(calibration_mapping, include) if not take],
                assumption="Clean, quality-qualified independent engine prefixes drawn under the fixed pseudo-truncation policy; not simultaneous trajectory coverage or field failure probability",
            )
            healthy = []
            for unit in split["fit"]:
                features, quality = transformed[unit]
                # Age is excluded from the detector. Every reference window has
                # the same context length as an ordinary inference window.
                start = min(cfg["window"] - 1, len(features) - 1)
                ends = np.arange(start, min(start + 11, len(features)))
                keep = quality["basic_quality"][ends]
                if keep.any():
                    healthy.append(features[ends[keep], :84])
            if not healthy or sum(len(v) for v in healthy) < 20:
                raise ValueError("Insufficient early healthy-prefix proxy windows for the independent anomaly detector")
            healthy_X = np.concatenate(healthy)
            detector = IsolationForest(n_estimators=128, max_samples=min(256, len(healthy_X)), contamination="auto", random_state=cfg["seed"], n_jobs=2).fit(healthy_X)
            reference = np.sort(-detector.score_samples(healthy_X))
            anomaly = {"detector": detector, "reference": reference, "feature_columns": list(range(84)),
                       "scope": "First complete-context windows of fit engines are a healthy proxy; no labelled fault-time ground truth exists",
                       "reference_windows": len(healthy_X), "reference_engine_ids": split["fit"]}
            report("calibration", int(include.sum()), "Independent engine conformal reference and separate early-prefix Isolation Forest frozen")
            _atomic_joblib(model, folder / "model.joblib")
            _atomic_joblib(processor, folder / "preprocessing.joblib")
            _atomic_joblib(anomaly, folder / "anomaly.joblib")
            write_json(folder / "calibration.json", calibration)
            write_json(folder / "feature_names.json", FEATURE_NAMES)
            write_json(folder / "training_mapping.json", mapping)
            write_json(folder / "validation_predictions.json", validation_predictions)
            immutable = ["model.joblib", "preprocessing.joblib", "anomaly.joblib", "config.json", "split.json", "calibration.json", "feature_names.json", "training_mapping.json", "validation_predictions.json"]
            write_json(folder / "bundle_hashes.json", {name: file_hash(folder / name) for name in immutable})
            metadata.update(
                training_status="complete", finished_at=utcnow(), duration_seconds=round(time.perf_counter() - begin, 4),
                validation_metrics=validation_metrics, calibration=calibration,
                training_samples=len(X), features=X.shape[1],
                parameters=int(sum(predictor.nodes.size for stage in model._predictors for predictor in stage)),
                parameter_unit="tree nodes", fit_engines=len(split["fit"]),
                validation_engines=len(split["selection"]), calibration_engines=int(include.sum()),
                anomaly_reference={k: anomaly[k] for k in ("scope", "reference_windows", "reference_engine_ids")},
            )
        # Tracking records the actual fitted artifacts; it is not a substitute
        # for the independently persisted model manifest.
        if cfg["track_mlflow"]:
            try:
                os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
                import mlflow
                tracking_root = (artifact_dir / "aeroguard").resolve()
                mlflow.set_tracking_uri("sqlite:///" + str(tracking_root / "mlflow.db"))
                experiment_name = "AeroGuard-CMAPSS-baseline"
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name, artifact_location=(tracking_root / "mlflow-artifacts").as_uri())
                else:
                    experiment_id = experiment.experiment_id
                with mlflow.start_run(experiment_id=experiment_id, run_name=run_id) as tracking:
                    metadata["mlflow_run_id"] = tracking.info.run_id
                    mlflow.log_params({k: v for k, v in cfg.items() if k != "subsets"})
                    mlflow.log_params({"train_sha256": metadata["source_sha256"], "feature_count": len(FEATURE_NAMES)})
                    mlflow.log_metrics({"validation_" + k: v for k, v in validation_metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)})
                    mlflow.log_dict(split, "split.json")
                    mlflow.log_dict(calibration, "calibration.json")
                    mlflow.log_artifact(str(folder / "bundle_hashes.json"))
                    mlflow.log_artifact(str(folder / "model.joblib"))
                metadata["tracking_status"] = "complete"
            except Exception as error:
                metadata.update(tracking_status="failed", tracking_error=f"{type(error).__name__}: {error}")
                LOGGER.exception("MLflow tracking failed for locally saved model %s", run_id)
        write_json(folder / "metadata.json", metadata)
        return metadata
    except BaseException as error:
        metadata.update(training_status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", finished_at=utcnow(), error=f"{type(error).__name__}: {error}")
        write_json(folder / "metadata.json", metadata)
        raise


class BaselineBundle:
    """Load only integrity-checked artifacts created by train_model."""
    def __init__(self, folder):
        self.folder = Path(folder)
        verify_bundle(self.folder)
        self.metadata = json.loads((self.folder / "metadata.json").read_text())
        if self.metadata.get("training_status") != "complete":
            raise ValueError("Model training has not completed")
        self.configuration = json.loads((self.folder / "config.json").read_text())
        if self.configuration.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Unsupported model feature contract")
        if self.metadata.get("dependencies", {}).get("scikit-learn") != sklearn.__version__:
            raise ValueError("Model was saved under a different scikit-learn version; use its pinned environment")
        self.model = joblib.load(self.folder / "model.joblib")
        self.processor = joblib.load(self.folder / "preprocessing.joblib")
        self.anomaly = joblib.load(self.folder / "anomaly.joblib")
        self.calibration = json.loads((self.folder / "calibration.json").read_text())

    def engine_predictions(self, engine, ends, uncertainty_gate=80):
        ends = [int(end) for end in ends]
        if not ends:
            return []
        if any(end < 0 or end >= len(engine["raw"]) for end in ends):
            raise ValueError("Prediction endpoint is outside the observed engine prefix")
        transformed = self.processor.transform(engine)
        features, quality = causal_features(transformed, self.configuration["window"], self.configuration["cap"])
        with threadpool_limits(limits=2):
            values = np.clip(self.model.predict(features[ends]), 0, self.configuration["cap"])
            scores = -self.anomaly["detector"].score_samples(features[ends, :84])
        reference = self.anomaly["reference"]
        timestamp = utcnow()
        output = []
        for end, point, score in zip(ends, values, scores):
            lower, upper = prediction_interval(float(point), self.calibration, self.configuration["cap"])
            good = bool(quality["basic_quality"][end])
            observed_fraction = float(quality["quality"][end])
            reason, exception = "", None
            if not good:
                reason, exception = "Missing sensors, long gaps, or an unsupported operating setting", "DATA" if quality["domain"][end] else "DOMAIN"
            elif observed_fraction < .99999:
                reason, exception = "The interval reference uses clean prefixes; missing observations require uncertainty review", "CALIBRATION"
            elif lower is None:
                reason, exception = "Too few independent calibration engines for a finite 90% interval", "CALIBRATION"
            elif upper - lower > uncertainty_gate:
                reason, exception = "Prediction interval exceeds the configured decision-width gate", "UNCERTAINTY"
            deviation = np.where(transformed["mask"][end], np.abs(transformed["sensors"][end]), 0)
            top = np.argsort(deviation)[-3:][::-1]
            output.append({
                "asset": f"{self.configuration['subset']}/test/{engine['unit']:03d}",
                "engine_id": int(engine["unit"]), "subset": self.configuration["subset"],
                "cycle": int(engine["cycles"][end]), "lifecycle": "1", "bundle_id": self.metadata["run_id"],
                "rul": float(point), "base_rul": float(point), "lower": lower, "upper": upper,
                "anomaly": float(np.searchsorted(reference, score, side="right") / len(reference)) if observed_fraction >= .8 else None,
                "anomaly_raw": float(score), "quality": observed_fraction,
                "quality_valid": not bool(reason), "data_quality_valid": good,
                "exception_type": exception, "quality_reason": reason,
                "history_length": int(quality["lengths"][end]), "complete_loss": bool(observed_fraction == 0),
                "required_history": self.configuration["window"],
                "horizon_censored": upper is not None and upper >= self.configuration["cap"],
                "interval_reference": "clean independent calibration prefixes", "nominal_coverage": .9,
                "calibration_engines": self.calibration["n_engines"], "trend": float(quality["trend"][end]),
                "timestamp": timestamp,
                "sensors": [{"index": int(j) + 1, "name": SENSOR_SCHEMA[j]["symbol"], "deviation": float(deviation[j]), "observed": bool(transformed["mask"][end, j])} for j in top],
            })
        return output


def evaluate_model(run_id, data_dir=DATA_DIR, artifact_dir=ARTIFACT_DIR):
    """Frozen official endpoint evaluation, excluded from promotion decisions."""
    folder = model_folder(run_id, artifact_dir)
    bundle = BaselineBundle(folder)
    subset = bundle.configuration["subset"]
    test_path = Path(data_dir) / f"test_{subset}.txt"
    truth_path = Path(data_dir) / f"RUL_{subset}.txt"
    hashes = {test_path.name: file_hash(test_path), truth_path.name: file_hash(truth_path)}
    previous = folder / "evaluation.json"
    if previous.exists() and json.loads(previous.read_text())["source_hashes"] != hashes:
        raise ValueError("This frozen run was already evaluated against different official files")
    engines = load_engines(test_path)
    ids = sorted(engines)
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError("Official endpoint truth requires consecutive test IDs starting at one")
    begin = time.perf_counter()
    predictions = []
    for unit in ids:
        predictions.extend(bundle.engine_predictions(engines[unit], [len(engines[unit]["raw"]) - 1]))
    # Endpoint truth is intentionally not parsed or exposed to the predictor.
    truth = parse_truth(truth_path, len(ids))
    metrics = regression_metrics(truth, [p["rul"] for p in predictions], bundle.configuration["cap"])
    supported = [(p, y) for p, y in zip(predictions, truth) if p["lower"] is not None]
    metrics.update(
        capped_interval_coverage=float(np.mean([p["lower"] <= min(y, bundle.configuration["cap"]) <= p["upper"] for p, y in supported])) if supported else None,
        interval_mean_width=float(np.mean([p["upper"] - p["lower"] for p, _ in supported])) if supported else None,
        interval_evaluated_engines=len(supported),
        decision_exception_rate=float(np.mean([not p["quality_valid"] or p["history_length"] < bundle.configuration["window"] for p in predictions])),
    )
    report = {"run_id": run_id, "subset": subset, "metrics": metrics, "source_hashes": hashes,
              "evaluated_at": utcnow(), "duration_seconds": round(time.perf_counter() - begin, 4), "decision_width_gate_cycles": 80,
              "protocol": "One frozen endpoint prediction per official test engine, then truth parsing. Raw-truth metrics; capped coverage separately labelled. Not used for promotion.",
              "anomaly_evaluation": "No per-cycle anomaly onset labels are supplied; score distribution and reference scope are reported, not an invented precision/recall."}
    write_json(folder / "predictions.json", predictions)
    write_json(folder / "evaluation_targets.json", [{"engine_id": i, "true_rul": y} for i, y in zip(ids, truth)])
    write_json(folder / "evaluation.json", report)
    with (folder / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["engine_id", "cycle", "prediction", "true_rul", "lower_capped", "upper_capped", "anomaly", "decision_valid"])
        writer.writeheader()
        for packet, target in zip(predictions, truth):
            writer.writerow({"engine_id": packet["engine_id"], "cycle": packet["cycle"], "prediction": packet["rul"], "true_rul": target,
                             "lower_capped": packet["lower"], "upper_capped": packet["upper"], "anomaly": packet["anomaly"],
                             "decision_valid": packet["quality_valid"] and packet["history_length"] >= bundle.configuration["window"]})
    if bundle.metadata.get("mlflow_run_id"):
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow
            mlflow.set_tracking_uri("sqlite:///" + str((Path(artifact_dir) / "aeroguard" / "mlflow.db").resolve()))
            with mlflow.start_run(run_id=bundle.metadata["mlflow_run_id"]):
                mlflow.log_metrics({"official_test_" + key: value for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)})
                mlflow.log_artifact(str(folder / "evaluation.json"))
        except Exception as error:
            report["tracking_error"] = f"{type(error).__name__}: {error}"
            write_json(folder / "evaluation.json", report)
    return report
