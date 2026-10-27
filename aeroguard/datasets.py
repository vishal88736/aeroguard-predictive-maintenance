"""Strict dataset ingestion, honest previews, and training-only preparation.

NASA C-MAPSS sensor columns follow the conventional simulator schema.  The
operating settings remain generically named because their interpretation varies
by subset. No vibration or electrical-current measurements are invented.
"""

import csv
import io
import json
import math
from pathlib import Path
import tempfile
import time

from fleetpm.config import ARTIFACT_DIR, SUBSETS, file_hash, write_json
from fleetpm.data import import_zip, inventory, parse_rows, parse_truth


SENSOR_SCHEMA = (
    ("T2", "Fan inlet temperature", "°R"),
    ("T24", "LPC outlet temperature", "°R"),
    ("T30", "HPC outlet temperature", "°R"),
    ("T50", "LPT outlet temperature", "°R"),
    ("P2", "Fan inlet pressure", "psia"),
    ("P15", "Bypass-duct pressure", "psia"),
    ("P30", "HPC outlet pressure", "psia"),
    ("Nf", "Physical fan speed", "rpm"),
    ("Nc", "Physical core speed", "rpm"),
    ("epr", "Engine pressure ratio", "ratio"),
    ("Ps30", "HPC outlet static pressure", "psia"),
    ("phi", "Fuel-flow to Ps30 ratio", "raw"),
    ("NRf", "Corrected fan speed", "rpm"),
    ("NRc", "Corrected core speed", "rpm"),
    ("BPR", "Bypass ratio", "ratio"),
    ("farB", "Burner fuel-air ratio", "ratio"),
    ("htBleed", "Bleed enthalpy", "raw"),
    ("Nf-demand", "Demanded fan speed", "raw"),
    ("NRf-demand", "Demanded corrected fan speed", "raw"),
    ("W31", "HPT coolant bleed flow", "lbm/s"),
    ("W32", "LPT coolant bleed flow", "lbm/s"),
)

COLUMN_NAMES = ["unit", "cycle", *[f"setting_{i}" for i in range(1, 4)],
                *[f"sensor_{i}" for i in range(1, 22)]]
COLUMNS = [
    {"name": "unit", "type": "integer", "label": "Engine unit", "unit": "identifier"},
    {"name": "cycle", "type": "integer", "label": "Observed operating cycle", "unit": "cycles"},
    *[{"name": f"setting_{i}", "type": "number", "label": f"Operating setting {i}", "unit": "raw"}
      for i in range(1, 4)],
    *[{"name": f"sensor_{i}", "index": i, "type": "number", "symbol": symbol,
       "label": label, "unit": unit}
      for i, (symbol, label, unit) in enumerate(SENSOR_SCHEMA, 1)],
]


def _identifier(subset, partition):
    if subset not in SUBSETS:
        raise ValueError("Choose FD001, FD002, FD003 or FD004")
    if partition not in {"train", "test", "RUL"}:
        raise ValueError("Partition must be train, test or RUL")


def _json_number(value):
    return value if math.isfinite(value) else None


class DatasetManager:
    def __init__(self, data_dir, artifact_dir=ARTIFACT_DIR):
        self.data_dir = Path(data_dir)
        self.artifact_dir = Path(artifact_dir)
        self._inventory_stamp = None
        self._inventory = None
        self._preview_cache = {}

    def _stamp(self):
        return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                     for p in sorted(self.data_dir.glob("*.txt"))) if self.data_dir.exists() else ()

    def summary(self):
        stamp = self._stamp()
        if stamp != self._inventory_stamp or self._inventory is None:
            self._inventory = inventory(self.data_dir, deep=True)
            self._inventory_stamp = stamp
            self._preview_cache.clear()
        path = self.data_dir / "provenance.json"
        provenance = json.loads(path.read_text()) if path.is_file() else {
            "dataset": "NASA C-MAPSS compatible inputs",
            "acquisition": "User-managed files; original provenance has not been verified",
        }
        return {"datasets": self._inventory, "provenance": provenance, "columns": COLUMNS,
                "schema_note": "26 columns: unit, cycle, three settings and 21 sensors. Conventional sensor mappings; raw indicates no calibrated engineering unit asserted.",
                "label_note": "Training RUL is derived from each complete training trajectory. RUL_*.txt supplies separate official test endpoint evaluation targets and is never a predictor input."}

    def _parsed(self, path):
        stamp = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        if stamp not in self._preview_cache:
            # Keep only one full parsed preview in memory; summaries are cached separately.
            self._preview_cache = {stamp: parse_rows(path)}
        return self._preview_cache[stamp]

    def preview(self, subset, partition="train", limit=20):
        _identifier(subset, partition)
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("Preview limit must be between 1 and 100")
        path = self.data_dir / f"{partition}_{subset}.txt"
        if not path.is_file():
            raise ValueError(f"{path.name} is missing. Import the original dataset or a compatible CSV.")
        prepared = self.artifact_dir / "aeroguard" / "prepared" / subset / "audit.json"
        audit = json.loads(prepared.read_text()) if prepared.is_file() else None
        # Never display a stale prepared split as if it described changed training inputs.
        train = self.data_dir / f"train_{subset}.txt"
        if audit and (not train.is_file() or audit.get("train_sha256") != file_hash(train)):
            audit = None
        if partition == "RUL":
            values = self._truth_values(path)
            test = self.data_dir / f"test_{subset}.txt"
            if test.is_file():
                ids = self._test_ids(self._parsed(test))
                parse_truth(path, len(ids))
            statistics = {"rows": len(values), "engine_count": len(values), "missing_values": 0,
                          "missing_by_column": {"RUL": 0}, "duplicate_records": 0,
                          "invalid_records": 0, "errors": [], "cycle_min": None, "cycle_max": None,
                          "label_role": "Official test endpoint evaluation only; never a feature or training label"}
            return {"subset": subset, "partition": partition,
                    "columns": [{"name": "RUL", "type": "number", "label": "Official endpoint RUL", "unit": "cycles"}],
                    "rows": [[v] for v in values[:limit]], "statistics": statistics,
                    "split": audit.get("split") if audit else None,
                    "source_label": "Historical benchmark endpoint evaluation targets"}
        rows = self._parsed(path)
        missing = {name: sum(not math.isfinite(row[i]) for row in rows)
                   for i, name in enumerate(COLUMN_NAMES)}
        ids = {int(row[0]) for row in rows}
        cycle_set = {(int(row[0]), int(row[1])) for row in rows}
        last_cycles = {}
        for row in rows:
            unit = int(row[0])
            last_cycles[unit] = max(last_cycles.get(unit, 0), int(row[1]))
        missing_cycles = sum(last_cycles.values()) - len(cycle_set)
        statistics = {"rows": len(rows), "engine_count": len(ids), "missing_values": sum(missing.values()),
                      "missing_by_column": missing, "duplicate_records": 0, "invalid_records": 0,
                      "errors": [], "cycle_min": int(min(row[1] for row in rows)),
                      "cycle_max": int(max(row[1] for row in rows)), "missing_cycle_slots": missing_cycles,
                      "label_role": "Training labels derive from complete engine endpoints" if partition == "train"
                      else "Observed test prefixes; truth is stored in the separate RUL file"}
        return {"subset": subset, "partition": partition, "columns": COLUMNS,
                "rows": [[_json_number(x) for x in row] for row in rows[:limit]],
                "statistics": statistics, "split": audit.get("split") if audit else None,
                "source_label": "Historical run-to-failure training trajectories" if partition == "train"
                else "Historical truncated benchmark test trajectories"}

    @staticmethod
    def _truth_values(path):
        fields = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not fields:
            raise ValueError("Endpoint RUL file is empty")
        return parse_truth(path, len(fields))

    @staticmethod
    def _test_ids(rows):
        ids = sorted({int(row[0]) for row in rows})
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError("Test engine IDs must be consecutive from 1 to map the endpoint RUL vector")
        return ids

    def import_file(self, filename, content, subset="FD001", partition="train"):
        _identifier(subset, partition)
        if not isinstance(content, bytes) or not content:
            raise ValueError("Upload a non-empty dataset file")
        if len(content) > 32 * 1024 * 1024:
            raise ValueError("Uploads are limited to 32 MB")
        suffix = Path(filename.replace("\\", "/")).suffix.lower()
        if suffix == ".zip" or content.startswith(b"PK\x03\x04"):
            result = import_zip(content, self.data_dir)
            self._inventory_stamp = None
            return {"message": "Validated and imported all twelve C-MAPSS files", "datasets": result,
                    "imported_files": 12, "subset": "ALL", "partition": "all"}
        if suffix not in {".csv", ".txt"}:
            raise ValueError("Upload a C-MAPSS ZIP, CSV or whitespace-delimited TXT file")
        text = content.decode("utf-8-sig")
        if not text.strip():
            raise ValueError("Dataset file is empty")
        if "\x00" in text:
            raise ValueError("Dataset file contains a null byte")
        if suffix == ".csv" or "," in text.splitlines()[0]:
            records = [[field.strip() for field in row] for row in csv.reader(io.StringIO(text)) if any(field.strip() for field in row)]
            if not records:
                raise ValueError("Dataset CSV is empty")
            try:
                [float(x) for x in records[0]]
            except ValueError:
                header = [x.lower() for x in records.pop(0)]
                expected = ["rul"] if partition == "RUL" else COLUMN_NAMES
                if header != expected:
                    raise ValueError("CSV header must be " + ", ".join(expected))
            expected_count = 1 if partition == "RUL" else 26
            if not records or any(len(row) != expected_count for row in records):
                raise ValueError(f"Expected {expected_count} values in every dataset row")
            normalized = "\n".join(" ".join(row) for row in records) + "\n"
        else:
            normalized = text
        self.data_dir.mkdir(parents=True, exist_ok=True)
        name = f"{partition}_{subset}.txt"
        with tempfile.TemporaryDirectory(prefix=".ag-import-", dir=self.data_dir) as directory:
            staged = Path(directory) / name
            staged.write_text(normalized, encoding="utf-8")
            if partition == "RUL":
                parsed = self._truth_values(staged)
                test = self.data_dir / f"test_{subset}.txt"
                if test.is_file():
                    parse_truth(staged, len(self._test_ids(parse_rows(test))))
            else:
                parsed = parse_rows(staged)
                if partition == "test":
                    ids = self._test_ids(parsed)
                    truth = self.data_dir / f"RUL_{subset}.txt"
                    if truth.is_file():
                        parse_truth(truth, len(ids))
            target = self.data_dir / name
            unchanged = target.exists() and file_hash(target) == file_hash(staged)
            if target.exists() and not unchanged:
                raise ValueError(f"{name} already exists with different contents. Choose a separate data directory; existing input is preserved.")
            digest = file_hash(staged)
            if not unchanged:
                staged.replace(target)
        self._inventory_stamp = None
        self._preview_cache.clear()
        return {"message": "Identical file already present" if unchanged else "Dataset file validated and imported",
                "filename": name, "subset": subset, "partition": partition,
                "rows": len(parsed), "sha256": digest, "unchanged": unchanged,
                "datasets": self.summary()["datasets"]}

    def prepare(self, subsets, progress=None, seed=42, window=30, cap=125):
        """Fit preprocessing only on reserved fit-engine IDs, never test/truth data.

        This produces an inspectable feature/preprocessing audit, not a trained
        RUL model. Missing ML dependencies raise an actionable import error.
        """
        import joblib
        import numpy as np
        from threadpoolctl import threadpool_limits
        from fleetpm.features import Preprocessor, load_engines, split_engines
        from .baseline import FEATURE_NAMES, causal_features

        if isinstance(subsets, str):
            subsets = [subsets]
        if not subsets or len(set(subsets)) != len(subsets):
            raise ValueError("Choose unique dataset subsets to prepare")
        if not isinstance(window, int) or not 5 <= window <= 60 or not math.isfinite(cap) or cap <= 0:
            raise ValueError("Use a 5–60 cycle window and a positive finite prediction cap")
        output = []
        for subset in subsets:
            _identifier(subset, "train")
            tick = time.perf_counter()
            path = self.data_dir / f"train_{subset}.txt"
            if not path.is_file():
                raise ValueError(f"{path.name} is missing; preprocessing requires complete training trajectories")
            engines = load_engines(path)
            split = split_engines(engines, int(seed))
            observed_rows = sum(int(np.isfinite(e["raw"][:, :3]).all(axis=1).sum()) for e in engines.values())
            if progress:
                progress("validation", observed_rows, f"{subset}: validated training trajectories; test files untouched")
            with threadpool_limits(limits=2):
                processor = Preprocessor(subset, int(seed)).fit(engines, split["fit"])
            baseline_features_finite = True
            for unit in split["fit"]:
                transformed = processor.transform(engines[unit])
                features, _ = causal_features(transformed, window, cap)
                baseline_features_finite = baseline_features_finite and bool(np.isfinite(features).all())
            folder = self.artifact_dir / "aeroguard" / "prepared" / subset
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder / "preprocessing.joblib.tmp"
            joblib.dump(processor, temporary)
            temporary.replace(folder / "preprocessing.joblib")
            audit = {"subset": subset, "train_sha256": file_hash(path), "split_seed": int(seed), "split": split,
                     "rows": observed_rows, "aligned_cycle_slots": sum(len(e["raw"]) for e in engines.values()), "engines": len(engines),
                     "fit_engines": len(split["fit"]), "selection_engines": len(split["selection"]),
                     "calibration_engines": len(split["calibration"]), "window": window, "cap": cap,
                     "input_shape": [len(FEATURE_NAMES)],
                     "baseline_feature_count": len(FEATURE_NAMES), "baseline_window": window,
                     "baseline_features_finite": baseline_features_finite,
                     "feature_contract": {
                         "baseline": {"kind": "past-only rolling tabular features", "features": len(FEATURE_NAMES),
                                      "window": window, "feature_names": FEATURE_NAMES}},
                     "features_finite": baseline_features_finite,
                     "preprocessing_fit_ids": processor.fit_ids,
                     "label_role": "RUL derived from complete training endpoints; no official test truth accessed",
                     "scope": "Training-only preprocessing and causal feature audit; no RUL model fit",
                     "duration_seconds": round(time.perf_counter() - tick, 4),
                     "artifact_path": str(folder), "preprocessor_sha256": file_hash(folder / "preprocessing.joblib")}
            write_json(folder / "split.json", split)
            write_json(folder / "audit.json", audit)
            output.append(audit)
            if progress:
                progress("features", len(split["fit"]), f"{subset}: fitted regime scaling on {len(split['fit'])} fit engines; causal feature audit saved")
        return {"subsets": output, "scope": "Input preparation only; train a model explicitly for RUL inference"}
