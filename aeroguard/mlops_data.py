"""Immutable incoming datasets and engine-disjoint, frozen comparison protocols.

Official test files and endpoint RUL labels are never copied into these stores.
An operator's complete-trajectory declaration is recorded, not inferred from a
truncated sensor stream or a maintenance-order resolution.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from fleetpm.config import file_hash, write_json
from fleetpm.data import parse_rows
from fleetpm.features import load_engines, prefix_end, split_engines
from fleetpm.store import now
from .datasets import COLUMN_NAMES

VERSION_PATTERN = re.compile(r"DV-FD00[1-4]-[a-f0-9]{20}")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def engine_digest(engine):
    raw = np.asarray(engine["raw"], dtype=np.float64)
    canonical = np.where(np.isnan(raw), 0, raw).astype("<f8").tobytes()
    mask = np.isnan(raw).astype(np.uint8).tobytes()
    return hashlib.sha256(canonical + mask).hexdigest()


def normalize_sensor_file(filename, content):
    if not isinstance(content, bytes) or not content or len(content) > 32 * 1024 * 1024:
        raise ValueError("Provide a non-empty TXT/CSV sensor file no larger than 32 MB")
    suffix = Path(filename.replace("\\", "/")).suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise ValueError("Incoming versions accept 26-column sensor TXT or CSV files")
    text = content.decode("utf-8-sig")
    if "\x00" in text or not text.strip():
        raise ValueError("Sensor file is empty or contains null bytes")
    converted = suffix == ".csv" or "," in text.splitlines()[0]
    if converted:
        records = [[v.strip() for v in row] for row in csv.reader(io.StringIO(text)) if any(v.strip() for v in row)]
        try:
            [float(v) for v in records[0]]
        except ValueError:
            if [v.lower() for v in records.pop(0)] != COLUMN_NAMES:
                raise ValueError("CSV header must match the documented C-MAPSS schema")
        if not records or any(len(row) != 26 for row in records):
            raise ValueError("Each sensor row must have 26 columns; RUL is not an input column")
        text = "\n".join(" ".join(row) for row in records) + "\n"
    return text.encode() if converted else content


class DatasetVersions:
    def __init__(self, artifact_dir):
        self.root = Path(artifact_dir) / "mlops" / "datasets"
        self.protocol_root = Path(artifact_dir) / "mlops" / "protocols"

    def add(self, subset, filename, content, label_status, provenance, namespace="incoming"):
        if subset not in {"FD001", "FD002", "FD003", "FD004"}:
            raise ValueError("Unknown dataset subset")
        if label_status not in {"complete_run_to_failure", "observed_prefix"}:
            raise ValueError("Declare complete run-to-failure trajectories or observed prefixes")
        if not provenance.strip() or len(provenance) > 1000:
            raise ValueError("Describe the source and label provenance in 1-1000 characters")
        if re.fullmatch(r"[A-Za-z0-9_-]{1,48}", namespace) is None:
            raise ValueError("Use a 1-48 character cohort name containing letters, numbers, underscores or hyphens")
        basename = Path(filename.replace("\\", "/")).name
        if label_status == "complete_run_to_failure" and basename.lower().startswith(("test_", "rul_")):
            raise ValueError("Official test files/targets cannot be used as labelled training data")
        normalized = normalize_sensor_file(basename, content)
        sha = hashlib.sha256(normalized).hexdigest()
        identity = {"subset": subset, "sha256": sha, "label_status": label_status, "namespace": namespace}
        version_id = f"DV-{subset}-{digest_json(identity)[:20]}"
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / version_id
        if destination.exists():
            return self.get(version_id) | {"reused": True}
        with tempfile.TemporaryDirectory(prefix=".incoming-", dir=self.root) as temporary:
            folder = Path(temporary)
            sensor = folder / f"train_{subset}.txt"
            sensor.write_bytes(normalized)
            rows = parse_rows(sensor)
            engines = load_engines(sensor)
            values = np.asarray(rows, dtype=np.float64)
            missing = float(np.isnan(values[:, 5:]).mean())
            missing_cycles = sum(len(e["raw"]) for e in engines.values()) - len(rows)
            manifest = {"id": version_id, **identity, "filename": basename,
                        "source_description": provenance.strip(), "created_at": now(),
                        "rows": len(rows), "engines": len(engines), "missing_sensor_fraction": missing,
                        "missing_cycles": missing_cycles, "file": sensor.name,
                        "label_note": "Completion is explicitly declared by the operator; sensor values alone do not verify failure.",
                        "engine_hashes": {str(k): engine_digest(v) for k, v in engines.items()},
                        "schema": "cmapss-26-v1"}
            manifest["integrity_sha256"] = digest_json(manifest)
            write_json(folder / "manifest.json", manifest)
            try:
                folder.rename(destination)
            except FileExistsError:
                return self.get(version_id) | {"reused": True}
        return manifest | {"reused": False}

    def get(self, version_id):
        if not isinstance(version_id, str) or VERSION_PATTERN.fullmatch(version_id) is None:
            raise ValueError("Invalid dataset version ID")
        folder = self.root / version_id
        if folder.is_symlink() or not folder.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Dataset snapshot must remain inside its version store")
        manifest = json.loads((folder / "manifest.json").read_text())
        expected = manifest.get("integrity_sha256")
        if expected != digest_json({k: v for k, v in manifest.items() if k != "integrity_sha256"}):
            raise ValueError("Dataset version manifest integrity check failed")
        if manifest.get("id") != version_id or manifest.get("file") != f"train_{manifest.get('subset')}.txt":
            raise ValueError("Dataset version manifest is invalid")
        if file_hash(folder / manifest["file"]) != manifest["sha256"]:
            raise ValueError("Immutable dataset version integrity check failed")
        return manifest

    def path(self, version_id):
        manifest = self.get(version_id)
        return self.root / version_id / manifest["file"]

    def list(self):
        result = []
        if self.root.exists():
            for folder in self.root.iterdir():
                if VERSION_PATTERN.fullmatch(folder.name):
                    try:
                        result.append(self.get(folder.name))
                    except (OSError, ValueError) as error:
                        result.append({"id": folder.name, "status": "invalid", "error": str(error)})
        return sorted(result, key=lambda x: x.get("created_at", ""), reverse=True)

    def protocol(self, subset):
        pointer = self.protocol_root / (subset + ".json")
        if not pointer.exists():
            return None
        value = json.loads(pointer.read_text())
        expected = value.pop("integrity_sha256", None)
        if expected != digest_json(value) or value.get("subset") != subset:
            raise ValueError("Frozen evaluation protocol integrity check failed")
        return value

    def freeze_protocol(self, version_id, configuration, split=None):
        manifest = self.get(version_id)
        if manifest["label_status"] != "complete_run_to_failure":
            raise ValueError("A comparison protocol requires declared complete training trajectories")
        existing = self.protocol(manifest["subset"])
        if existing:
            return existing
        engines = load_engines(self.path(version_id))
        split = split or split_engines(engines, configuration["split_seed"], .2)
        groups = [set(split[k]) for k in ("fit", "selection", "calibration")]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)) or set.union(*groups) != set(engines):
            raise ValueError("Protocol engine cohorts must be disjoint and cover the reference snapshot")
        minimum = min(15, configuration["window"])
        held = sorted(groups[1] | groups[2])
        contract = {k: configuration[k] for k in ("split_seed", "window", "cap")}
        contract["minimum_history"] = minimum
        value = {"subset": manifest["subset"], "reference_version_id": version_id,
                 "configuration": contract, "split": split,
                 "heldout_engine_hashes": {str(k): engine_digest(engines[k]) for k in held},
                 "prefixes": {str(k): int(engines[k]["cycles"][prefix_end(engines[k], configuration["split_seed"], minimum)]) for k in held}}
        value["id"] = f"EP-{manifest['subset']}-{digest_json(value)[:20]}"
        value["validation_cohort_sha256"] = digest_json({str(k): {"hash": value["heldout_engine_hashes"][str(k)], "cycle": value["prefixes"][str(k)]} for k in split["selection"]})
        value["created_at"] = now()
        self.protocol_root.mkdir(parents=True, exist_ok=True)
        write_json(self.protocol_root / (manifest["subset"] + ".json"), value | {"integrity_sha256": digest_json(value)})
        return value


def protocol_split(engines, protocol, configuration):
    for key in ("split_seed", "window", "cap", "minimum_history"):
        if configuration[key] != protocol["configuration"][key]:
            raise ValueError(f"Frozen comparison protocol requires the original {key}")
    held = set(protocol["split"]["selection"]) | set(protocol["split"]["calibration"])
    for unit in held:
        if unit not in engines or engine_digest(engines[unit]) != protocol["heldout_engine_hashes"][str(unit)]:
            raise ValueError("Frozen validation/calibration trajectories changed or are missing; candidate cannot be compared")
    fit = sorted(set(engines) - held)
    if not fit or not set(protocol["split"]["fit"]) <= set(fit):
        raise ValueError("Retain the original fitting engines and append new complete engines to the fitting pool")
    return {"fit": fit, "selection": list(protocol["split"]["selection"]), "calibration": list(protocol["split"]["calibration"])}
