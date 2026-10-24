"""Project-local configuration. Importing this module downloads nothing."""

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
COUNTS = {"FD001": (100, 100, 1, 1), "FD002": (260, 259, 6, 1),
          "FD003": (100, 100, 1, 2), "FD004": (249, 248, 6, 2)}


def load_env(root=ROOT):
    """Parse simple KEY=value settings without shell evaluation or overwrites."""
    path = Path(root) / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.replace("_", "").isalnum():
                os.environ.setdefault(key, value.strip().strip("\"'"))
    for key in ("UV_CACHE_DIR", "PIP_CACHE_DIR", "TORCH_HOME", "HF_HOME", "MPLCONFIGDIR",
                "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "JUPYTER_RUNTIME_DIR"):
        if key in os.environ and not Path(os.environ[key]).is_absolute():
            os.environ[key] = str(Path(root) / os.environ[key])


load_env()


def project_path(key, fallback):
    p = Path(os.environ.get(key, fallback)).expanduser()
    return p if p.is_absolute() else ROOT / p


DATA_DIR = project_path("FLEET_DATA_DIR", "data/cmapss")
ARTIFACT_DIR = project_path("FLEET_ARTIFACT_DIR", "artifacts")
RUNTIME_DIR = project_path("FLEET_RUNTIME_DIR", "runtime")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_status():
    modules = {"numpy": "numpy", "pandas": "pandas", "torch": "torch",
               "scikit-learn": "sklearn", "matplotlib": "matplotlib", "joblib": "joblib"}
    return {name: importlib.util.find_spec(module) is not None for name, module in modules.items()}


@dataclass(frozen=True)
class TrainConfig:
    subset: str = "FD001"
    seed: int = 42
    split_seed: int = 42
    window: int = 30
    minimum_history: int = 15
    cap: float = 125.0
    batch_size: int = 128
    epochs: int = 60
    clean_epochs: int = 3
    repair_epochs: int = 3
    patience: int = 12
    folds: int = 3
    calibration_fraction: float = 0.20
    alpha: float = 0.10
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    repair_weight: float = 0.10
    temporal_weight: float = 0.01
    sparse_weight: float = 0.0001
    correction_bound: float = 10.0
    device: str = "cuda"
    amp: bool = True
    workers: int = 0
    profile: str = "balanced"
    model_version: str = "hybrid-v1"

    def __post_init__(self):
        if self.subset not in SUBSETS:
            raise ValueError("Choose FD001, FD002, FD003 or FD004")
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ValueError("Device must be cuda, cpu or auto")
        if self.window < 15 or not 1 <= self.minimum_history <= self.window:
            raise ValueError("Unsupported history/window length")
        if self.folds < 2 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Invalid fold, epoch or batch count")
        if min(self.clean_epochs,self.repair_epochs,self.patience)<1 or self.cap<=0 or self.learning_rate<=0:
            raise ValueError("Phase lengths, patience, cap and learning rate must be positive")
        if self.workers!=0:
            raise ValueError("Use workers=0 for deterministic epoch-resume and notebook portability")
        if not 0 < self.alpha < 1 or not 0.05 <= self.calibration_fraction <= 0.4:
            raise ValueError("Invalid calibration configuration")

    def changed(self, **values):
        return replace(self, **values)

    def to_dict(self):
        return asdict(self)

    def fingerprint(self, source_hash):
        payload = {"configuration": self.to_dict(), "train_source_sha256": source_hash}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def profile_config(profile="balanced", **overrides):
    profiles = {
        "balanced": {},
        "rigorous": {"epochs": 150, "patience": 15, "folds": 5,
                     "clean_epochs": 10, "repair_epochs": 5},
        "smoke": {"epochs": 1, "clean_epochs": 1, "repair_epochs": 1,
                  "folds": 2, "batch_size": 32},
    }
    if profile not in profiles:
        raise ValueError("Unknown profile")
    return TrainConfig(**(profiles[profile] | {"profile": profile} | overrides))
