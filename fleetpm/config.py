"""Project-local configuration. Importing this module downloads nothing."""

import hashlib
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
