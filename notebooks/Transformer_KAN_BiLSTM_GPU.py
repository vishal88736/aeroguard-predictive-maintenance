# %% [markdown] id=introduction
# # Transformer → KAN → BiLSTM: GPU training on C-MAPSS
#
# **One selected paper, a controlled extension, four dataset subsets.**
#
# Kejie Xu, Yingqing Guo and Qifan Zhou, *Research on the Remaining Useful Life
# Prediction Algorithm for Aero-Engines Based on Transformer–KAN–BiLSTM*,
# Aerospace 12 (2025), 998. [Paper / DOI](https://doi.org/10.3390/aerospace12110998).
# Supplied PDF SHA-256: `1fc996612dd16e8c150d38349e81f410bf2abcc87d1c48d2f11fdaf1aabf8e8f`.
#
# This standalone notebook implements the paper's Transformer → spline KAN →
# bidirectional LSTM architecture and compares it with an **operating-condition
# gated KAN** extension. It supports **FD001, FD002, FD003 and FD004**,
# CUDA mixed precision, gradient accumulation, train-only spline-grid adaptation,
# resumable checkpoints, validation-based variant selection, official endpoint
# evaluation, portable website exports, and **SHAP + LIME as the final analysis stage**.
# All implementation code is inline;
# neither `fleetpm/` nor the website is needed.
#
# **Authored and unexecuted. No trained weights or measured neural results are
# included.** Training starts when **you** run the training cell. Opening this
# notebook installs nothing, downloads nothing and starts no training.

# %% [markdown] id=paper-fidelity
# ## 1. What follows the paper, and what needed an explicit choice
#
# ```text
# Historical sensor/settings window (cycles <= t)
#   → embedding + sinusoidal positions
#   → 6 Transformer encoders + 6 masked decoders
#   → 2 spline KAN layers, applied to each historical token
#   → BiLSTM over that observed window
#   → forward/backward final states → RUL regression (10 horizons)
# ```
#
# | Item | Paper | This implementation |
# |:---|:---|:---|
# | Module order | Transformer → KAN → BiLSTM (pp. 14–15, Fig. 5) | Same order; no CNN, GRU, GAN, RL or baseline-model substitution |
# | Transformer | Embedding 32, 8 heads, 6 encoder/decoder pairs, dropout 0.2 (p. 14) | Same for FD001/FD003; encoder and decoder each have six independent layers |
# | Complex subsets | Dimensions 64/128, heads 8/16, grids 40–50 (p. 18) | FD002 = 64/8/40, FD004 = 128/16/50; this pairing is inferred from the text |
# | KAN | SiLU residual + trainable B-spline edges, initially near zero; grid updates (pp. 8–10) | Cubic Cox–de Boor basis, two learnable edge layers, train-only quantile-grid updates and curve refitting |
# | Sparsity | Activation L1 + entropy, μ1 = μ2 = 1 (p. 16) | Same terms on a bounded sample of current training activations; λ = 1e-5 is configurable |
# | Training | LR 1e-4, batch 256, 50 iterations; early stopping discussed (pp. 14, 21) | 50 epochs, effective batch 256, Adam; optimizer, clipping and patience are explicit implementation choices |
# | Unspecified interfaces | No complete tensor-level code, input-window length or BiLSTM width | Window 30; KAN widths [d, d, d]; one BiLSTM layer with 64 units per direction; feed-forward width = d |
# | Ten-step prediction | Subsequent-window ten-step description, indexing unspecified (p. 14) | Heads predict RUL at t, t+1, …, t+9; post-failure horizons are masked; official scoring uses **t only** |
# | Decoder input / softmax | Generic sequence-to-sequence classification diagram | Shifted **observed sensor embeddings**, no target teacher forcing; nonnegative regression replaces classification softmax |
# | Data / split | Describes 22 selected features and merging official train/test before 70/20/10 repartition (p. 14) | Actual 3 settings + 21 sensors; remove constants using fit engines only; official training engines split 80/20; official test stays untouched |
# | RUL / smoothing | Cap, normalization formula and smoothing width unspecified | Uncapped RUL by default, min–max to [-1, 1], trailing 5-cycle mean, target divided by 100 for optimization |
#
# **This is a transparent implementation of the published architecture, not an
# exact numerical reproduction.** The paper's Table 4 reports RMSE
# **3.6784 / 4.9498 / 3.7988 / 7.5202** under its repartitioned protocol. Those are
# author-reported reference values, not outputs of this notebook or guaranteed
# scores on the untouched official test engines. No test data or test labels are
# used for scaling, grid fitting, checkpoint selection or hyperparameter tuning.
# BiLSTM sees both directions **inside an already observed window only**.
#
# ### Proposed extension and ablation
#
# `paper` is the reference architecture above. `conditioned_kan` adds a small
# gate `1 + 0.5 tanh(MLP(operating_settings_t))` to each token's KAN output before
# BiLSTM. The three settings are observed inputs; removed constant settings map
# to zero. Gates start at one and stay in [0.5, 1.5]. The hypothesis is that an
# explicit setting-conditioned modulation can help FD002/FD004's changing
# conditions. This is a **proposed engineering extension, with no established
# novelty priority or performance gain**. No future cycles or target labels enter
# the gate. Both variants share splits, preprocessing and training settings.
#
# Train both variants, freeze the winner using **mean validation RMSE across the
# requested seeds**, and only then evaluate the official test set. A tie or worse
# extension retains `paper`; no test score chooses the winner. SHAP and LIME run
# after these stages and never influence fitting or variant selection.

# %% [markdown] id=manual-environment
# ## 2. Manual environment setup
#
# Use a CUDA-enabled PyTorch kernel (local NVIDIA GPU, Colab or Kaggle).
# In a new local terminal at the project root, the supplied Python 3.12 environment
# can be installed separately from the website:
#
# ```bash
# python3.12 -m venv .venv-kan
# .venv-kan/bin/python -m pip install --cache-dir .venv-kan/cache/pip torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
# .venv-kan/bin/python -m pip install --cache-dir .venv-kan/cache/pip -r requirements-kan.txt
# .venv-kan/bin/python -m ipykernel install --prefix .venv-kan --name kan-bilstm --display-name "KAN BiLSTM CUDA"
# .venv-kan/bin/python -m jupyterlab
# ```
#
# Select **KAN BiLSTM CUDA** as the kernel. CUDA wheels can download several GB.
# The command above targets CUDA 13.0; choose a wheel compatible with your NVIDIA
# driver using the [official PyTorch installer](https://pytorch.org/get-started/locally/)
# if your machine differs. On Windows use `.venv-kan\Scripts\python.exe`.
# In Colab/Kaggle, select a GPU runtime and use its compatible preinstalled torch;
# the notebook prints versions and fails clearly if CUDA is unavailable.
#
# This notebook uses [PyTorch Transformer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Transformer.html)
# and [torch.amp mixed precision](https://docs.pytorch.org/docs/2.14/amp.html).
# Code cells never invoke package installers or automatically fetch datasets.

# %% id=configuration
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd


IMPLEMENTATION_VERSION = "xu2025-transformer-kan-bilstm-v2"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
EXPECTED_ENGINES = {"FD001": (100, 100), "FD002": (260, 259),
                    "FD003": (100, 100), "FD004": (249, 248)}
PAPER_RMSE = {"FD001": 3.6784, "FD002": 4.9498, "FD003": 3.7988, "FD004": 7.5202}
RAW_COLUMNS = ["engine_id", "cycle", *[f"setting_{i}" for i in range(1, 4)],
               *[f"sensor_{i}" for i in range(1, 22)]]
FEATURE_COLUMNS = RAW_COLUMNS[2:]  # Never include engine ID, cycle or RUL as a feature.

# Edit these paths for Colab / Kaggle / another computer. A standalone copy works.
_project_root = next((p for p in [Path.cwd(), *Path.cwd().parents]
                      if (p / "data" / "cmapss").is_dir()), Path.cwd())
DATA_DIR = _project_root / "data" / "cmapss"
OUTPUT_DIR = _project_root / "artifacts" / "kan_bilstm"
SELECTED_SUBSETS = list(SUBSETS)  # Use ["FD001"] for your first complete training run.
SEEDS = [42]  # e.g. [42, 43, 44] for independently trained repeated runs.
VARIANTS = ["paper", "conditioned_kan"]  # Matched ablation, eight runs with four subsets and one seed.
RESUME = True  # Resume only an identical configuration and identical training file.
if not SELECTED_SUBSETS or len(set(SELECTED_SUBSETS)) != len(SELECTED_SUBSETS):
    raise ValueError("Choose a nonempty list of unique dataset subsets")
if not SEEDS or len(set(SEEDS)) != len(SEEDS):
    raise ValueError("Choose a nonempty list of unique training seeds")


@dataclass(frozen=True)
class Config:
    subset: str = "FD001"
    variant: str = "paper"
    seed: int = 42
    split_seed: int = 2025  # Kept fixed across training seeds.
    validation_fraction: float = 0.20
    validation_prefixes: tuple = (0.25, 0.40, 0.55, 0.70, 0.85, 1.0)
    window: int = 30
    stride: int = 1
    smoothing_window: int = 5
    normalization: str = "minmax"
    constant_tolerance: float = 1e-8
    rul_cap: float | None = None  # P4 does not specify a 125-cycle target cap.
    target_scale: float = 100.0
    horizons: int = 10
    d_model: int = 32
    n_heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    feedforward_dim: int = 32
    dropout: float = 0.2
    kan_hidden: int = 32
    grid_size: int = 200
    spline_degree: int = 3
    grid_update_epochs: tuple = (1, 5, 10)
    grid_sample_windows: int = 24
    grid_max_points: int = 512
    grid_uniform_mix: float = 0.10
    grid_ridge: float = 1e-5
    regularization_points: int = 32
    kan_regularization: float = 1e-5
    bilstm_hidden: int = 64
    bilstm_layers: int = 1
    learning_rate: float = 1e-4
    epochs: int = 50
    patience: int = 12
    min_delta: float = 1e-4
    effective_batch_size: int = 256
    micro_batch_size: int = 32  # 8 accumulated batches = 256; lower this for less VRAM.
    eval_batch_size: int = 32
    gradient_clip: float = 1.0
    amp: bool = True
    deterministic: bool = True
    num_workers: int = 0  # Notebook-safe on Linux and Windows.


def config_for(subset, seed=42, variant="paper"):
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset: {subset}")
    dimensions, heads, grid = {"FD001": (32, 8, 200), "FD002": (64, 8, 40),
                               "FD003": (32, 8, 200), "FD004": (128, 16, 50)}[subset]
    return Config(subset=subset, variant=variant, seed=seed, d_model=dimensions, n_heads=heads,
                  feedforward_dim=dimensions, kan_hidden=dimensions, grid_size=grid)


def validate_config(cfg):
    if not 0 <= cfg.seed <= 2147483647 or not 0 <= cfg.split_seed <= 2147483647:
        raise ValueError("Seeds must be nonnegative 32-bit integers")
    if cfg.variant not in {"paper", "conditioned_kan"}:
        raise ValueError("variant must be paper or conditioned_kan")
    if cfg.subset not in SUBSETS or cfg.d_model % cfg.n_heads:
        raise ValueError("Check dataset and attention-head divisibility")
    positive = ("window", "stride", "smoothing_window", "horizons", "d_model", "n_heads",
                "encoder_layers", "decoder_layers", "feedforward_dim", "kan_hidden", "grid_size",
                "spline_degree", "grid_sample_windows", "grid_max_points", "regularization_points",
                "bilstm_hidden", "bilstm_layers", "epochs", "patience", "effective_batch_size",
                "micro_batch_size", "eval_batch_size", "target_scale", "learning_rate", "gradient_clip")
    if any(getattr(cfg, field) <= 0 for field in positive):
        raise ValueError("All dimensions, batch sizes, learning rates and epochs must be positive")
    if cfg.effective_batch_size % cfg.micro_batch_size:
        raise ValueError("effective_batch_size must be a multiple of micro_batch_size")
    if not 0 < cfg.validation_fraction < 0.5 or not 0 <= cfg.dropout < 1:
        raise ValueError("Invalid validation fraction or dropout")
    if not cfg.validation_prefixes or any(not 0 < x <= 1 for x in cfg.validation_prefixes):
        raise ValueError("Validation prefix fractions must be in (0, 1]")
    if cfg.normalization not in ("minmax", "standard") or not 0 < cfg.grid_uniform_mix <= 1:
        raise ValueError("Invalid normalization or spline-grid uniform mix")
    if cfg.grid_ridge <= 0 or cfg.kan_regularization < 0 or cfg.num_workers != 0:
        raise ValueError("Use positive grid ridge, nonnegative regularization, and num_workers=0")
    if cfg.rul_cap is not None and cfg.rul_cap <= 0:
        raise ValueError("rul_cap must be positive or None")


CONFIGS = {subset: config_for(subset, SEEDS[0]) for subset in SELECTED_SUBSETS}
for _cfg in CONFIGS.values():
    validate_config(_cfg)
print("Data:", DATA_DIR)
print("Outputs:", OUTPUT_DIR)
print(pd.DataFrame([{k: asdict(c)[k] for k in
                     ("subset", "d_model", "n_heads", "grid_size", "epochs", "micro_batch_size",
                      "effective_batch_size", "rul_cap")} for c in CONFIGS.values()]).to_string(index=False))

# %% [markdown] id=data-protocol
# ## 3. Data, targets and train-only preprocessing
#
# Place the 12 supplied NASA files in `DATA_DIR`: `train_FD001.txt`,
# `test_FD001.txt`, `RUL_FD001.txt`, and the corresponding three files for each of
# FD002–FD004. Folder paths are configurable above; nothing is downloaded.
#
# Each sensor file must have exactly **26 columns**: engine ID, cycle, three
# settings and 21 sensors. Raw sensor names are preserved instead of inventing
# engineering labels from the paper's inconsistent feature table. Training files
# contain complete run-to-failure trajectories; a training target at cycle t is
# `last_training_cycle - t`. Official test trajectories are truncated; the supplied
# `RUL_FDxxx.txt` contains one **endpoint** target per test engine, read only during
# final evaluation.
#
# Engine identities are split **before** scaling/windowing. Constant-column
# detection and normalization use fit engines only. Smoothing is a trailing mean
# within each engine, so it never uses future readings. Window padding repeats
# the first observed row only when fewer than `window` cycles are available.
# No interpolation, centered smoothing, trajectory-wide normalization or random
# splitting of overlapping windows is used. Validation prefixes are fixed and
# saved; early stopping uses current-cycle RMSE at these held-out engine prefixes.

# %% id=data-functions
def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_sensor_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; set DATA_DIR to your C-MAPSS folder")
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != 26 or frame.empty:
        raise ValueError(f"{path.name}: expected nonempty 26-column C-MAPSS data, got {frame.shape}")
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name}: all 26 columns must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: missing/infinite values; repair the source file explicitly")
    integer_values = values[:, :2]
    if np.any(integer_values < 1) or np.any(integer_values != np.floor(integer_values)):
        raise ValueError(f"{path.name}: engine IDs and cycles must be positive integers")
    frame = pd.DataFrame(values, columns=RAW_COLUMNS)
    frame[["engine_id", "cycle"]] = frame[["engine_id", "cycle"]].astype(np.int64)
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError(f"{path.name}: duplicate engine/cycle records")
    frame = frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    ids = frame.engine_id.unique()
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise ValueError(f"{path.name}: engine IDs must be consecutive starting at 1")
    for engine_id, group in frame.groupby("engine_id", sort=True):
        if not np.array_equal(group.cycle.to_numpy(), np.arange(1, len(group) + 1)):
            raise ValueError(f"{path.name}: engine {engine_id} has missing/noncontiguous cycles")
    return frame


def split_training_engines(frame, cfg):
    ids = frame.engine_id.unique().astype(int)
    if len(ids) < 10:
        raise ValueError("At least 10 complete engines are required for this experiment")
    shuffled = np.random.default_rng(cfg.split_seed).permutation(ids)
    n_validation = max(2, int(math.ceil(len(ids) * cfg.validation_fraction)))
    valid = sorted(map(int, shuffled[:n_validation]))
    fit = sorted(map(int, shuffled[n_validation:]))
    assert not set(fit) & set(valid)
    assert set(fit) | set(valid) == set(ids)
    return {"fit_engine_ids": fit, "validation_engine_ids": valid,
            "split_seed": cfg.split_seed, "source": f"train_{cfg.subset}.txt"}


def smooth_engine_features(frame, cfg):
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    # groupby.transform preserves row order and never crosses engine boundaries.
    return features.groupby(frame.engine_id, sort=False).transform(
        lambda group: group.rolling(cfg.smoothing_window, min_periods=1).mean())


class TrainOnlyPreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, frame, fit_ids):
        fit = frame.loc[frame.engine_id.isin(fit_ids)]
        if fit.empty or set(fit.engine_id.unique()) != set(fit_ids):
            raise ValueError("Fit engine IDs are missing from the training file")
        smoothed = smooth_engine_features(fit, self.cfg)
        spread = smoothed.max() - smoothed.min()
        self.columns = spread[spread > self.cfg.constant_tolerance].index.tolist()
        if not self.columns:
            raise ValueError("All feature columns are constant")
        selected = smoothed[self.columns]
        if self.cfg.normalization == "minmax":
            self.center = ((selected.max() + selected.min()) / 2).to_numpy(dtype=np.float64)
            self.scale = ((selected.max() - selected.min()) / 2).to_numpy(dtype=np.float64)
        else:
            self.center = selected.mean().to_numpy(dtype=np.float64)
            self.scale = selected.std(ddof=0).to_numpy(dtype=np.float64)
        self.scale = np.maximum(self.scale, self.cfg.constant_tolerance)
        self.fit_ids = sorted(map(int, fit_ids))
        return self

    def transform(self, frame):
        smoothed = smooth_engine_features(frame, self.cfg)
        result = (smoothed[self.columns].to_numpy(dtype=np.float64) - self.center) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("Preprocessing produced a nonfinite value")
        # Do not re-fit or clip using validation/test distribution information.
        return result.astype(np.float32)

    def to_dict(self):
        return {"columns": self.columns, "center": self.center.tolist(), "scale": self.scale.tolist(),
                "fit_engine_ids": self.fit_ids, "normalization": self.cfg.normalization,
                "smoothing_window": self.cfg.smoothing_window,
                "removed_columns": [column for column in FEATURE_COLUMNS if column not in self.columns]}

    @classmethod
    def from_dict(cls, cfg, state):
        obj = cls(cfg)
        obj.columns = list(state["columns"])
        obj.center = np.asarray(state["center"], dtype=np.float64)
        obj.scale = np.asarray(state["scale"], dtype=np.float64)
        obj.fit_ids = list(state["fit_engine_ids"])
        if (not obj.columns or len(obj.center) != len(obj.columns) or
                len(obj.scale) != len(obj.columns) or not np.isfinite(obj.center).all() or
                not np.isfinite(obj.scale).all() or np.any(obj.scale <= 0)):
            raise ValueError("Invalid saved preprocessor")
        return obj


def make_engine_arrays(frame, preprocessor):
    frame = frame.reset_index(drop=True)
    transformed = preprocessor.transform(frame)
    return {int(engine_id): {"x": transformed[group.index.to_numpy()].copy(),
                             "cycles": group.cycle.to_numpy(dtype=np.int64)}
            for engine_id, group in frame.groupby("engine_id", sort=True)}


def condition_indices(columns):
    return [columns.index(f"setting_{i}") if f"setting_{i}" in columns else -1 for i in range(1, 4)]


def mix_sensor_windows(window, backgrounds, masks, indexes):
    """Grouped explanation perturbations; keep setting columns and within-sensor history intact."""
    window = np.asarray(window, dtype=np.float32)
    backgrounds = np.asarray(backgrounds, dtype=np.float32)
    masks = np.atleast_2d(np.asarray(masks, dtype=np.float32))
    indexes = np.asarray(indexes, dtype=int)
    if (window.ndim != 2 or backgrounds.ndim != 3 or backgrounds.shape[1:] != window.shape or
            masks.shape[1] != len(indexes) or len(set(indexes)) != len(indexes) or
            np.any(indexes < 0) or np.any(indexes >= window.shape[1]) or
            not np.isfinite(masks).all() or np.any((masks < 0) | (masks > 1))):
        raise ValueError("Invalid grouped-window perturbation inputs")
    perturbed = np.broadcast_to(window, (len(masks), len(backgrounds), *window.shape)).copy()
    keep = masks[:, None, None, :]
    perturbed[..., indexes] = (keep * np.take(window, indexes, axis=-1)[None, None, :, :]
                               + (1 - keep) * np.take(backgrounds, indexes, axis=-1)[None, :, :, :])
    return perturbed


def historical_window(features, end_index, window):
    if not 0 <= end_index < len(features):
        raise IndexError("Window endpoint is outside the observed trajectory")
    values = features[max(0, end_index - window + 1):end_index + 1]
    if len(values) < window:
        values = np.concatenate([np.repeat(values[:1], window - len(values), axis=0), values], axis=0)
    return np.ascontiguousarray(values, dtype=np.float32)


def horizon_targets(remaining_cycles, cfg):
    horizons = np.arange(cfg.horizons, dtype=np.float32)
    valid = (horizons <= remaining_cycles).astype(np.float32)
    target = np.maximum(float(remaining_cycles) - horizons, 0)
    if cfg.rul_cap is not None:
        target = np.minimum(target, cfg.rul_cap)
    return (target / cfg.target_scale).astype(np.float32), valid


def regression_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if (truth.ndim != 1 or truth.shape != prediction.shape or not len(truth) or
            not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError("Metrics need aligned, finite, nonempty 1-D arrays")
    error = prediction - truth
    denominator = float(np.square(truth - truth.mean()).sum())
    # Positive error = overestimating life (late), penalized more strongly.
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    with np.errstate(over="ignore"):
        score = float(np.expm1(exponent).sum())
    nonzero = truth != 0
    return {"n_engines": int(len(truth)), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "r2": float(1 - np.square(error).sum() / denominator) if denominator else None,
            "mape_percent_nonzero": float(np.mean(np.abs(error[nonzero] / truth[nonzero])) * 100)
            if nonzero.any() else None,
            "mape_nonzero_count": int(nonzero.sum()), "nasa_score": score if math.isfinite(score) else None,
            "nasa_overflow": not math.isfinite(score)}

# %% id=data-audit
if not SELECTED_SUBSETS or len(SELECTED_SUBSETS) != len(set(SELECTED_SUBSETS)):
    raise ValueError("Choose a nonempty list of unique dataset subsets")
if not SEEDS or len(SEEDS) != len(set(SEEDS)):
    raise ValueError("Choose a nonempty list of unique training seeds")

TRAIN_FRAMES = {}
audit_rows = []
for subset in SELECTED_SUBSETS:
    cfg = CONFIGS[subset]
    for filename in (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"):
        if not (DATA_DIR / filename).is_file():
            raise FileNotFoundError(f"Missing {DATA_DIR / filename}")
    # This cell intentionally does not read test features or evaluation targets.
    frame = load_sensor_file(DATA_DIR / f"train_{subset}.txt")
    TRAIN_FRAMES[subset] = frame
    split = split_training_engines(frame, cfg)
    preprocessor = TrainOnlyPreprocessor(cfg).fit(frame, split["fit_engine_ids"])
    audit_rows.append({"subset": subset, "training_rows": len(frame),
                       "training_engines": frame.engine_id.nunique(),
                       "fit_engines": len(split["fit_engine_ids"]),
                       "validation_engines": len(split["validation_engine_ids"]),
                       "selected_features": len(preprocessor.columns),
                       "removed_constants": ", ".join(preprocessor.to_dict()["removed_columns"])})
    assert frame.engine_id.nunique() == EXPECTED_ENGINES[subset][0], "Check your NASA dataset version"
print(pd.DataFrame(audit_rows).to_string(index=False))

# %% id=data-sanity-checks
def check_data_invariants():
    cfg = config_for("FD001")
    synthetic_rows = []
    for engine_id in range(1, 11):
        for cycle in range(1, 41):
            synthetic_rows.append([engine_id, cycle, *[engine_id + cycle * (j + 1) / 100
                                                       for j in range(24)]])
    synthetic = pd.DataFrame(synthetic_rows, columns=RAW_COLUMNS)
    split = split_training_engines(synthetic, cfg)
    fit_ids = split["fit_engine_ids"]
    p = TrainOnlyPreprocessor(cfg).fit(synthetic, fit_ids)
    changed = synthetic.copy()
    changed.loc[changed.engine_id.isin(split["validation_engine_ids"]), FEATURE_COLUMNS] += 1e6
    p2 = TrainOnlyPreprocessor(cfg).fit(changed, fit_ids)
    assert p.to_dict() == p2.to_dict(), "Held-out engines influenced fitted preprocessing"
    one = synthetic[synthetic.engine_id == 1].reset_index(drop=True)
    full = p.transform(one)
    prefix = p.transform(one.iloc[:25].reset_index(drop=True))
    np.testing.assert_allclose(full[:25], prefix, rtol=0, atol=0)
    np.testing.assert_array_equal(historical_window(full, 24, 30), historical_window(prefix, 24, 30))
    targets, mask = horizon_targets(2, cfg)
    np.testing.assert_allclose(targets[:4] * cfg.target_scale, [2, 1, 0, 0])
    np.testing.assert_array_equal(mask[:4], [1, 1, 1, 0])
    assert regression_metrics([10], [11])["nasa_score"] > regression_metrics([10], [9])["nasa_score"]
    assert regression_metrics([10, 20], [10, 20])["rmse"] == 0
    assert not set(fit_ids) & set(split["validation_engine_ids"])
    print("Passed: engine split, train-only normalization, causal smoothing/windows, targets and metric direction.")


check_data_invariants()

# %% [markdown] id=cuda-notes
# ## 4. CUDA runtime
#
# An NVIDIA CUDA device is required; there is no silent CPU training fallback.
# AMP uses BF16 where supported, otherwise FP16 with GradScaler. Spline recursion,
# edge regularization, grid refitting and BiLSTM run in FP32 for numerical stability.
# Validation, final inference, SHAP and LIME use FP32 for consistent reported outputs.
# The microbatch is 32 with accumulation to an effective 256. For limited VRAM,
# change `micro_batch_size` to 16 or 8 in `Config`, and `eval_batch_size` likewise.
# This preserves the architecture and effective batch. No runtime/VRAM claim is
# made before you run it. Subsets and seeds are trained sequentially.

# %% id=cuda-runtime
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError("Select your CUDA kernel after installing requirements-kan.txt manually") from exc

if not torch.cuda.is_available():
    raise RuntimeError(f"CUDA unavailable in {sys.executable}. Check GPU runtime, driver and CUDA torch wheel.")

DEVICE = torch.device("cuda:0")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print({"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0), "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
       "amp_dtype": str(AMP_DTYPE)})


def seed_everything(seed, deterministic=True):
    # Configure before the first CUDA matrix multiplication.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


seed_everything(SEEDS[0])

# %% id=window-datasets
class TrainingWindows(Dataset):
    """Lazy windows: store each engine once instead of materializing all overlapping windows."""
    def __init__(self, engines, ids, cfg, validation=False):
        self.engines, self.cfg = engines, cfg
        self.records = []
        for engine_id in sorted(ids):
            count = len(engines[engine_id]["x"])
            if validation:
                ends = sorted({max(0, min(count - 1, int(round(count * fraction)) - 1))
                               for fraction in cfg.validation_prefixes})
            else:
                # Include short-prefix windows, matching the padding used at inference.
                ends = list(range(0, count, cfg.stride))
                if ends[-1] != count - 1:
                    ends.append(count - 1)
            self.records.extend((int(engine_id), int(end)) for end in ends)
        if not self.records:
            raise ValueError("No windows available")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        engine_id, end = self.records[index]
        engine = self.engines[engine_id]
        remaining = int(engine["cycles"][-1] - engine["cycles"][end])
        target, mask = horizon_targets(remaining, self.cfg)
        x = historical_window(engine["x"], end, self.cfg.window)
        return torch.from_numpy(x), torch.from_numpy(target), torch.from_numpy(mask)


class InferenceWindows(Dataset):
    """Sensor windows only: no evaluation RUL labels can become model inputs."""
    def __init__(self, engines, records, cfg):
        self.engines, self.records, self.cfg = engines, list(records), cfg

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        engine_id, end = self.records[index]
        return torch.from_numpy(historical_window(self.engines[engine_id]["x"], end, self.cfg.window))

# %% [markdown] id=kan-mathematics
# ## 5. Spline KAN: learned univariate edges
#
# For each layer, `y_j = Σ_i [w_ji SiLU(x_i) + Σ_k c_jik B_ik(x_i)]`.
# The B-spline basis uses the paper's Cox–de Boor recurrence (Eq. 3); the spline
# coefficients and SiLU branch weights are trainable. Knots are buffers saved in
# checkpoints, not trainable parameters. Scheduled quantile updates use **only fit
# engine activations**, blend in a uniform grid to prevent collapsed intervals,
# and refit old spline curves with ridge-stabilized least squares. Optimizer state
# for the refitted coefficients is reset after regridding.
#
# Activation regularization computes mean absolute **edge outputs**, including
# both SiLU and spline terms, then an L1 sum and normalized edge entropy (p. 16).
# A deterministic sample of at most 32 token activations bounds the cost. This is
# activation regularization, not a coefficient-only substitute. Grid-update timing,
# λ, cubic degree and numerical stabilization are documented engineering choices.

# %% id=kan-model
class SplineKANLayer(nn.Module):
    def __init__(self, inputs, outputs, cfg):
        super().__init__()
        self.inputs, self.outputs, self.cfg = inputs, outputs, cfg
        self.grid_size, self.degree = cfg.grid_size, cfg.spline_degree
        knots = torch.arange(-self.degree, self.grid_size + self.degree + 1, dtype=torch.float32)
        knots = knots * (2.0 / self.grid_size) - 1.0
        self.register_buffer("knots", knots.expand(inputs, -1).clone())
        self.base_weight = nn.Parameter(torch.empty(outputs, inputs))
        self.coefficients = nn.Parameter(torch.empty(outputs, inputs, self.grid_size + self.degree))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.coefficients, std=0.005)
        self.last_regularization = None

    def basis(self, x):
        # x: [points, inputs], basis: [points, inputs, grid_size + degree].
        x = x.float().unsqueeze(-1)
        knots = self.knots
        values = ((x >= knots[:, :-1]) & (x < knots[:, 1:])).float()
        for degree in range(1, self.degree + 1):
            left_den = (knots[:, degree:-1] - knots[:, :-degree - 1]).clamp_min(1e-8)
            right_den = (knots[:, degree + 1:] - knots[:, 1:-degree]).clamp_min(1e-8)
            values = ((x - knots[:, :-degree - 1]) / left_den * values[..., :-1]
                      + (knots[:, degree + 1:] - x) / right_den * values[..., 1:])
        return values.contiguous()

    def edges(self, x):
        # [points, inputs, outputs], used on a small sample for the paper's penalty.
        basis = self.basis(x)
        spline = torch.einsum("nik,oik->nio", basis, self.coefficients)
        return F.silu(x).unsqueeze(-1) * self.base_weight.T.unsqueeze(0) + spline

    def forward(self, x):
        shape = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            flat = x.float().reshape(-1, self.inputs)
            basis = self.basis(flat)
            result = (F.linear(F.silu(flat), self.base_weight, self.bias)
                      + F.linear(basis.flatten(1), self.coefficients.flatten(1)))
            if self.training:
                n = min(self.cfg.regularization_points, len(flat))
                indexes = torch.linspace(0, len(flat) - 1, n, device=flat.device).long()
                amplitudes = self.edges(flat[indexes]).abs().mean(0)
                l1 = amplitudes.sum()
                probabilities = amplitudes / l1.clamp_min(1e-12)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                self.last_regularization = l1 + entropy
            else:
                self.last_regularization = None
            return result.reshape(*shape[:-1], self.outputs)

    @torch.no_grad()
    def adapt_grid(self, inputs):
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            x = inputs.float().reshape(-1, self.inputs)
            indexes = torch.linspace(0, len(x) - 1, min(len(x), self.cfg.grid_max_points), device=x.device).long()
            x = x[indexes]
            old_curves = torch.einsum("nik,oik->nio", self.basis(x), self.coefficients)
            fractions = torch.linspace(0, 1, self.grid_size + 1, device=x.device)
            quantiles = torch.quantile(x, fractions, dim=0).T.contiguous()
            low, high = x.amin(0) - 0.02, x.amax(0) + 0.02
            uniform = low[:, None] + (high - low)[:, None] * fractions[None, :]
            interior = ((1 - self.cfg.grid_uniform_mix) * quantiles
                        + self.cfg.grid_uniform_mix * uniform)
            step = (high - low) / self.grid_size
            left = interior[:, :1] - step[:, None] * torch.arange(self.degree, 0, -1, device=x.device)
            right = interior[:, -1:] + step[:, None] * torch.arange(1, self.degree + 1, device=x.device)
            new_knots = torch.cat([left, interior, right], dim=1)
            old_knots = self.knots.clone()
            old_coefficients = self.coefficients.detach().clone()
            try:
                self.knots.copy_(new_knots)
                design = self.basis(x).transpose(0, 1).contiguous()  # [input, point, basis]
                target = old_curves.transpose(0, 1).contiguous()  # [input, point, output]
                count = design.shape[-1]
                # Ridge via augmentation keeps the CUDA least-squares system full rank.
                ridge = math.sqrt(self.cfg.grid_ridge) * torch.eye(count, device=x.device)
                augmented = torch.cat([design, ridge.expand(self.inputs, -1, -1)], dim=1)
                target = torch.cat([target, torch.zeros(self.inputs, count, self.outputs, device=x.device)], dim=1)
                solution = torch.linalg.lstsq(augmented, target).solution
                replacement = solution.permute(2, 0, 1).contiguous()
                if not torch.isfinite(replacement).all():
                    raise FloatingPointError("Nonfinite spline-grid refit")
                self.coefficients.copy_(replacement)
                residual = (torch.einsum("nik,oik->nio", self.basis(x), self.coefficients) - old_curves)
                return float(torch.sqrt(residual.square().mean()).cpu())
            except Exception:
                self.knots.copy_(old_knots)
                self.coefficients.copy_(old_coefficients)
                raise


class TemporalKAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        widths = [cfg.d_model, cfg.kan_hidden, cfg.d_model]
        self.norms = nn.ModuleList(nn.LayerNorm(width) for width in widths[:-1])
        self.layers = nn.ModuleList(SplineKANLayer(a, b, cfg) for a, b in zip(widths[:-1], widths[1:]))

    def forward(self, sequence):
        for norm, layer in zip(self.norms, self.layers):
            sequence = layer(torch.tanh(norm(sequence.float())))
        return sequence

    def regularization(self):
        terms = [layer.last_regularization for layer in self.layers if layer.last_regularization is not None]
        if not terms:
            return next(self.parameters()).new_zeros(())
        return torch.stack(terms).sum()

    @torch.no_grad()
    def adapt_grids(self, sequence):
        residuals = []
        for norm, layer in zip(self.norms, self.layers):
            inputs = torch.tanh(norm(sequence.float()))
            residuals.append(layer.adapt_grid(inputs))
            sequence = layer(inputs)
        return residuals

# %% id=transformer-bilstm
class SinusoidalPosition(nn.Module):
    def __init__(self, dimensions, length):
        super().__init__()
        position = torch.arange(length).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, dimensions, 2).float() * (-math.log(10000.0) / dimensions))
        encoding = torch.zeros(1, length, dimensions)
        encoding[0, :, 0::2] = torch.sin(position * frequency)
        encoding[0, :, 1::2] = torch.cos(position * frequency[:encoding[0, :, 1::2].shape[1]])
        self.register_buffer("encoding", encoding)

    def forward(self, sequence):
        return sequence + self.encoding[:, :sequence.shape[1]].to(sequence.dtype)


class TransformerKANBiLSTM(nn.Module):
    def __init__(self, input_features, cfg, setting_indices=None):
        super().__init__()
        self.cfg = cfg
        self.projection = nn.Linear(input_features, cfg.d_model)
        self.position = SinusoidalPosition(cfg.d_model, cfg.window)
        self.start_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.transformer = nn.Transformer(
            d_model=cfg.d_model, nhead=cfg.n_heads, num_encoder_layers=cfg.encoder_layers,
            num_decoder_layers=cfg.decoder_layers, dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout, activation="relu", batch_first=True, norm_first=False)
        self.kan = TemporalKAN(cfg)
        self.setting_indices = tuple(setting_indices if setting_indices is not None else (0, 1, 2))
        if len(self.setting_indices) != 3 or any(i < -1 or i >= input_features for i in self.setting_indices):
            raise ValueError("Three valid setting feature indices are required; use -1 for a removed constant")
        self.bilstm = nn.LSTM(cfg.d_model, cfg.bilstm_hidden, num_layers=cfg.bilstm_layers,
                              batch_first=True, bidirectional=True,
                              dropout=cfg.dropout if cfg.bilstm_layers > 1 else 0)
        self.dropout = nn.Dropout(cfg.dropout)
        self.regression = nn.Linear(2 * cfg.bilstm_hidden, cfg.horizons)
        self.register_buffer("decoder_mask", torch.triu(torch.ones(cfg.window, cfg.window, dtype=torch.bool), 1))
        # Construct after shared layers so both variants start with matched shared weights for a fixed seed.
        if cfg.variant == "conditioned_kan":
            self.condition_gate = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, cfg.d_model))
            nn.init.zeros_(self.condition_gate[-1].weight)
            nn.init.zeros_(self.condition_gate[-1].bias)

    def temporal_embedding(self, x):
        embedded = self.projection(x) * math.sqrt(self.cfg.d_model)
        source = self.position(embedded)
        # Decoder receives historical sensor tokens, NEVER RUL labels or future sensor values.
        shifted = torch.cat([self.start_token.expand(x.shape[0], -1, -1), embedded[:, :-1]], dim=1)
        target = self.position(shifted)
        return self.transformer(src=source, tgt=target,
                                tgt_mask=self.decoder_mask[:x.shape[1], :x.shape[1]])

    def forward(self, x):
        sequence = self.temporal_embedding(x)
        sequence = self.kan(sequence)
        if self.cfg.variant == "conditioned_kan":
            with torch.autocast(device_type=x.device.type, enabled=False):
                settings = torch.stack([x[..., i].float() if i >= 0 else torch.zeros_like(x[..., 0]).float()
                                        for i in self.setting_indices], dim=-1)
                sequence = sequence * (1 + 0.5 * torch.tanh(self.condition_gate(settings)))
        # FP32 cuDNN LSTM avoids unsupported mixed-precision recurrent kernels.
        with torch.autocast(device_type=x.device.type, enabled=False):
            _, (hidden, _) = self.bilstm(sequence.float())
            # hidden[-2] and hidden[-1] are the final forward/backward states, not output[:, -1].
            fused = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            return F.softplus(self.regression(self.dropout(fused)))

    @torch.no_grad()
    def adapt_kan_grids(self, x):
        was_training = self.training
        self.eval()
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self.kan.adapt_grids(self.temporal_embedding(x.float()))
        finally:
            self.train(was_training)

# %% [markdown] id=gpu-checks-notes
# ## 6. GPU correctness checks (no optimizer or model fitting)
#
# These tests check the real spline basis, finite gradients, tensor shapes and
# checkpoint round-trip when **you** run them. They do not measure prediction
# quality. The tiny test architecture is temporary; training uses the full
# per-subset configuration above.

# %% id=gpu-sanity-checks
def check_gpu_implementation():
    cfg = replace(config_for("FD001"), window=8, d_model=16, n_heads=4, feedforward_dim=16,
                  encoder_layers=1, decoder_layers=1, kan_hidden=16, grid_size=5, bilstm_hidden=8)
    layer = SplineKANLayer(3, 2, cfg).to(DEVICE)
    x = torch.linspace(-0.95, 0.95, 64, device=DEVICE)[:, None].repeat(1, 3)
    torch.testing.assert_close(layer.basis(x).sum(-1), torch.ones_like(x), rtol=1e-5, atol=1e-5)
    layer(x).square().mean().backward()
    assert layer.coefficients.grad is not None and torch.isfinite(layer.coefficients.grad).all()
    layer.eval()
    before = layer(x).detach().clone()
    residual = layer.adapt_grid(x)
    assert math.isfinite(residual)
    torch.testing.assert_close(before, layer(x), rtol=0.15, atol=0.02)
    model = TransformerKANBiLSTM(7, cfg).to(DEVICE)
    sample = torch.randn(2, cfg.window, 7, device=DEVICE)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        prediction = model(sample)
        objective = prediction.square().mean() + cfg.kan_regularization * model.kan.regularization()
    assert prediction.shape == (2, 10) and torch.isfinite(prediction).all()
    objective.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    model.eval()
    replica = TransformerKANBiLSTM(7, cfg).to(DEVICE).eval()
    replica.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(sample), replica(sample))
    novel = TransformerKANBiLSTM(7, replace(cfg, variant="conditioned_kan")).to(DEVICE).eval()
    novel.load_state_dict(model.state_dict(), strict=False)
    torch.testing.assert_close(model(sample), novel(sample))  # Initial gate is exactly one.
    del novel
    del model, replica, sample, layer, x, prediction, objective
    gc.collect()
    torch.cuda.empty_cache()
    print("Passed GPU spline, gradient, AMP, shape and state_dict round-trip checks. No model training performed.")


check_gpu_implementation()

# %% [markdown] id=training-notes
# ## 7. Training and checkpointing
#
# Loss is the mean, per training window, of squared error over valid forecast
# horizons, plus the paper's KAN L1/entropy term. Targets are divided by
# `target_scale`; exported predictions and every reported RUL metric are in
# **cycles**. Post-failure horizons contribute no loss. The checkpoint with lowest
# held-out **current-cycle validation RMSE** is selected; official test labels
# never enter this function.
#
# `best.pt` contains inference weights; `last.pt` also contains Adam, AMP and RNG
# states for an epoch-boundary resume. Each run saves configuration, split IDs,
# validation endpoints, preprocessing, source-data hashes, environment, training
# history and status. An interrupted epoch is rerun from its previous completed
# checkpoint. Different configurations/data hashes get separate directories.
# Re-running an already completed identical run reuses it when `RESUME=True`.
# Set `RESUME=False` to refuse reuse; change the seed/config to make another run.

# %% id=training-functions
def cpu_model_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def capture_rng():
    name, state, position, has_gauss, cached_gaussian = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": {"name": name, "state": state.tolist(), "position": position,
                      "has_gauss": has_gauss, "cached_gaussian": cached_gaussian}}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    value = state["numpy"]
    np.random.set_state((value["name"], np.asarray(value["state"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))


def make_loader(dataset, cfg, shuffle=False, epoch=0):
    generator = torch.Generator().manual_seed(cfg.seed + epoch)
    return DataLoader(dataset, batch_size=cfg.micro_batch_size if shuffle else cfg.eval_batch_size,
                      shuffle=shuffle, drop_last=False, pin_memory=True,
                      num_workers=0, generator=generator)


@torch.inference_mode()
def validate_current_cycle(model, dataset, cfg):
    model.eval()
    truths, predictions = [], []
    for x, target, _ in make_loader(dataset, cfg):
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=False):
            output = model(x)
        predictions.append(output[:, 0].float().cpu().numpy() * cfg.target_scale)
        truths.append(target[:, 0].numpy() * cfg.target_scale)
    truth, prediction = np.concatenate(truths), np.concatenate(predictions)
    metrics = regression_metrics(truth, prediction)
    if not math.isfinite(metrics["rmse"]):
        raise FloatingPointError("Nonfinite validation RMSE")
    # Each validation engine contributes up to six fixed historical prefixes.
    metrics["n_prefixes"] = metrics.pop("n_engines")
    records = pd.DataFrame([{"engine_id": engine_id,
                              "cycle": int(dataset.engines[engine_id]["cycles"][end])}
                             for engine_id, end in dataset.records])
    records["true_rul"] = truth
    records["predicted_rul"] = prediction
    return metrics, records


def update_training_grids(model, optimizer, dataset, cfg, epoch):
    rng = np.random.default_rng(cfg.seed + 1009 * epoch)
    indexes = rng.choice(len(dataset), min(cfg.grid_sample_windows, len(dataset)), replace=False)
    batch = torch.stack([dataset[int(index)][0] for index in indexes]).to(DEVICE)
    residuals = model.adapt_kan_grids(batch)
    for layer in model.kan.layers:
        optimizer.state.pop(layer.coefficients, None)
    del batch
    return residuals


def train_epoch(model, dataset, optimizer, scaler, cfg, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loader = make_loader(dataset, cfg, shuffle=True, epoch=epoch)
    accumulation = cfg.effective_batch_size // cfg.micro_batch_size
    seen, total_mse, total_regularization, successful_steps, skipped_steps = 0, 0.0, 0.0, 0, 0
    group_size = 0
    for batch_index, (x, target, valid) in enumerate(loader):
        if batch_index % accumulation == 0:
            group_size = min(cfg.effective_batch_size, len(dataset) - seen)
        x, target, valid = [value.to(DEVICE, non_blocking=True) for value in (x, target, valid)]
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=cfg.amp):
            prediction = model(x)
            per_window_mse = ((prediction.float() - target.float()).square() * valid).sum(1) / valid.sum(1).clamp_min(1)
            regularization = model.kan.regularization()
            # Weight the last partial accumulation group correctly, by its actual sample count.
            objective = (per_window_mse.sum() + cfg.kan_regularization * regularization * len(x)) / group_size
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite objective at epoch {epoch}, batch {batch_index}")
        scaler.scale(objective).backward()
        count = len(x)
        seen += count
        total_mse += float(per_window_mse.detach().sum().cpu())
        total_regularization += float(regularization.detach().cpu()) * count
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip, error_if_nonfinite=False)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise FloatingPointError("Nonfinite gradient; try amp=False and inspect input scaling")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped_steps += 1
            else:
                successful_steps += 1
            optimizer.zero_grad(set_to_none=True)
        # Break references to saved activation graphs between microbatches.
        for layer in model.kan.layers:
            layer.last_regularization = None
        if (batch_index + 1) % 200 == 0:
            print(f"  epoch {epoch}: {seen:,}/{len(dataset):,} windows", flush=True)
    if successful_steps == 0:
        raise FloatingPointError("Every optimizer update was skipped; disable AMP and inspect numerics")
    return {"train_mse_scaled": total_mse / seen, "kan_regularization": total_regularization / seen,
            "windows": seen, "optimizer_steps": successful_steps, "amp_skipped_steps": skipped_steps}


def environment_snapshot():
    packages = {}
    for package in ("torch", "numpy", "pandas", "matplotlib", "jupyterlab", "ipykernel", "nbformat"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "amp_dtype": str(AMP_DTYPE), "created_at": utc_now()}


def train_subset(cfg, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, resume=RESUME):
    """Reads train_<subset>.txt ONLY. Does not access official test data or RUL targets."""
    validate_config(cfg)
    source = Path(data_dir) / f"train_{cfg.subset}.txt"
    identity = {"implementation": IMPLEMENTATION_VERSION, "config": asdict(cfg),
                "training_file_sha256": sha256_file(source)}
    signature = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    run_dir = Path(output_dir) / cfg.subset / f"seed-{cfg.seed}-{signature[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing.get("signature") != signature:
            raise RuntimeError("Existing artifact identity differs; select a new output directory")
        if not resume:
            raise FileExistsError(f"Run already exists: {run_dir}; use RESUME=True or change seed/config")
        if existing.get("status") == "completed":
            for filename, expected in existing.get("artifact_hashes", {}).items():
                if not (run_dir / filename).is_file() or sha256_file(run_dir / filename) != expected:
                    raise RuntimeError(f"Completed artifact is missing or modified: {filename}")
            if sha256_file(run_dir / "best.pt") != existing.get("best_sha256"):
                raise RuntimeError("Completed model checkpoint is modified")
            print(f"Reusing completed run: {run_dir}")
            return run_dir

    frame = load_sensor_file(source)
    split = split_training_engines(frame, cfg)
    preprocessor = TrainOnlyPreprocessor(cfg).fit(frame, split["fit_engine_ids"])
    engines = make_engine_arrays(frame, preprocessor)
    training = TrainingWindows(engines, split["fit_engine_ids"], cfg)
    validation = TrainingWindows(engines, split["validation_engine_ids"], cfg, validation=True)
    seed_everything(cfg.seed, cfg.deterministic)
    model = TransformerKANBiLSTM(len(preprocessor.columns), cfg, condition_indices(preprocessor.columns)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and AMP_DTYPE == torch.float16)
    history, best, best_epoch, bad_epochs, start_epoch = [], float("inf"), 0, 0, 1
    best_validation_metrics = None
    if resume and (run_dir / "last.pt").is_file():
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise RuntimeError("Resume checkpoint identity differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        history, best, best_epoch = saved["history"], saved["best_rmse"], saved["best_epoch"]
        bad_epochs, start_epoch = saved["bad_epochs"], saved["epoch"] + 1
        best_validation_metrics = saved["best_validation_metrics"]
        print(f"Resuming {cfg.subset} seed {cfg.seed} at epoch {start_epoch}")
        del saved

    save_json(run_dir / "config.json", asdict(cfg))
    save_json(run_dir / "split.json", split)
    save_json(run_dir / "preprocessor.json", preprocessor.to_dict())
    save_json(run_dir / "data_manifest.json", {"training_file": source.name, "sha256": identity["training_file_sha256"],
                                             "rows": len(frame), "engines": len(engines),
                                             "training_source": "NASA C-MAPSS run-to-failure benchmark"})
    pd.DataFrame([{"engine_id": engine_id, "cycle": int(engines[engine_id]["cycles"][end]),
                   "true_rul": int(engines[engine_id]["cycles"][-1] - engines[engine_id]["cycles"][end])}
                  for engine_id, end in validation.records]).to_csv(run_dir / "validation_endpoints.csv", index=False)
    save_json(run_dir / "environment.json", environment_snapshot())
    metadata = {"signature": signature, **identity, "status": "running", "updated_at": utc_now(),
                "input_features": len(preprocessor.columns), "parameter_count": sum(p.numel() for p in model.parameters()),
                "training_windows": len(training), "validation_prefixes": len(validation),
                "fit_engines": len(split["fit_engine_ids"]), "validation_engines": len(split["validation_engine_ids"])}
    save_json(metadata_path, metadata)
    print(f"Training {cfg.subset}, seed {cfg.seed}: {len(training):,} windows, "
          f"{metadata['parameter_count']:,} parameters, full {cfg.encoder_layers}/{cfg.decoder_layers}-layer Transformer")
    print("Artifacts:", run_dir)
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            if bad_epochs >= cfg.patience:
                break
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            grid_residuals = update_training_grids(model, optimizer, training, cfg, epoch) if epoch in cfg.grid_update_epochs else []
            training_metrics = train_epoch(model, training, optimizer, scaler, cfg, epoch)
            validation_metrics, validation_predictions = validate_current_cycle(model, validation, cfg)
            rmse = validation_metrics["rmse"]
            if rmse < best - cfg.min_delta:
                best, best_epoch, bad_epochs = rmse, epoch, 0
                best_validation_metrics = validation_metrics
                validation_predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
                save_checkpoint(run_dir / "best.pt", {"model": cpu_model_state(model), "config": asdict(cfg),
                                                      "input_features": len(preprocessor.columns), "signature": signature,
                                                      "setting_indices": condition_indices(preprocessor.columns),
                                                      "implementation": IMPLEMENTATION_VERSION, "epoch": epoch,
                                                      "validation_rmse": rmse})
            else:
                bad_epochs += 1
            row = {"epoch": epoch, **training_metrics, "validation_rmse": rmse,
                   "validation_mae": validation_metrics["mae"], "validation_r2": validation_metrics["r2"],
                   "seconds": time.perf_counter() - started,
                   "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "grid_refit_rmse_max": max(grid_residuals) if grid_residuals else None}
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            save_checkpoint(run_dir / "last.pt", {"model": cpu_model_state(model), "optimizer": optimizer.state_dict(),
                                                  "scaler": scaler.state_dict(), "rng": capture_rng(),
                                                  "signature": signature, "epoch": epoch, "history": history,
                                                  "best_validation_metrics": best_validation_metrics,
                                                  "best_rmse": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs})
            metadata.update(updated_at=utc_now(), last_completed_epoch=epoch, best_epoch=best_epoch,
                            best_validation_rmse=best, bad_epochs=bad_epochs)
            save_json(metadata_path, metadata)
            print(f"  {epoch:02d}/{cfg.epochs} | MSE {row['train_mse_scaled']:.5f} | "
                  f"val RMSE {rmse:.3f} cycles | best {best:.3f} | {row['seconds']:.1f}s | "
                  f"{row['peak_vram_gib']:.2f} GiB", flush=True)
        if not (run_dir / "best.pt").is_file():
            raise RuntimeError("No valid checkpoint was produced")
        metadata.update(status="completed", updated_at=utc_now(), best_epoch=best_epoch,
                        best_validation_rmse=best, last_completed_epoch=history[-1]["epoch"],
                        best_validation_metrics=best_validation_metrics,
                        stopped_early=bad_epochs >= cfg.patience, best_sha256=sha256_file(run_dir / "best.pt"))
        metadata["artifact_hashes"] = {filename: sha256_file(run_dir / filename) for filename in
                                       ("best.pt", "config.json", "preprocessor.json", "split.json",
                                        "data_manifest.json", "validation_endpoints.csv", "validation_predictions.csv", "history.csv")}
        save_json(metadata_path, metadata)
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        updated_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        save_json(metadata_path, metadata)
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            print("CUDA OOM: lower micro_batch_size/eval_batch_size to 16 or 8, keep effective batch 256, "
                  "restart the kernel and rerun. A changed configuration creates a separate run.")
        raise
    finally:
        del model, optimizer, scaler, engines, training, validation
        gc.collect()
        torch.cuda.empty_cache()
    return run_dir

# %% [markdown] id=start-training-notes
# ## 8. Start GPU training here
#
# **This next cell starts real training.** Run preceding cells first, then run this
# cell when you are ready. By default it trains both variants on all four subsets,
# sequentially, for up to 50 epochs each (eight runs for one seed). For a first run, set
# `SELECTED_SUBSETS = ["FD001"]` in the configuration cell. Stop/interrupt at any
# time; rerunning the same configuration resumes at the last complete epoch.
#
# To change a hyperparameter, edit `Config`/`config_for` and rerun configuration
# and subsequent cells. Increasing epochs or changing microbatch size changes
# the run identity; identical settings resume. Keep your output directory on
# persistent storage if using a temporary cloud runtime.

# %% id=start-training
TRAINED_RUNS = {}
if not VARIANTS or len(set(VARIANTS)) != len(VARIANTS) or any(v not in {"paper", "conditioned_kan"} for v in VARIANTS):
    raise ValueError("Select unique known VARIANTS")
for subset in SELECTED_SUBSETS:
    for variant in VARIANTS:
        for seed in SEEDS:
            cfg = config_for(subset, seed, variant)
            # Optional local override example: cfg = replace(cfg, micro_batch_size=16, eval_batch_size=16)
            run_dir = train_subset(cfg)
            TRAINED_RUNS[(subset, variant, seed)] = run_dir
print("Completed training runs:")
for key, path in TRAINED_RUNS.items():
    print(key, "→", path)

# %% [markdown] id=variant-selection-notes
# ## 8b. Freeze the architecture choice before reading official test labels
#
# This comparison uses matched fit/validation engine IDs and the same recorded
# validation prefixes. The extension is retained only if its mean validation RMSE
# strictly beats the paper reference. We do not select the luckiest training seed.
# The selection file is frozen before the evaluation cells below read test labels.

# %% id=variant-selection
def select_variants_on_validation(trained_runs):
    selections = {}
    previous_file = OUTPUT_DIR / "variant_selection.json"
    previous = json.loads(previous_file.read_text()) if previous_file.is_file() else {}
    for subset in SELECTED_SUBSETS:
        groups = {}
        reference_split, reference_endpoints, reference_data, reference_config = None, None, None, None
        for (part, variant, seed), folder in trained_runs.items():
            if part != subset:
                continue
            record = json.loads((folder / "run.json").read_text())
            split = json.loads((folder / "split.json").read_text())
            endpoints = (folder / "validation_endpoints.csv").read_text()
            source = record["training_file_sha256"]
            comparable_config = {key: value for key, value in record["config"].items() if key not in {"variant", "seed"}}
            if reference_split is None:
                reference_split, reference_endpoints, reference_data = split, endpoints, source
                reference_config = comparable_config
            if (split != reference_split or endpoints != reference_endpoints or source != reference_data or
                    comparable_config != reference_config):
                raise ValueError("Ablation runs must share training configuration, corpus, engine split and validation prefixes")
            value = float(record["best_validation_rmse"])
            if record["status"] != "completed" or not math.isfinite(value):
                raise ValueError("Ablation requires completed runs with finite validation evidence")
            groups.setdefault(variant, []).append((seed, value))
        if any(sorted(seed for seed, _ in entries) != sorted(SEEDS) for entries in groups.values()):
            raise ValueError("Every compared variant must contain the same requested seeds")
        means = {variant: float(np.mean([value for _, value in entries])) for variant, entries in groups.items()}
        if not means:
            continue
        winner = "paper" if "paper" in means else next(iter(means))
        if "conditioned_kan" in means and means["conditioned_kan"] < means[winner] - 1e-10:
            winner = "conditioned_kan"
        selections[subset] = {"subset": subset, "selected_variant": winner, "mean_validation_rmse": means,
                              "seeds": list(SEEDS), "criterion": "strictly_lower_mean_validation_rmse; ties retain paper",
                              "matched_paper_comparison": "paper" in groups and "conditioned_kan" in groups,
                              "official_test_used_for_selection": False, "frozen_at": utc_now()}
        old = previous.get(subset, {})
        if {k: v for k, v in old.items() if k != "frozen_at"} == {k: v for k, v in selections[subset].items() if k != "frozen_at"}:
            selections[subset]["frozen_at"] = old["frozen_at"]
    save_json(OUTPUT_DIR / "variant_selection.json", selections)
    return selections


VARIANT_SELECTION = select_variants_on_validation(TRAINED_RUNS)
print(json.dumps(VARIANT_SELECTION, indent=2))

# %% [markdown] id=evaluation-notes
# ## 9. Official endpoint evaluation
#
# Now load `best.pt`, make and save one current-cycle prediction for each official
# test engine, **then** read the endpoint ground-truth vector. There is no
# retraining, fitting or checkpoint selection here. This is the official
# last-observed-window evaluation, not a random-window score.
#
# Report RMSE and MAE in cycles; R² is dimensionless. MAPE is a **percentage excluding zero targets**
# (with its denominator count saved). If a non-default `rul_cap` was configured,
# raw and capped-target metrics are both saved and clearly named. The default is
# uncapped. NASA score is an additional benchmark utility, not a P4 model
# component: for error `d = prediction - truth`, early `d < 0` gives
# `exp(-d/13)-1`; late `d >= 0` gives `exp(d/10)-1`. **Lower is better**, and life
# overestimation is penalized more. Overflow is explicitly flagged, never clipped
# to a flattering score. Evaluation timing includes batch inference, not training
# or preprocessing; it must not be compared to the paper's timing without matching
# hardware and methodology.

# %% id=evaluation-functions
def cached_official_evaluation(run_dir, data_dir):
    """Reuse a sealed, unchanged evaluation on notebook resume; preserve frozen evidence/timestamps."""
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    if not (run_dir / "research_export.json").is_file() or not (run_dir / "metrics.json").is_file():
        return None
    export = json.loads((run_dir / "research_export.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    record = json.loads((run_dir / "run.json").read_text())
    if record.get("status") != "completed" or record.get("implementation") != IMPLEMENTATION_VERSION:
        return None
    subset = record["config"]["subset"]
    for filename in ("run.json", "metrics.json", "test_predictions.csv", "test_predictions_sensor_only.csv"):
        path = run_dir / filename
        if not path.is_file() or sha256_file(path) != export.get("files", {}).get(filename, {}).get("sha256"):
            return None
    for path, expected in [(run_dir / "best.pt", metrics.get("best_checkpoint_sha256")),
                           (data_dir / f"test_{subset}.txt", metrics.get("test_file_sha256")),
                           (data_dir / f"RUL_{subset}.txt", metrics.get("rul_file_sha256"))]:
        if not path.is_file() or sha256_file(path) != expected:
            return None
    return metrics


def load_trained_run(run_dir):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "run.json").read_text())
    if metadata.get("status") != "completed":
        raise RuntimeError(f"Training is not completed: {run_dir}")
    if metadata.get("implementation") != IMPLEMENTATION_VERSION:
        raise RuntimeError("Artifact implementation version differs from this notebook")
    if sha256_file(run_dir / "best.pt") != metadata.get("best_sha256"):
        raise RuntimeError("Checkpoint is missing or modified")
    required_artifacts = {"best.pt", "config.json", "preprocessor.json", "split.json",
                          "data_manifest.json", "validation_endpoints.csv", "validation_predictions.csv", "history.csv"}
    if set(metadata.get("artifact_hashes", {})) != required_artifacts:
        raise RuntimeError("Saved artifact integrity manifest is incomplete")
    for filename, expected in metadata["artifact_hashes"].items():
        if sha256_file(run_dir / filename) != expected:
            raise RuntimeError(f"Artifact changed after training: {filename}")
    cfg = Config(**json.loads((run_dir / "config.json").read_text()))
    validate_config(cfg)
    saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if saved["signature"] != metadata["signature"] or canonical_json(saved["config"]) != canonical_json(asdict(cfg)):
        raise RuntimeError("Checkpoint/config identity differs")
    preprocessor = TrainOnlyPreprocessor.from_dict(cfg, json.loads((run_dir / "preprocessor.json").read_text()))
    if len(preprocessor.columns) != saved["input_features"]:
        raise RuntimeError("Preprocessor dimension does not match the trained model")
    if saved["setting_indices"] != condition_indices(preprocessor.columns):
        raise RuntimeError("Operating-setting indices differ from the trained model")
    model = TransformerKANBiLSTM(saved["input_features"], cfg, saved["setting_indices"]).to(DEVICE).eval()
    model.load_state_dict(saved["model"])
    return model, preprocessor, cfg


@torch.inference_mode()
def predict_sensor_windows(model, dataset, cfg):
    model.eval()
    predictions = []
    for x in DataLoader(dataset, batch_size=cfg.eval_batch_size, num_workers=0, pin_memory=True):
        with torch.autocast("cuda", enabled=False):
            result = model(x.to(DEVICE, non_blocking=True))
        predictions.append(result.float().cpu().numpy() * cfg.target_scale)
    values = np.concatenate(predictions, axis=0)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite test predictions")
    return values


def evaluate_official_test(run_dir, data_dir=DATA_DIR):
    run_dir = Path(run_dir)
    cached = cached_official_evaluation(run_dir, data_dir)
    if cached is not None:
        print("Reusing verified frozen evaluation:", run_dir)
        return cached
    model, preprocessor, cfg = load_trained_run(run_dir)
    try:
        sensor_path = Path(data_dir) / f"test_{cfg.subset}.txt"
        frame = load_sensor_file(sensor_path)
        engines = make_engine_arrays(frame, preprocessor)
        records = [(engine_id, len(engine["x"]) - 1) for engine_id, engine in sorted(engines.items())]
        dataset = InferenceWindows(engines, records, cfg)
        torch.cuda.synchronize()
        started = time.perf_counter()
        forecast = predict_sensor_windows(model, dataset, cfg)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        predictions = pd.DataFrame({"engine_id": [record[0] for record in records],
                                     "cycle": [int(engines[e]["cycles"][end]) for e, end in records],
                                     "predicted_rul": forecast[:, 0]})
        for horizon in range(1, cfg.horizons):
            predictions[f"predicted_rul_t_plus_{horizon}"] = forecast[:, horizon]
        # Freeze sensor-only outputs before opening the label file.
        predictions.to_csv(run_dir / "test_predictions_sensor_only.csv", index=False)
        truth_path = Path(data_dir) / f"RUL_{cfg.subset}.txt"
        truth = np.loadtxt(truth_path, dtype=np.float64, ndmin=1)
        if (truth.ndim != 1 or len(truth) != len(records) or not np.isfinite(truth).all() or
                np.any(truth < 0) or np.any(truth != np.floor(truth))):
            raise ValueError("Official endpoint RUL vector must be one nonnegative integer per test engine")
        predictions["true_rul"] = truth
        predictions["error_cycles"] = predictions.predicted_rul - truth
        predictions.to_csv(run_dir / "test_predictions.csv", index=False)
        target_truth = np.minimum(truth, cfg.rul_cap) if cfg.rul_cap is not None else truth
        metrics = {"subset": cfg.subset, "variant": cfg.variant, "seed": cfg.seed, "protocol": "official_test_last_window_per_engine",
                   "target_mode": "uncapped" if cfg.rul_cap is None else f"capped_{cfg.rul_cap}",
                   "raw_target_metrics": regression_metrics(truth, forecast[:, 0]),
                   "configured_target_metrics": regression_metrics(target_truth, forecast[:, 0]),
                   "inference_seconds": seconds, "inference_precision": "float32", "batch_size": cfg.eval_batch_size,
                   "mean_batch_inference_ms_per_engine": 1000 * seconds / len(records),
                   "test_file_sha256": sha256_file(sensor_path), "rul_file_sha256": sha256_file(truth_path),
                   "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"), "evaluated_at": utc_now(),
                   "paper_reported_rmse_different_protocol": PAPER_RMSE[cfg.subset],
                   "comparison_note": "Paper merges/repartitions original train/test; values are not directly comparable."}
        save_json(run_dir / "metrics.json", metrics)
        print(cfg.subset, "official endpoint results:", json.dumps(metrics["raw_target_metrics"], indent=2))
        return metrics
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

# %% id=run-evaluation
if not globals().get("TRAINED_RUNS"):
    raise RuntimeError("Run the training cell first, or populate TRAINED_RUNS with your completed run directories")
EVALUATIONS = {key: evaluate_official_test(path) for key, path in TRAINED_RUNS.items()}
summary_rows = []
for (subset, variant, seed), report in EVALUATIONS.items():
    summary_rows.append({"subset": subset, "variant": variant, "seed": seed, **report["raw_target_metrics"],
                         "selected_on_validation": VARIANT_SELECTION[subset]["selected_variant"] == variant,
                         "run_dir": str(TRAINED_RUNS[(subset, variant, seed)])})
summary = pd.DataFrame(summary_rows)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT_DIR / "official_test_summary.csv", index=False)
print(summary[["subset", "variant", "seed", "n_engines", "rmse", "mae", "r2", "nasa_score"]].to_string(index=False))
if len(SEEDS) > 1:
    repeated = summary.groupby(["subset", "variant"])[["rmse", "mae", "r2", "nasa_score"]].agg(["mean", "std", "count"])
    repeated.to_csv(OUTPUT_DIR / "repeated_seed_summary.csv")
    print("Repeated independent training seeds (same engine split):\n", repeated)

# %% [markdown] id=plots-notes
# ## 10. Learning curves and official test predictions
#
# Plots read your saved results. Engine-ID plots preserve engine order; the
# optional scatter shows prediction error without sorting targets into a fake
# degradation trajectory. The paper's reference values are not plotted as measured
# notebook results.

# %% id=result-plots
for (subset, variant, seed), run_dir in TRAINED_RUNS.items():
    history = pd.read_csv(run_dir / "history.csv")
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), constrained_layout=True)
    axes[0].plot(history.epoch, history.train_mse_scaled, color="#3478F6")
    axes[0].set(title="Training MSE (scaled targets)", xlabel="Epoch", ylabel="Mean squared error")
    axes[1].plot(history.epoch, history.validation_rmse, color="#FF8A24")
    axes[1].set(title="Held-out engine prefixes", xlabel="Epoch", ylabel="Current-cycle RMSE (cycles)")
    axes[2].plot(predictions.engine_id, predictions.true_rul, label="True endpoint RUL", color="#3478F6", linewidth=1)
    axes[2].plot(predictions.engine_id, predictions.predicted_rul, label="Predicted RUL", color="#FF8A24", linewidth=1)
    axes[2].set(title="Official test endpoints", xlabel="Engine ID", ylabel="RUL (cycles)")
    axes[2].legend()
    fig.suptitle(f"{subset} · {variant} · seed {seed} · Transformer–KAN–BiLSTM")
    fig.savefig(run_dir / "training_and_predictions.png", dpi=160)
    plt.show()
    plt.close(fig)

# %% [markdown] id=interpretation-notes
# ## 11. Inspect a learned KAN edge
#
# The KAN is **after** the Transformer: its coordinates are learned latent
# features, not raw temperatures/pressures. This plot shows an actual learned
# univariate edge and its finite-difference derivative; it is not a sensor-causal
# explanation. Automatic pruning and symbolic replacement are not performed.

# %% id=kan-edge-plot
selected_run = next(iter(TRAINED_RUNS.values()))
model, preprocessor, saved_cfg = load_trained_run(selected_run)
try:
    layer = model.kan.layers[0]
    input_edge, output_edge = 0, 0  # Change to another latent edge if desired.
    with torch.inference_mode():
        grid = torch.linspace(-0.99, 0.99, 300, device=DEVICE)
        values = torch.zeros(300, layer.inputs, device=DEVICE)
        values[:, input_edge] = grid
        curve = layer.edges(values)[:, input_edge, output_edge].cpu().numpy()
        knots = layer.knots[input_edge].cpu().numpy()
        grid = grid.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    axes[0].plot(grid, curve, color="#3478F6")
    axes[1].plot(grid, np.gradient(curve, grid), color="#FF8A24")
    for axis in axes:
        axis.set_xlabel("Normalized latent coordinate")
        for knot in knots[::max(1, len(knots) // 15)]:
            if -1 <= knot <= 1:
                axis.axvline(knot, color="#E1EAF4", linewidth=0.7)
    axes[0].set(title=f"KAN layer 1, edge {input_edge} → {output_edge}", ylabel="Edge output")
    axes[1].set(title="Edge derivative", ylabel="Numerical derivative")
    fig.savefig(selected_run / "kan_latent_edge.png", dpi=160)
    plt.show()
    plt.close(fig)
finally:
    del model
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown] id=reload-notes
# ## 12. Reload for sensor-only inference
#
# Rerun the configuration/definition cells in a fresh kernel, set a completed run
# directory, and call this function. A prediction needs the full observed prefix
# for trailing smoothing and only its final window for the neural network. No
# evaluation target or future observation is passed to the model.

# %% id=reload-inference
def predict_observed_prefix(run_dir, observed_prefix):
    """observed_prefix: one engine, columns RAW_COLUMNS, sorted cycles 1..current."""
    prefix = observed_prefix.sort_values("cycle").reset_index(drop=True).copy()
    if (prefix.empty or list(prefix.columns) != RAW_COLUMNS or prefix.engine_id.nunique() != 1 or
            not np.array_equal(prefix.cycle.to_numpy(), np.arange(1, len(prefix) + 1)) or
            not np.isfinite(prefix.to_numpy(dtype=np.float64)).all()):
        raise ValueError("Pass one finite engine prefix with the exact C-MAPSS schema and consecutive cycles")
    model, preprocessor, cfg = load_trained_run(run_dir)
    try:
        engine_id = int(prefix.engine_id.iloc[0])
        engines = make_engine_arrays(prefix, preprocessor)
        dataset = InferenceWindows(engines, [(engine_id, len(prefix) - 1)], cfg)
        forecast = predict_sensor_windows(model, dataset, cfg)[0]
        return {"engine_id": engine_id, "subset": cfg.subset, "cycle": int(prefix.cycle.iloc[-1]),
                "predicted_rul": float(forecast[0]), "forecast_rul": forecast.tolist(),
                "model_directory": str(run_dir), "inference_at": utc_now()}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


example_key = next(iter(TRAINED_RUNS))
example_run = TRAINED_RUNS[example_key]
example_test = load_sensor_file(DATA_DIR / f"test_{example_key[0]}.txt")
example_prefix = example_test[example_test.engine_id == 1].reset_index(drop=True)
print(predict_observed_prefix(example_run, example_prefix))

# %% [markdown] id=artifacts-and-next-steps
# ## Saved files and run checklist
#
# Each `artifacts/kan_bilstm/FDxxx/seed-…/` directory contains:
#
# - `best.pt`, `last.pt`: inference weights and resumable epoch state.
# - `config.json`, `preprocessor.json`, `split.json`, `validation_endpoints.csv`:
#   complete numerical and data-split configuration.
# - `data_manifest.json`, `environment.json`, `run.json`: provenance, environment,
#   architecture/version, status, parameter count and checkpoint checksum.
# - `history.csv`: every completed epoch's loss, validation metrics, duration,
#   grid-refit residual and measured GPU memory.
# - `test_predictions_sensor_only.csv`: predictions saved before labels are read.
# - `test_predictions.csv`, `metrics.json`: actual official endpoint evaluation.
# - `training_and_predictions.png`, `kan_latent_edge.png`: plots from your weights.
#
# `validation_predictions.csv` freezes best-checkpoint development predictions.
# `variant_selection.json` records the validation-only architecture choice.
# `official_test_summary.csv` collects actual completed runs. With several seeds,
# `repeated_seed_summary.csv` reports mean/std/count without choosing the best test
# seed. Keep the notebook with your exported artifacts to reload class definitions.
# Checkpoints from this notebook are separate from the website's CPU baseline
# artifacts; there is no automatic model promotion or website replacement here.
#
# **Run order:** install/select a CUDA kernel manually → set paths and subsets →
# Run All → inspect validation history → review official endpoint metrics.
# For a new experiment, tune on validation engines and freeze settings before
# reviewing official test scores. A low test error is a measured outcome, not a
# guarantee from selecting KAN. This simulated benchmark does not establish a
# real engine failure date or a maintenance safety guarantee.

# %% [markdown] id=website-export-notes
# ## 13. Export the trained model for the AeroGuard website
#
# The website reads the checksummed `research_export.json` bundle automatically
# from `artifacts/kan_bilstm/`. If training elsewhere, download the bundle ZIP
# produced below and use **Model Registry → GPU research → Import training bundle**.
# Existing CPU benchmark models remain available separately. An untrained notebook
# never creates a successful model entry.
#
# The inference export is a TorchScript model verified against the loaded weights
# on actual observed windows at two batch sizes. It uses FP32 so the local API can
# run it on CPU after PyTorch is installed. The trace is valid only for the saved
# window length and feature schema. The website verifies hashes and compares
# predictions to exported FP32 probe outputs before reporting inference readiness.
# This export does not re-fit the model or alter which variant was selected.

# %% id=website-export-functions
import zipfile


def public_run_id(run_dir):
    record = json.loads((Path(run_dir) / "run.json").read_text())
    cfg = record["config"]
    return f"KAN-{cfg['subset']}-{cfg['variant']}-{cfg['seed']}-{record['signature'][:12]}"


def seal_website_bundle(run_dir, selection):
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    cfg = Config(**record["config"])
    explanations = sorted((run_dir / "explanations").glob("engine-*-cycle-*.json"))
    names = ["run.json", "config.json", "preprocessor.json", "split.json", "data_manifest.json",
             "validation_endpoints.csv", "validation_predictions.csv", "history.csv", "environment.json",
             "metrics.json", "test_predictions.csv", "test_predictions_sensor_only.csv",
             "inference.ts", "inference_probes.npz", "selection_record.json"]
    save_json(run_dir / "selection_record.json", selection)
    files = {}
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required website artifact is missing: {name}")
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    for path in explanations:
        files[path.relative_to(run_dir).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    export = {"schema_version": 1, "run_id": public_run_id(run_dir), "subset": cfg.subset,
              "variant": cfg.variant, "seed": cfg.seed, "implementation": IMPLEMENTATION_VERSION,
              "architecture": "Transformer-KAN-BiLSTM", "status": "completed", "exported_at": utc_now(),
              "source": "NASA C-MAPSS benchmark; user-trained GPU research model",
              "selected_on_validation": selection["selected_variant"] == cfg.variant,
              "selection": selection, "explanation_status": "available" if explanations else "not_generated",
              "explanation_files": [path.relative_to(run_dir).as_posix() for path in explanations],
              "files": files}
    save_json(run_dir / "research_export.json", export)
    archive_path = run_dir.parent / f"{export['run_id']}.zip"
    temporary = archive_path.with_name(archive_path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ["research_export.json", *files]:
            archive.write(run_dir / name, name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Export archive integrity check failed")
    temporary.replace(archive_path)
    return archive_path


def export_inference_model(run_dir, selection):
    run_dir = Path(run_dir)
    previous = run_dir / "research_export.json"
    if previous.is_file():
        existing = json.loads(previous.read_text())
        unchanged = existing.get("run_id") == public_run_id(run_dir)
        for filename in ("run.json", "inference.ts", "inference_probes.npz"):
            path = run_dir / filename
            unchanged = unchanged and path.is_file() and sha256_file(path) == existing.get("files", {}).get(filename, {}).get("sha256")
        if unchanged:
            return seal_website_bundle(run_dir, selection)
    model, preprocessor, cfg = load_trained_run(run_dir)
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        frame = load_sensor_file(DATA_DIR / f"test_{cfg.subset}.txt")
        engines = make_engine_arrays(frame, preprocessor)
        records = [(engine_id, len(engine["x"]) - 1) for engine_id, engine in sorted(engines.items())][:2]
        example = torch.from_numpy(np.stack([historical_window(engines[e]["x"], end, cfg.window) for e, end in records]))
        model = model.cpu().eval()
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            traced = torch.jit.trace(model, example, check_trace=True, check_inputs=[(example[:1],)])
            expected = model(example)
            torch.testing.assert_close(traced(example), expected, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(traced(example[:1]), model(example[:1]), rtol=1e-4, atol=1e-5)
        path = Path(run_dir) / "inference.ts"
        temporary = path.with_name(path.name + ".tmp")
        traced.save(str(temporary))
        temporary.replace(path)
        np.savez_compressed(Path(run_dir) / "inference_probes.npz", windows=example.numpy(),
                            predictions_cycles=expected.numpy() * cfg.target_scale)
        return seal_website_bundle(run_dir, selection)
    finally:
        torch.backends.mha.set_fastpath_enabled(old_fastpath)
        del model
        gc.collect()
        torch.cuda.empty_cache()


WEBSITE_BUNDLES = {}
for (subset, variant, seed), folder in TRAINED_RUNS.items():
    WEBSITE_BUNDLES[(subset, variant, seed)] = export_inference_model(folder, VARIANT_SELECTION[subset])
print("Website bundles (SHAP/LIME files are added after the final stage):")
for key, path in WEBSITE_BUNDLES.items():
    print(key, "→", path)

# %% [markdown] id=post-training-explanation-notes
# ## 14. Final analysis only: genuine SHAP and LIME
#
# Training, validation selection, official evaluation and inference export are
# complete before this stage. These explanations **do not feed back into training**.
# Install the SHAP/LIME dependencies from `requirements-kan.txt` manually if absent.
# The default explains engine 1 for the validation-selected variant and the first
# requested seed of each subset. Increase the engine list or sampling budget
# explicitly; these are **local** explanations, not a fleet-wide importance study.
#
# [Kernel SHAP](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html)
# explains groups consisting of each selected sensor's **whole historical window**.
# A zero mask replaces that sensor trajectory with one from a fitting-engine
# background, averaged over four nearby-setting backgrounds. Operating settings
# stay fixed. Background selection uses observed settings only, never labels.
# [LIME's LimeBase](https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_base)
# fits its actual locally weighted regression to the **same grouped-mask prediction
# function**, with an independent held-out perturbation set for fidelity checks.
# This avoids replacing individual timesteps with independent random noise.
# Cross-sensor mixtures can still be off-manifold; neither method establishes
# physical causality, sensor fault diagnosis, nor a guaranteed failure date.
#
# SHAP contributions (cycles) sum with the background expectation to approximate
# the model prediction; the residual is reported. LIME coefficients, intercept,
# local surrogate prediction, fitting R², **held-out weighted R²/MAE**, kernel and
# perturbation counts are recorded separately. A poor fidelity score is shown,
# not hidden. No SHAP/LIME values are substituted with occlusion scores or examples.

# %% id=post-training-explanation-config
RUN_EXPLANATIONS = True
EXPLAIN_ENGINE_IDS = [1]
EXPLANATION_BACKGROUND_CANDIDATES = 128
EXPLANATION_BACKGROUNDS = 4
SHAP_SAMPLES = 256
LIME_SAMPLES = 512
LIME_HELDOUT_SAMPLES = 128
LIME_KERNEL_WIDTH = 0.35

if RUN_EXPLANATIONS:
    if (min(EXPLANATION_BACKGROUND_CANDIDATES, EXPLANATION_BACKGROUNDS, SHAP_SAMPLES,
            LIME_SAMPLES, LIME_HELDOUT_SAMPLES) < 2 or LIME_KERNEL_WIDTH <= 0 or
            EXPLANATION_BACKGROUND_CANDIDATES < EXPLANATION_BACKGROUNDS):
        raise ValueError("Use positive explanation budgets, at least two samples, and enough background candidates")
    try:
        import shap
        from lime.lime_base import LimeBase
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise RuntimeError("Final explanation stage needs shap, lime and scikit-learn. Install requirements-kan.txt manually, then rerun this stage; your completed models are retained.") from exc

# %% id=post-training-explanation-functions
def explanation_backgrounds(run_dir, preprocessor, cfg, observed_window):
    record = json.loads((Path(run_dir) / "run.json").read_text())
    source = DATA_DIR / f"train_{cfg.subset}.txt"
    if sha256_file(source) != record["training_file_sha256"]:
        raise ValueError("Explanation backgrounds must use the unchanged fitting corpus")
    split = json.loads((Path(run_dir) / "split.json").read_text())
    frame = load_sensor_file(source)
    fit_ids = split["fit_engine_ids"]
    if fit_ids != preprocessor.fit_ids:
        raise ValueError("Background fit IDs differ from preprocessing fit IDs")
    frame = frame[frame.engine_id.isin(fit_ids)].reset_index(drop=True)
    engines = make_engine_arrays(frame, preprocessor)
    candidates = TrainingWindows(engines, fit_ids, cfg)
    rng = np.random.default_rng(cfg.seed + 9029)
    chosen = rng.choice(len(candidates), min(EXPLANATION_BACKGROUND_CANDIDATES, len(candidates)), replace=False)
    windows, provenance = [], []
    for index in chosen:
        engine_id, end = candidates.records[int(index)]
        windows.append(historical_window(engines[engine_id]["x"], end, cfg.window))
        provenance.append({"partition": "fit", "engine_id": engine_id, "cycle": int(engines[engine_id]["cycles"][end])})
    windows = np.stack(windows)
    setting_indexes = [i for i in condition_indices(preprocessor.columns) if i >= 0]
    if setting_indexes:
        distances = np.mean((windows[:, -1, setting_indexes] - observed_window[-1, setting_indexes]) ** 2, axis=1)
    else:
        distances = np.zeros(len(windows))
    nearest = np.argsort(distances, kind="stable")[:EXPLANATION_BACKGROUNDS]
    return windows[nearest], [provenance[int(index)] for index in nearest]


def heldout_weighted_fidelity(truth, prediction, weights):
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.average(truth, weights=weights)
    denominator = np.sum(weights * (truth - mean) ** 2)
    numerator = np.sum(weights * (truth - prediction) ** 2)
    return {"weighted_r2": float(1 - numerator / denominator) if denominator > 1e-12 else None,
            "weighted_mae_cycles": float(np.average(np.abs(truth - prediction), weights=weights))}


def explain_trained_engine(run_dir, engine_id):
    # No optimizer, checkpoint selection, or evaluation-target access in this function.
    model, preprocessor, cfg = load_trained_run(run_dir)
    try:
        frame = load_sensor_file(DATA_DIR / f"test_{cfg.subset}.txt")
        engines = make_engine_arrays(frame, preprocessor)
        if engine_id not in engines:
            raise ValueError(f"Engine {engine_id} is not in {cfg.subset}")
        end = len(engines[engine_id]["x"]) - 1
        window = historical_window(engines[engine_id]["x"], end, cfg.window)
        columns = [name for name in preprocessor.columns if name.startswith("sensor_")]
        indexes = [preprocessor.columns.index(name) for name in columns]
        if not columns:
            raise ValueError("No varying sensor groups to explain")
        backgrounds, provenance = explanation_backgrounds(run_dir, preprocessor, cfg, window)
        mask_columns = np.asarray(indexes, dtype=int)

        @torch.inference_mode()
        def predict_masks(masks):
            masks = np.atleast_2d(np.asarray(masks, dtype=np.float32))
            if masks.shape[1] != len(columns) or not np.isfinite(masks).all() or np.any((masks < 0) | (masks > 1)):
                raise ValueError("Expected finite sensor-group masks in [0, 1]")
            outputs = []
            # Limit every neural forward batch, including SHAP's large evaluation arrays.
            for offset in range(0, len(masks), max(1, cfg.eval_batch_size // len(backgrounds))):
                group = masks[offset:offset + max(1, cfg.eval_batch_size // len(backgrounds))]
                perturbed = mix_sensor_windows(window, backgrounds, group, mask_columns)
                batch = torch.from_numpy(perturbed.reshape(-1, *window.shape)).to(DEVICE)
                with torch.autocast("cuda", enabled=False):
                    prediction = model(batch)[:, 0].float().cpu().numpy() * cfg.target_scale
                outputs.extend(prediction.reshape(len(group), len(backgrounds)).mean(1).tolist())
            return np.asarray(outputs, dtype=np.float64)

        ones = np.ones((1, len(columns)), dtype=np.float32)
        zeros = np.zeros_like(ones)
        prediction = float(predict_masks(ones)[0])
        # Kernel SHAP computes genuine sampled Shapley estimates for this grouped game.
        np.random.seed(cfg.seed + engine_id)
        explainer = shap.KernelExplainer(predict_masks, zeros, feature_names=columns)
        contributions = np.asarray(explainer.shap_values(ones, nsamples=SHAP_SAMPLES, l1_reg=0.0, silent=True)).reshape(-1)
        baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        if contributions.shape != (len(columns),) or not np.isfinite(contributions).all():
            raise FloatingPointError("SHAP returned invalid feature contributions")

        rng = np.random.default_rng(cfg.seed + engine_id * 7919)
        neighborhood = (rng.random((LIME_SAMPLES, len(columns))) < 0.8).astype(np.float64)
        neighborhood[0] = 1
        neighborhood[1] = 0
        distances = np.sqrt(np.mean((neighborhood - 1) ** 2, axis=1))
        labels = predict_masks(neighborhood)
        kernel = lambda distance: np.exp(-(np.asarray(distance) ** 2) / LIME_KERNEL_WIDTH ** 2)
        lime_base = LimeBase(kernel_fn=kernel, verbose=False, random_state=cfg.seed + engine_id)
        intercept, local_exp, fitting_r2, local_prediction = lime_base.explain_instance_with_data(
            neighborhood, labels[:, None], distances, label=0, num_features=len(columns),
            feature_selection="none", model_regressor=Ridge(alpha=1.0, fit_intercept=True))
        coefficients = np.zeros(len(columns), dtype=np.float64)
        for index, value in local_exp:
            coefficients[int(index)] = float(value)
        heldout = (rng.random((LIME_HELDOUT_SAMPLES, len(columns))) < 0.8).astype(np.float64)
        heldout_truth = predict_masks(heldout)
        heldout_prediction = float(intercept) + heldout @ coefficients
        heldout_distances = np.sqrt(np.mean((heldout - 1) ** 2, axis=1))
        fidelity = heldout_weighted_fidelity(heldout_truth, heldout_prediction, kernel(heldout_distances))
        result = {"schema_version": 1, "run_id": public_run_id(run_dir), "subset": cfg.subset,
                  "variant": cfg.variant, "engine_id": int(engine_id), "cycle": int(engines[engine_id]["cycles"][end]),
                  "target": "current_cycle_predicted_rul", "units": "cycles", "prediction": prediction,
                  "created_at": utc_now(), "model_sha256": sha256_file(Path(run_dir) / "best.pt"),
                  "feature_groups": columns, "background_source": "fit_engines_only", "backgrounds": provenance,
                  "perturbation": "replace_whole_sensor_windows; observed_operating_settings_fixed",
                  "shap": {"method": "shap.KernelExplainer", "package_version": importlib.metadata.version("shap"),
                           "samples": SHAP_SAMPLES, "baseline": baseline, "values": contributions.tolist(),
                           "additivity_residual_cycles": float(prediction - baseline - contributions.sum())},
                  "lime": {"method": "lime.lime_base.LimeBase", "package_version": importlib.metadata.version("lime"),
                           "samples": LIME_SAMPLES, "heldout_samples": LIME_HELDOUT_SAMPLES,
                           "kernel": "exp(-distance_squared/width_squared)", "kernel_width": LIME_KERNEL_WIDTH,
                           "intercept": float(intercept), "coefficients": coefficients.tolist(),
                           "local_prediction": float(np.asarray(local_prediction).reshape(-1)[0]),
                           "fitting_r2": float(fitting_r2) if math.isfinite(float(fitting_r2)) else None,
                           "heldout_fidelity": fidelity},
                  "limitations": ["Local grouped-mask explanation, not sensor-causal diagnosis.",
                                  "Whole histories retain within-sensor structure; cross-sensor mixtures may be off-manifold.",
                                  "SHAP contributions and LIME coefficients have different meanings."]}
        save_json(Path(run_dir) / "explanations" / f"engine-{engine_id:03d}-cycle-{result['cycle']}.json", result)
        return result
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

# %% id=run-final-explanations
EXPLANATIONS = []
if RUN_EXPLANATIONS:
    for (subset, variant, seed), run_dir in TRAINED_RUNS.items():
        if variant != VARIANT_SELECTION[subset]["selected_variant"] or seed != SEEDS[0]:
            continue
        for engine_id in EXPLAIN_ENGINE_IDS:
            report = explain_trained_engine(run_dir, int(engine_id))
            EXPLANATIONS.append(report)
            print(subset, variant, engine_id, "SHAP residual:", report["shap"]["additivity_residual_cycles"],
                  "LIME held-out fidelity:", report["lime"]["heldout_fidelity"])
            order = np.argsort(np.abs(report["shap"]["values"]))[-10:]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            for axis, values, title in [(axes[0], report["shap"]["values"], "SHAP contributions"),
                                        (axes[1], report["lime"]["coefficients"], "LIME coefficients")]:
                vals = np.asarray(values)[order]
                axis.barh(np.asarray(report["feature_groups"])[order], vals,
                          color=np.where(vals >= 0, "#FF8A24", "#3478F6"))
                axis.axvline(0, color="#64748B", linewidth=0.7)
                axis.set(title=title, xlabel="Cycles; group retained versus fit-background replacement")
            fig.suptitle(f"{subset} · engine {engine_id} · {variant} · local explanations")
            fig.savefig(run_dir / "explanations" / f"engine-{engine_id:03d}-shap-lime.png", dpi=160)
            plt.show()
            plt.close(fig)
else:
    print("Explanation stage explicitly skipped; website will show no SHAP/LIME results for these runs.")

# Re-seal exports after explanations. This only packages results; it never re-fits or selects a model.
for (subset, variant, seed), run_dir in TRAINED_RUNS.items():
    WEBSITE_BUNDLES[(subset, variant, seed)] = seal_website_bundle(run_dir, VARIANT_SELECTION[subset])
print("Finished. Import these completed training bundles into AeroGuard:")
for path in WEBSITE_BUNDLES.values():
    print(path)
