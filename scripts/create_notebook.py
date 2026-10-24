"""Generate the authored, unexecuted CUDA training notebook without nbformat."""

import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
cells=[]


def markdown(source):
    cells.append({"cell_type":"markdown","metadata":{},"source":textwrap.dedent(source).strip()+"\n"})


def code(source):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
                  "source":textwrap.dedent(source).strip()+"\n"})


markdown('''
# FleetPulse · C-MAPSS FD001–FD004
## CUDA training, independent calibration and website integration

Train one research-integrated model per NASA subset: task-aware sensor repair,
causal CNN-GRU, compact Transformer attention and a small spline/KAN readout.
Development-only bounded residual correction, two independent-engine calibration
references and an Isolation Forest complete each saved inference bundle.

**This notebook is authored but unexecuted. It contains no claimed training results.**
Opening it installs/downloads nothing. Running the training cell performs real training.
The companion website reads these same bundles, endpoint predictions and SQLite work orders.
Keep the `fleetpm/` package with this notebook; it is the shared training/serving implementation.
''')
markdown('''
### 1 · Manual environment setup

A project-local Python 3.12 `.venv` has been created. Install dependencies manually in a terminal:

```bash
uv pip install --python .venv/bin/python --cache-dir .venv/cache/uv torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python --cache-dir .venv/cache/uv -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name fleetpulse --display-name "FleetPulse CUDA (Python 3.12)"
.venv/bin/python -m jupyterlab
```

Select **FleetPulse CUDA (Python 3.12)** in Jupyter. CUDA packages may require several GB;
there is no automatic installer in a code cell. The explicit CPU alternative is documented
in the README. Official API references: [PyTorch installation](https://pytorch.org/get-started/locally/),
[mixed precision](https://docs.pytorch.org/docs/2.14/amp.html).
''')
code('''
from pathlib import Path
import importlib.util
import os
import sys

candidates = [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next((p for p in candidates if (p / "fleetpm" / "config.py").is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Open this notebook inside the extracted FleetPulse project, with fleetpm/ present.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from fleetpm.config import load_env
load_env(PROJECT_ROOT)
required = ["torch", "numpy", "pandas", "sklearn", "matplotlib", "joblib"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError(f"Missing packages: {missing}. Install requirements.txt manually in the project .venv and select its kernel.")
print("Project:", PROJECT_ROOT)
print("Kernel:", sys.executable)
''')
markdown('''
### 2 · CUDA and numerical precision

The default requires CUDA rather than silently switching to CPU. Set `DEVICE="cpu"`
deliberately if needed. Training uses BF16 when supported, otherwise FP16 with gradient
scaling. Spline basis calculations remain float32. A low-memory batch of 128 and sequential
subset execution target laptop GPUs; actual runtime and memory use must be measured.
''')
code('''
import gc
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from fleetpm.config import ARTIFACT_DIR, DATA_DIR, RUNTIME_DIR, SUBSETS, profile_config
from fleetpm.learning import choose_device

DEVICE = os.environ.get("FLEET_TRAIN_DEVICE", "cuda")  # cuda | cpu | auto
device = choose_device(DEVICE)
print({"torch": torch.__version__, "device": str(device), "cuda_runtime": torch.version.cuda})
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(device))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
''')
markdown('''
### 3 · Data: all four official subsets

Manually download NASA's C-MAPSS archive from the [official portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
Use the website's ZIP import or place all twelve `train_`, `test_` and `RUL_` files in `data/cmapss/`.
There are 26 columns per sensor row: unit, cycle, three settings and 21 measurements.
The available FD004 files contain 249 training and 248 test engines; NASA portal descriptive metadata reverses these counts.

The following audit checks schema and label-vector length, not the authenticity of arbitrary uploaded data.
''')
code('''
from fleetpm.data import inventory
audit = inventory(DATA_DIR, deep=True)
display(pd.DataFrame([{k: v for k, v in item.items() if k not in {"files", "errors"}} for item in audit]))
if not all(item["ready"] for item in audit):
    problems = {item["subset"]: item["errors"] or "Required files missing" for item in audit if not item["ready"]}
    raise RuntimeError(f"Data is not ready: {problems}. No files are downloaded automatically.")
if any(not item.get("official_counts_match") for item in audit):
    print("NOTICE: engine counts differ from the official NASA benchmark. Label results as a custom dataset.")
''')
markdown('''
### 4 · Training profile and compute budget

The balanced profile uses 60 joint epochs, three engine-grouped OOF folds and one seed.
The rigorous profile uses 150 joint epochs and five folds; set multiple seeds explicitly
when budget permits. The smoke profile trains only two batches per phase for wiring checks
and is never accepted as a deployable trained bundle.

Each subset trains OOF models for correction, then one final model. Checkpoints resume only
when training-file hash, configuration and numerical-source code match. A completed matching
run is reused; report-only operations do not cause retraining.
''')
code('''
PROFILE = os.environ.get("FLEET_PROFILE", "balanced")  # balanced | rigorous | smoke
SEEDS = [42]          # for a multi-seed study: [13, 29, 42, 71, 101]
SELECTED_SUBSETS = list(SUBSETS)
TRACK_MLFLOW = True   # local SQLite/artifacts only; requires the full requirements file
if TRACK_MLFLOW and importlib.util.find_spec("mlflow") is None:
    raise RuntimeError("Install the full requirements file for MLflow, or set TRACK_MLFLOW=False explicitly.")

configurations = [profile_config(PROFILE, subset=subset, seed=seed, device=DEVICE)
                  for subset in SELECTED_SUBSETS for seed in SEEDS]
display(pd.DataFrame([cfg.to_dict() for cfg in configurations]))
print("Model fits:", sum(cfg.folds + 1 for cfg in configurations))
print("Each fit has clean pretraining, repair pretraining and joint fine-tuning.")
''')
markdown('''
### 5 · Isolation protocol

Before windowing, reserve about **60% fit engines, 20% development-selection engines and
20% independent calibration engines**. This is a concrete implementation of the architecture's
80% development / 20% calibration split, with a selection cohort inside development.
Every OOF fold refits preprocessing and uses a fixed epoch budget without selecting its
checkpoint on the held-out fold. Residual correction fits only OOF fit-engine predictions.
Selection chooses checkpoint/correction; calibration is untouched until the predictor is frozen.

Training imports **only `train_*.txt`**. The official `test_*.txt` and `RUL_*.txt` are used later
by the evaluator, never to normalize data, repair inputs, fit correction, or choose a seed.
''')
code('''
from fleetpm.features import load_engines, split_engines
split_preview = {}
for cfg in configurations:
    engines = load_engines(DATA_DIR / f"train_{cfg.subset}.txt")
    groups = split_engines(engines, cfg.split_seed, cfg.calibration_fraction)
    split_preview[f"{cfg.subset}/seed{cfg.seed}"] = {name: len(ids) for name, ids in groups.items()}
display(pd.DataFrame(split_preview).T)
''')
markdown('''
### 6 · Model interface and optional synthetic CUDA probe

Inputs are `[batch,30,66]`: 21 sensor values, three settings, 21 trusted masks and 21 gaps.
Causal convolutions feed a GRU and one attention encoder; their fusion joins a 16-value
statistical projection and three quality features. A small 32->8->1 cubic-spline head produces
capped RUL. Trusted observations are preserved by an explicit repair merge.

Historical prefixes shorter than 30 cycles are right-padded with a separate valid-token mask.
The last **valid** token is selected. Such official endpoints are included in numerical
evaluation but remain warm-up exceptions for automatic maintenance decisions.
''')
code('''
from fleetpm.models import HybridRUL
from fleetpm.features import STAT_DIM

probe = HybridRUL().to(device)
probe.eval()
x = torch.zeros(2, 30, 66, device=device)
x[:, :, 24:45] = 1  # fully observed synthetic compatibility input
with torch.inference_mode():
    point, generated = probe(x, torch.zeros(2, STAT_DIM, device=device),
        torch.ones(2, 3, device=device), torch.tensor([30, 30], device=device),
        torch.ones(2, 30, dtype=torch.bool, device=device))
assert point.shape == (2,) and generated.shape == (2, 30, 21)
assert torch.isfinite(point).all()
print("Synthetic shape/device probe passed. Parameters:", sum(p.numel() for p in probe.parameters()))
print("This checks tensor compatibility, not prediction accuracy.")
del probe, x, point, generated
gc.collect()
if device.type == "cuda": torch.cuda.empty_cache()
''')
markdown('''
### 7 · Train FD001, FD002, FD003 and FD004

**Running the next cell starts training.** Leave it unexecuted to keep this as a prepared notebook.
No environment or dataset download occurs here. Losses, fold membership, source hashes,
model weights, optimizer/AMP state and per-epoch progress are saved under `artifacts/runs/`.
Interrupt at any time; only atomically completed epoch checkpoints are resumed.
''')
code('''
from fleetpm.train import train_subset

trained_runs = []
for cfg in configurations:
    print(f"\\n=== {cfg.subset} / seed {cfg.seed} / {cfg.profile} ===")
    result = train_subset(cfg, DATA_DIR, ARTIFACT_DIR, track_mlflow=TRACK_MLFLOW)
    trained_runs.append(result)
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
display(pd.DataFrame(trained_runs)[["run_id", "subset", "selection_rmse", "parameters", "device", "smoke_only"]])
''')
markdown('''
### 8 · Inspect measured learning curves and calibration references

Selection RMSE uses capped targets and is development evidence. Calibration has separate
clean and declared-missingness references on the same held-out engines: these are not
twice as many independent engines. Nominal 90% coverage assumes matching engine-prefix
and corruption distributions; it is not a per-machine failure probability.
''')
code('''
fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
for subset, ax in zip(SELECTED_SUBSETS, axes.flat):
    matches = [r for r in trained_runs if r["subset"] == subset]
    for run in matches:
        history = json.loads((ARTIFACT_DIR / "runs" / run["run_id"] / "history.json").read_text())
        joint = [h for h in history if h["phase"] == "joint"]
        ax.plot([h["epoch"] for h in joint], [h["selection_rmse"] for h in joint], label=f"seed {run['configuration']['seed']}")
    ax.set(title=subset, xlabel="Joint epoch", ylabel="Selection capped RMSE (cycles)")
    ax.legend(); ax.grid(alpha=.2)
plt.show()
calibration_table = []
for run in trained_runs:
    calibration = json.loads((ARTIFACT_DIR / "runs" / run["run_id"] / "calibration.json").read_text())
    for name in ("clean", "supported_missing"):
        calibration_table.append({"run_id": run["run_id"], "reference": name, **calibration[name]})
display(pd.DataFrame(calibration_table))
''')
markdown('''
### 9 · Freeze the selected run before opening official endpoint truth

Selection uses development-selection RMSE only. It never uses test RMSE, NASA score or
calibration residuals. All runs remain available for later multi-seed reporting.
''')
code('''
selected_runs = {
    subset: min([r for r in trained_runs if r["subset"] == subset], key=lambda r: r["selection_rmse"])
    for subset in SELECTED_SUBSETS
}
selection_manifest = {s: r["run_id"] for s, r in selected_runs.items()}
from fleetpm.config import write_json
write_json(ARTIFACT_DIR / "selection_before_test.json", selection_manifest)
display(selection_manifest)
''')
markdown('''
### 10 · Locked official endpoint evaluation

Predict one endpoint per official test engine **before parsing endpoint truth**. Report raw-truth
RMSE/MAE/NASA score, with capped-target RMSE and capped interval coverage labelled separately.
Positive error means optimistic/late prediction and receives the stronger NASA penalty.
All short endpoints are counted; their decision warm-up/exception status is reported.

Repeated parameter tuning after viewing these results invalidates an untouched-test claim.
''')
code('''
from fleetpm.evaluate import evaluate_run

evaluation_reports = []
for subset, run in selected_runs.items():
    if run["smoke_only"]:
        print("Skipping performance evaluation/activation for smoke-only run", run["run_id"])
        continue
    evaluation_reports.append(evaluate_run(run["run_id"], DATA_DIR, ARTIFACT_DIR, device=DEVICE))
metric_table = pd.DataFrame([{"subset": r["subset"], "run_id": r["run_id"], **r["metrics"]} for r in evaluation_reports])
display(metric_table)
if len(metric_table): metric_table.to_csv(ARTIFACT_DIR / "all_subsets_metrics.csv", index=False)
''')
code('''
fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
for subset, ax in zip(SELECTED_SUBSETS, axes.flat):
    path = ARTIFACT_DIR / "runs" / selected_runs[subset]["run_id"] / "predictions.csv"
    if not path.exists():
        ax.text(.5, .5, "No final evaluation", ha="center"); ax.set_title(subset); continue
    frame = pd.read_csv(path)
    ax.scatter(frame.engine_id, frame.true_rul, s=13, label="Official endpoint truth", alpha=.7)
    ax.scatter(frame.engine_id, frame.prediction, s=13, label="Frozen model prediction", alpha=.7)
    ax.set(title=subset, xlabel="Engine ID (not sorted by truth)", ylabel="Remaining cycles")
    ax.legend(); ax.grid(alpha=.2)
plt.show()
''')
markdown('''
### 11 · Optional fault stress tests and sensor explanations

These operations use the frozen model and do not fit it again. Stress testing includes
unsupported conditions and reports abstention/exception rate, rather than hiding rejected
cases. Sensor-window occlusion is an approximate diagnostic, **not SHAP or causal diagnosis**.
''')
code('''
RUN_STRESS_TESTS = False
RUN_EXPLANATIONS = False
from fleetpm.evaluate import stress_run
from fleetpm.inference import Bundle

if RUN_STRESS_TESTS:
    for run in selected_runs.values():
        if not run["smoke_only"]:
            display(stress_run(run["run_id"], DATA_DIR, ARTIFACT_DIR, DEVICE))
if RUN_EXPLANATIONS:
    for subset, run in selected_runs.items():
        if run["smoke_only"]: continue
        bundle = Bundle(ARTIFACT_DIR / "runs" / run["run_id"], DEVICE)
        engines = load_engines(DATA_DIR / f"test_{subset}.txt")
        engine = engines[min(engines)]
        explanation = bundle.sensor_occlusion(engine, len(engine["raw"])-1)
        write_json(ARTIFACT_DIR / "runs" / run["run_id"] / "occlusion.json", explanation)
        display(pd.DataFrame(explanation["sensors"]).sort_values("rul_change"))
        del bundle
''')
markdown('''
### 12 · Connect actual bundles to the website and replay maintenance decisions

This local replay uses model predictions on observed test sensor prefixes; truth never
enters the triage packet. The rule agent applies persistence, exception handling and one
active order per incident. Delivery is a durable local queue, not an external CMMS.
Ordinary service is not a failure label, and closing an order is not physical self-healing.
''')
code('''
from fleetpm.repository import activate_run
from fleetpm.store import Store
from fleetpm.evaluate import replay_run

RUN_LOCAL_REPLAY = True
store = Store(RUNTIME_DIR / "fleetpulse.sqlite")
for subset, run in selected_runs.items():
    if run["smoke_only"]: continue
    activate_run(run["run_id"], ARTIFACT_DIR)
    if RUN_LOCAL_REPLAY:
        display(replay_run(run["run_id"], store, DATA_DIR, ARTIFACT_DIR, DEVICE, max_engines=12, last_cycles=60))
display(pd.DataFrame(store.orders("dataset")))
''')
markdown('''
### 13 · Open the bright, white-theme dashboard

From a separate terminal in this project:

```bash
.venv/bin/python -m fleetpm.webapp
```

Open **http://127.0.0.1:8765**. Dataset mode reads the active bundles and the actual exported
endpoint/replay records. The clearly marked **Guided demo** uses separate synthetic records
and never enters training. The dashboard also offers data import, configuration, training-job
status, run comparison, asset histories and persistent local work orders.

To view MLflow separately after opting into tracking:

```bash
.venv/bin/mlflow ui --backend-store-uri "sqlite:///artifacts/mlflow.db"
```

**Evidence limits:** this is an executable implementation prepared for training, not an
assertion of achieved benchmark accuracy, calibrated field coverage, runtime or maintenance benefit.
''')

notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"FleetPulse CUDA (Python 3.12)","language":"python","name":"fleetpulse"},
          "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
for n,cell in enumerate(cells): cell["id"]=f"fleetpulse-{n:03d}"
path=ROOT/"notebooks"/"C_MAPSS_All_Subsets_CUDA.ipynb"
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n")
print(path)
