# Selected research workflow: Transformer–KAN–BiLSTM

The selected research model follows Xu, Guo and Zhou, *Aerospace* 12 (2025), 998, [DOI 10.3390/aerospace12110998](https://doi.org/10.3390/aerospace12110998). Its module order is Transformer → spline KAN → BiLSTM. The earlier four-paper hybrid remains available as an archived experiment. Neither neural model has been trained here, so no measured conclusion that one neural design outperforms another is claimed. The paper model was selected in response to the user's preference and its reported architecture/results.

The [standalone notebook](../notebooks/Transformer_KAN_BiLSTM_GPU.ipynb) contains all model, data, training, evaluation, export and explanation code. Its percent-format source is [the adjacent Python file](../notebooks/Transformer_KAN_BiLSTM_GPU.py), rebuilt using `python scripts/build_kan_notebook.py`. All delivered notebook cells have empty outputs and null execution counts. Training remains for the user to run.

## Architecture and proposed extension

The reference uses six encoder layers, six decoder layers, historical sensor embeddings as decoder inputs, two cubic B-spline KAN layers, a BiLSTM, and a ten-horizon RUL regression head. FD001/FD003 use embedding 32, eight heads and 200 spline intervals. FD002 uses 64/eight/40 and FD004 uses 128/16/50, an explicitly documented interpretation of the paper's complex-subset settings. The paper does not fully specify window length, tensor interfaces, recurrent widths, normalization, optimizer or target cap; implementation choices are recorded in the notebook, rather than described as exact numerical reproduction.

The proposed `conditioned_kan` variant adds a bounded setting-conditioned gate to KAN features before BiLSTM. It uses the three observed operating settings at each token. Its multiplicative factor is `1 + 0.5*tanh(MLP(settings))`, initialized to one. Constant settings removed by preprocessing map to zero. It adds no future observations, engine identity, target, repair generator, CNN/GRU branch or reinforcement-learning component. Explicit operating-condition modulation is a testable engineering hypothesis, particularly for FD002/FD004; no priority claim or measured improvement is made.

`VARIANTS = ["paper", "conditioned_kan"]` runs the matched ablation. The same source file, train/validation engine IDs, validation prefixes and hyperparameters are required. The extension is selected only if its mean validation RMSE across the requested seeds is strictly lower. Ties and worse performance retain the paper reference. Selection is saved before official test evaluation; seeds are not selected from their test scores. The website independently checks the selection rule in imported reports.

See the [architecture diagram](research-architecture.svg).

## Manual GPU environment and training

Use Python 3.12 and an NVIDIA GPU with a suitable driver. No installation or training is started automatically by the website or this documentation.

```bash
cd aeroguard-predictive-maintenance
python3.12 -m venv .venv-kan
.venv-kan/bin/python -m pip install --cache-dir .venv-kan/cache/pip torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv-kan/bin/python -m pip install --cache-dir .venv-kan/cache/pip -r requirements-kan.txt
.venv-kan/bin/python -m ipykernel install --prefix .venv-kan --name kan-bilstm --display-name "KAN BiLSTM CUDA"
.venv-kan/bin/python -m jupyterlab
```

The wheel index above is CUDA 13.0; use the [official PyTorch installer](https://pytorch.org/get-started/locally/) for a different compatible driver/platform. CUDA dependencies can require several GB. Caches remain inside the environment. Windows uses `.venv-kan\Scripts\python.exe`. A compatible Colab/Kaggle GPU runtime can use the same notebook after changing paths, without requiring the rest of this repository.

Open `notebooks/Transformer_KAN_BiLSTM_GPU.ipynb` and select the CUDA kernel. Set `DATA_DIR`, `OUTPUT_DIR`, `SELECTED_SUBSETS`, `SEEDS`, and `VARIANTS`. The default is all four subsets and two variants: eight sequential training runs for one seed. A first experiment can select FD001 only. The default microbatch is 32, accumulated to 256; use 16 or 8 for less memory while retaining effective batch 256. CUDA execution and actual VRAM requirements have not been measured here.

Place `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, and the equivalent files for FD002–FD004 in `DATA_DIR`. The current local corpus is already present at `data/cmapss/`. Training files must contain complete engine trajectories. Evaluation targets are a separate endpoint vector and never become predictor inputs.

Run cells in order. The training cell performs real GPU fitting; other definition/preflight cells do not. `RESUME=True` reuses completed identical runs or resumes at the last complete epoch. Changed data/configuration/implementation creates a separate run. Stopping mid-epoch preserves the last completed checkpoint. Training uses mixed precision; validation, final inference and explanations use FP32 for consistent reported outputs. The final stages evaluate official endpoints, verify a CPU TorchScript export, and then compute SHAP/LIME explanations.

## Evaluation boundary

The paper explicitly describes combining original train/test data and repartitioning 70/20/10 (p. 14). This notebook retains the official held-out test boundary, splits official training engines 80/20 before fitting, and evaluates one last-observed window per test engine. The paper's RMSE values 3.6784/4.9498/3.7988/7.5202 are author-reported values under its protocol, not expected scores under this protocol.

Targets are uncapped by default because this paper does not specify a 125-cycle cap. An optional cap is recorded and raw/capped evaluation is kept distinct. RMSE and MAE use cycles, R² is dimensionless, and MAPE is a percentage over nonzero targets with the denominator reported. NASA score uses predicted-minus-true error: negative errors have scale 13 and positive late errors scale 10; lower is better. Predictions are saved before evaluation targets are opened.

Only fitting-engine data determines constant removal, normalization and scheduled spline grids. Smoothing uses a trailing mean per engine. BiLSTM can traverse an observed window in both directions but never sees readings after the prediction cycle. Validation windows and test trajectories do not fit normalization or KAN grids.

## SHAP and LIME are the final analysis stage

The notebook imports actual `shap.KernelExplainer` and `lime.lime_base.LimeBase` only in the post-training stage. There is no SHAP/LIME loss, model selection based on explanations, or occlusion score relabelled as SHAP/LIME. Dependencies are explicitly listed in `requirements-kan.txt` and never installed by code cells.

Each mask feature represents the whole observed history of one selected raw sensor. A removed sensor is replaced by that sensor's full trajectory from fitting-engine background windows, averaged over four backgrounds selected with observed operating settings. Settings remain fixed. This preserves within-sensor temporal structure; cross-sensor mixtures can still be unrealistic. Interpret these as local grouped-mask model explanations with operating settings fixed, not causal fault diagnoses.

Default budgets are 256 SHAP evaluations, 512 LIME neighborhood masks, 128 independent held-out LIME masks, and a kernel width of 0.35. Actual neural calls include background averaging. Engine 1, the first requested seed, and the validation-selected variant are explained for each subset. Change `EXPLAIN_ENGINE_IDS` and budgets explicitly for more coverage. There is no claim of global importance from this local sample.

SHAP exports contributions, background expectation and additivity residual in cycles. LIME exports coefficients, intercept, local prediction, weighted fitting R² and independently held-out weighted R²/MAE. The website displays the methods separately and retains poor fidelity scores. A missing explanation is shown as missing, never filled with example bars.

## Website integration

Run the existing app with `.venv/bin/python -m aeroguard --port 8765`, then visit `/models`. The blue/orange/white research section includes the notebook, requirements, architecture and ZIP import. Local exports under `artifacts/kan_bilstm/FDxxx/seed-…/research_export.json` are discovered automatically; cloud exports can be imported using their ZIP. This starts no training and sends no maintenance dispatch.

The website validates required files, hashes, configuration identity, disjoint cohorts, frozen validation prefixes, recomputed metrics, local source-file hashes, selection evidence and explanation provenance. Importing the same bundle is idempotent. Explanations can be added to the same immutable model/evaluation bundle after training; previous reports are retained when updated. Models and frozen predictions cannot be silently replaced under an existing run ID.

Research detail pages provide actual endpoint predictions, learning curves, selected/comparison variant labels, engine selection, observed sensor trends, SHAP/LIME evidence, artifact downloads, and optional sensor-prefix inference. Reports/explanations require no PyTorch in the API environment. Prefix inference loads the verified `inference.ts` model and checks the exported FP32 probes before use. If PyTorch or matching local corpus is unavailable, the API responds with an explicit dependency status and the UI explains what is missing.

To enable that inference in the CUDA environment, install the application dependencies there manually as well:

```bash
.venv-kan/bin/python -m pip install --cache-dir .venv-kan/cache/pip -r requirements-app.txt
.venv-kan/bin/python -m aeroguard --port 8765
```

The app's existing CPU baseline, independent anomaly model and simulated work-order pipeline retain their own registry and labels. Research models use a different 80/20 uncapped development protocol, so cross-family automatic promotion into the baseline's calibrated triage is deliberately blocked by keeping the research registry separate. No uncalibrated interval or invented anomaly score is attached to neural RUL results. This is a completed training/export/viewing integration; GPU fitting, measured model superiority and operational maintenance validation remain pending the user's training and evaluation.

## API additions

| Endpoint | Purpose |
| --- | --- |
| `GET /api/research` | Export catalog, pending/error states, dependency status |
| `POST /api/research/import` | Validate/import the notebook ZIP; same-origin client header required |
| `GET /api/research/runs/{run_id}` | Actual metrics, learning curves, endpoint predictions, artifact metadata |
| `GET /api/research/runs/{run_id}/engines/{engine_id}` | Frozen engine prediction, observed sensors, saved SHAP/LIME |
| `GET /api/research/runs/{run_id}/predict?engine_id=1&cycle=30` | Actual optional TorchScript inference; no targets as inputs |
| `GET /api/research/runs/{run_id}/download` | Verified portable training/result bundle |
| `GET /downloads/kan-notebook` | Current standalone GPU notebook |
| `GET /downloads/kan-requirements` | GPU/explanation requirements |
| `GET /documents/research.svg` | Selected architecture and explanation order |

Tests using isolated export fixtures verify contracts and failure handling; their made-up weights/metrics are never installed into the real catalog or presented as benchmark results. These tests do not establish GPU training, TorchScript numerical compatibility or actual SHAP/LIME performance in the absent research environment.
