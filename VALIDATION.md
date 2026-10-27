# Validation and evidence boundaries

The source, fixture tests and website workflows were verified locally before public publication. This repository excludes the original datasets, model weights, databases and private execution logs; the counts below describe the recorded local run, not a promise that those artifacts are bundled.

## Executed checks

| Check | Recorded result |
| --- | --- |
| Python suite with the local C-MAPSS corpus | 119 discovered, 114 passed, 5 optional PyTorch tests skipped |
| Source-only publication checkout | 119 discovered, 112 passed, 7 optional corpus/PyTorch checks skipped |
| Existing application browser workflows | 63 passed |
| Research/theme browser workflows | 28 passed; disposable publication runner also passed |
| Browser viewports | Desktop 1440, tablet 1024, mobile 390 pixels |
| JavaScript errors | None in the recorded browser runs |
| Corpus preservation | All 12 selected source file hashes retained |
| Supplied document preservation | All 5 original PDF hashes retained; documents are not redistributed |
| Selected GPU notebook | 38 cells, empty outputs and null execution counts |

The Python tests cover schema validation, causal features and prefix invariance, fit-only preprocessing, metric direction, actual isolated CPU fitting/calibration, anomaly detection, triage persistence, work-order idempotency, jobs/failure handling, promotion, API validation and research export integrity.

The existing browser suite checks routes/refresh, filters, recorded histories, labelled demo order workflows, data controls, model comparison and responsive navigation. The research suite checks import, model/result/engine views, separate SHAP/LIME panels, source labels, missing-artifact/dependency states and the requested palette.

Research-result UI tests use a **visibly labelled isolated contract fixture with dummy model bytes**. They validate interfaces and failure handling; they do not establish neural performance or actual explanation quality. No fixture model was placed in the real registry.

## Source-only checkout and CI

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/build_kan_notebook.py
node scripts/check_frontend.cjs
python3 scripts/audit_public_tree.py
```

On a checkout without NASA files, the two real-corpus checks explicitly skip. Without PyTorch, five optional numerical tests skip. Other data/model/API tests create temporary fixtures. CI installs the lightweight application stack and runs these source-only tests, JavaScript checks, publication checks and isolated research browser contracts. The Actions badge reports the actual current workflow outcome.

See [browser setup](README.md#browser-contract-tests) for the disposable test runner. It does not train a real model or change your active registry.

## Historical CPU benchmark evidence

Four real CPU baseline models were trained and evaluated locally, one per subset, with separate Isolation Forests and independent calibration. A genuinely weaker retraining candidate was rejected by the development-metric promotion gate; the active model was retained. Repeat replay/idempotency checks created zero duplicate incidents.

Official endpoint metrics use one last observed test window per engine and uncapped supplied truth. The baseline head is capped at 125, which limits raw-target performance. Raw and capped metrics are distinct. The historical table is in the [README](README.md#evaluation-boundaries); exact feature/split/calibration assumptions are in [ML_PROTOCOL.md](docs/ML_PROTOCOL.md).

## Not executed or established

- GPU fitting, the selected model's tensor/CUDA execution and numerical TorchScript compatibility.
- Actual neural SHAP/LIME calculations, or a measured benefit from the condition gate.
- Numerical reproduction of the paper's differently repartitioned results.
- Live physical telemetry, maintenance safety, real dispatch, field validation of drift thresholds or measured field benefit.
- Public hosting of the application; a public source repository is separate from application deployment.

The notebook contains GPU sanity checks and export probe comparisons for the user's training run. Missing training remains an explicit pending state. See [RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md).

## Versioned MLOps implementation verification, 27 October 2026

- Current local suite: **140 discovered, 135 passed, 5 optional PyTorch checks skipped**.
- Disposable source-only checkout: **140 discovered, 133 passed, 7 optional corpus/PyTorch checks skipped**.
- Research/theme browser regression: **28 checks passed**, no JavaScript errors.
- New MLOps browser workflows: **30 checks passed**, no JavaScript errors, covering real CPU fitting/MLflow registration on isolated synthetic fixtures, observed-batch inference, fixed-cohort retraining/rejection, scheduling, review resolution and responsive layouts at 1440/1024/390 pixels.
- Real local workspace: the four retained CPU models were packaged, loaded, numerically checked and registered with champion aliases. Four immutable training snapshots and frozen protocols were created. The existing five training-run records remain; no CPU benchmark refitting or GPU training occurred during initialization.
- Twelve source dataset files, five supplied PDFs, original CSS/HTML files and the active-model pointer retained their pre-change hashes. Existing model bundles pass their saved integrity manifests.
- Airflow Compose configuration validates. The HTTP adapter is exercised against a real local API in automated tests. The container/DAG runtime remains unexecuted here because Docker-daemon access is unavailable.

Monitoring now implements local regime-conditioned engine-level KS screening with explicit independence, prefix-length, sample-size and threshold limitations. Field drift validity, maintenance effectiveness and distributed availability remain unestablished. Neural exports can be tracked in MLflow, but numerical GPU execution and calibrated maintenance promotion of neural models remain separate verification requirements. See [MLOPS.md](docs/MLOPS.md).
