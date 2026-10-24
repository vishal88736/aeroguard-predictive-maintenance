# Contributing to AeroGuard

Read the [README](README.md), [ML protocol](docs/ML_PROTOCOL.md) and [research workflow](docs/RESEARCH_WORKFLOW.md) before changing model or data behavior.

## Local development

Use Python 3.12 and `requirements-app.txt` for the application. Install CUDA dependencies manually from `requirements-kan.txt`; do not add installation/download code to notebook cells. Node.js is only needed for browser-development tests.

Run the Python tests and source checks in the README. Fixtures must stay in temporary directories and be labelled as fixtures. Browser tests must not retrain real benchmark models, replace active versions or dispatch external maintenance actions.

## ML changes

- Split by engine before preprocessing/windowing; keep official test targets out of fitting and selection.
- Preserve causal windows; distinguish raw/capped targets, anomaly scores, calibration and quality gates.
- Compare extensions on matched cohorts/prefixes. Never present a published paper score as a reproduced result.
- Keep SHAP/LIME after evaluation. Record background provenance and fidelity; do not relabel occlusion.
- Update implementation versions/contracts when changing model interfaces or checkpoint/export semantics.

## Notebook source

Edit `notebooks/Transformer_KAN_BiLSTM_GPU.py`, then run `python scripts/build_kan_notebook.py`. Public notebooks have empty outputs and execution counts. Keep raw experiments local and publish reviewed summaries separately when appropriate.

## Public source boundaries

Do not commit `.env`, credentials, supplied PDFs, datasets, weights, databases, private logs or original reference images. Keep third-party notices. Run `python scripts/audit_public_tree.py` before publishing. Describe changes and verification in a focused pull request; distinguish fixture, local-corpus and actual GPU evidence.
