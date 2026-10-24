# AeroGuard implementation audit

Inspected before modification, 23 October 2026.

The existing project uses a buildless JavaScript/SVG client, Python standard-library HTTP server, SQLite persistence, strict NASA C-MAPSS parsing, and a CUDA research pipeline. The notebook orchestrates causal repair, CNN–GRU/attention, a KAN head, out-of-fold correction, calibration, and Isolation Forest. No research model was trained or ML dependency installed at the start of this task.

Useful implementations retained: `fleetpm/data.py`, `features.py`, `metrics.py`, `models.py`, `train.py`, `inference.py`, the CUDA notebook, original paper reviews, source PDFs, and transactional work-order/event storage. The legacy website and API remain in `web/` and `fleetpm/webapp.py` for compatibility. AeroGuard uses its own frontend assets and FastAPI entry point.

The earlier website had seven hash routes, fixed policy thresholds, limited work-order actions, manual activation, synchronous evaluation/replay, and no landing page, analytics, operational settings, CSV import, persistent pipeline stage tracking, or retraining promotion gate.

The new application keeps the established framework-free frontend to preserve the working client approach and avoid a framework migration. It adds a FastAPI/Pydantic service layer, typed browser contracts, local Inter/Lucide assets, a CPU histogram-gradient-boosting RUL baseline, a separately fitted Isolation Forest, persistent background jobs, explicit pipeline stages, input-only causal features, guarded promotion, work-order ownership/audit, and genuine MLflow tracking. The existing PyTorch workflow is retained as an optional research path, rather than replaced with synthetic predictions.

Dataset discovery found all twelve C-MAPSS files in another local project. Only those selected files are copied after strict validation. Their source paths, hashes, expected/observed counts, and provenance limitations are recorded in `data/cmapss/provenance.json`; no other project context or artifacts are imported.

The user requested execution and end-to-end verification. The lightweight application requirements are installed into this project's `.venv`, with `.venv/cache/uv`; CUDA packages remain optional. Tests and benchmark metrics are reported separately from synthetic fixtures. Public deployment, live physical sensors, actual maintenance dispatch, and field reliability are outside the local case-study implementation.
