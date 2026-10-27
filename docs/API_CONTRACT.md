# AeroGuard application API

All application endpoints use `/api`. Responses are JSON. Errors are `{error:{code,message,details?}}` with meaningful HTTP status codes. Mutations require `X-AeroGuard-Client: dashboard` from the browser. Long-running POSTs accept `Idempotency-Key` and return HTTP 202 `{job_id,status}`. Defaults: `mode=benchmark`, `subset=ALL`; demo records use the separate `demo` namespace.

## Queries

- `GET /health`: application, status, Python/version, dependencies, hardware, data/model readiness.
- `GET /dashboard?mode=&subset=&engine=&severity=&cycle_min=&cycle_max=`: `{mode,source_label,summary,assets,orders,runs,jobs,active,last_successful_run}`. Summary: `total,healthy,attention,critical,unknown,open_orders,mean_rul,models_active`.
- `GET /engines` with the same filters: `{items,total,source_label}`.
- `GET /engines/{asset}` where asset is `FD001/test/003` or `FD001/demo/003`: `{asset,history,sensors,operating_settings,decisions,orders,source_label,demo}`. Sensor query `limit` defaults to 60 (max 150). Each sensor is `{key,index,symbol,label,unit,value,series:[{cycle,value}]}`. History has cycle/rul/lower/upper and optional `true_rul`; policy packets never receive ground truth.
- `GET /analytics`: dashboard plus model metrics and monitoring notes. Client derives histogram/donut/trends from returned actual assets/history.
- `GET /data`: `{datasets,provenance,columns}`; dataset items reuse fleetpm inventory keys (`subset,ready,expected_train,expected_test,train_engines,test_engines,rows,files,errors,official_counts_match`).
- `GET /data/{subset}/preview?partition=train|test|RUL&limit=20`: `{columns,rows,statistics,split,source_label}`. Statistics include counts, missing values, validation errors, cycle bounds; split is null until prepared/model fitted.
- `GET /models`: `{runs,active,promotions,research_runs}`. Each run has `run_id,subset,training_status,created_at,finished_at,configuration,family,validation_metrics,evaluation,calibration,parameters,artifact_path,mlflow_run_id,promotion_eligible,promotion_reason` where available. Evaluation is `{metrics,predictions?}`. Metrics keys: `rmse,mae,r2,nasa_score,capped_rmse,n_engines`.
- `GET /models/{run_id}`: full run, split, artifact file names and hashes.
- `GET /models/{run_id}/download`: immutable bundle ZIP.
- `GET /jobs`: `{items:[job]}`. Job: `id,kind,status,configuration,created_at,started_at,finished_at,error,result`. Status queued/running/complete/failed/interrupted.
- `GET /jobs/{id}`: job plus `stages`. Stage: `id,name,subset,status,started_at,finished_at,duration_seconds,records,detail,error`.
- `GET /orders?mode=&q=&status=&priority=`: `{items}`. Order: existing Store fields plus `subset,predicted_rul,owner,notes,simulated`. States open/acknowledged/in_progress/resolved.
- `GET /orders/{id}?mode=`: order plus `audit`.
- `GET /settings`: validated Settings below.
- `GET /notifications?mode=`: `{items}` with id/title/message/asset/timestamp/severity.
- `GET /search?q=&mode=`: `{engines,orders}`.

Engine fields: `asset,engine_id,subset,cycle,rul,lower,upper,anomaly,quality_valid,quality,bundle_id,timestamp,state,health,trend`. `health` is healthy/attention/critical/unknown; `state` preserves NORMAL/WATCH/MEDIUM/HIGH/CRITICAL/DATA_EXCEPTION/NOT_REPLAYED/WARMING_UP. RUL is in cycles; anomaly is an empirical healthy-prefix percentile, not a labelled-fault probability.

## Mutations

- `POST /data/import`: multipart `file`, `subset` (FD001..4), `partition` (train/test/RUL). ZIP contains all 12 original files. CSV supports the documented 26-column schema or a single RUL column. No silent overwrite of different data.
- `POST /data/prepare`: PipelineRequest (below), background input validation/fit-only preprocessing audit.
- `POST /models/train`: TrainRequest (below), background fit → calibration → development-gated promotion → locked official evaluation → replay, recording every stage. Official evaluation never decides promotion.
- `POST /models/promote`: `{run_id,rollback:false}`; only eligible candidates. Explicit rollback `{run_id,rollback:true}` must refer to a previously promoted version; requires user confirmation in UI.
- `POST /pipeline/run`: PipelineRequest, background replay of active model(s); missing model is an actionable failure, never synthetic fallback.
- `POST /jobs/{id}/retry`: empty object, copies prior job inputs, new idempotency key; retains immutable artifacts/events.
- `PATCH /orders/{id}?mode=`: `{action:acknowledge|assign|start|resolve,owner?:string,notes?:string,resolution?:preventive_service|sensor_repair|confirmed_failure|no_fault_found|deferred}`. Assign requires owner. Resolve requires notes and explicit resolution, with confirmation dialog.
- `PUT /settings`: full validated Settings.

TrainRequest defaults: `{subsets:["FD001"],iterations:100,learning_rate:.08,max_leaf_nodes:15,seed:42,split_seed:42,window:30,stride:3,cap:125,replay_cycles:30,promote:true,track_mlflow:true}`. CPU baseline is HistGradientBoosting RUL + independent Isolation Forest. Optional CUDA notebook remains a separate research workflow.

PipelineRequest: `{mode:"benchmark",subsets:["FD001"],cycles:30}`.

Settings: `{workspace_name:"AeroGuard Research",researcher_name:"Local researcher",default_subset:"ALL",policy:{critical_rul:5,critical_lower:10,critical_upper:10,attention_lower:30,watch_lower:50,anomaly_attention:.99,anomaly_extreme:.995,trend_threshold:.04,persistence:3,critical_persistence:2,rearm_cycles:5},promotion_rmse_tolerance_percent:0,promotion_nasa_tolerance_percent:5,replay_cycles:30,uncertainty_gate:80}`.

Benchmark engine detail paths are URL-encoded in API calls. Browser routes may use an encoded asset as one identifier, e.g. `/engines/FD001%2Ftest%2F003`. All ten specified browser routes must work with history navigation and direct refresh. `/` is a standalone landing page; the remaining routes share the sidebar/top bar. Static assets are served at `/assets/` and `/vendor/`. Downloads `/downloads/notebook`, `/downloads/requirements`, `/downloads/environment`, `/downloads/architecture` remain available.
## GPU research export APIs

The research endpoints and download routes are documented in [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md#api-additions). `ResearchManifest` in `aeroguard/research.py` validates source/model/selection/explanation contracts; browser types are in `frontend/api.d.ts`. Invalid archives or inconsistent evidence return structured 400 responses, conflicting immutable artifacts return 409, and missing optional neural dependencies return 503 with `research_dependency_unavailable`. Query/path validation remains Pydantic 422. Research imports never replace the CPU active-model map.


## Versioned MLOps endpoints

The `/api/mlops` endpoint family manages immutable incoming versions, monitoring, fixed-cohort training, registration, scheduling and reviews. Typed mutation payloads and background job IDs follow the existing API conventions. See [MLOPS.md](MLOPS.md#api-and-verification) and the generated [OpenAPI schema](openapi.json) for exact contracts.
