# AeroGuard baseline ML protocol

This describes the implemented CPU baseline in `aeroguard/baseline.py`, the actual four-subset execution, and its limits. It is separate from the preserved CUDA research implementation in `fleetpm/` and the unexecuted research notebook. No published paper scores are substituted for local measurements.

## Data and provenance

NASA C-MAPSS is a historical, simulated turbofan degradation benchmark, not a live physical telemetry feed. Each record contains an engine identifier, observed cycle, three operating settings, and 21 sensor values. The application replays observed prefixes or ingests compatible batches. The selected local source corpus and every file SHA-256 are recorded in `data/cmapss/provenance.json`.

Observed, schema-validated engine counts are:

| Subset | Complete training trajectories | Truncated official test trajectories |
| --- | ---: | ---: |
| FD001 | 100 | 100 |
| FD002 | 260 | 259 |
| FD003 | 100 | 100 |
| FD004 | **249** | **248** |

NASA's [portal description](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) lists FD004 counts in reverse. The supplied numeric files and their 248-element endpoint truth vector are internally consistent; the application uses their observed counts and documents the discrepancy. These files were copied from the user's existing local corpus, preserving the source project. This is not a claim of an independently verified official archive checksum.

For a complete training trajectory, the retrospective target at observed cycle `t` is `last_training_cycle - t`; the regressor fits `min(125, target)`. Eventual trajectory length is a training label construction step, never a prediction feature. Test files are truncated. Their separately supplied `RUL_FD00n.txt` values refer to each engine's final observed test cycle, ordered by consecutive engine ID.

## Engine isolation and preprocessing

Before any learned normalization or feature selection, fixed `split_seed=42` assigns approximately 60% of training engines to fit, 20% to development selection, and 20% to independent interval calibration. All windows from an engine remain in its cohort. The executed fits used:

| Subset | Fit engines | Development engines | Calibration engines | Sampled fit windows |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 60 | 20 | 20 | 3,789 |
| FD002 | 156 | 52 | 52 | 10,236 |
| FD003 | 60 | 20 | 20 | 4,728 |
| FD004 | 149 | 50 | 50 | 11,572 |

`Preprocessor.fit` sees fit-engine sensor trajectories only. K-means identifies one operating regime for FD001/FD003 and six for FD002/FD004, using standardized operating settings. Sensor means and standard deviations are fit within those regimes. Constant or unavailable variation uses a finite scale guard. At inference, regime assignment and scaling use those frozen training parameters. Missing values remain represented by observation masks and observation-gap counters; there is no interpolation from future cycles or other held-out engines.

Schema ingestion may inspect test-file structure and the endpoint truth vector's length/finite values. It does not use those values to fit preprocessing, choose hyperparameters, calculate development metrics, or promote models. The fitting function itself opens only `train_*.txt`. The evaluator freezes every test endpoint prediction before parsing truth for the numerical comparison.

## Causal feature contract

The baseline computes 90 features at each observed cycle:

| Feature group | Count | Definition |
| --- | ---: | --- |
| Current standardized sensor values | 21 | Trusted measurements at cycle `t`; unavailable values carry zero plus separate quality evidence |
| Rolling sensor means | 21 | Trusted observations in `[t-29, t]` |
| Rolling sensor standard deviations | 21 | Same trailing context |
| Rolling sensor slopes | 21 | Least-squares slopes over observed cycle positions in the same context |
| Standardized operating settings | 3 | Frozen fit-only transform at cycle `t` |
| Observed fraction | 1 | Trusted sensor values divided by available context positions |
| Maximum observation gap | 1 | Largest gap in the trailing context, represented in the declared 30-cycle units |
| Capped observed cycle | 1 | `min(observed_cycle,125)/125`; not divided by eventual lifetime |

Statistics require at least five trusted observations per channel; unavailable statistics become zero. Cumulative-sum computations only subtract past prefix sums. The minimum fitting history is 15 cycles at the default 30-cycle window. Short test histories still receive numerical predictions, while maintenance decisions wait for the configured complete context (`required_history`, default 30).

The tests mutate every future sensor suffix and also truncate the provided trajectory exactly at the requested prediction cycle. Past features, RUL predictions, and anomaly scores remain identical. Sensor names and units are centralized in `SENSOR_SCHEMA`; for example sensor 3 is T30, HPC outlet total temperature, and sensor 7 is P30, HPC outlet total pressure. No arbitrary channel is renamed “vibration” or “motor current.” Uncertain unit mappings are labelled dataset units rather than invented physical units.

## RUL estimator and reproducibility

Each subset has its own [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). The executed baseline used 100 fixed-budget boosting iterations, learning rate 0.08, at most 15 leaf nodes, 127 histogram bins, L2 penalty 1, minimum leaf size 15, `early_stopping=False`, seed 42, and stride 3. Per-window sample weights give each fit engine equal total training weight. CPU numerical threads are limited to two.

No internal random window-level validation is used for early stopping. Development performance is measured at one deterministic historical cutoff per development engine, selected by the fixed split seed. The cutoff is an offline evaluation policy; it never becomes a predictor input. Changing the training seed preserves development/calibration engine cohorts. The model is not refit on development or calibration engines after evaluation.

Each fit saves joblib model/preprocessor/anomaly objects, configuration, split memberships, feature names, per-window engine/cycle mapping, development predictions, calibration reference, versions, source hashes, and a bundle integrity manifest. Every run has a new immutable ID. Loading verifies the feature contract, saved scikit-learn version, required files, and hashes before deserialization. Only locally generated trusted artifacts are accepted; model/joblib upload is not an ingestion feature.

## Independent anomaly detection

A separate Isolation Forest uses the first 84 sensor/statistic features. It deliberately excludes observed age, operating settings, quality counts, and RUL predictions, avoiding a trivial anomaly rule that labels every old engine because it is older than the reference. It is fit on the first 11 complete-context windows of fit engines, with up to 256 sampled windows per tree, 128 trees, and seed 42. Those early windows are an explicitly declared healthy-prefix proxy, not verified healthy ground truth.

The stored reference contains sorted negative Isolation Forest scores. A new score is shown as its empirical percentile against that reference. A high percentile means unusual relative to the selected early-prefix population. It is **not** the probability of failure, a physical diagnosis, or a supervised fault classifier. C-MAPSS supplies RUL endpoints but no trustworthy per-cycle anomaly onset labels, so the application does not invent anomaly precision, recall, or AUC. Thresholds are configurable engineering assumptions.

## Calibration and quality exceptions

The frozen RUL predictor is calibrated on one fixed historical prefix per reserved calibration engine. Only prefixes passing input-quality/domain checks enter the reference. Absolute errors use the capped target. At nominal 90%, the split-conformal radius is order statistic `ceil((n+1)*0.9)`. If that rank exceeds the number of independent engines, the radius is unavailable; the app explicitly reports a calibration exception.

The predicted interval is `[max(0,prediction-radius), min(125,prediction+radius)]`. Boundary clipping makes its width vary by engine. Nominal coverage requires exchangeable independent engine-prefix samples under the declared cutoff and quality population. It is not simultaneous coverage over a replayed trajectory, an engine-specific failure probability, or a field guarantee. Missing sensor cases are not silently treated as calibrated clean data. Unsupported missingness/domain conditions, insufficient calibration, or intervals wider than the configurable decision gate become review exceptions.

The default interval-width gate is 80 cycles. The evaluator records that engineering setting explicitly; changing a policy setting affects subsequent inference/decisions, not an already archived decision. Point RUL and calibrated bounds are capped at 125 cycles, so `125+` is a horizon limit rather than a precisely estimated longer lifetime.

## Metrics and NASA score

Official evaluation produces one endpoint prediction per test engine. RMSE, MAE, and R² use the **raw supplied test truth**. Capped-target RMSE and interval coverage are separately named. The regressor's 125-cycle cap can create large safe-side underestimation on engines whose raw RUL is much larger; those examples are not removed.

With error `d = predicted_RUL - actual_RUL`, the summed NASA score is:

```text
if d < 0: exp(-d/13) - 1
else:     exp( d/10) - 1
```

Positive error is optimistic overprediction, so it receives the stronger penalty. **Lower NASA score is better.** The score is a sum, so subsets with different engine counts should not be compared as if they had equal sample sizes. R² is supplementary and may be unavailable when all targets are constant. MAPE is not used as the primary metric near zero RUL.

## Measured initial baseline run

The following are real local results of the initial 100-iteration, single-seed execution through the application service on 23 October 2026. The [README reference table](../README.md#evaluation-boundaries) records the public summary; raw local logs and artifact snapshots are excluded from this repository. These are measured CPU baseline results, not the CUDA research model or a claimed reproduction of the papers.

| Subset | Test engines | Raw RMSE | Raw MAE | R² | Summed NASA score | Capped RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FD001 | 100 | 14.3561 | 10.6238 | 0.88065 | 263.6018 | 12.9679 |
| FD002 | 259 | 25.8850 | 17.1969 | 0.76833 | 7,053.6293 | 13.2351 |
| FD003 | 100 | 14.3739 | 11.2620 | 0.87943 | 311.4315 | 12.9346 |
| FD004 | 248 | 27.0292 | 19.1435 | 0.75425 | 5,389.0739 | 14.7075 |

| Subset | Calibration radius | Measured capped interval coverage | Mean capped interval width |
| --- | ---: | ---: | ---: |
| FD001 | 37.1477 | 98.00% | 60.7869 cycles |
| FD002 | 29.2990 | 96.14% | 47.8148 cycles |
| FD003 | 23.2167 | 93.00% | 39.2995 cycles |
| FD004 | 34.1034 | 96.77% | 53.8138 cycles |

Coverage must be read together with these substantial interval widths. The measurements do not establish well-localized failure dates or good field calibration. Development and official test metrics are not directly comparable populations: development cutoffs follow a declared pseudo-truncation policy, while official test truncation follows the benchmark generation process.

Initial run IDs are `AG-FD001-bb895fcc403d`, `AG-FD002-12f3decf5b59`, `AG-FD003-dc9b72fe76a6`, and `AG-FD004-75fa8fc2e9f5`. All four real MLflow run records were queried through `MlflowClient`: each is `FINISHED` and contains both development RMSE and official-test RMSE. Any later retraining is a separate candidate/version; this table remains a snapshot of the initial experiment.

## Promotion, policy, and limits

The application chooses/promotes models using fixed development-engine metrics **before** official endpoint evaluation. A candidate must satisfy comparable data/cohort/horizon rules and configurable development RMSE/NASA tolerances. Official test RMSE is displayed after the choice and does not decide promotion. Previous versions are retained. Repeated manual tuning after viewing the same official test scores invalidates an untouched-test claim.

The plain Python triage agent receives model predictions, anomaly percentiles, normalized sensor trend evidence, quality flags, and the configured threshold snapshot. It does not receive test truth. Persistence, hysteresis, quality precedence, warm-up, and an active-order latch control alerts. Every archived decision includes input values, evaluated/triggered rules, severity, explanation, recommended action, policy version, and timestamp. Decisions and orders are simulated records of a historical replay; no actuator self-repair or external maintenance dispatch occurs.

The initial execution is one seed, not a repeated-seed study. No matched neural ablation, real-world fault-onset validation, CPU p95 service benchmark, deployed drift detector, or field maintenance benefit is claimed. The retained PyTorch/CUDA research model still requires its optional dependencies and its own actual training/validation. The CPU baseline tests and real benchmark results do not verify CUDA execution.
