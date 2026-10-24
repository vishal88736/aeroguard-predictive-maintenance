# Retained FleetPulse experiment

The original standard-library web client and integrated ML implementation remain for reference and compatibility. AeroGuard's selected public interface is `frontend/` and its entry point is `python -m aeroguard`; follow the [root README](../README.md) first.

## Legacy entry point

```bash
.venv/bin/python -m fleetpm.webapp
```

The legacy client defaults to loopback port 8765. Do not run it on the same port as AeroGuard. Its explicitly labelled synthetic demo is separate from actual benchmark inference and work orders.

## Earlier integrated notebook

[C_MAPSS_All_Subsets_CUDA.ipynb](../notebooks/C_MAPSS_All_Subsets_CUDA.ipynb) uses the shared `fleetpm` package. Its compact model includes causal repair, CNN–GRU, attention and a spline readout, with development-only residual correction and independent calibration. These are paper-inspired adaptations, not numerical reproductions. This neural experiment remains untrained in the documented validation snapshot.

The newer [Transformer–KAN–BiLSTM notebook](../notebooks/Transformer_KAN_BiLSTM_GPU.ipynb) is the selected research path, with a controlled condition-gate comparison and actual SHAP/LIME code at the final stage. See [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md).

## Data and theme

Both workflows use user-provided C-MAPSS files and keep labels out of predictor inputs. No dataset or model is downloaded automatically. The earlier client retains its navy/peach styling with an authored generic bridge SVG. The supplied university reference image is excluded from the public repository; the current AeroGuard theme is blue, orange and white.
