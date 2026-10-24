# Research references and adaptation boundaries

The papers below informed the design. Their PDFs, the original case-study brief and user reference imagery are not redistributed. Consult each publisher/source for access and licensing terms.

## Selected architecture

Kejie Xu, Yingqing Guo and Qifan Zhou. **Research on the Remaining Useful Life Prediction Algorithm for Aero-Engines Based on Transformer–KAN–BiLSTM.** Aerospace 12 (2025), 998. [DOI](https://doi.org/10.3390/aerospace12110998).

The notebook follows Transformer → spline KAN → BiLSTM. Tensor interfaces and unspecified training details are explicit implementation choices. The paper combines/repartitions original train/test data; this project retains the official held-out boundary, so scores are not directly comparable. The proposed condition gate is selected using validation evidence only. No neural result has been reproduced here.

## Earlier integrated workflow

| Source | Idea considered | Treatment in the earlier `fleetpm` experiment |
| --- | --- | --- |
| Yildirim Ozupak. *A hybrid deep learning framework for reliable and interpretable remaining life estimation in turbofan Engines: Focusing on Robustness, Generalization, and Sensor-Level interpretability.* Mechanical Systems and Signal Processing 258 (2026), 114642. [DOI](https://doi.org/10.1016/j.ymssp.2026.114642) | Temporal features, repeated-run reliability and sensor explanations | Compact causal CNN–GRU with explicit evaluation boundaries |
| Dongdong Tang. *A deep learning framework for remaining useful life prediction of turbofan engines with partial sensor failure.* PLOS One 21(4) (2026), e0347312. [DOI](https://doi.org/10.1371/journal.pone.0347312) | Task-aware repair and downstream RUL feedback | Compact causal repair adaptation; not the full GAN |
| Jorge Paredes, Danilo Chavez, Ramiro Isa-Jara and Diego Vargas. *Trans-D3: A Novel Hybrid Transformer-Based Actor-Critic Approach for Remaining Useful Life Prediction.* Sensors 26 (2026), 2949. [DOI](https://doi.org/10.3390/s26102949) | Attention and controlled residual adjustment | Bounded supervised correction from development data; not a TD3 reproduction |

Combining modules does not establish an additive benefit. The earlier hybrid is preserved; the selected notebook is a separate paper-based workflow. Neither neural architecture has been trained in the documented local validation snapshot.

## Dataset and implementation sources

- [NASA C-MAPSS dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
- [NASA damage-propagation reference](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf).
- [PyTorch installation](https://pytorch.org/get-started/locally/) and [mixed precision](https://docs.pytorch.org/docs/2.14/amp.html).
- [SHAP KernelExplainer](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html).
- [LIME API](https://lime-ml.readthedocs.io/en/latest/lime.html).

Implementation, benchmark evaluation and field performance are different evidence. See [VALIDATION.md](../VALIDATION.md) for what was executed.
