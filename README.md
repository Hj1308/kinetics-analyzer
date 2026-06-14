# EISForge 🔬⚡

> **Open-source Electrochemical Impedance Spectroscopy (EIS) and Cyclic Voltammetry (CV) analysis toolkit with Physics-Informed Machine Learning.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-alpha-orange)

EISForge combines classical electrochemical analysis (CNLS fitting, Kramers–Kronig validation, ECSA, Koutecky–Levich) with a Physics-Informed Transformer architecture (**EIS-GPT**, in development) for automatic circuit interpretation.

---

## Author & Citation

**Hoda Jafari** — 📧 hoda.jaafari@gmail.com — 🔗 https://github.com/Hj1308
First published: May 2026

If you use EISForge in your research, please cite:

```bibtex
@software{jafari2026eisforge,
  author    = {Jafari, Hoda},
  title     = {EISForge: Physics-Informed Transformer for Electrochemical Impedance Spectroscopy},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Hj1308/EISforge-},
  note      = {Open-source EIS analysis with ML}
}
```

---

## Features

### ✅ Available now

| Module | Description |
|---|---|
| **CNLS circuit fitting** | Complex Non-Linear Least Squares with bounds, outlier detection, and IUPAC modulus weighting (`eisforge/core/fitter.py`) |
| **Kramers–Kronig validation** | linKK (via `impedance.py`) with a custom Voigt-circuit fallback (`eisforge/core/validators.py`) |
| **CV / LSV analysis** | Peak detection, onset potentials, automatic interpretation (`eisforge/analysis/`) |
| **ECSA calculator** | H-UPD, CO stripping, and double-layer capacitance (Cdl) methods (`eisforge/analysis/ecsa_calculator.py`) |
| **Koutecky–Levich analysis** | Rotating-disk electrode kinetics (`eisforge/analysis/koutecky_levich.py`) |
| **EIS–CV correlation** | Cross-technique analysis (`eisforge/analysis/eis_cv_correlator.py`) |
| **Batch processing** | Multi-file analysis with parallel processing (`eisforge/analysis/batch_analyzer.py`) |
| **Uncertainty quantification** | Monte-Carlo dropout and active-learning query strategies (`eisforge/ml/uncertainty/`) |
| **EIS-GPT architecture** | Transformer with spectrum tokenizer and physics-informed loss (Kramers–Kronig, causality, passivity terms) — architecture implemented, pretrained weights coming soon (`eisforge/ml/eis_gpt/`) |
| **Streamlit web app** | Interactive UI for the full pipeline (`app.py`) |

### 🚧 Planned (see Roadmap)

DRT analysis · BioLogic (.mpt/.mpr) and Zahner (.ism) parsers · pretrained EIS-GPT foundation model · interactive Nyquist/Bode visualization module · band-edge (Ecb/Evb) calculator.

---

## How EISForge compares

| Feature | ZView | EC-Lab | EISForge |
| --- | --- | --- | --- |
| Circuit fitting (CNLS) | ✅ | ✅ | ✅ |
| Kramers–Kronig validation | ✅ | ✅ | ✅ |
| CV/LSV + ECSA analysis | ❌ | partial | ✅ |
| ML-assisted interpretation | ❌ | ❌ | ✅ (in development) |
| Physics-informed deep learning | ❌ | ❌ | 🚧 architecture ready |
| DRT analysis | ❌ | partial | 🚧 planned |
| Free & open source | ❌ | ❌ | ✅ MIT |

---

## Core innovation: EIS-GPT 🧠

EISForge introduces a **Physics-Informed Transformer** for EIS analysis. Instead of requiring the user to pre-select an equivalent circuit, the model is designed to:

1. **Tokenize** EIS spectra (each frequency point = one token)
2. **Enforce** Kramers–Kronig relations as physics constraints in the loss
3. **Predict** circuit topology with uncertainty estimates
4. **Estimate** initial parameters for CNLS refinement

```
L_total = L_reconstruction
        + λ₁ × L_kramers_kronig
        + λ₂ × L_causality
        + λ₃ × L_passivity
```

> **Status:** the tokenizer, transformer, and physics-informed loss are implemented and unit-tested. Training on a large synthetic dataset and releasing pretrained weights is the next milestone — see the Roadmap.

---

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/Hj1308/EISforge-.git
cd EISforge-
pip install -r requirements.txt
pip install -e .
```

---

## Quick start

### Python API

```python
from eisforge.core.analyzer import EisAnalyzer

ana = EisAnalyzer()

# Load data (Gamry .DTA, Autolab .idf, or generic .csv/.txt)
dataset = ana.load("my_data.DTA")

# Validate data quality with Kramers-Kronig
kk = ana.validate_kk(dataset)

# Fit an equivalent circuit (impedance.py syntax)
result = ana.fit(dataset, circuit="R0-p(R1,CPE1)-Wo1",
                 initial_guess=[15, 300, 1e-5, 0.9, 50, 1])
print(result.parameter_table())
```

### Web interface

```bash
streamlit run app.py
```

---

## Supported instruments

| Vendor | Format | Status |
|---|---|---|
| Gamry Instruments | `.DTA` | ✅ |
| Metrohm Autolab | `.idf` (CV + EIS, multi-scan) | ✅ |
| Generic | `.csv` / `.txt` (auto-detection) | ✅ |
| BioLogic | `.mpt` / `.mpr` | 🚧 planned |
| Zahner | `.ism` | 🚧 planned |

---

## Project structure

```
eisforge/
├── core/          # CNLS fitting engine, K-K validation, preprocessing
├── parsers/       # Gamry, Autolab, generic CSV importers
├── analysis/      # CV/LSV, ECSA, Koutecky-Levich, batch, EIS-CV correlation
├── ml/
│   ├── eis_gpt/   # Physics-Informed Transformer (tokenizer, model, loss)
│   └── uncertainty/  # MC dropout, active learning
├── knowledge/     # Literature-driven parameter guessing
├── standards/     # Reference data for carbon materials
└── utils/         # Experimental conditions, file helpers
app.py             # Streamlit web application
tests/             # pytest test suite
```

---

## Roadmap

- [x] Core EIS analysis engine (CNLS + Kramers–Kronig)
- [x] Gamry / Autolab / CSV parsers
- [x] CV, LSV, ECSA, and Koutecky–Levich analysis
- [x] EIS-GPT architecture with physics-informed loss
- [x] Streamlit web interface
- [ ] BioLogic and Zahner parsers
- [ ] DRT analysis (Tikhonov regularization)
- [ ] Train EIS-GPT on synthetic spectra → release pretrained weights
- [ ] Band-edge (Ecb/Evb) calculator for semiconductor photocatalysts
- [ ] Zenodo DOI + first tagged release

---

## Contributing

Bug reports, feature requests, and pull requests are welcome — please open an [issue](https://github.com/Hj1308/EISforge-/issues).

## License

MIT License — Copyright (c) 2026 Hoda Jafari.
Free to use in research and commercial applications. **Please cite this work if you use it in your publications.**
