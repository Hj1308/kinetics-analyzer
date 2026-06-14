# CatalystAnalytics 🔬

**Integrated Catalyst & Reaction Analysis Suite**  
Author: Hoda Jafari | [github.com/Hj1308](https://github.com/Hj1308)

---

## Overview

`CatalystAnalytics` is an open-source Python toolkit for researchers in heterogeneous catalysis, water treatment, and nanomaterials characterisation. It provides:

| Module | Description |
|---|---|
| `convert_to_mmol_L()` | Universal unit converter: ppmS, ppm, mg/L, g/L, mmol/L, mol/L |
| `SampleInfo` | Structured dataclass for experimental metadata |
| `KineticsAnalyser` | Zero/first/second/pseudo-first order model fitting |
| `calc_conversion()` | Conversion X(%) profile over time |
| `calc_tof()` | Turnover Frequency (h⁻¹) |
| `calc_toc_removal()` | Total Organic Carbon removal (%) |
| `BETAnalyser` | BET surface area from N₂ physisorption |
| `TPlotAnalyser` | T-Plot (Harkins-Jura): micropore/mesopore/macropore % |

---

## Installation

```bash
git clone https://github.com/Hj1308/kinetics-analyzer.git
cd kinetics-analyzer
pip install -r requirements.txt
```

---

## Quick Start

### Desulfurization example (ppmS)

```python
from catalyst_analytics import SampleInfo, KineticsAnalyser, convert_to_mmol_L
import numpy as np

info = SampleInfo(
    sample_name         = "MoS2/Al2O3",
    process_type        = "desulfurization",
    catalyst_mass_g     = 0.05,
    solution_vol_L      = 0.050,
    c0_value            = 500.0,
    c0_unit             = "ppmS",       # auto-converts using MW_S = 32.06 g/mol
    active_sites_mmol_g = 0.32,         # from NH3-TPD or CO chemisorption
    notes               = "DBT in n-decane, 300 °C, 30 bar H2"
)

time_h = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
c_ppmS = np.array([500, 420, 350, 250, 160, 100, 45])
c_mmol = np.array([convert_to_mmol_L(v, "ppmS") for v in c_ppmS])

analyzer = KineticsAnalyser(time_h, c_mmol, info)
report   = analyzer.full_report()
print(f"Conversion X = {report['Conversion X (%)']:.1f}%")
print(f"Best model:   {report['Best Fit Model']}")
print(f"TOF =         {report['TOF (h⁻¹)']} h⁻¹")

analyzer.plot_kinetics(save_path="MoS2_kinetics.png")
```

### Water treatment (mg/L)

```python
info = SampleInfo(
    sample_name    = "TiO2-P25",
    process_type   = "photocatalysis",
    catalyst_mass_g = 0.1,
    solution_vol_L  = 0.1,
    c0_value        = 50.0,
    c0_unit         = "mg/L",
    mw_pollutant    = 180.0,           # g/mol
    active_sites_mmol_g = 0.15
)
```

### BET + T-Plot (pore distribution)

```python
from catalyst_analytics import BETAnalyser, TPlotAnalyser
import numpy as np

p_rel = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 0.70, 0.90, 0.99])
v_ads = np.array([85,  102,  115,  126,  135,  143,  150,  172,  200,  280,  520])

# BET surface area
bet     = BETAnalyser(p_rel, v_ads)
bet_res = bet.fit_bet()
print(bet_res)
# → {'S_BET_m2g': 442.02, 'C_BET': 110.92, 'R2': 0.99744, ...}

# T-Plot: pore distribution
tplot  = TPlotAnalyser(p_rel, v_ads, s_bet=bet_res["S_BET_m2g"], total_pore_volume=0.52)
report = tplot.full_tplot_report()
print(report)
# → {'V_micro_cm3g': 0.08, 'V_meso_cm3g': 0.44, 'Micropore_%': 15.4, 'Mesopore_%': 84.6, ...}

tplot.plot_tplot(save_path="tplot.png")
```

### TOC Removal

```python
from catalyst_analytics import calc_toc_removal

toc_removal = calc_toc_removal(toc0=85.0, toc_t=12.0)
print(f"TOC removal: {toc_removal}%")  # → 85.88%
```

---

## Supported Concentration Units

| Unit | Description | MW Required? |
|---|---|---|
| `mol/L` | Molar | No |
| `mmol/L` | Millimolar | No |
| `ppmS` | ppm Sulfur (auto MW_S = 32.06) | No |
| `ppm` | Parts per million (aqueous ≈ mg/L) | ✅ Yes |
| `mg/L` | Milligrams per litre | ✅ Yes |
| `g/L` | Grams per litre | ✅ Yes |

---

## Requirements

```
numpy
pandas
scipy
matplotlib
```

---

## Citation

If you use this tool in your research, please cite:

```
Jafari, H. (2026). CatalystAnalytics: Integrated Catalyst & Reaction Analysis Suite.
GitHub. https://github.com/Hj1308/kinetics-analyzer
```

---

## License
MIT — free to use, modify, and distribute.
