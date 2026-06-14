# kinetics-analyzer

A Python tool for fitting concentration-vs-time data to kinetic models used in
catalysis, adsorption, and oxidative desulfurization (ODS) research.

Supports **multiple samples simultaneously** and outputs a comparative results
table with rate constants, R² values, and half-lives for all models.

---

## Features

- **Multi-sample analysis** — one time column + N catalyst/sample columns
- **Six kinetic models** — Zero Order, First Order, Pseudo-First Order (Lagergren), Second Order, Weber-Morris, Elovich
- **Weber-Morris intraparticle diffusion** — rate constant kid, boundary layer intercept C, R²
- **Elovich chemisorption** — initial adsorption rate α, surface heterogeneity constant β, R²
- **qt auto-calculation** — pass V_L (litres) and m_g (grams) to enable diffusion models
- **Half-life (t½)** calculated for every standard kinetic model
- **Zero-guard** — safe handling of complete removal data (Ce → 0) via clipping with warnings
- **Auto-detect removal fraction** — columns with values 0–1 are automatically treated as removal fractions
- **Unit conversion** — supports ppm (mg/L), ppm-S (sulfur analyser output), or ppm of intact substrate (GC output)
- **Lagergren diagnostic** — reports ΔC0% to detect deviation from ideal pseudo-first order kinetics
- **Mass-balance validation** — cross-checks ppm-S vs ppm-substrate measurements
- **File I/O** — reads CSV / Excel, writes results to CSV / Excel
- **Auto-plots** — removal efficiency + 4 linearised fits + R² bar chart + t½ chart + Weber-Morris + Elovich per sample

---

## Installation

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

---

## Quick Start

### From a file (recommended)

Prepare your data as a CSV or Excel file with this structure.
Concentrations can be in **mg/L (ppm)** or **removal fraction (0–1)**:

| time | Cat-A | Cat-B | Cat-C |
|------|-------|-------|-------|
| 0    | 500   | 500   | 500   |
| 10   | 350   | 300   | 430   |
| 30   | 200   | 130   | 340   |
| 60   | 110   | 55    | 250   |
| 120  | 30    | 8     | 120   |

Then run:

```python
from kinetics import read_data, analyze_dataframe, save_results, plot_results

df = read_data("ODS_data.xlsx", time_col="time")

# Standard kinetic models only
results, raw = analyze_dataframe(df, time_col="time", conc_basis="ppm")

# With Weber-Morris and Elovich (provide V_L in litres and m_g in grams)
results, raw = analyze_dataframe(
    df,
    time_col="time",
    conc_basis="ppm",
    V_L=0.05,    # e.g. 50 mL solution
    m_g=0.01,    # e.g. 10 mg catalyst
)

save_results(results, "kinetics_results.xlsx")
plot_results(raw, "kinetics_plot.png")
print(results)
```

### Interactive CLI

```bash
python kinetics.py
```

Choose between:
- `[1]` Load from CSV or Excel file
- `[2]` Enter one sample manually
- `[3]` Run built-in demo (3-catalyst ODS example)

The CLI will also prompt whether to enable **Weber-Morris and Elovich** models
and ask for V_L and m_g if selected.

---

## Kinetic Models

### Standard Models

| Model | Linearised Form | Slope | t½ Formula |
|---|---|---|---|
| Zero Order | Ce = C0 − k0·t | −k0 | C0 / (2·k0) |
| First Order | ln(Ce/C0) = −k1·t | −k1 | ln(2) / k1 |
| Pseudo-First Order | ln(Ce) = −k_pfo·t + ln(C0) | −k_pfo | ln(2) / k_pfo |
| Second Order | 1/Ce = k2·t + 1/C0 | k2 | 1 / (k2·C0) |

### Diffusion / Surface Models

| Model | Linearised Form | Parameters | When to Use |
|---|---|---|---|
| Weber-Morris | qt = kid·√t + C | kid (mg/g/min⁰·⁵), C (mg/g) | Intraparticle diffusion is rate-limiting (porous catalysts) |
| Elovich | qt = (1/β)·ln(α·β) + (1/β)·ln(t) | α (mg/g/min), β (g/mg) | Chemisorption on heterogeneous surfaces |

**Weber-Morris intercept C interpretation:**
- `C = 0` → intraparticle diffusion is the sole rate-limiting step
- `C > 0` → film diffusion (boundary layer) also contributes
- `C < 0` → multi-stage diffusion (micropore + mesopore) or surface diffusion dominance

**Elovich β interpretation:**
- Larger β → more homogeneous surface (uniform adsorption energy)
- Smaller β → more heterogeneous surface (wider distribution of active site energies)

---

## qt Calculation

The adsorption capacity qt (mg/g) is required for Weber-Morris and Elovich:

```
qt = (C0 - Ce) × V_L / m_g
```

Where:
- `C0` = initial concentration (mg/L)
- `Ce` = residual concentration at time t (mg/L)
- `V_L` = solution volume (litres)
- `m_g` = catalyst/adsorbent mass (grams)

```python
from kinetics import compute_qt

qt = compute_qt(C0=250, Ce=ce_array, V_L=0.05, m_g=0.01)
```

---

## Pseudo-First Order vs First Order

Both models give the same k value from the slope. The difference is that in
Pseudo-First Order the intercept is left free, so `C0_fit = exp(intercept)` is
computed from the data. If `ΔC0% = |C0_fit − C0| / C0 × 100` is large (> 15%),
this signals deviation from ideal Lagergren kinetics and should be noted in
your analysis.

---

## Unit Conversion

The `conc_basis` parameter controls how concentrations are handled:

| conc_basis | Input unit | Conversion | Affected parameters |
|---|---|---|---|
| `"ppm"` | mg/L | None | k0 in mg/L/min, k2 in L/mg/min |
| `"sulfur"` | ppm-S | ppm / (32.06 × 1000) / n_sulfur | k0 in mol/L/min, k2 in L/mol/min |
| `"substrate"` | ppm molecule | ppm / (MW × 1000) | k0 in mol/L/min, k2 in L/mol/min |

> **Note:** k_app, R², and t½ are insensitive to the choice of conc_basis.
> Only k0 and k2 (which carry molar dimensions) change.

---

## Mass-Balance Validation

Cross-check your ppm-S measurement (XRF) against your ppm-substrate
measurement (GC) to catch unit errors early:

```python
from kinetics import validate_mass_balance

validate_mass_balance(
    ppm_S=500,
    ppm_substrate=2873,        # ppm DBT measured by GC
    molar_mass_substrate=184.26,
    n_sulfur=1
)
```

Expected output:
```
Mass-balance check:
  Measured ppm-S  = 500.00 mg/L
  Expected ppm-S  = 499.88 mg/L  (from 2873.00 ppm substrate, MW=184.26, n_S=1)
  Deviation       = 0.0%
  PASS (within 5% tolerance)
```

---

## Output Example

```
==========================================================================================
  KINETIC ANALYSIS  --  COMPARATIVE RESULTS
==========================================================================================
        C0          k0 (Zero)  R2 (Zero) ...  kid (W-M)  C_wm    R2 (W-M)  alpha    beta    R2 (Elovich)
Sample
Cat-A   500 mg/L    3.59       0.8371    ...  4.25       +22.01  0.9930    6.77     0.0522  0.9912
Cat-B   500 mg/L    3.57       0.7447    ...  4.32       +11.38  0.9970    3.94     0.0516  0.9873
```

---

## File Structure

```
kinetics-analyzer/
├── kinetics.py        # Main module — all functions
└── README.md
```

> **Note:** Data files are not stored in this repository.
> Keep your experimental data locally and pass the file path to `read_data()`.

---

## Common Use Cases

**Oxidative Desulfurization (ODS)**
```python
# Data measured as ppm-S by sulfur analyser (e.g. XRF, ANTEK)
results, raw = analyze_dataframe(df, conc_basis="sulfur", n_sulfur=1,
                                  V_L=0.002, m_g=0.005)
```

**Adsorption / Water Treatment**
```python
# Data measured as ppm of contaminant (e.g. methylene blue, heavy metals)
results, raw = analyze_dataframe(df, conc_basis="ppm",
                                  V_L=0.05, m_g=0.01)
```

**GC-measured substrate (e.g. DBT, BT, thiophene)**
```python
results, raw = analyze_dataframe(df, conc_basis="substrate",
                                  molar_mass=184.26,  # DBT
                                  n_sulfur=1,
                                  V_L=0.002, m_g=0.005)
```

---

## License

MIT License

---

## Author

Hj1308 / Hoda Jaafari
