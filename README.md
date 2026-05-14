# Chemical Kinetics Analyzer
### تحلیل سینتیک شیمیایی — سولفورزدایی و حذف آلاینده

A Python tool for fitting pollutant removal data to kinetic models and plotting the results.

---

## Features

- Accepts experimental data: **C₀** (initial), **Cₑ** (final), **t** (time)
- Automatically calculates **removal efficiency** (%)
- Fits data to **4 kinetic models**:
  | Model | Linearized Form | k Unit |
  |---|---|---|
  | Zero Order | Cₑ vs t | mg/L·min |
  | First Order | ln(Cₑ/C₀) vs t | min⁻¹ |
  | Second Order | 1/Cₑ vs t | L/mg·min |
  | Pseudo First Order | ln(Cₑ) vs t | min⁻¹ |
- Compares **R²** values and identifies the best-fit model
- Generates two plots:
  - Removal % vs Time
  - All model fits + linearized sub-plots

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/kinetics-analyzer.git
cd kinetics-analyzer
pip install -r requirements.txt
```

---

## Usage

```bash
python kinetics.py
```

You will be asked to either:
1. Use the built-in example dataset (desulfurization experiment)
2. Enter your own data interactively (C₀, Cₑ, t for each point)

### Example Session
```
  [1] Use example dataset
  [2] Enter my own data manually
  Choice (1/2): 2

  Number of data points: 4
  Initial concentration C0: 500

  Point 1 →  t [s], Ce [mg/L]: 5, 380
  Point 2 →  t [s], Ce [mg/L]: 10, 280
  Point 3 →  t [s], Ce [mg/L]: 20, 170
  Point 4 →  t [s], Ce [mg/L]: 30, 100
```

---

## Output

```
╭─────────┬───────────┬───────────┬───────────╮
│ t (min) │ C0 (mg/L) │ Ce (mg/L) │ Removal % │
├─────────┼───────────┼───────────┼───────────┤
│    5.0  │   100.00  │   80.00   │   20.0%   │
│   10.0  │   100.00  │   62.00   │   38.0%   │
│   ...   │   ...     │   ...     │   ...     │
╰─────────┴───────────┴───────────┴───────────╯

╭──────────────────────┬──────────┬────────────┬────────╮
│ Model                │ k        │ Unit       │ R²     │
├──────────────────────┼──────────┼────────────┼────────┤
│ Zero Order           │ 0.61234  │ mg/L·min   │ 0.9134 │
│ First Order          │ 0.02891  │ min⁻¹      │ 0.9978 │  ← best
│ Second Order         │ 0.00421  │ L/mg·min   │ 0.9712 │
│ Pseudo First Order   │ 0.02891  │ min⁻¹      │ 0.9965 │
╰──────────────────────┴──────────┴────────────┴────────╯

  ✓ Best fit model: First Order  (R² = 0.9978)
```

---

## Project Structure

```
kinetics-analyzer/
├── kinetics.py        # main script
├── requirements.txt   # dependencies
└── README.md
```

---

## How it Works

1. **Removal %** = (C₀ − Cₑ) / C₀ × 100
2. Each model is **linearized** and a linear regression is applied
3. **R²** (coefficient of determination) is used to rank models
4. The best-fit model is highlighted

---

## License
MIT — free to use and modify.
