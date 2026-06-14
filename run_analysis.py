#!/usr/bin/env python3
"""
run_analysis.py -- Ready-to-run ODS experiment analysis
========================================================
Experiment conditions:
  - Catalyst mass : 5 mg  (m_g = 0.005 g)
  - Fuel volume   : 2 mL  (V_L = 0.002 L)
  - Initial S conc: 250 ppm
  - 7 catalysts tested

Data file: data/ODS_removal_data.csv
  - Column 'time' : reaction time in minutes
  - Columns Cat1..Cat7 : sulfur removal fraction (0 to 1)

Outputs (saved to results/ folder):
  - results/ODS_kinetics.xlsx   : all model parameters
  - results/ODS_plot_<Cat>.png  : kinetic plots per catalyst
"""

import pandas as pd
from pathlib import Path
from kinetics import analyze_dataframe, save_results, plot_results

# -------------------------------------------------------
# Experiment parameters
# -------------------------------------------------------
V_L   = 0.002   # 2 mL solution volume in litres
m_g   = 0.005   # 5 mg catalyst mass in grams
C0_ppm = 250.0  # initial sulfur concentration (ppm)

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
data_path = Path("data/ODS_removal_data.csv")
df = pd.read_csv(data_path)

print("Loaded data:")
print(df.to_string(index=False))
print()

# -------------------------------------------------------
# Run all kinetic models
# Weber-Morris and Elovich are enabled via V_L and m_g
# -------------------------------------------------------
results_df, raw_results = analyze_dataframe(
    df,
    time_col="time",
    conc_basis="ppm",   # removal fractions auto-detected; C0 = 100 (arbitrary)
    V_L=V_L,
    m_g=m_g,
)

# -------------------------------------------------------
# Print summary to console
# -------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
print("=" * 90)
print("  KINETIC ANALYSIS RESULTS  --  ODS Experiment  (250 ppm S, 5 mg cat, 2 mL fuel)")
print("=" * 90)
print(results_df.to_string())
print("=" * 90)

# -------------------------------------------------------
# Save results
# -------------------------------------------------------
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

save_results(results_df, out_dir / "ODS_kinetics.xlsx")
plot_results(raw_results, out_dir / "ODS_plot.png")

print()
print("All outputs saved to results/")
print("  - ODS_kinetics.xlsx")
for cat in raw_results:
    print("  - ODS_plot_{}.png".format(cat))
