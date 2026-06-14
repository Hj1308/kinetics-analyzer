#!/usr/bin/env python3
"""
kinetics.py  v2.1 -- Multi-sample Chemical Kinetics Analyzer
=============================================================
Fits concentration-vs-time data to Zero Order, First Order,
Pseudo-First Order, and Second Order kinetic models for N samples
simultaneously.

Usage
-----
    python kinetics.py

    Or import as a module:
        from kinetics import analyze_dataframe, read_data, save_results

Author : Hj1308 / Hoda Jaafari
License: MIT
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
MW_S = 32.06          # g/mol -- atomic mass of sulfur


# -------------------------------------------------------
# Unit conversion
# -------------------------------------------------------

def to_molar(C_ppm, conc_basis="sulfur", molar_mass=None, n_sulfur=1):
    """
    Convert concentration array from ppm (mg/L) to mol/L.

    Parameters
    ----------
    C_ppm       : array-like
        Concentration values in mg/L (ppm).
    conc_basis  : str
        "sulfur"    -- C_ppm represents ppm-S (XRF / sulfur analyser output).
                       mol(analyte)/L = ppm / (MW_S * 1000) / n_sulfur
        "substrate" -- C_ppm represents ppm of the intact molecule (e.g. GC output).
                       mol/L = ppm / (molar_mass * 1000)
    molar_mass  : float, optional
        Molecular weight of the substrate in g/mol.
        Required when conc_basis="substrate". Example: 184.26 for DBT.
    n_sulfur    : int
        Number of sulfur atoms per molecule (default 1).
        Used only when conc_basis="sulfur".

    Returns
    -------
    numpy.ndarray
        Concentration in mol/L.

    Notes
    -----
    k_app, R2, and t_half are insensitive to the choice of conc_basis
    because it applies only a constant scale factor to Ce.
    Only k0 [mol/L/min] and k2 [L/mol/min] change with this choice.
    """
    C = np.asarray(C_ppm, dtype=float)
    if conc_basis == "sulfur":
        return C / (MW_S * 1000.0) / n_sulfur
    elif conc_basis == "substrate":
        if molar_mass is None:
            raise ValueError(
                "molar_mass is required when conc_basis='substrate'. "
                "Example: molar_mass=184.26 for dibenzothiophene (DBT)."
            )
        return C / (molar_mass * 1000.0)
    else:
        raise ValueError("conc_basis must be 'sulfur' or 'substrate'.")


# -------------------------------------------------------
# Core fitting function -- single sample
# -------------------------------------------------------

def fit_all_models(t, Ce_input, C0_override=None,
                   conc_basis="ppm", molar_mass=None,
                   n_sulfur=1, eps=1e-12):
    """
    Fit Zero, First, Pseudo-First, and Second Order kinetic models
    to a single concentration-vs-time dataset using linear regression.

    Parameters
    ----------
    t            : array-like
        Time points (any consistent unit, e.g. minutes).
    Ce_input     : array-like
        Equilibrium / residual concentrations (ppm or mol/L).
    C0_override  : float, optional
        Override the initial concentration C0. If None, Ce_input[0] is used.
    conc_basis   : str
        "ppm"       -- keep k0/k2 in mg/L-based units (no conversion).
        "sulfur"    -- convert ppm-S to mol/L before regression.
        "substrate" -- convert ppm(molecule) to mol/L before regression.
    molar_mass   : float, optional
        Molecular weight (g/mol). Required when conc_basis="substrate".
    n_sulfur     : int
        Number of S atoms per molecule (used when conc_basis="sulfur").
    eps          : float
        Floor value applied via np.clip to prevent log(0) and 1/0 errors.

    Returns
    -------
    dict
        Keys: k0, R2_0, t_half_0, k1, R2_1, t_half_1,
              k_pfo, R2_pfo, t_half_pfo, C0_fit, delta_C0_pct,
              k2, R2_2, t_half_2, C0, conc_unit, best_model,
              _t, _Ce, _Ce_raw, _s0, _s1, _s2, _s3 (private, for plotting).

    Model Summary
    -------------
    Zero Order        : Ce  = C0 - k0 * t         -->  slope = -k0
    First Order       : ln(Ce/C0) = -k1 * t        -->  slope = -k1
    Pseudo-First Order: ln(Ce) = -k_pfo*t + ln(C0) -->  intercept free (Lagergren)
    Second Order      : 1/Ce = k2 * t + 1/C0       -->  slope = k2

    Half-life Formulas
    ------------------
    Zero Order        : t_half = C0 / (2 * k0)
    First / PFO       : t_half = ln(2) / k
    Second Order      : t_half = 1 / (k2 * C0)

    Diagnostic -- Pseudo-First Order
    ---------------------------------
    When the PFO intercept is left free, C0_fit = exp(intercept).
    If delta_C0_pct = |C0_fit - C0| / C0 * 100 is large (>15%),
    this indicates deviation from ideal Lagergren kinetics.
    """
    t      = np.asarray(t,       dtype=float)
    Ce_raw = np.asarray(Ce_input, dtype=float)

    # -- Zero-guard: clip Ce <= 0 to avoid log(0) and 1/0 --
    zero_pts = int(np.sum(Ce_raw <= 0))
    if zero_pts > 0:
        warnings.warn(
            "{} data point(s) with Ce <= 0 were clipped to eps={:.2e}. "
            "These may correspond to complete contaminant removal (>99%). "
            "Verify detection limits or exclude those points from regression.".format(
                zero_pts, eps),
            UserWarning, stacklevel=3
        )
    Ce_safe = np.clip(Ce_raw, eps, None)

    # -- Optional unit conversion --
    if conc_basis in ("sulfur", "substrate"):
        Ce        = to_molar(Ce_safe, conc_basis, molar_mass, n_sulfur)
        conc_unit = "mol/L"
    else:
        Ce        = Ce_safe
        conc_unit = "mg/L"

    C0 = float(C0_override) if C0_override is not None else Ce[0]

    # -- Linear regressions --

    # Zero Order: Ce vs t
    s0  = linregress(t, Ce)
    k0  = -s0.slope
    R2_0 = s0.rvalue ** 2

    # First Order: ln(Ce/C0) vs t
    s1  = linregress(t, np.log(Ce / C0))
    k1  = -s1.slope
    R2_1 = s1.rvalue ** 2

    # Pseudo-First Order: ln(Ce) vs t  (free intercept -- Lagergren test)
    s2         = linregress(t, np.log(Ce))
    k_pfo      = -s2.slope
    C0_fit     = np.exp(s2.intercept)
    R2_pfo     = s2.rvalue ** 2
    delta_C0_pct = abs(C0_fit - C0) / C0 * 100

    # Second Order: 1/Ce vs t
    s3  = linregress(t, 1.0 / Ce)
    k2  = s3.slope
    R2_2 = s3.rvalue ** 2

    # -- Half-lives --
    t_half_0   = C0 / (2.0 * k0)   if k0   > 0 else float("nan")
    t_half_1   = np.log(2) / k1    if k1   > 0 else float("nan")
    t_half_pfo = np.log(2) / k_pfo if k_pfo > 0 else float("nan")
    t_half_2   = 1.0 / (k2 * C0)   if k2   > 0 else float("nan")

    # -- Best model by R2 --
    r2_map = {
        "Zero Order":         R2_0,
        "First Order":        R2_1,
        "Pseudo-First Order": R2_pfo,
        "Second Order":       R2_2,
    }
    best_model = max(r2_map, key=r2_map.get)

    return {
        # Zero Order
        "k0":            k0,
        "R2_0":          R2_0,
        "t_half_0":      t_half_0,
        # First Order
        "k1":            k1,
        "R2_1":          R2_1,
        "t_half_1":      t_half_1,
        # Pseudo-First Order
        "k_pfo":         k_pfo,
        "R2_pfo":        R2_pfo,
        "t_half_pfo":    t_half_pfo,
        "C0_fit":        C0_fit,
        "delta_C0_pct":  delta_C0_pct,
        # Second Order
        "k2":            k2,
        "R2_2":          R2_2,
        "t_half_2":      t_half_2,
        # Metadata
        "C0":            C0,
        "conc_unit":     conc_unit,
        "best_model":    best_model,
        # Private -- used by plot_results()
        "_t":    t,
        "_Ce":   Ce,
        "_Ce_raw": Ce_raw,
        "_s0":   s0,
        "_s1":   s1,
        "_s2":   s2,
        "_s3":   s3,
    }


# -------------------------------------------------------
# Multi-sample DataFrame analyser
# -------------------------------------------------------

def analyze_dataframe(df, time_col="time",
                      conc_basis="ppm", molar_mass=None, n_sulfur=1):
    """
    Run fit_all_models on every sample column in a DataFrame.

    Parameters
    ----------
    df          : pandas.DataFrame
        One time column + N concentration columns (one per sample/catalyst).
    time_col    : str
        Name of the time column. Default "time".
    conc_basis  : str
        Passed to fit_all_models. Options: "ppm", "sulfur", "substrate".
    molar_mass  : float, optional
        Passed to fit_all_models (required when conc_basis="substrate").
    n_sulfur    : int
        Passed to fit_all_models.

    Returns
    -------
    results_df  : pandas.DataFrame
        Comparative summary table with one row per sample.
    raw_results : dict
        Full result dicts keyed by sample name, e.g. for plot_results().

    Example
    -------
    >>> df = pd.read_excel("ODS_data.xlsx")
    >>> results, raw = analyze_dataframe(df, time_col="time",
    ...                                  conc_basis="sulfur", n_sulfur=1)
    >>> print(results)
    """
    if time_col not in df.columns:
        raise ValueError(
            "Time column '{}' not found. Available columns: {}".format(
                time_col, list(df.columns))
        )

    t           = df[time_col].values
    sample_cols = [c for c in df.columns if c != time_col]
    raw_results = {}
    rows        = []

    for col in sample_cols:
        res = fit_all_models(
            t, df[col].values,
            conc_basis=conc_basis,
            molar_mass=molar_mass,
            n_sulfur=n_sulfur
        )
        raw_results[col] = res

        unit  = res["conc_unit"]
        k0_u  = "{}/min".format(unit)
        k2_u  = "L/{}/min".format(unit)

        def fmt(v, decimals=2):
            return "{:.{}f}".format(v, decimals) if not np.isnan(v) else "N/A"

        rows.append({
            "Sample":           col,
            "C0":               "{:.4g} {}".format(res["C0"], unit),
            # Zero Order
            "k0 (Zero)":        "{:.4g} {}".format(res["k0"], k0_u),
            "R2 (Zero)":        "{:.4f}".format(res["R2_0"]),
            "t_half_0 (min)":   fmt(res["t_half_0"]),
            # First Order
            "k1 (1st)":         "{:.4g} /min".format(res["k1"]),
            "R2 (1st)":         "{:.4f}".format(res["R2_1"]),
            "t_half_1 (min)":   fmt(res["t_half_1"]),
            # Pseudo-First Order
            "k_pfo (PFO)":      "{:.4g} /min".format(res["k_pfo"]),
            "R2 (PFO)":         "{:.4f}".format(res["R2_pfo"]),
            "t_half_pfo (min)": fmt(res["t_half_pfo"]),
            "C0_fit (PFO)":     "{:.4g}".format(res["C0_fit"]),
            "dC0% (PFO)":       "{:.1f}%".format(res["delta_C0_pct"]),
            # Second Order
            "k2 (2nd)":         "{:.4g} {}".format(res["k2"], k2_u),
            "R2 (2nd)":         "{:.4f}".format(res["R2_2"]),
            "t_half_2 (min)":   fmt(res["t_half_2"]),
            # Best fit
            "Best Model":       res["best_model"],
        })

    results_df = pd.DataFrame(rows).set_index("Sample")
    return results_df, raw_results


# -------------------------------------------------------
# File I/O
# -------------------------------------------------------

def read_data(filepath, time_col="time"):
    """
    Load a CSV or Excel file into a DataFrame.

    All columns except time_col are treated as sample columns.

    Parameters
    ----------
    filepath : str or Path
        Path to a .csv, .xlsx, or .xls file.
    time_col : str
        Name of the time column. Default "time".

    Returns
    -------
    pandas.DataFrame
    """
    p = Path(filepath)
    if p.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError("Unsupported file format: {}. Use .csv or .xlsx".format(p.suffix))
    if time_col not in df.columns:
        raise ValueError(
            "Column '{}' not found in {}. Available: {}".format(
                time_col, p.name, list(df.columns))
        )
    return df


def save_results(results_df, out_path):
    """
    Save the comparative results table to a CSV or Excel file.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Output of analyze_dataframe().
    out_path   : str or Path
        Destination file path (.csv or .xlsx).
    """
    p = Path(out_path)
    if p.suffix in (".xlsx", ".xls"):
        results_df.to_excel(p)
    else:
        results_df.to_csv(p)
    print("Results saved: {}".format(p.resolve()))


# -------------------------------------------------------
# Plotting
# -------------------------------------------------------

COLORS = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B"
]


def _fit_line(slope, intercept, x):
    return slope * x + intercept


def plot_results(raw_results, out_path=None):
    """
    Generate a multi-panel figure for each sample showing:
      - Row 0 : Removal efficiency (%) vs time
      - Row 1 : Four linearised kinetic plots (Zero, 1st, PFO, 2nd order)
      - Row 2 : R2 bar chart  +  half-life (t_half) horizontal bar chart

    Parameters
    ----------
    raw_results : dict
        Output from analyze_dataframe() or {name: fit_all_models(...)}.
    out_path    : str or Path, optional
        If provided, figures are saved as PNG files using the pattern:
        <stem>_<sample_name>.png
        If None, figures are displayed interactively.
    """
    for i, (name, res) in enumerate(raw_results.items()):
        t      = res["_t"]
        Ce     = res["_Ce"]
        C0     = res["C0"]
        unit   = res["conc_unit"]
        color  = COLORS[i % len(COLORS)]

        removal = (C0 - Ce) / C0 * 100.0
        t_fit   = np.linspace(t[0], t[-1], 200)

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle("Kinetic Analysis -- {}".format(name),
                     fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.42)

        # -- Row 0: Removal efficiency --
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(t, removal, "o-", color=color, lw=2, ms=6)
        ax0.set_xlabel("Time (min)")
        ax0.set_ylabel("Removal (%)")
        ax0.set_title("Removal Efficiency vs Time")
        ax0.set_ylim(0, 108)
        ax0.grid(True, alpha=0.3)

        # -- Row 1: Linearised fits --
        s0, s1, s2, s3 = res["_s0"], res["_s1"], res["_s2"], res["_s3"]

        panels = [
            (gs[1, 0],
             Ce,
             "Ce ({})".format(unit),
             s0.slope, s0.intercept,
             "Zero Order\nk={:.3g}  R2={:.4f}".format(res["k0"], res["R2_0"])),

            (gs[1, 1],
             np.log(Ce / C0),
             "ln(Ce / C0)",
             s1.slope, s1.intercept,
             "First Order\nk={:.3g}  R2={:.4f}".format(res["k1"], res["R2_1"])),

            (gs[1, 2],
             np.log(Ce),
             "ln(Ce)",
             s2.slope, s2.intercept,
             "Pseudo-First Order\nk={:.3g}  R2={:.4f}\ndC0={:.1f}%".format(
                 res["k_pfo"], res["R2_pfo"], res["delta_C0_pct"])),

            (gs[1, 3],
             1.0 / Ce,
             "1/Ce (1/{})".format(unit),
             s3.slope, s3.intercept,
             "Second Order\nk={:.3g}  R2={:.4f}".format(res["k2"], res["R2_2"])),
        ]

        for (gs_spec, y, ylabel, slope, intercept, title) in panels:
            ax = fig.add_subplot(gs_spec)
            ax.plot(t, y, "o", color=color, ms=6)
            ax.plot(t_fit, _fit_line(slope, intercept, t_fit), "--k", lw=1.5)
            ax.set_xlabel("t (min)")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=8.5)
            ax.grid(True, alpha=0.3)

        # -- Row 2a: R2 bar chart --
        ax5 = fig.add_subplot(gs[2, :2])
        labels = ["Zero", "First", "PFO", "Second"]
        r2vals = [res["R2_0"], res["R2_1"], res["R2_pfo"], res["R2_2"]]
        best_label = res["best_model"].split()[0]
        bar_colors = [
            "#FFD700" if lbl == best_label else color for lbl in labels
        ]
        bars = ax5.bar(labels, r2vals, color=bar_colors,
                       edgecolor="white", linewidth=0.8)
        ax5.set_ylim(0, 1.10)
        ax5.set_ylabel("R2")
        ax5.set_title("Model Comparison (R2)  [gold = best fit]")
        ax5.axhline(0.99, ls=":", color="gray", lw=1)
        for bar, v in zip(bars, r2vals):
            ax5.text(bar.get_x() + bar.get_width() / 2.0,
                     v + 0.012, "{:.4f}".format(v),
                     ha="center", va="bottom", fontsize=8)
        ax5.grid(True, alpha=0.3, axis="y")

        # -- Row 2b: Half-life horizontal bar --
        ax6 = fig.add_subplot(gs[2, 2:])
        t_half_map = {
            "Zero":   res["t_half_0"],
            "First":  res["t_half_1"],
            "PFO":    res["t_half_pfo"],
            "Second": res["t_half_2"],
        }
        valid = {k: v for k, v in t_half_map.items() if not np.isnan(v)}
        if valid:
            ax6.barh(list(valid.keys()), list(valid.values()),
                     color=color, alpha=0.85, edgecolor="white")
            for idx, (k, v) in enumerate(valid.items()):
                ax6.text(v + max(valid.values()) * 0.01, idx,
                         "{:.1f} min".format(v),
                         va="center", fontsize=8)
            ax6.set_xlabel("t_half (min)")
            ax6.set_title("Half-Life (t1/2) per Model")
            ax6.grid(True, alpha=0.3, axis="x")
        else:
            ax6.text(0.5, 0.5,
                     "t1/2 undefined\n(all k <= 0)",
                     ha="center", va="center",
                     transform=ax6.transAxes,
                     fontsize=10, color="gray")
            ax6.set_title("Half-Life (t1/2) per Model")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if out_path:
            stem  = Path(out_path).stem
            fname = "{}_{}.png".format(stem, name.replace(" ", "_"))
            fpath = Path(out_path).parent / fname
            plt.savefig(str(fpath), dpi=150, bbox_inches="tight")
            print("Plot saved: {}".format(fpath.resolve()))
        else:
            plt.show()

        plt.close(fig)


# -------------------------------------------------------
# Mass-balance validation
# -------------------------------------------------------

def validate_mass_balance(ppm_S, ppm_substrate,
                           molar_mass_substrate,
                           n_sulfur=1,
                           tol_pct=5.0):
    """
    Verify that ppm-S (from sulfur analyser) is consistent with
    ppm of the intact substrate molecule (from GC).

    Expected relationship:
        ppm_S = ppm_substrate * (MW_S * n_sulfur / MW_substrate)

    Parameters
    ----------
    ppm_S                : float
        Measured total sulfur concentration (mg-S/L).
    ppm_substrate        : float
        Measured substrate concentration (mg/L), e.g. ppm DBT from GC.
    molar_mass_substrate : float
        Molecular weight of substrate (g/mol). DBT = 184.26.
    n_sulfur             : int
        Number of sulfur atoms per substrate molecule (default 1).
    tol_pct              : float
        Acceptable deviation threshold in percent (default 5%).

    Example
    -------
    >>> validate_mass_balance(500, 2873, molar_mass_substrate=184.26, n_sulfur=1)
    # Checks if 500 ppm-S is consistent with 2873 ppm-DBT
    """
    expected_ppm_S = ppm_substrate * (MW_S * n_sulfur / molar_mass_substrate)
    deviation = abs(ppm_S - expected_ppm_S) / expected_ppm_S * 100.0

    print("Mass-balance check:")
    print("  Measured ppm-S  = {:.2f} mg/L".format(ppm_S))
    print("  Expected ppm-S  = {:.2f} mg/L  "
          "(from {:.2f} ppm substrate, MW={:.2f}, n_S={})".format(
              expected_ppm_S, ppm_substrate, molar_mass_substrate, n_sulfur))
    print("  Deviation       = {:.1f}%".format(deviation))
    if deviation <= tol_pct:
        print("  PASS (within {:.0f}% tolerance)".format(tol_pct))
    else:
        print("  WARNING: deviation > {:.0f}% -- "
              "check measurement units or analytical method.".format(tol_pct))


# -------------------------------------------------------
# Console display
# -------------------------------------------------------

def print_results(results_df):
    """Print the comparative results table to the console."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    sep = "=" * 90
    print("\n" + sep)
    print("  KINETIC ANALYSIS  --  COMPARATIVE RESULTS")
    print(sep)
    print(results_df.to_string())
    print(sep + "\n")


# -------------------------------------------------------
# Interactive entry-point
# -------------------------------------------------------

def _manual_entry():
    """Prompt user to enter a single sample interactively."""
    print("\n--- Manual Data Entry ---")
    C0 = float(input("  Initial concentration C0 (mg/L): "))
    n  = int(input("  Number of data points: "))
    ts, Ces = [], []
    for i in range(1, n + 1):
        raw = input("  Point {:2d}  ->  t [min], Ce [mg/L]: ".format(i))
        parts = raw.split(",")
        ts.append(float(parts[0]))
        Ces.append(float(parts[1]))
    return np.array(ts), np.array(Ces), C0


def _choose_conc_basis():
    """Prompt user to choose a concentration basis for unit conversion."""
    print("\nConcentration basis:")
    print("  [a]  ppm (mg/L)   -- k0/k2 kept in mg/L units  [default]")
    print("  [b]  ppm-S        -- convert to mol/L via MW(S) = 32.06 g/mol")
    print("  [c]  ppm substrate -- convert to mol/L via substrate MW")

    choice = input("Choice (a/b/c) [default a]: ").strip().lower()

    conc_basis = "ppm"
    molar_mass = None
    n_sulfur   = 1

    if choice == "b":
        conc_basis = "sulfur"
        ns = input("  n_sulfur -- S atoms per molecule [default 1]: ").strip()
        n_sulfur = int(ns) if ns else 1

    elif choice == "c":
        conc_basis = "substrate"
        molar_mass = float(
            input("  Molar mass of substrate (g/mol) [e.g. 184.26 for DBT]: ")
        )
        ns = input("  n_sulfur [default 1]: ").strip()
        n_sulfur = int(ns) if ns else 1

    return conc_basis, molar_mass, n_sulfur


def main():
    print("\n+--------------------------------------+")
    print("|  Chemical Kinetics Analyzer  v2.1   |")
    print("|  Zero / 1st / PFO / 2nd Order       |")
    print("+--------------------------------------+\n")

    print("Input mode:")
    print("  [1]  Load data from CSV or Excel file")
    print("  [2]  Enter one sample manually (interactive)")
    print("  [3]  Run built-in demo  (3-catalyst ODS example)\n")
    mode = input("Choice (1/2/3): ").strip()

    conc_basis, molar_mass, n_sulfur = _choose_conc_basis()

    out_prefix = input(
        "\nOutput file prefix (no extension, or press Enter to skip): "
    ).strip()

    # ---- Mode 1: load from file ----------------------------------------
    if mode == "1":
        fpath  = input("  File path (.csv or .xlsx): ").strip()
        tcol   = input("  Time column name [default 'time']: ").strip() or "time"
        df     = read_data(fpath, tcol)
        res_df, raw = analyze_dataframe(df, tcol, conc_basis, molar_mass, n_sulfur)
        print_results(res_df)
        if out_prefix:
            save_results(res_df, out_prefix + "_kinetics.xlsx")
            plot_results(raw, out_prefix + "_plot.png")

    # ---- Mode 2: manual single sample -----------------------------------
    elif mode == "2":
        t, Ce, C0 = _manual_entry()
        res = fit_all_models(t, Ce, C0_override=C0,
                             conc_basis=conc_basis,
                             molar_mass=molar_mass,
                             n_sulfur=n_sulfur)
        public_keys = [k for k in res if not k.startswith("_")]
        row = {k: res[k] for k in public_keys}
        df_row = pd.DataFrame([row], index=pd.Index(["Sample_1"], name="Sample"))
        print_results(df_row)
        if out_prefix:
            save_results(df_row, out_prefix + "_kinetics.xlsx")
            plot_results({"Sample_1": res}, out_prefix + "_plot.png")

    # ---- Mode 3: built-in demo ------------------------------------------
    elif mode == "3":
        demo_data = {
            "time":  [0,   5,  10,  20,  30,  60,  90, 120],
            "Cat-A": [500, 420, 350, 260, 200, 110,  60,  30],
            "Cat-B": [500, 390, 300, 200, 130,  55,  20,   8],
            "Cat-C": [500, 460, 430, 380, 340, 250, 180, 120],
        }
        df     = pd.DataFrame(demo_data)
        res_df, raw = analyze_dataframe(
            df, "time", conc_basis=conc_basis,
            molar_mass=molar_mass, n_sulfur=n_sulfur
        )
        print("\nDemo: three catalysts (Cat-A, Cat-B, Cat-C) over 120 min")
        print_results(res_df)
        if out_prefix:
            save_results(res_df, out_prefix + "_kinetics.xlsx")
            plot_results(raw, out_prefix + "_plot.png")
        else:
            plot_results(raw)

    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
