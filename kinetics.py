#!/usr/bin/env python3
"""
kinetics.py  v3.0 -- Multi-sample Chemical Kinetics Analyzer
=============================================================
Fits concentration-vs-time data to kinetic models used in
catalysis, adsorption, and oxidative desulfurization (ODS).

Supported models
----------------
  Standard kinetic models:
    - Zero Order
    - First Order
    - Pseudo-First Order  (Lagergren, free intercept)
    - Second Order

  Diffusion / surface models:
    - Weber-Morris  (Intraparticle Diffusion)
    - Elovich       (Heterogeneous Surface Chemisorption)

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
        raise ValueError("conc_basis must be 'ppm', 'sulfur', or 'substrate'.")


# -------------------------------------------------------
# qt computation  (required for Weber-Morris and Elovich)
# -------------------------------------------------------

def compute_qt(C0, Ce, V_L, m_g):
    """
    Compute adsorption capacity qt at each time point.

    qt represents how much contaminant (mg) has been removed
    per gram of catalyst/adsorbent at time t.

    Parameters
    ----------
    C0  : float
        Initial concentration (mg/L).
    Ce  : array-like
        Residual concentration at each time point (mg/L).
    V_L : float
        Volume of solution in litres (e.g. 0.05 for 50 mL).
    m_g : float
        Mass of catalyst/adsorbent in grams (e.g. 0.01 for 10 mg).

    Returns
    -------
    numpy.ndarray
        qt values in mg/g.

    Formula
    -------
        qt = (C0 - Ce) * V_L / m_g

    Notes
    -----
    - C0 and Ce must use the same concentration unit (mg/L).
    - If your concentrations are in mol/L, multiply by molar mass first.
    - qt is required as input for fit_weber_morris() and fit_elovich().
    """
    Ce = np.asarray(Ce, dtype=float)
    return (C0 - Ce) * V_L / m_g


# -------------------------------------------------------
# Weber-Morris intraparticle diffusion model
# -------------------------------------------------------

def fit_weber_morris(t, qt):
    """
    Fit the Weber-Morris intraparticle diffusion model.

    The model assumes that the rate-limiting step is diffusion of the
    contaminant into the internal pores of the catalyst/adsorbent particle.

    Model equation
    --------------
        qt = kid * sqrt(t) + C

    where:
        kid  = intraparticle diffusion rate constant (mg/g/min^0.5)
        C    = intercept related to the boundary layer thickness (mg/g)

    Interpretation of intercept C
    ------------------------------
        C = 0      : intraparticle diffusion is the sole rate-limiting step.
        C > 0      : film diffusion (boundary layer) also contributes.
        C < 0      : uncommon; may indicate surface diffusion dominance
                     or multi-stage diffusion (micropore + mesopore).

    Multi-stage behaviour
    ---------------------
    For porous catalysts with both micropores and mesopores, the qt vs sqrt(t)
    plot may show two or three linear regions with different slopes:
        Region 1 (steep) : film diffusion on external surface
        Region 2          : diffusion into mesopores
        Region 3 (flat)   : diffusion into micropores / equilibrium

    Parameters
    ----------
    t   : array-like
        Time points (min). Must not include t=0 to avoid sqrt(0) ambiguity.
    qt  : array-like
        Adsorption capacity at each time point (mg/g). Compute with compute_qt().

    Returns
    -------
    dict with keys:
        kid      : float  -- rate constant (mg/g/min^0.5)
        C        : float  -- intercept (mg/g)
        R2       : float  -- coefficient of determination
        slope    : float  -- raw regression slope (= kid)
        intercept: float  -- raw regression intercept (= C)

    References
    ----------
    Weber, W.J.; Morris, J.C. J. Sanit. Eng. Div. 1963, 89, 31-60.
    """
    t  = np.asarray(t,  dtype=float)
    qt = np.asarray(qt, dtype=float)
    reg = linregress(np.sqrt(t), qt)
    return {
        "kid":       reg.slope,
        "C":         reg.intercept,
        "R2":        reg.rvalue ** 2,
        "slope":     reg.slope,
        "intercept": reg.intercept,
    }


# -------------------------------------------------------
# Elovich model (heterogeneous surface chemisorption)
# -------------------------------------------------------

def fit_elovich(t, qt):
    """
    Fit the Elovich kinetic model for chemisorption on heterogeneous surfaces.

    The Elovich model describes adsorption where the activation energy
    increases linearly with surface coverage, typical for catalysts with
    non-uniform active sites (e.g. doped carbons, metal oxides).

    Differential form
    -----------------
        dq/dt = alpha * exp(-beta * q)

    Linearised form (used here)
    ----------------------------
        qt = (1/beta) * ln(alpha * beta) + (1/beta) * ln(t)

    Which simplifies to:
        qt = A + B * ln(t)

    where:
        B = 1/beta    --> beta  = 1/B    (desorption constant, g/mg)
        A = B*ln(alpha*beta) --> alpha = exp(A/B) / beta  (initial adsorption rate, mg/g/min)

    Parameters
    ----------
    t   : array-like
        Time points (min). t > 0 required (ln(t) is undefined at t=0).
    qt  : array-like
        Adsorption capacity (mg/g). Compute with compute_qt().

    Returns
    -------
    dict with keys:
        alpha    : float  -- initial adsorption rate (mg/g/min)
        beta     : float  -- desorption constant / surface heterogeneity (g/mg)
                             Larger beta  --> more homogeneous surface
                             Smaller beta --> more heterogeneous surface
        R2       : float  -- coefficient of determination
        slope    : float  -- raw regression slope (= 1/beta)
        intercept: float  -- raw regression intercept

    Notes
    -----
    - Good fit (R2 > 0.98) confirms chemisorption on a heterogeneous surface.
    - Compare beta across samples: lower beta = more varied adsorption energies.
    - Unlike PSO, Elovich does not assume a specific surface reaction mechanism.

    References
    ----------
    Chien, S.H.; Clayton, W.R. Soil Sci. Soc. Am. J. 1980, 44, 265-268.
    """
    t  = np.asarray(t,  dtype=float)
    qt = np.asarray(qt, dtype=float)

    if np.any(t <= 0):
        raise ValueError(
            "All time values must be > 0 for Elovich fitting (ln(t) undefined at t=0)."
        )

    reg = linregress(np.log(t), qt)
    B   = reg.slope
    A   = reg.intercept
    R2  = reg.rvalue ** 2

    if B <= 0:
        warnings.warn(
            "Elovich regression slope <= 0. "
            "alpha and beta cannot be computed. "
            "The model may not be appropriate for this dataset.",
            UserWarning, stacklevel=2
        )
        return {"alpha": np.nan, "beta": np.nan, "R2": R2,
                "slope": B, "intercept": A}

    beta  = 1.0 / B
    alpha = np.exp(A * beta) / beta

    return {
        "alpha":     alpha,
        "beta":      beta,
        "R2":        R2,
        "slope":     B,
        "intercept": A,
    }


# -------------------------------------------------------
# Core fitting function -- single sample (standard models)
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
    """
    t      = np.asarray(t,        dtype=float)
    Ce_raw = np.asarray(Ce_input, dtype=float)

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

    if conc_basis in ("sulfur", "substrate"):
        Ce        = to_molar(Ce_safe, conc_basis, molar_mass, n_sulfur)
        conc_unit = "mol/L"
    else:
        Ce        = Ce_safe
        conc_unit = "mg/L"

    C0 = float(C0_override) if C0_override is not None else Ce[0]

    s0   = linregress(t, Ce)
    k0   = -s0.slope
    R2_0 = s0.rvalue ** 2

    s1   = linregress(t, np.log(Ce / C0))
    k1   = -s1.slope
    R2_1 = s1.rvalue ** 2

    s2           = linregress(t, np.log(Ce))
    k_pfo        = -s2.slope
    C0_fit       = np.exp(s2.intercept)
    R2_pfo       = s2.rvalue ** 2
    delta_C0_pct = abs(C0_fit - C0) / C0 * 100

    s3   = linregress(t, 1.0 / Ce)
    k2   = s3.slope
    R2_2 = s3.rvalue ** 2

    t_half_0   = C0 / (2.0 * k0)   if k0   > 0 else float("nan")
    t_half_1   = np.log(2) / k1    if k1   > 0 else float("nan")
    t_half_pfo = np.log(2) / k_pfo if k_pfo > 0 else float("nan")
    t_half_2   = 1.0 / (k2 * C0)   if k2   > 0 else float("nan")

    r2_map = {
        "Zero Order":         R2_0,
        "First Order":        R2_1,
        "Pseudo-First Order": R2_pfo,
        "Second Order":       R2_2,
    }
    best_model = max(r2_map, key=r2_map.get)

    return {
        "k0": k0, "R2_0": R2_0, "t_half_0": t_half_0,
        "k1": k1, "R2_1": R2_1, "t_half_1": t_half_1,
        "k_pfo": k_pfo, "R2_pfo": R2_pfo, "t_half_pfo": t_half_pfo,
        "C0_fit": C0_fit, "delta_C0_pct": delta_C0_pct,
        "k2": k2, "R2_2": R2_2, "t_half_2": t_half_2,
        "C0": C0, "conc_unit": conc_unit, "best_model": best_model,
        "_t": t, "_Ce": Ce, "_Ce_raw": Ce_raw,
        "_s0": s0, "_s1": s1, "_s2": s2, "_s3": s3,
    }


# -------------------------------------------------------
# Multi-sample DataFrame analyser
# -------------------------------------------------------

def analyze_dataframe(df, time_col="time",
                      conc_basis="ppm", molar_mass=None, n_sulfur=1,
                      V_L=None, m_g=None):
    """
    Run all kinetic models on every sample column in a DataFrame.

    Standard models (Zero, First, PFO, Second Order) are always computed.
    Weber-Morris and Elovich models are computed when V_L and m_g are provided.

    Parameters
    ----------
    df          : pandas.DataFrame
        One time column + N concentration columns (one per sample/catalyst).
        Concentrations can be in mg/L (ppm) or removal fraction (0-1).
        If all values are <= 1.0, they are interpreted as removal fractions
        and converted to Ce automatically (C0 assumed = 100 arbitrary units).
    time_col    : str
        Name of the time column. Default "time".
    conc_basis  : str
        Passed to fit_all_models. Options: "ppm", "sulfur", "substrate".
    molar_mass  : float, optional
        Passed to fit_all_models (required when conc_basis="substrate").
    n_sulfur    : int
        Passed to fit_all_models.
    V_L         : float, optional
        Volume of solution in litres. Required for Weber-Morris and Elovich.
        Example: 0.05 for 50 mL.
    m_g         : float, optional
        Mass of catalyst/adsorbent in grams. Required for Weber-Morris and Elovich.
        Example: 0.01 for 10 mg.

    Returns
    -------
    results_df  : pandas.DataFrame
        Comparative summary table with one row per sample.
    raw_results : dict
        Full result dicts keyed by sample name (for plot_results()).

    Example
    -------
    >>> df = pd.read_excel("ODS_data.xlsx")
    >>> results, raw = analyze_dataframe(df, time_col="time",
    ...                                  conc_basis="sulfur", n_sulfur=1,
    ...                                  V_L=0.002, m_g=0.005)
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

    compute_diffusion = (V_L is not None) and (m_g is not None)

    for col in sample_cols:
        raw_vals = df[col].values.astype(float)

        # Auto-detect removal fraction input (values between 0 and 1)
        if np.all(raw_vals <= 1.0) and np.all(raw_vals >= 0.0):
            Ce_vals = 100.0 * (1.0 - raw_vals)
        else:
            Ce_vals = raw_vals

        res = fit_all_models(
            t, Ce_vals,
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

        row = {
            "Sample":            col,
            "C0":                "{:.4g} {}".format(res["C0"], unit),
            "k0 (Zero)":         "{:.4g} {}".format(res["k0"], k0_u),
            "R2 (Zero)":         "{:.4f}".format(res["R2_0"]),
            "t_half_0 (min)":    fmt(res["t_half_0"]),
            "k1 (1st)":          "{:.4g} /min".format(res["k1"]),
            "R2 (1st)":          "{:.4f}".format(res["R2_1"]),
            "t_half_1 (min)":    fmt(res["t_half_1"]),
            "k_pfo (PFO)":       "{:.4g} /min".format(res["k_pfo"]),
            "R2 (PFO)":          "{:.4f}".format(res["R2_pfo"]),
            "t_half_pfo (min)":  fmt(res["t_half_pfo"]),
            "C0_fit (PFO)":      "{:.4g}".format(res["C0_fit"]),
            "dC0% (PFO)":        "{:.1f}%".format(res["delta_C0_pct"]),
            "k2 (2nd)":          "{:.4g} {}".format(res["k2"], k2_u),
            "R2 (2nd)":          "{:.4f}".format(res["R2_2"]),
            "t_half_2 (min)":    fmt(res["t_half_2"]),
            "Best Model":        res["best_model"],
        }

        # Weber-Morris and Elovich (optional -- requires V_L and m_g)
        if compute_diffusion:
            Ce_for_qt = res["_Ce"]
            C0_for_qt = res["C0"]
            qt = compute_qt(C0_for_qt, Ce_for_qt, V_L, m_g)
            raw_results[col]["_qt"] = qt

            wm = fit_weber_morris(t, qt)
            el = fit_elovich(t, qt)

            raw_results[col]["_wm"] = wm
            raw_results[col]["_el"] = el

            row["kid (W-M mg/g/min^0.5)"] = "{:.4f}".format(wm["kid"])
            row["C_wm (mg/g)"]            = "{:.4f}".format(wm["C"])
            row["R2 (Weber-Morris)"]       = "{:.4f}".format(wm["R2"])
            row["alpha (Elovich)"]         = fmt(el["alpha"], 4)
            row["beta (Elovich g/mg)"]     = fmt(el["beta"],  4)
            row["R2 (Elovich)"]            = "{:.4f}".format(el["R2"])
        else:
            raw_results[col]["_qt"] = None
            raw_results[col]["_wm"] = None
            raw_results[col]["_el"] = None

        rows.append(row)

    results_df = pd.DataFrame(rows).set_index("Sample")
    return results_df, raw_results


# -------------------------------------------------------
# File I/O
# -------------------------------------------------------

def read_data(filepath, time_col="time"):
    """
    Load a CSV or Excel file into a DataFrame.

    Parameters
    ----------
    filepath : str or Path  -- path to .csv, .xlsx, or .xls file.
    time_col : str          -- name of the time column. Default "time".

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
    out_path   : str or Path  (.csv or .xlsx)
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
      - Row 2 : R2 bar chart  +  half-life horizontal bar chart
      - Row 3 : Weber-Morris and Elovich plots (if V_L and m_g were provided)

    Parameters
    ----------
    raw_results : dict  -- output from analyze_dataframe()
    out_path    : str or Path, optional
        If provided, saved as PNG: <stem>_<sample_name>.png
        If None, displayed interactively.
    """
    for i, (name, res) in enumerate(raw_results.items()):
        t      = res["_t"]
        Ce     = res["_Ce"]
        C0     = res["C0"]
        unit   = res["conc_unit"]
        color  = COLORS[i % len(COLORS)]
        qt     = res.get("_qt")
        wm     = res.get("_wm")
        el     = res.get("_el")

        has_diffusion = (qt is not None) and (wm is not None) and (el is not None)
        n_rows = 4 if has_diffusion else 3

        removal = (C0 - Ce) / C0 * 100.0
        t_fit   = np.linspace(t[0], t[-1], 200)

        fig = plt.figure(figsize=(14, 4 * n_rows))
        fig.suptitle("Kinetic Analysis -- {}".format(name),
                     fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.55, wspace=0.42)

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
            (gs[1, 0], Ce,            "Ce ({})".format(unit),
             s0.slope, s0.intercept,
             "Zero Order\nk={:.3g}  R2={:.4f}".format(res["k0"], res["R2_0"])),
            (gs[1, 1], np.log(Ce/C0), "ln(Ce / C0)",
             s1.slope, s1.intercept,
             "First Order\nk={:.3g}  R2={:.4f}".format(res["k1"], res["R2_1"])),
            (gs[1, 2], np.log(Ce),    "ln(Ce)",
             s2.slope, s2.intercept,
             "Pseudo-First Order\nk={:.3g}  R2={:.4f}\ndC0={:.1f}%".format(
                 res["k_pfo"], res["R2_pfo"], res["delta_C0_pct"])),
            (gs[1, 3], 1.0/Ce,        "1/Ce (1/{})".format(unit),
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
        if has_diffusion:
            labels += ["W-M", "Elovich"]
            r2vals += [wm["R2"], el["R2"]]
        best_label = res["best_model"].split()[0]
        bar_colors = ["#FFD700" if lbl == best_label else color for lbl in labels]
        bars = ax5.bar(labels, r2vals, color=bar_colors, edgecolor="white", linewidth=0.8)
        ax5.set_ylim(0, 1.10)
        ax5.set_ylabel("R2")
        ax5.set_title("Model R2 Comparison  [gold = best]")
        ax5.axhline(0.99, ls=":", color="gray", lw=1)
        for bar, v in zip(bars, r2vals):
            ax5.text(bar.get_x() + bar.get_width()/2.0,
                     v + 0.012, "{:.4f}".format(v),
                     ha="center", va="bottom", fontsize=7)
        ax5.grid(True, alpha=0.3, axis="y")

        # -- Row 2b: Half-life --
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
                ax6.text(v + max(valid.values())*0.01, idx,
                         "{:.1f} min".format(v), va="center", fontsize=8)
            ax6.set_xlabel("t_half (min)")
            ax6.set_title("Half-Life (t1/2) per Model")
            ax6.grid(True, alpha=0.3, axis="x")
        else:
            ax6.text(0.5, 0.5, "t1/2 undefined\n(all k <= 0)",
                     ha="center", va="center", transform=ax6.transAxes,
                     fontsize=10, color="gray")
            ax6.set_title("Half-Life (t1/2) per Model")

        # -- Row 3: Weber-Morris + Elovich (only if qt available) --
        if has_diffusion:
            sqrt_t   = np.sqrt(t)
            sqrt_fit = np.sqrt(t_fit)

            ax_wm = fig.add_subplot(gs[3, :2])
            ax_wm.plot(sqrt_t, qt, "o", color=color, ms=7, label="Data")
            ax_wm.plot(sqrt_fit,
                       _fit_line(wm["slope"], wm["intercept"], sqrt_fit),
                       "--k", lw=1.5,
                       label="kid={:.3f}  C={:.3f}  R2={:.4f}".format(
                           wm["kid"], wm["C"], wm["R2"]))
            ax_wm.set_xlabel("sqrt(t)  (min^0.5)")
            ax_wm.set_ylabel("qt  (mg/g)")
            ax_wm.set_title("Weber-Morris  (Intraparticle Diffusion)")
            ax_wm.legend(fontsize=8)
            ax_wm.grid(True, alpha=0.3)

            ax_el = fig.add_subplot(gs[3, 2:])
            ax_el.plot(np.log(t), qt, "o", color=color, ms=7, label="Data")
            ln_fit = np.log(t_fit)
            ax_el.plot(ln_fit,
                       _fit_line(el["slope"], el["intercept"], ln_fit),
                       "--k", lw=1.5,
                       label="alpha={:.3f}  beta={:.4f}  R2={:.4f}".format(
                           el["alpha"] if not np.isnan(el["alpha"]) else 0,
                           el["beta"]  if not np.isnan(el["beta"])  else 0,
                           el["R2"]))
            ax_el.set_xlabel("ln(t)  (ln·min)")
            ax_el.set_ylabel("qt  (mg/g)")
            ax_el.set_title("Elovich  (Heterogeneous Surface)")
            ax_el.legend(fontsize=8)
            ax_el.grid(True, alpha=0.3)

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
    Verify consistency between ppm-S (sulfur analyser) and
    ppm of intact substrate (GC).

    Expected: ppm_S = ppm_substrate * (MW_S * n_sulfur / MW_substrate)

    Parameters
    ----------
    ppm_S                : float  -- measured total sulfur (mg-S/L)
    ppm_substrate        : float  -- measured substrate (mg/L)
    molar_mass_substrate : float  -- MW of substrate g/mol (DBT = 184.26)
    n_sulfur             : int    -- S atoms per molecule (default 1)
    tol_pct              : float  -- acceptable deviation % (default 5)

    Example
    -------
    >>> validate_mass_balance(500, 2873, molar_mass_substrate=184.26, n_sulfur=1)
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
    """Prompt user to choose a concentration basis."""
    print("\nConcentration basis:")
    print("  [a]  ppm (mg/L)    -- k0/k2 in mg/L units  [default]")
    print("  [b]  ppm-S         -- convert to mol/L via MW(S)=32.06")
    print("  [c]  ppm substrate -- convert to mol/L via substrate MW")
    choice = input("Choice (a/b/c) [default a]: ").strip().lower()
    conc_basis = "ppm"
    molar_mass = None
    n_sulfur   = 1
    if choice == "b":
        conc_basis = "sulfur"
        ns = input("  n_sulfur [default 1]: ").strip()
        n_sulfur = int(ns) if ns else 1
    elif choice == "c":
        conc_basis = "substrate"
        molar_mass = float(input("  Molar mass (g/mol) [e.g. 184.26 for DBT]: "))
        ns = input("  n_sulfur [default 1]: ").strip()
        n_sulfur = int(ns) if ns else 1
    return conc_basis, molar_mass, n_sulfur


def _ask_diffusion_params():
    """Optionally prompt for V_L and m_g to enable Weber-Morris and Elovich."""
    print("\nWeber-Morris and Elovich models require solution volume and catalyst mass.")
    ans = input("Enable diffusion models? (y/n) [default n]: ").strip().lower()
    if ans == "y":
        V_L = float(input("  Solution volume V (litres) [e.g. 0.05 for 50 mL]: "))
        m_g = float(input("  Catalyst mass m (grams)  [e.g. 0.01 for 10 mg]:  "))
        return V_L, m_g
    return None, None


def main():
    print("\n+------------------------------------------+")
    print("|  Chemical Kinetics Analyzer  v3.0        |")
    print("|  Zero / 1st / PFO / 2nd Order            |")
    print("|  Weber-Morris  /  Elovich                 |")
    print("+------------------------------------------+\n")

    print("Input mode:")
    print("  [1]  Load data from CSV or Excel file")
    print("  [2]  Enter one sample manually (interactive)")
    print("  [3]  Run built-in demo  (3-catalyst example)\n")
    mode = input("Choice (1/2/3): ").strip()

    conc_basis, molar_mass, n_sulfur = _choose_conc_basis()
    V_L, m_g = _ask_diffusion_params()

    out_prefix = input(
        "\nOutput file prefix (no extension, or Enter to skip): "
    ).strip()

    # ---- Mode 1: load from file ----------------------------------------
    if mode == "1":
        fpath = input("  File path (.csv or .xlsx): ").strip()
        tcol  = input("  Time column name [default 'time']: ").strip() or "time"
        df    = read_data(fpath, tcol)
        res_df, raw = analyze_dataframe(
            df, tcol, conc_basis, molar_mass, n_sulfur, V_L, m_g
        )
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
        if V_L is not None and m_g is not None:
            qt = compute_qt(C0, res["_Ce"], V_L, m_g)
            res["_qt"] = qt
            res["_wm"] = fit_weber_morris(t, qt)
            res["_el"] = fit_elovich(t, qt)
        else:
            res["_qt"] = res["_wm"] = res["_el"] = None

        public_keys = [k for k in res if not k.startswith("_")]
        row    = {k: res[k] for k in public_keys}
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
            molar_mass=molar_mass, n_sulfur=n_sulfur,
            V_L=V_L, m_g=m_g
        )
        print("\nDemo: three generic catalysts over 120 min")
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
