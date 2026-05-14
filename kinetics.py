"""
Chemical Kinetics Analyzer — Desulfurization & Pollutant Removal
================================================================
Fits experimental data (C0, Ce, t) to four kinetic models:
  - Zero order
  - First order
  - Second order
  - Pseudo first order

Author  : Your Name
License : MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from tabulate import tabulate


# ──────────────────────────────────────────────
# 1. DATA ENTRY
# ──────────────────────────────────────────────

def input_data_manual():
    """
    Interactive entry: user types C0, Ce, t for each data point.
    Returns arrays: t (s), C0 (mg/L or any unit), Ce (mg/L), removal (%)
    """
    print("\n" + "="*55)
    print("  Chemical Kinetics Analyzer — Data Entry")
    print("="*55)
    n = int(input("  Number of data points: "))
    t_vals, ce_vals = [], []

    c0 = float(input("  Initial concentration C0 (same for all points): "))

    print(f"\n  Enter (time, final concentration Ce) for each point:")
    for i in range(n):
        row = input(f"    Point {i+1}  →  t [s], Ce [mg/L]: ").split(",")
        t_vals.append(float(row[0].strip()))
        ce_vals.append(float(row[1].strip()))

    t   = np.array(t_vals)
    Ce  = np.array(ce_vals)
    C0  = np.full_like(Ce, c0)
    removal = (C0 - Ce) / C0 * 100.0

    return t, C0, Ce, removal


def input_data_hardcoded():
    """
    Example dataset — desulfurization experiment.
    Replace these values with your own measurements.

    Columns: time (min), C0 (mg/L), Ce (mg/L)
    """
    data = [
        #  t,   C0,    Ce
        (  5,  100,  80.0),
        ( 10,  100,  62.0),
        ( 20,  100,  42.0),
        ( 30,  100,  30.0),
        ( 45,  100,  20.0),
        ( 60,  100,  14.0),
        ( 90,  100,   8.5),
        (120,  100,   5.0),
    ]
    t   = np.array([d[0] for d in data], dtype=float)
    C0  = np.array([d[1] for d in data], dtype=float)
    Ce  = np.array([d[2] for d in data], dtype=float)
    removal = (C0 - Ce) / C0 * 100.0
    return t, C0, Ce, removal


# ──────────────────────────────────────────────
# 2. KINETIC MODEL FITTING
# ──────────────────────────────────────────────

def fit_zero_order(t, C0, Ce):
    """
    Zero order: Ce = C0 - k*t
    Linearized: Ce vs t  →  slope = -k
    """
    slope, intercept, r, *_ = linregress(t, Ce)
    k   = -slope
    C0_fit = intercept          # should be ≈ C0[0]
    R2  = r**2
    Ce_pred = C0_fit - k * t
    return dict(k=k, intercept=C0_fit, R2=R2, Ce_pred=Ce_pred,
                x_label="t", y_label="Ce",
                x_data=t, y_data=Ce,
                model="Zero Order",  unit_k="mg/L·min")


def fit_first_order(t, C0, Ce):
    """
    First order: ln(Ce/C0) = -k*t   →   Ce = C0·exp(-k·t)
    Linearized: ln(Ce/C0) vs t  →  slope = -k
    """
    y = np.log(Ce / C0)
    slope, intercept, r, *_ = linregress(t, y)
    k   = -slope
    R2  = r**2
    Ce_pred = C0 * np.exp(-k * t)
    return dict(k=k, intercept=intercept, R2=R2, Ce_pred=Ce_pred,
                x_label="t", y_label="ln(Ce/C0)",
                x_data=t, y_data=y,
                model="First Order", unit_k="min⁻¹")


def fit_second_order(t, C0, Ce):
    """
    Second order: 1/Ce = 1/C0 + k*t
    Linearized: 1/Ce vs t  →  slope = k
    """
    y = 1.0 / Ce
    slope, intercept, r, *_ = linregress(t, y)
    k  = slope
    R2 = r**2
    Ce_pred = 1.0 / (1.0/C0 + k * t)
    return dict(k=k, intercept=intercept, R2=R2, Ce_pred=Ce_pred,
                x_label="t", y_label="1/Ce",
                x_data=t, y_data=y,
                model="Second Order", unit_k="L/mg·min")


def fit_pseudo_first_order(t, C0, Ce):
    """
    Pseudo first order (Lagergren): ln(Ce) = ln(C0) - k'*t
    Often used when Ce is small and external [B] >> [A].
    Linearized: ln(Ce) vs t  →  slope = -k'
    """
    y = np.log(Ce)
    slope, intercept, r, *_ = linregress(t, y)
    k   = -slope
    C0_fit = np.exp(intercept)
    R2  = r**2
    Ce_pred = C0_fit * np.exp(-k * t)
    return dict(k=k, intercept=intercept, R2=R2, Ce_pred=Ce_pred,
                x_label="t", y_label="ln(Ce)",
                x_data=t, y_data=y,
                model="Pseudo First Order", unit_k="min⁻¹")


# ──────────────────────────────────────────────
# 3. SUMMARY TABLE
# ──────────────────────────────────────────────

def print_summary(t, C0, Ce, removal):
    print("\n" + "="*55)
    print("  Experimental Data Summary")
    print("="*55)
    rows = []
    for i in range(len(t)):
        rows.append([
            f"{t[i]:.1f}",
            f"{C0[i]:.2f}",
            f"{Ce[i]:.2f}",
            f"{removal[i]:.1f}%"
        ])
    print(tabulate(rows,
                   headers=["t (min)", "C0 (mg/L)", "Ce (mg/L)", "Removal %"],
                   tablefmt="rounded_outline"))


def print_kinetic_results(results):
    print("\n" + "="*55)
    print("  Kinetic Model Fitting Results")
    print("="*55)
    rows = []
    for r in results:
        rows.append([r["model"], f"{r['k']:.5f}", r["unit_k"], f"{r['R2']:.4f}"])
    print(tabulate(rows,
                   headers=["Model", "k", "Unit", "R²"],
                   tablefmt="rounded_outline"))

    best = max(results, key=lambda x: x["R2"])
    print(f"\n  ✓ Best fit model: {best['model']}  (R² = {best['R2']:.4f})\n")
    return best


# ──────────────────────────────────────────────
# 4. PLOTTING
# ──────────────────────────────────────────────

COLORS = ["#378ADD", "#1D9E75", "#7F77DD", "#BA7517"]
MODEL_COLORS = {
    "Zero Order":        "#378ADD",
    "First Order":       "#1D9E75",
    "Second Order":      "#7F77DD",
    "Pseudo First Order":"#BA7517",
}

def plot_all(t, C0, Ce, removal, results):
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top: Ce vs t with all model predictions ──────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor("#1a1d26")
    ax0.scatter(t, Ce, color="white", s=60, zorder=5, label="Measured Ce")
    t_smooth = np.linspace(t.min(), t.max(), 300)
    for r in results:
        col = MODEL_COLORS[r["model"]]
        c0_val = C0[0]
        if r["model"] == "Zero Order":
            y_smooth = c0_val - r["k"] * t_smooth
        elif r["model"] == "First Order":
            y_smooth = c0_val * np.exp(-r["k"] * t_smooth)
        elif r["model"] == "Second Order":
            y_smooth = 1.0 / (1.0/c0_val + r["k"] * t_smooth)
        else:  # Pseudo
            C0_fit = np.exp(r["intercept"])
            y_smooth = C0_fit * np.exp(-r["k"] * t_smooth)
        ax0.plot(t_smooth, y_smooth, color=col, linewidth=1.8,
                 label=f"{r['model']}  (R²={r['R2']:.3f})", alpha=0.85)
    _style_ax(ax0, "Concentration vs Time — All Models",
              "Time (min)", "Ce  (mg/L)")
    ax0.legend(fontsize=8, facecolor="#1a1d26", labelcolor="white",
               framealpha=0.6, loc="upper right")

    # ── Bottom 4 panels: linearized plots ────────────────────────
    positions = [gs[1, 0], gs[1, 1], gs[2, 0], gs[2, 1]]
    for i, r in enumerate(results):
        ax = fig.add_subplot(positions[i])
        ax.set_facecolor("#1a1d26")
        col = MODEL_COLORS[r["model"]]

        # linearized data points
        ax.scatter(r["x_data"], r["y_data"], color="white", s=40, zorder=5)

        # regression line
        x_fit = np.linspace(r["x_data"].min(), r["x_data"].max(), 200)
        if r["model"] == "Zero Order":
            y_fit = r["intercept"] - r["k"] * x_fit
        elif r["model"] in ("First Order", "Pseudo First Order"):
            y_fit = r["intercept"] - r["k"] * x_fit
        else:  # Second
            y_fit = r["intercept"] + r["k"] * x_fit
        ax.plot(x_fit, y_fit, color=col, linewidth=1.8)

        title = f"{r['model']}\nk = {r['k']:.5f} {r['unit_k']}   R² = {r['R2']:.4f}"
        _style_ax(ax, title, r["x_label"], r["y_label"])

    plt.suptitle("Chemical Kinetics Analysis — Pollutant Removal",
                 color="white", fontsize=13, y=0.98, fontweight="bold")
    plt.savefig("kinetics_result.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  Plot saved → kinetics_result.png")
    plt.show()


def plot_removal(t, removal):
    """Separate chart: Removal % vs Time"""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d26")
    ax.plot(t, removal, color="#1D9E75", linewidth=2, marker="o",
            markersize=6, markerfacecolor="white")
    for xi, yi in zip(t, removal):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), color="#aaaaaa", fontsize=8, ha="center")
    _style_ax(ax, "Pollutant Removal Efficiency vs Time",
              "Time (min)", "Removal  (%)")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig("removal_curve.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  Plot saved → removal_curve.png")
    plt.show()


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, color="white", fontsize=9, pad=6)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(color="#222233", linewidth=0.6)


# ──────────────────────────────────────────────
# 5. ENTRY POINT
# ──────────────────────────────────────────────

def main():
    # ── choose mode ──────────────────────────────────────────────
    print("\n  [1] Use example dataset")
    print("  [2] Enter my own data manually")
    mode = input("  Choice (1/2): ").strip()

    if mode == "2":
        t, C0, Ce, removal = input_data_manual()
    else:
        t, C0, Ce, removal = input_data_hardcoded()

    # ── print data table ─────────────────────────────────────────
    print_summary(t, C0, Ce, removal)

    # ── fit all models ───────────────────────────────────────────
    results = [
        fit_zero_order(t, C0, Ce),
        fit_first_order(t, C0, Ce),
        fit_second_order(t, C0, Ce),
        fit_pseudo_first_order(t, C0, Ce),
    ]

    # ── results table + best model ───────────────────────────────
    print_kinetic_results(results)

    # ── plots ─────────────────────────────────────────────────────
    plot_removal(t, removal)
    plot_all(t, C0, Ce, removal, results)


if __name__ == "__main__":
    main()
