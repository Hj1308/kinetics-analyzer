# catalyst_analytics.py
# CatalystAnalytics — Integrated Catalyst & Reaction Analysis Suite
# Author: Hoda Jafari | github.com/Hj1308
# Version: 2.0.0
#
# Modules:
#   1. Unit Converter        — ppmS, ppm, mg/L, g/L, mmol/L, mol/L
#   2. SampleInfo            — structured sample metadata dataclass
#   3. KineticsAnalyser      — zero/first/second/pseudo-first order fitting
#   4. Conversion Calculator — X(%) profile over time
#   5. TOF Calculator        — Turnover Frequency (h⁻¹)
#   6. TOC Removal           — Total Organic Carbon removal (%)
#   7. BETAnalyser           — BET surface area from N2 physisorption
#   8. TPlotAnalyser         — T-Plot (Harkins-Jura): micropore/mesopore/macropore %

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
N_AV   = 6.02214076e23   # Avogadro number
SIGMA  = 0.162e-18       # N2 molecular cross-section (m²)
V_MOL  = 22414.0         # molar volume at STP (cm³/mol)
MW_S   = 32.06           # molar mass of sulfur (g/mol)

# ─────────────────────────────────────────
# 1. UNIT CONVERTER
# ─────────────────────────────────────────
def convert_to_mmol_L(value: float, unit: str, mw: Optional[float] = None) -> float:
    """
    Convert any concentration unit to mmol/L.

    Supported units:
        mol/L   — molar
        mmol/L  — millimolar
        mg/L    — milligrams per litre (requires mw in g/mol)
        ppm     — parts per million, aqueous ≈ mg/L (requires mw)
        g/L     — grams per litre (requires mw)
        ppmS    — ppm sulfur, auto-converted using MW_S = 32.06 g/mol

    Parameters
    ----------
    value : float  — numeric concentration
    unit  : str    — unit string (case-insensitive stripped)
    mw    : float  — molecular weight (g/mol), required for mg/L, ppm, g/L

    Returns
    -------
    float — concentration in mmol/L
    """
    unit = unit.strip()
    if unit == "mol/L":
        return value * 1000.0
    elif unit == "mmol/L":
        return value
    elif unit in ("mg/L", "ppm"):
        if mw is None:
            raise ValueError("Molecular weight (g/mol) is required for mg/L or ppm units.")
        return value / mw
    elif unit == "g/L":
        if mw is None:
            raise ValueError("Molecular weight (g/mol) is required for g/L unit.")
        return (value * 1000.0) / mw
    elif unit == "ppmS":
        return value / MW_S
    else:
        raise ValueError(f"Unknown concentration unit: '{unit}'. "
                         f"Supported: mol/L, mmol/L, mg/L, ppm, g/L, ppmS")


# ─────────────────────────────────────────
# 2. SAMPLE METADATA
# ─────────────────────────────────────────
@dataclass
class SampleInfo:
    """
    Stores structured experimental metadata for a catalytic reaction.

    Parameters
    ----------
    sample_name          : str   — catalyst or sample identifier
    process_type         : str   — 'desulfurization' | 'water_treatment' |
                                   'photocatalysis' | 'oxidation' | 'other'
    catalyst_mass_g      : float — mass of catalyst used (g)
    solution_vol_L       : float — volume of reaction solution (L)
    c0_value             : float — initial concentration (raw numeric)
    c0_unit              : str   — unit of c0_value
    mw_pollutant         : float — molecular weight of pollutant (g/mol),
                                   required if unit is mg/L, ppm, or g/L
    active_sites_mmol_g  : float — active site density (mmol/g),
                                   from NH3-TPD, CO chemisorption, etc.
                                   Required for TOF calculation.
    notes                : str   — free-text experimental notes
    """
    sample_name          : str
    process_type         : str
    catalyst_mass_g      : float
    solution_vol_L       : float
    c0_value             : float
    c0_unit              : str
    mw_pollutant         : Optional[float] = None
    active_sites_mmol_g  : Optional[float] = None
    notes                : str = ""

    @property
    def c0_mmol_L(self) -> float:
        """Initial concentration converted to mmol/L."""
        return convert_to_mmol_L(self.c0_value, self.c0_unit, self.mw_pollutant)

    @property
    def catalyst_loading_g_L(self) -> float:
        """Catalyst loading in g/L."""
        return self.catalyst_mass_g / self.solution_vol_L

    @property
    def n0_mmol(self) -> float:
        """Initial moles of pollutant in the reactor (mmol)."""
        return self.c0_mmol_L * self.solution_vol_L

    def summary(self) -> dict:
        """Returns a printable dictionary summary of sample metadata."""
        return {
            "Sample"           : self.sample_name,
            "Process"          : self.process_type,
            "C0 (input)"       : f"{self.c0_value} {self.c0_unit}",
            "C0 (mmol/L)"      : round(self.c0_mmol_L, 4),
            "n0 (mmol)"        : round(self.n0_mmol, 4),
            "Catalyst (g)"     : self.catalyst_mass_g,
            "Volume (L)"       : self.solution_vol_L,
            "Cat. loading g/L" : round(self.catalyst_loading_g_L, 3),
            "Active sites mmol/g": self.active_sites_mmol_g or "N/A",
            "Notes"            : self.notes,
        }


# ─────────────────────────────────────────
# 3. CONVERSION CALCULATOR
# ─────────────────────────────────────────
def calc_conversion(c0_mmol_L: float, ct_mmol_L: float) -> float:
    """
    Calculate pollutant conversion.

    X (%) = (C0 - Ct) / C0 × 100

    Parameters
    ----------
    c0_mmol_L : float — initial concentration (mmol/L)
    ct_mmol_L : float — concentration at time t (mmol/L)

    Returns
    -------
    float — conversion percentage
    """
    return round((c0_mmol_L - ct_mmol_L) / c0_mmol_L * 100, 2)


# ─────────────────────────────────────────
# 4. TOF — Turnover Frequency
# ─────────────────────────────────────────
def calc_tof(
    converted_mmol      : float,
    catalyst_mass_g     : float,
    active_sites_mmol_g : float,
    time_h              : float
) -> float:
    """
    Calculate Turnover Frequency (TOF).

    TOF (h⁻¹) = n_converted / (n_active_sites × time)

    where n_active_sites = catalyst_mass_g × active_sites_mmol_g  [mmol]

    Parameters
    ----------
    converted_mmol      : float — moles of pollutant converted (mmol)
    catalyst_mass_g     : float — catalyst mass (g)
    active_sites_mmol_g : float — active site density (mmol/g)
    time_h              : float — reaction time (h)

    Returns
    -------
    float — TOF in h⁻¹
    """
    n_sites = catalyst_mass_g * active_sites_mmol_g
    if n_sites <= 0 or time_h <= 0:
        return float("nan")
    return round(converted_mmol / (n_sites * time_h), 4)


# ─────────────────────────────────────────
# 5. TOC REMOVAL
# ─────────────────────────────────────────
def calc_toc_removal(toc0: float, toc_t: float) -> float:
    """
    Calculate Total Organic Carbon (TOC) removal efficiency.

    TOC removal (%) = (TOC0 - TOC_t) / TOC0 × 100

    Parameters
    ----------
    toc0  : float — initial TOC (mg/L or ppm)
    toc_t : float — TOC at time t

    Returns
    -------
    float — TOC removal percentage
    """
    if toc0 <= 0:
        return float("nan")
    return round((toc0 - toc_t) / toc0 * 100, 2)


# ─────────────────────────────────────────
# 6. KINETICS ANALYSER
# ─────────────────────────────────────────
class KineticsAnalyser:
    """
    Fits kinetic models to concentration–time data and calculates
    conversion, rate constants, TOF, and TOC removal.

    Supported kinetic models:
        - Zero-order          :  C = C0 - k·t
        - First-order         :  ln(C/C0) = -k·t
        - Second-order        :  1/C = 1/C0 + k·t
        - Pseudo-first-order  :  ln(qe - qt) = ln(qe) - k1·t  (Lagergren)

    Usage
    -----
    >>> info = SampleInfo(...)
    >>> an = KineticsAnalyser(time_h, c_mmol_L, info)
    >>> report = an.full_report()
    >>> an.plot_kinetics(save_path="output.png")
    """

    def __init__(self, time: np.ndarray, concentration: np.ndarray,
                 sample_info: SampleInfo):
        """
        Parameters
        ----------
        time          : array-like — time points (hours)
        concentration : array-like — concentration in mmol/L
                        (use convert_to_mmol_L() before passing)
        sample_info   : SampleInfo instance
        """
        self.t    = np.array(time,          dtype=float)
        self.c    = np.array(concentration, dtype=float)
        self.info = sample_info
        self.c0   = self.c[0]

    # ── Conversion profile
    def conversion_profile(self) -> np.ndarray:
        """Returns X(%) at each time point."""
        return np.array([calc_conversion(self.c0, ct) for ct in self.c])

    # ── Kinetic model fits
    def fit_zero_order(self) -> dict:
        slope, _, r, *_ = linregress(self.t, self.c)
        return {"model": "Zero-order",
                "k (mmol/L/h)": round(-slope, 5), "R2": round(r**2, 5)}

    def fit_first_order(self) -> dict:
        y = np.log(self.c / self.c0)
        slope, _, r, *_ = linregress(self.t, y)
        return {"model": "First-order",
                "k (h⁻¹)": round(-slope, 5), "R2": round(r**2, 5)}

    def fit_second_order(self) -> dict:
        y = 1.0 / self.c
        slope, _, r, *_ = linregress(self.t, y)
        return {"model": "Second-order",
                "k (L/mmol/h)": round(slope, 5), "R2": round(r**2, 5)}

    def fit_pseudo_first_order(self) -> dict:
        """Lagergren pseudo-first-order model for adsorption kinetics."""
        qe_est = self.c[0] - self.c[-1]
        qt     = self.c[0] - self.c
        y      = np.log(np.clip(qe_est - qt, 1e-12, None))
        slope, intercept, r, *_ = linregress(self.t[:-1], y[:-1])
        return {"model": "Pseudo-first-order",
                "k1 (h⁻¹)": round(-slope, 5),
                "qe (mmol/g)": round(np.exp(intercept), 5),
                "R2": round(r**2, 5)}

    def best_fit(self) -> dict:
        """Returns the model with highest R²."""
        results = [
            self.fit_zero_order(), self.fit_first_order(),
            self.fit_second_order(), self.fit_pseudo_first_order()
        ]
        return max(results, key=lambda r: r["R2"])

    # ── TOF
    def calc_tof_val(self, time_h: float) -> Optional[float]:
        if self.info.active_sites_mmol_g is None:
            return None
        converted_mmol = (self.c0 - self.c[-1]) * self.info.solution_vol_L
        return calc_tof(converted_mmol, self.info.catalyst_mass_g,
                        self.info.active_sites_mmol_g, time_h)

    # ── Full report
    def full_report(self) -> dict:
        """Returns a comprehensive analysis dictionary."""
        X_final = calc_conversion(self.c0, self.c[-1])
        best    = self.best_fit()
        tof_val = self.calc_tof_val(self.t[-1])
        return {
            "Sample Info"      : self.info.summary(),
            "Conversion X (%)" : X_final,
            "Best Fit Model"   : best,
            "All Models"       : [
                self.fit_zero_order(), self.fit_first_order(),
                self.fit_second_order(), self.fit_pseudo_first_order()
            ],
            "TOF (h⁻¹)"        : tof_val if tof_val is not None
                                  else "N/A — provide active_sites_mmol_g in SampleInfo",
        }

    # ── Plot
    def plot_kinetics(self, save_path: str = "kinetics_plot.png") -> str:
        """Generates and saves a two-panel kinetics figure."""
        X    = self.conversion_profile()
        best = self.best_fit()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].plot(self.t, self.c, "o-", color="steelblue", lw=2, ms=7)
        axes[0].set_xlabel("Time (h)", fontsize=12)
        axes[0].set_ylabel("Concentration (mmol/L)", fontsize=12)
        axes[0].set_title(f"Concentration Profile\n{self.info.sample_name}", fontsize=12)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.t, X, "s-", color="darkorange", lw=2, ms=7)
        axes[1].axhline(y=X[-1], color="crimson", ls="--", alpha=0.6,
                        label=f"X_final = {X[-1]:.1f}%")
        axes[1].set_xlabel("Time (h)", fontsize=12)
        axes[1].set_ylabel("Conversion X (%)", fontsize=12)
        axes[1].set_title(
            f"Conversion Profile\nBest model: {best['model']}  R²={best['R2']}",
            fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"CatalystAnalytics | {self.info.sample_name}",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path


# ─────────────────────────────────────────
# 7. BET ANALYSER
# ─────────────────────────────────────────
class BETAnalyser:
    """
    Calculates BET surface area from N2 physisorption data.

    Uses the BET linear equation:
        P / [V(P0 - P)] = 1/(Vm·C) + (C-1)/(Vm·C) · (P/P0)

    Parameters
    ----------
    pressure        : array — relative pressure (P/P0)
    volume_adsorbed : array — volume adsorbed (cm³/g STP)
    p0              : float — saturation pressure (default 1.0 for P/P0 input)
    """
    def __init__(self, pressure: np.ndarray, volume_adsorbed: np.ndarray, p0: float = 1.0):
        self.p  = np.array(pressure,        dtype=float)
        self.v  = np.array(volume_adsorbed, dtype=float)
        self.p0 = p0

    def bet_transform(self):
        x = self.p / self.p0
        return x, x / (self.v * (1 - x))

    def fit_bet(self, p_min: float = 0.05, p_max: float = 0.35) -> dict:
        """
        Fits BET equation in the standard linear range.

        Returns
        -------
        dict with: Vm_cm3g, C_BET, S_BET_m2g, R2, slope, intercept
        """
        x, y = self.bet_transform()
        mask = (x >= p_min) & (x <= p_max)
        slope, intercept, r, *_ = linregress(x[mask], y[mask])
        c    = 1 + slope / intercept
        vm   = 1 / (slope + intercept)
        sbet = (vm * N_AV * SIGMA) / V_MOL
        return {
            "Vm_cm3g"   : round(vm,       4),
            "C_BET"     : round(c,        2),
            "S_BET_m2g" : round(sbet,     2),
            "R2"        : round(r**2,     5),
            "slope"     : round(slope,    6),
            "intercept" : round(intercept,6),
        }


# ─────────────────────────────────────────
# 8. T-PLOT ANALYSER
# ─────────────────────────────────────────
class TPlotAnalyser:
    """
    T-Plot analysis using the Harkins-Jura equation to determine:
        - Micropore volume (V_micro, cm³/g)
        - External (mesopore + macropore) surface area (S_ext, m²/g)
        - Pore type distribution: Micropore / Mesopore / Macropore (%)

    Harkins-Jura t-equation:
        t = sqrt(13.99 / (0.034 - log10(P/P0)))   [Angstrom]

    In the t-plot:
        - Linear through origin → no micropores
        - Positive intercept    → micropore volume present
        - Slope ∝ external surface area

    Parameters
    ----------
    pressure            : array — relative pressure (P/P0)
    volume_adsorbed     : array — volume adsorbed (cm³/g STP)
    s_bet               : float — BET surface area (m²/g)
    total_pore_volume   : float — total pore volume at P/P0 ≈ 0.99 (cm³/g)
    """
    def __init__(self, pressure: np.ndarray, volume_adsorbed: np.ndarray,
                 s_bet: float, total_pore_volume: float):
        self.p    = np.array(pressure,        dtype=float)
        self.v    = np.array(volume_adsorbed, dtype=float)
        self.sbet = s_bet
        self.vtot = total_pore_volume

    @staticmethod
    def harkins_jura_t(p_rel: np.ndarray) -> np.ndarray:
        """Statistical film thickness t in Angstrom (Harkins-Jura)."""
        return np.sqrt(13.99 / (0.034 - np.log10(p_rel)))

    def fit_tplot(self, t_min: float = 3.5, t_max: float = 5.0) -> dict:
        """
        Fits linear region of the t-plot.

        Default range 3.5–5.0 Å is the standard micropore analysis window.
        If insufficient points, range is auto-expanded.

        Returns
        -------
        dict with: S_ext_m2g, V_micro_cm3g, R2_tplot
        """
        t = self.harkins_jura_t(self.p)
        mask = (t >= t_min) & (t <= t_max)
        if mask.sum() < 2:
            t_min, t_max = t.min() + 0.1, t.max() - 0.1
            mask = (t >= t_min) & (t <= t_max)
        slope, intercept, r, *_ = linregress(t[mask], self.v[mask])
        s_ext   = slope * 15.47
        v_micro = max(intercept * 1e-3 / 1.547, 0.0)
        return {
            "S_ext_m2g"     : round(s_ext,   2),
            "V_micro_cm3g"  : round(v_micro, 4),
            "R2_tplot"      : round(r**2,    5),
        }

    def pore_distribution(self, v_micro: float) -> dict:
        """
        Calculates pore volume fractions.

        V_meso  = V_total - V_micro
        V_macro = 0 by default (requires Hg porosimetry for accurate value)

        Returns
        -------
        dict with volumes (cm³/g) and percentages for each pore type
        """
        v_meso  = max(self.vtot - v_micro, 0.0)
        v_macro = 0.0
        total   = v_micro + v_meso + v_macro
        if total <= 0:
            return {"error": "Total pore volume is zero or negative."}
        return {
            "V_micro_cm3g" : round(v_micro, 4),
            "V_meso_cm3g"  : round(v_meso,  4),
            "V_macro_cm3g" : round(v_macro, 4),
            "V_total_cm3g" : round(total,   4),
            "Micropore_%"  : round(100 * v_micro / total, 1),
            "Mesopore_%"   : round(100 * v_meso  / total, 1),
            "Macropore_%"  : round(100 * v_macro / total, 1),
        }

    def full_tplot_report(self) -> dict:
        """Convenience method: runs fit and pore distribution together."""
        fit = self.fit_tplot()
        dist = self.pore_distribution(fit["V_micro_cm3g"])
        return {**fit, **dist}

    def plot_tplot(self, save_path: str = "tplot.png") -> str:
        """Generates and saves the t-plot figure."""
        t   = self.harkins_jura_t(self.p)
        res = self.fit_tplot()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(t, self.v, color="royalblue", s=45, zorder=5, label="Experimental")
        t_line    = np.linspace(t.min(), t.max(), 300)
        slope_val = res["S_ext_m2g"] / 15.47
        int_val   = res["V_micro_cm3g"] * 1.547e3
        ax.plot(t_line, slope_val * t_line + int_val,
                color="crimson", lw=2, label=f"Linear fit  R²={res['R2_tplot']}")
        ax.set_xlabel("Statistical film thickness  t (Å)", fontsize=12)
        ax.set_ylabel("Volume adsorbed  (cm³/g STP)",      fontsize=12)
        ax.set_title("T-Plot Analysis (Harkins-Jura)", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path
