# tests/test_catalyst_analytics.py
# Automated tests for CatalystAnalytics
# Run with: python -m pytest tests/

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catalyst_analytics import (
    convert_to_mmol_L, SampleInfo, KineticsAnalyser,
    calc_conversion, calc_tof, calc_toc_removal,
    BETAnalyser, TPlotAnalyser
)

# ────────────────────────────────────────────
# Unit converter tests
# ────────────────────────────────────────────
class TestUnitConverter:
    def test_mol_L(self):
        assert convert_to_mmol_L(1.0, "mol/L") == 1000.0

    def test_mmol_L(self):
        assert convert_to_mmol_L(5.0, "mmol/L") == 5.0

    def test_ppmS(self):
        # 500 ppmS = 500/32.06 mmol/L
        result = convert_to_mmol_L(500.0, "ppmS")
        assert abs(result - 500.0 / 32.06) < 1e-4

    def test_mg_L_with_mw(self):
        # 180 mg/L, MW=180 → 1.0 mmol/L
        assert abs(convert_to_mmol_L(180.0, "mg/L", mw=180.0) - 1.0) < 1e-6

    def test_g_L_with_mw(self):
        # 1 g/L, MW=100 → 10 mmol/L
        assert abs(convert_to_mmol_L(1.0, "g/L", mw=100.0) - 10.0) < 1e-6

    def test_mg_L_missing_mw_raises(self):
        with pytest.raises(ValueError):
            convert_to_mmol_L(50.0, "mg/L")

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError):
            convert_to_mmol_L(1.0, "xyz")

# ────────────────────────────────────────────
# SampleInfo tests
# ────────────────────────────────────────────
class TestSampleInfo:
    def setup_method(self):
        self.info = SampleInfo(
            sample_name="TestCat", process_type="desulfurization",
            catalyst_mass_g=0.05, solution_vol_L=0.05,
            c0_value=500.0, c0_unit="ppmS",
            active_sites_mmol_g=0.32
        )

    def test_c0_mmol_L(self):
        assert abs(self.info.c0_mmol_L - 500.0 / 32.06) < 1e-3

    def test_catalyst_loading(self):
        assert self.info.catalyst_loading_g_L == 1.0

    def test_n0_mmol(self):
        assert abs(self.info.n0_mmol - self.info.c0_mmol_L * 0.05) < 1e-6

# ────────────────────────────────────────────
# Kinetics tests
# ────────────────────────────────────────────
class TestKinetics:
    def setup_method(self):
        self.info = SampleInfo(
            sample_name="MoS2", process_type="desulfurization",
            catalyst_mass_g=0.05, solution_vol_L=0.05,
            c0_value=500.0, c0_unit="ppmS",
            active_sites_mmol_g=0.32
        )
        time_h = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
        c_ppmS = np.array([500, 420, 350, 250, 160, 100, 45])
        c_mmol = np.array([convert_to_mmol_L(v, "ppmS") for v in c_ppmS])
        self.an = KineticsAnalyser(time_h, c_mmol, self.info)

    def test_conversion_final(self):
        report = self.an.full_report()
        assert report["Conversion X (%)"] > 80

    def test_best_fit_has_r2(self):
        best = self.an.best_fit()
        assert "R2" in best
        assert best["R2"] > 0.90

    def test_tof_positive(self):
        report = self.an.full_report()
        assert isinstance(report["TOF (h⁻¹)"], float)
        assert report["TOF (h⁻¹)"] > 0

    def test_conversion_profile_length(self):
        X = self.an.conversion_profile()
        assert len(X) == 7

# ────────────────────────────────────────────
# TOF and TOC tests
# ────────────────────────────────────────────
class TestHelpers:
    def test_conversion(self):
        assert calc_conversion(100.0, 10.0) == 90.0

    def test_tof(self):
        tof = calc_tof(converted_mmol=1.0, catalyst_mass_g=0.05,
                       active_sites_mmol_g=0.32, time_h=6.0)
        assert abs(tof - 1.0 / (0.016 * 6.0)) < 0.01

    def test_toc_removal(self):
        assert abs(calc_toc_removal(85.0, 12.0) - 85.88) < 0.1

    def test_toc_zero_raises(self):
        import math
        assert math.isnan(calc_toc_removal(0.0, 5.0))

# ────────────────────────────────────────────
# BET tests
# ────────────────────────────────────────────
class TestBET:
    def setup_method(self):
        self.p = np.array([0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.50,0.70,0.90,0.99])
        self.v = np.array([85,  102,  115,  126,  135,  143, 150, 172,  200,  280, 520])

    def test_sbet_positive(self):
        bet = BETAnalyser(self.p, self.v)
        res = bet.fit_bet()
        assert res["S_BET_m2g"] > 0

    def test_r2_quality(self):
        bet = BETAnalyser(self.p, self.v)
        res = bet.fit_bet()
        assert res["R2"] > 0.99

# ────────────────────────────────────────────
# T-Plot tests
# ────────────────────────────────────────────
class TestTPlot:
    def setup_method(self):
        self.p = np.array([0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.50,0.70,0.90,0.99])
        self.v = np.array([85,  102,  115,  126,  135,  143, 150, 172,  200,  280, 520])
        self.tp = TPlotAnalyser(self.p, self.v, s_bet=442.0, total_pore_volume=0.52)

    def test_pore_dist_sums_100(self):
        res  = self.tp.fit_tplot()
        dist = self.tp.pore_distribution(res["V_micro_cm3g"])
        total_pct = dist["Micropore_%"] + dist["Mesopore_%"] + dist["Macropore_%"]
        assert abs(total_pct - 100.0) < 0.2

    def test_vtot_conserved(self):
        res  = self.tp.fit_tplot()
        dist = self.tp.pore_distribution(res["V_micro_cm3g"])
        assert abs(dist["V_total_cm3g"] - 0.52) < 0.01
