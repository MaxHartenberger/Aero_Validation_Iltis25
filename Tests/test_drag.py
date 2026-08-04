"""Tests for the drag computation module.

These tests verify the core scientific functions in aero_validation.drag
using synthetic data with known properties.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import json

from aero_validation.drag import (
    VehicleConstants,
    load_constants,
    assign_config,
    robust_linear_fit,
    robust_linear_fit_multi,
    derive_drag_metrics,
    fd_table_from_summary,
    summarize_by_config,
)


# ---------------------------------------------------------------------------
# VehicleConstants & load_constants
# ---------------------------------------------------------------------------

class TestLoadConstants:
    """Tests for JSON config loading."""

    def test_loads_valid_json(self):
        """Should load all fields from a valid JSON."""
        data = {
            "vehicle": {
                "mass_kg": 280.0,
                "rho_kgm3": 1.225,
                "frontal_area_m2_by_config": {"full": 1.20, "no_front_rear": 1.20},
            },
            "analysis": {
                "fd_speeds_ms": [10.0, 15.0, 20.0],
                "baseline_config": "no_front_rear",
                "config_ranges_by_run_id": {
                    "full": [1, 12],
                    "no_rear": [13, 24],
                    "no_front_rear": [25, 36],
                    "no_front": [37, 46],
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = Path(f.name)

        try:
            vc = load_constants(path)
            assert vc.mass_kg == 280.0
            assert vc.rho_kgm3 == 1.225
            assert vc.baseline_config == "no_front_rear"
            assert len(vc.config_ranges_by_run_id) == 4
            assert vc.config_ranges_by_run_id["full"] == (1, 12)
        finally:
            path.unlink()

    def test_missing_ranges_raises(self):
        """Should raise ValueError when config_ranges_by_run_id is missing."""
        data = {
            "vehicle": {"mass_kg": 280.0, "rho_kgm3": 1.225},
            "analysis": {"baseline_config": "no_front_rear"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="config_ranges_by_run_id"):
                load_constants(path)
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# assign_config
# ---------------------------------------------------------------------------

class TestAssignConfig:
    def test_assigns_correct_config(self):
        ranges = {"full": (1, 12), "no_rear": (13, 24)}
        assert assign_config(5, ranges) == "full"
        assert assign_config(13, ranges) == "no_rear"
        assert assign_config(24, ranges) == "no_rear"

    def test_unknown_run_returns_unknown(self):
        ranges = {"full": (1, 12)}
        assert assign_config(99, ranges) == "unknown"


# ---------------------------------------------------------------------------
# robust_linear_fit
# ---------------------------------------------------------------------------

class TestRobustLinearFit:
    def test_perfect_linear(self, rng):
        """Perfect y = 2*x + 3 with no noise should recover coefficients."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 3.0
        m, b, r2, n = robust_linear_fit(x, y)
        assert abs(m - 2.0) < 1e-6
        assert abs(b - 3.0) < 1e-6
        assert abs(r2 - 1.0) < 1e-10
        assert n == 100

    def test_rejects_outliers(self, rng):
        """A dataset with a few extreme outliers should still recover the slope."""
        x = np.linspace(5, 30, 200)
        y_true = 0.01 * x + 0.05  # realistic decel ~ k2*v² + c
        noise = rng.normal(0, 0.002, size=200)
        y = y_true + noise
        # Inject outliers
        y[0] = 0.5
        y[1] = -0.3
        m, b, r2, n = robust_linear_fit(x, y)
        # Slope should be near 0.01 (k2 ~ 0.001-0.005 for Formula Student)
        assert 0.005 < m < 0.015, f"Slope m={m:.6f} outside expected range"
        assert n < 200, "Should have rejected some outlier points"


# ---------------------------------------------------------------------------
# derive_drag_metrics
# ---------------------------------------------------------------------------

class TestDeriveDragMetrics:
    def test_known_case(self):
        """Test with hand-calculated values.
        k2 = (rho * CdA) / (2 * m)  →  CdA = 2*m*k2/rho
        """
        k2 = 0.002  # m⁻¹
        c = 0.15    # m/s²
        mass = 280.0
        rho = 1.225
        area = 1.20

        metrics = derive_drag_metrics(
            k2=k2, c=c, mass_kg=mass, rho_kgm3=rho, area_m2=area
        )
        # CdA = 2 * 280 * 0.002 / 1.225 = 0.9143...
        assert abs(metrics["CdA_m2"] - 0.9142857) < 1e-4
        # Cr = c / g = 0.15 / 9.80665 = 0.01530...
        assert abs(metrics["Cr"] - 0.0152956) < 1e-4
        # Cd = CdA / area = 0.9143 / 1.20 = 0.7619...
        assert abs(metrics["Cd"] - 0.7619048) < 1e-4
        # Fd at 10 m/s = 0.5 * 1.225 * 0.9143 * 100 = 56.0...
        assert abs(metrics["Fd_10ms_N"] - 56.0) < 0.5

    def test_cd_nan_when_area_none(self):
        metrics = derive_drag_metrics(
            k2=0.002, c=0.15, mass_kg=280.0, rho_kgm3=1.225, area_m2=None
        )
        assert np.isnan(metrics["Cd"])
        assert not np.isnan(metrics["CdA_m2"])


# ---------------------------------------------------------------------------
# fd_table_from_summary
# ---------------------------------------------------------------------------

class TestFdTableFromSummary:
    def test_generates_table(self):
        summary = pd.DataFrame(
            {
                "config": ["full", "no_rear"],
                "CdA_m2_mean": [1.0, 0.8],
            }
        )
        table = fd_table_from_summary(
            summary, rho_kgm3=1.225, speeds_ms=[10.0, 20.0]
        )
        assert len(table) == 4  # 2 configs × 2 speeds
        assert set(table.columns) == {"config", "v_ms", "Fd_N"}
        # Fd = 0.5 * 1.225 * 1.0 * 100 = 61.25
        full_10 = table[(table["config"] == "full") & (table["v_ms"] == 10.0)]
        assert abs(float(full_10["Fd_N"].iloc[0]) - 61.25) < 0.01


# ---------------------------------------------------------------------------
# summarize_by_config
# ---------------------------------------------------------------------------

class TestSummarizeByConfig:
    def test_computes_stats_and_delta(self):
        coeffs = pd.DataFrame(
            {
                "config": ["full", "full", "no_front_rear", "no_front_rear"],
                "run_id": [1, 2, 25, 26],
                "CdA_m2": [1.0, 1.1, 0.5, 0.55],
                "Cr": [0.015, 0.016, 0.014, 0.015],
                "r2": [0.95, 0.96, 0.93, 0.94],
                "n_points": [200, 210, 180, 190],
            }
        )
        stats, contrib = summarize_by_config(
            coeffs, baseline_config="no_front_rear"
        )
        assert len(stats) == 2
        assert "delta_CdA_vs_base" in stats.columns
        # full CdA_mean = 1.05, no_front_rear CdA_mean = 0.525
        # delta = 1.05 - 0.525 = 0.525
        full_row = stats[stats["config"] == "full"]
        assert abs(float(full_row["delta_CdA_vs_base"].iloc[0]) - 0.525) < 0.01

        assert len(contrib) == 3
        assert set(contrib["component"]) == {
            "front_wing",
            "rear_wing",
            "both_wings",
        }
