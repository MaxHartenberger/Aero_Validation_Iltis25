"""Tests for the signals module."""

import numpy as np
import pandas as pd
import pytest

from aero_validation.signals import sanitize_columns, smooth, add_derived_signals


class TestSanitizeColumns:
    def test_removes_unnamed_columns(self):
        df = pd.DataFrame({"t[s]": [1, 2], "Unnamed: 2": [3, 4]})
        result = sanitize_columns(df)
        assert "Unnamed: 2" not in result.columns

    def test_maps_known_headers(self):
        df = pd.DataFrame({"t[s]": [1, 2], "ICU_Speedfl[km/h]": [30, 40]})
        result = sanitize_columns(df)
        assert "t_s" in result.columns
        assert "ICU_Speedfl_kmh" in result.columns

    def test_fixes_encoding(self):
        """Degree symbol encoded as replacement char should be fixed."""
        df = pd.DataFrame({"ICU_SteeringAngle[�]": [0, 1]})
        result = sanitize_columns(df)
        assert "ICU_SteeringAngle_deg" in result.columns


class TestSmooth:
    def test_odd_window(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smooth(x, win=3)
        assert len(result) == len(x)
        # center-aligned: first element averages (1,2)/2 → but with min_periods=1
        # Actually for 3-element centered: [1.5, 2.0, 3.0, 4.0, 4.5]
        assert abs(result[2] - 3.0) < 1e-6

    def test_enforces_min_window(self):
        """Even small windows should be bumped to at least 3."""
        x = np.array([1.0, 2.0, 1.0, 2.0])
        result = smooth(x, win=1)  # should become 3
        assert len(result) == 4


class TestAddDerivedSignals:
    def test_adds_wheel_speed(self):
        df = pd.DataFrame(
            {
                "ICU_Speedfl_kmh": [30.0, 40.0],
                "ICU_Speedfr_kmh": [32.0, 42.0],
                "ICU_Speedrl_kmh": [31.0, 41.0],
                "ICU_Speedrr_kmh": [31.0, 41.0],
            }
        )
        result = add_derived_signals(df)
        assert "WHEEL_SPEED_kmh" in result.columns
        # mean of 30,32,31,31 = 31.0
        assert abs(result["WHEEL_SPEED_kmh"].iloc[0] - 31.0) < 0.1
