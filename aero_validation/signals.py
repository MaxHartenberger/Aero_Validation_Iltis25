from __future__ import annotations

"""Signal processing utilities: column sanitization, derived signals, smoothing.

Provides helpers to normalize raw column headers, compute wheel and GPS speeds,
estimate and apply IMU stationary bias, and produce a longitudinal acceleration
signal aligned to speed derivatives.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def _column_mapping() -> Dict[str, str]:
    """Return a mapping from raw column headers to sanitized names.

    Handles mis-encoded characters (e.g., '�' to '°') and converts to
    snake_case-like canonical names used across the project.
    """
    return {
        "t[s]": "t_s",
        "ICU_Speedfl[km/h]": "ICU_Speedfl_kmh",
        "ICU_Speedfr[km/h]": "ICU_Speedfr_kmh",
        "ICU_Speedrl[km/h]": "ICU_Speedrl_kmh",
        "ICU_Speedrr[km/h]": "ICU_Speedrr_kmh",
        "ICU_Apps[]": "ICU_Apps_pct",
        # Steering angle
        "ICU_SteeringAngle[�]": "ICU_SteeringAngle_deg",
        "ICU_SteeringAngle[°]": "ICU_SteeringAngle_deg",
        # GPS velocities
        "SBG_GPS1_VELOCITY_D[m.s-1]": "GPS1_VEL_D_ms",
        "SBG_GPS1_VELOCITY_E[m.s-1]": "GPS1_VEL_E_ms",
        "SBG_GPS1_VELOCITY_N[m.s-1]": "GPS1_VEL_N_ms",
        # GPS lat/lon
        "SBG_GPS1_LATITUDE[�]": "GPS1_LAT_deg",
        "SBG_GPS1_LATITUDE[°]": "GPS1_LAT_deg",
        "SBG_GPS1_LONGITUDE[�]": "GPS1_LON_deg",
        "SBG_GPS1_LONGITUDE[°]": "GPS1_LON_deg",
        # IMU accelerations
        "SBG_IMU_ACCEL_X[m.s-2]": "IMU_ACCEL_X_ms2",
        "SBG_IMU_ACCEL_Y[m.s-2]": "IMU_ACCEL_Y_ms2",
        "SBG_IMU_ACCEL_Z[m.s-2]": "IMU_ACCEL_Z_ms2",
        # Brakes and R2D
        "ICU_Brakpresf[bar]": "ICU_Brakpresf_bar",
        "ICU_Brakpresr[bar]": "ICU_Brakpresr_bar",
        "ICU_R2D_active[]": "ICU_R2D_active",
    }


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column headers and drop unnamed trailing columns.

    Replaces mis-encoded degree symbols, strips whitespace, and maps known raw
    headers to canonical names via `_column_mapping()`.

    Args:
        df: Input DataFrame.

    Returns:
        A new DataFrame with sanitized columns.
    """
    cols = [c.strip() for c in df.columns]
    df = df.copy()
    df.columns = cols
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]  # drop empty trailing columns

    mapping = _column_mapping()
    new_cols: List[str] = []
    for c in df.columns:
        c_fixed = c.replace("�", "°")
        new_cols.append(mapping.get(c_fixed, c_fixed))
    df.columns = new_cols
    return df


def smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Return a centered rolling-mean smoothed signal.

    Ensures an odd window size of at least 3 samples.

    Args:
        x: Input array.
        win: Window length in samples.

    Returns:
        Smoothed array of the same length as `x`.
    """
    win = max(3, int(win) | 1)
    return pd.Series(x).rolling(window=win, center=True, min_periods=1).mean().to_numpy()


def add_derived_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add wheel speed, GPS speed magnitude, and IMU longitudinal acceleration.

    - `WHEEL_SPEED_kmh`: mean of available wheel speed channels.
    - `GPS1_VEL_MAG_ms`: magnitude from GPS D/E/N components when available.
    - `IMU_ACCEL_LONG_ms2`: longitudinal acceleration from IMU axes after
      stationary-bias removal and axis alignment to match speed derivative.

    Args:
        df: Input DataFrame with raw/sanitized columns.

    Returns:
        Copy of `df` with derived columns added when possible.
    """
    df = df.copy()

    # Wheel speed (km/h)
    wheel_cols = [
        c
        for c in [
            "ICU_Speedfl_kmh",
            "ICU_Speedfr_kmh",
            "ICU_Speedrl_kmh",
            "ICU_Speedrr_kmh",
        ]
        if c in df.columns
    ]
    if wheel_cols:
        df["WHEEL_SPEED_kmh"] = df[wheel_cols].mean(axis=1, skipna=True)

    # GPS speed magnitude (m/s)
    gps_components = [c for c in ["GPS1_VEL_D_ms", "GPS1_VEL_E_ms", "GPS1_VEL_N_ms"] if c in df.columns]
    if len(gps_components) == 3:
        df["GPS1_VEL_MAG_ms"] = np.sqrt((df[gps_components] ** 2).sum(axis=1))

    # Longitudinal acceleration (m/s^2)
    accel_components = [c for c in ["IMU_ACCEL_X_ms2", "IMU_ACCEL_Y_ms2", "IMU_ACCEL_Z_ms2"] if c in df.columns]
    if len(accel_components) == 3 and "t_s" in df.columns:
        ax = df["IMU_ACCEL_X_ms2"].to_numpy()
        ay = df["IMU_ACCEL_Y_ms2"].to_numpy()
        az = df["IMU_ACCEL_Z_ms2"].to_numpy()
        t = df["t_s"].to_numpy()

        # Zero-velocity mask combining GPS and wheel speeds where available
        zero_th = 0.05  # m/s
        zero_mask = None
        v_gps = df["GPS1_VEL_MAG_ms"].to_numpy() if "GPS1_VEL_MAG_ms" in df.columns else None
        v_wh = (df["WHEEL_SPEED_kmh"].to_numpy() * (1000.0 / 3600.0)) if "WHEEL_SPEED_kmh" in df.columns else None
        if v_gps is not None:
            zero_mask = np.isfinite(v_gps) & (v_gps < zero_th)
        if v_wh is not None:
            zero_mask_wh = np.isfinite(v_wh) & (v_wh < zero_th)
            zero_mask = zero_mask_wh if zero_mask is None else (zero_mask & zero_mask_wh)

        # Estimate stationary bias per axis
        min_samples = 20
        if zero_mask is not None and zero_mask.sum() >= min_samples:
            bx = float(np.nanmedian(ax[zero_mask]))
            by = float(np.nanmedian(ay[zero_mask]))
            bz = float(np.nanmedian(az[zero_mask]))
        else:
            bx = float(np.nanmedian(ax))
            by = float(np.nanmedian(ay))
            bz = float(np.nanmedian(az))

        ax_cal = ax - bx
        ay_cal = ay - by
        az_cal = az - bz

        # Build dv/dt target from the best available speed
        v_base = v_gps if v_gps is not None else v_wh
        if v_base is not None and np.isfinite(v_base).any():
            dt = np.diff(t)
            dt_med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.05
            win = max(3, int(round(0.5 / dt_med)))
            v_s = pd.Series(v_base).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
            dvdt = np.gradient(v_s, t)

            move_mask = v_s > 0.5
            straight_mask = np.ones_like(move_mask, dtype=bool)
            if "ICU_SteeringAngle_deg" in df.columns:
                steer = df["ICU_SteeringAngle_deg"].to_numpy()
                straight_mask = np.isfinite(steer) & (np.abs(steer) <= 5.0)

            valid = (
                np.isfinite(dvdt)
                & np.isfinite(ax_cal)
                & np.isfinite(ay_cal)
                & np.isfinite(az_cal)
                & move_mask
                & straight_mask
            )

            if valid.sum() >= 50:
                A = np.vstack([ax_cal[valid], ay_cal[valid], az_cal[valid]]).T
                y = dvdt[valid]
                try:
                    w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                except Exception:
                    w = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                w = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            w = np.array([1.0, 0.0, 0.0], dtype=float)

        df["IMU_ACCEL_LONG_ms2"] = w[0] * ax_cal + w[1] * ay_cal + w[2] * az_cal

    return df
