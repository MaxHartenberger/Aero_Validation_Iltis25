from __future__ import annotations

"""Signal processing utilities: column sanitization, derived signals, smoothing.

Provides helpers to normalize raw column headers, compute wheel and GPS speeds,
estimate and apply IMU stationary bias, and produce a longitudinal acceleration
signal aligned to speed derivatives.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

__all__ = ["sanitize_columns", "add_derived_signals", "smooth"]


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

    Delegates to the private helper functions below. Kept as a single public
    entry point for backward compatibility.

    Args:
        df: Input DataFrame with raw/sanitized columns.

    Returns:
        Copy of `df` with derived columns added when possible.
    """
    df = df.copy()
    _add_wheel_speed(df)
    _add_gps_speed_magnitude(df)
    _add_imu_longitudinal_accel(df)
    return df


# ---------------------------------------------------------------------------
# Private sub-functions (testable individually)
# ---------------------------------------------------------------------------

def _add_wheel_speed(df: pd.DataFrame) -> None:
    """Compute ``WHEEL_SPEED_kmh`` as the row-wise mean of available wheel
    speed channels. Modifies *df* in place."""
    wheel_cols = [
        c for c in [
            "ICU_Speedfl_kmh", "ICU_Speedfr_kmh",
            "ICU_Speedrl_kmh", "ICU_Speedrr_kmh",
        ]
        if c in df.columns
    ]
    if wheel_cols:
        df["WHEEL_SPEED_kmh"] = df[wheel_cols].mean(axis=1, skipna=True)


def _add_gps_speed_magnitude(df: pd.DataFrame) -> None:
    """Compute ``GPS1_VEL_MAG_ms`` from D/E/N components when all three are
    present. Modifies *df* in place."""
    gps = [c for c in ["GPS1_VEL_D_ms", "GPS1_VEL_E_ms", "GPS1_VEL_N_ms"] if c in df.columns]
    if len(gps) == 3:
        df["GPS1_VEL_MAG_ms"] = np.sqrt((df[gps] ** 2).sum(axis=1))


def _add_imu_longitudinal_accel(df: pd.DataFrame) -> None:
    """Compute ``IMU_ACCEL_LONG_ms2`` via stationary-bias removal and
    least-squares axis alignment to the best-available speed derivative.

    Requires ``t_s`` and the three ``IMU_ACCEL_{X,Y,Z}_ms2`` columns.
    Modifies *df* in place.
    """
    accel_components = [c for c in ["IMU_ACCEL_X_ms2", "IMU_ACCEL_Y_ms2", "IMU_ACCEL_Z_ms2"] if c in df.columns]
    if len(accel_components) != 3 or "t_s" not in df.columns:
        return

    ax = df["IMU_ACCEL_X_ms2"].to_numpy()
    ay = df["IMU_ACCEL_Y_ms2"].to_numpy()
    az = df["IMU_ACCEL_Z_ms2"].to_numpy()
    t = df["t_s"].to_numpy()

    zero_mask = _build_zero_velocity_mask(df)

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

    w = _fit_accel_alignment_weights(ax_cal, ay_cal, az_cal, df, t)
    df["IMU_ACCEL_LONG_ms2"] = w[0] * ax_cal + w[1] * ay_cal + w[2] * az_cal


def _build_zero_velocity_mask(df: pd.DataFrame) -> np.ndarray | None:
    """Return a boolean mask of samples where the vehicle is likely stationary.

    Combines GPS speed and wheel speed (converted to m/s) with a threshold of
    0.05 m/s. Returns ``None`` if neither speed source is available.
    """
    zero_th = 0.05  # m/s
    zero_mask = None
    v_gps = df["GPS1_VEL_MAG_ms"].to_numpy() if "GPS1_VEL_MAG_ms" in df.columns else None
    v_wh = (df["WHEEL_SPEED_kmh"].to_numpy() * (1000.0 / 3600.0)) if "WHEEL_SPEED_kmh" in df.columns else None
    if v_gps is not None:
        zero_mask = np.isfinite(v_gps) & (v_gps < zero_th)
    if v_wh is not None:
        zero_mask_wh = np.isfinite(v_wh) & (v_wh < zero_th)
        zero_mask = zero_mask_wh if zero_mask is None else (zero_mask & zero_mask_wh)
    return zero_mask


def _fit_accel_alignment_weights(
    ax_cal: np.ndarray,
    ay_cal: np.ndarray,
    az_cal: np.ndarray,
    df: pd.DataFrame,
    t: np.ndarray,
) -> np.ndarray:
    """Fit weights ``[wx, wy, wz]`` such that ``wx*ax + wy*ay + wz*az`` best
    matches the speed derivative ``dv/dt`` during straight, moving segments.

    Falls back to ``[1, 0, 0]`` (pure X-axis) if there are too few valid
    samples.
    """
    # Best-available speed for dv/dt
    v_gps = df["GPS1_VEL_MAG_ms"].to_numpy() if "GPS1_VEL_MAG_ms" in df.columns else None
    v_wh = (df["WHEEL_SPEED_kmh"].to_numpy() * (1000.0 / 3600.0)) if "WHEEL_SPEED_kmh" in df.columns else None
    v_base = v_gps if v_gps is not None else v_wh

    if v_base is None or not np.isfinite(v_base).any():
        return np.array([1.0, 0.0, 0.0], dtype=float)

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
        & np.isfinite(ax_cal) & np.isfinite(ay_cal) & np.isfinite(az_cal)
        & move_mask & straight_mask
    )

    if valid.sum() >= 50:
        A = np.vstack([ax_cal[valid], ay_cal[valid], az_cal[valid]]).T
        y = dvdt[valid]
        try:
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            return w.astype(float)
        except Exception:
            pass
    return np.array([1.0, 0.0, 0.0], dtype=float)
