#!/usr/bin/env python3
"""Estimate steering zero-offset and assess its impact on IMU longitudinal acceleration.

- Uses straight-line coastdown intervals to estimate the mean steering angle when the car is going straight.
- Optionally uses lateral acceleration vs. steering regression to infer an offset.
- Recomputes IMU-based longitudinal acceleration with and without applying the offset to steering and compares them.

Usage (from repo root):
  python Code/Calibration/steering_offset_analysis.py

Outputs:
  - Prints estimated steering offsets.
  - Prints correlation and difference statistics between baseline and offset-corrected IMU_ACCEL_LONG_ms2.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure repo root on sys.path for aero_validation imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from aero_validation.io import load_data, repo_root  # type: ignore
from aero_validation.intervals import load_intervals_csv  # type: ignore
from aero_validation.signals import add_derived_signals  # type: ignore


def estimate_offset_from_straights(df: pd.DataFrame, intervals, v_min: float = 5.0) -> float:
    """Estimate steering offset as median steering angle in straight coastdown intervals.

    Uses drag intervals as a proxy for straight-line segments, with optional speed cut.
    """
    if "ICU_SteeringAngle_deg" not in df.columns:
        raise RuntimeError("ICU_SteeringAngle_deg not found in data")

    steer = df["ICU_SteeringAngle_deg"].to_numpy()
    v = df["GPS1_VEL_MAG_ms"].to_numpy() if "GPS1_VEL_MAG_ms" in df.columns else None

    mask = np.zeros(len(df), dtype=bool)
    for i0, i1 in intervals:
        i1_eff = min(len(df), i1)
        mask[i0:i1_eff] = True

    if v is not None:
        mask &= (v >= v_min)

    finite = np.isfinite(steer) & mask
    if finite.sum() == 0:
        raise RuntimeError("No finite steering samples in selected intervals for offset estimation")

    offset = float(np.nanmedian(steer[finite]))
    return offset


def estimate_offset_from_ay(df: pd.DataFrame, v_min: float = 8.0) -> float | None:
    """Estimate steering offset via lateral acceleration regression: a_y ~= K * steer + b.

    Returns delta0 = -b / K when possible, else None.
    """
    cols = ["ICU_SteeringAngle_deg", "IMU_ACCEL_Y_ms2", "GPS1_VEL_MAG_ms"]
    if not all(c in df.columns for c in cols):
        return None

    steer = df["ICU_SteeringAngle_deg"].to_numpy()
    ay = df["IMU_ACCEL_Y_ms2"].to_numpy()
    v = df["GPS1_VEL_MAG_ms"].to_numpy()

    mask = np.isfinite(steer) & np.isfinite(ay) & np.isfinite(v)
    mask &= (v >= v_min)

    # Avoid very large lateral accelerations where simple linearity may break down
    ay_clip = np.clip(ay, -10.0, 10.0)
    mask &= np.isfinite(ay_clip)

    if mask.sum() < 1000:
        return None

    x = steer[mask]
    y = ay_clip[mask]

    # Fit y = K * x + b
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        K, b = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        return None

    if not np.isfinite(K) or abs(K) < 1e-6:
        return None

    delta0 = -b / K
    return float(delta0)


def recompute_longitudinal_with_offset(df_raw: pd.DataFrame, offset_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Recompute IMU_ACCEL_LONG_ms2 with and without applying a steering offset.

    Returns (a_long_base, a_long_offset) aligned to df_raw.index.
    """
    # Baseline: as-is
    df_base = add_derived_signals(df_raw)
    if "IMU_ACCEL_LONG_ms2" not in df_base.columns:
        raise RuntimeError("IMU_ACCEL_LONG_ms2 not available after add_derived_signals")
    a_base = df_base["IMU_ACCEL_LONG_ms2"].to_numpy()

    # Offset-corrected steering
    df_corr = df_raw.copy()
    if "ICU_SteeringAngle_deg" in df_corr.columns:
        df_corr["ICU_SteeringAngle_deg"] = df_corr["ICU_SteeringAngle_deg"] - offset_deg
    df_corr = add_derived_signals(df_corr)
    if "IMU_ACCEL_LONG_ms2" not in df_corr.columns:
        raise RuntimeError("IMU_ACCEL_LONG_ms2 not available after add_derived_signals with corrected steering")
    a_corr = df_corr["IMU_ACCEL_LONG_ms2"].to_numpy()

    return a_base, a_corr


def summarize_difference(a_base: np.ndarray, a_corr: np.ndarray) -> dict:
    mask = np.isfinite(a_base) & np.isfinite(a_corr)
    if mask.sum() == 0:
        return {"n": 0}
    dx = a_corr[mask] - a_base[mask]
    corr = np.corrcoef(a_base[mask], a_corr[mask])[0, 1]
    return {
        "n": int(mask.sum()),
        "mean_diff": float(np.nanmean(dx)),
        "std_diff": float(np.nanstd(dx)),
        "max_abs_diff": float(np.nanmax(np.abs(dx))),
        "corr": float(corr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate steering offset and check its impact on IMU longitudinal acceleration.")
    parser.add_argument("--csv", type=Path, default=None, help="Input CSV (default: Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned.csv if present)")
    parser.add_argument("--intervals", type=Path, default=None, help="Intervals CSV (default: Outputs/02_Run_Extraction/csv/drag_intervals.csv)")
    args = parser.parse_args()

    repo = repo_root()
    default_csv = repo / "Outputs" / "01_Data_Preparation" / "Aero_Validation_Signals_cleaned.csv"
    fallback_csv = repo / "Data" / "Aero_Validation_Signals_cleaned.csv"
    csv_path = args.csv if args.csv is not None else (default_csv if default_csv.exists() else (fallback_csv if fallback_csv.exists() else None))
    if csv_path is None:
        raise SystemExit("No input CSV provided and default cleaned CSV not found")

    intervals_default = repo / "Outputs" / "02_Run_Extraction" / "csv" / "drag_intervals.csv"
    intervals_path = args.intervals if args.intervals is not None else intervals_default

    df_raw = load_data(csv_path)
    intervals, meta, run_ids = load_intervals_csv(intervals_path)

    print(f"Loaded data from: {csv_path}")
    print(f"Loaded intervals from: {intervals_path} (n_intervals={len(intervals)})")

    try:
        offset_straight = estimate_offset_from_straights(df_raw, intervals)
        print(f"Estimated steering offset from straight coastdown segments: {offset_straight:.3f} deg")
    except RuntimeError as e:
        print(f"Offset from straights: {e}")
        offset_straight = None

    offset_ay = estimate_offset_from_ay(df_raw)
    if offset_ay is not None:
        print(f"Estimated steering offset from a_y vs steer regression: {offset_ay:.3f} deg")
    else:
        print("Estimated steering offset from a_y vs steer: not available (insufficient data or columns)")

    # Choose an offset to test impact on IMU longitudinal acceleration
    offset_for_test = None
    if offset_straight is not None:
        offset_for_test = offset_straight
    elif offset_ay is not None:
        offset_for_test = offset_ay

    if offset_for_test is None:
        print("No steering offset available to test impact on IMU_ACCEL_LONG_ms2; exiting.")
        return

    print(f"\nRecomputing IMU_ACCEL_LONG_ms2 with steering offset {offset_for_test:.3f} deg...")
    a_base, a_corr = recompute_longitudinal_with_offset(df_raw, offset_for_test)
    stats = summarize_difference(a_base, a_corr)
    if stats["n"] == 0:
        print("No overlapping finite samples between baseline and corrected longitudinal acceleration.")
        return

    print("Impact on IMU_ACCEL_LONG_ms2:")
    print(f"  n samples:     {stats['n']}")
    print(f"  mean diff:     {stats['mean_diff']:.4f} m/s^2 (corr - base) ")
    print(f"  std diff:      {stats['std_diff']:.4f} m/s^2")
    print(f"  max |diff|:    {stats['max_abs_diff']:.4f} m/s^2")
    print(f"  correlation r: {stats['corr']:.5f}")


if __name__ == "__main__":
    main()
