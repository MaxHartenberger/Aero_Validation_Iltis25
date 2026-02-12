#!/usr/bin/env python3
"""Combine exported per-channel 10 ms CSVs into a cleaned dataset + derived signals.

Inputs:
- Per-channel CSVs produced by export_channels_10ms_to_signals.py (in Outputs/01_Data_Preparation/Signals)

Outputs:
- Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned_10ms.csv
  (canonical column names; intended to be stable for downstream scripts)

This script is intentionally standalone (does not depend on other data-prep scripts).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Ensure repo root on sys.path for package imports (aero_validation/*)
sys.path.append(str(Path(__file__).resolve().parents[2]))


def list_channel_csvs(channels_dir: Path) -> List[Path]:
    files = [p for p in channels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    files.sort()
    return files


def read_channel_csv(path: Path) -> Tuple[np.ndarray, str, np.ndarray]:
    df = pd.read_csv(path)
    if df.shape[1] < 2 or df.columns[0] != "time":
        raise ValueError(f"Unexpected schema in {path}: columns={list(df.columns)}")
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    col = df.columns[1]
    y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t)
    return t[mask], col, y[mask]


def combine_channels(channels_dir: Path) -> pd.DataFrame:
    paths = list_channel_csvs(channels_dir)
    if not paths:
        raise FileNotFoundError(f"No per-channel CSVs found in: {channels_dir}")

    df_out: Optional[pd.DataFrame] = None
    for p in paths:
        t, col, y = read_channel_csv(p)
        s = pd.Series(y, index=t, name=col)
        if not s.index.is_unique:
            s = s.groupby(level=0).mean()
        s = s.sort_index()
        df_s = s.to_frame()
        if df_out is None:
            df_out = df_s
        else:
            df_out = df_out.join(df_s, how="outer")

    assert df_out is not None
    df_out = df_out.sort_index()
    df_out.index.name = "t_s"
    df_out = df_out.reset_index()
    return df_out


def _rename_extracted_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known extracted channels to canonical names with units."""
    rename_map: Dict[str, str] = {
        # IMU accel
        "SBG_IMU_ACCEL_X": "IMU_ACCEL_X_ms2",
        "SBG_IMU_ACCEL_Y": "IMU_ACCEL_Y_ms2",
        "SBG_IMU_ACCEL_Z": "IMU_ACCEL_Z_ms2",
        # GPS velocity components
        "SBG_GPS1_VELOCITY_D": "GPS1_VEL_D_ms",
        "SBG_GPS1_VELOCITY_E": "GPS1_VEL_E_ms",
        "SBG_GPS1_VELOCITY_N": "GPS1_VEL_N_ms",
        # Apps (often percent)
        "ICU_Apps": "ICU_Apps_pct",
        # Wheel speeds
        "ICU_Speed": "ICU_Speed_kmh",
        "ICU_Speedfl": "ICU_Speedfl_kmh",
        "ICU_Speedfr": "ICU_Speedfr_kmh",
        "ICU_Speedrl": "ICU_Speedrl_kmh",
        "ICU_Speedrr": "ICU_Speedrr_kmh",
        # Steering + brake pressure
        "ICU_SteeringAngle": "ICU_SteeringAngle_deg",
        "ICU_Brakpresr": "ICU_Brakpresr_bar",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def _ensure_gps_speed(df: pd.DataFrame) -> pd.DataFrame:
    if all(c in df.columns for c in ("GPS1_VEL_D_ms", "GPS1_VEL_E_ms", "GPS1_VEL_N_ms")):
        v2 = df[["GPS1_VEL_D_ms", "GPS1_VEL_E_ms", "GPS1_VEL_N_ms"]].astype(float) ** 2
        df["GPS1_VEL_MAG_ms"] = np.sqrt(v2.sum(axis=1))
    return df


def _add_longitudinal_accel(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IMU_ACCEL_LONG_ms2 using the shared project utility if possible."""
    if "IMU_ACCEL_LONG_ms2" in df.columns:
        return df

    try:
        from aero_validation.signals import add_derived_signals  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import aero_validation.signals.add_derived_signals; cannot compute IMU_ACCEL_LONG_ms2"
        ) from e

    df2 = add_derived_signals(df)
    if "IMU_ACCEL_LONG_ms2" not in df2.columns:
        raise RuntimeError(
            "IMU_ACCEL_LONG_ms2 could not be computed. Ensure IMU accel axes are present and renamed "
            "(IMU_ACCEL_X_ms2/IMU_ACCEL_Y_ms2/IMU_ACCEL_Z_ms2), and that t_s exists."
        )
    return df2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Combine per-channel 10 ms exports into a cleaned dataset with derived signals")
    parser.add_argument(
        "--channels-dir",
        default=os.path.join("Outputs", "01_Data_Preparation", "Signals"),
        help="Directory containing per-channel CSVs (default: Outputs/01_Data_Preparation/Signals)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("Outputs", "01_Data_Preparation", "Aero_Validation_Signals_cleaned_10ms.csv"),
        help="Canonical cleaned output CSV path",
    )

    args = parser.parse_args(argv)

    channels_dir = Path(args.channels_dir)
    out = Path(args.output)

    df = combine_channels(channels_dir)
    df = _rename_extracted_channels(df)

    df = _ensure_gps_speed(df)
    df = _add_longitudinal_accel(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
