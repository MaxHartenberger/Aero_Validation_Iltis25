from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "Data"


def outputs_drag_dir() -> Path:
    return repo_root() / "Outputs" / "03_Drag_Calculation"


@dataclass(frozen=True)
class VehicleConstants:
    mass_kg: float
    rho_kgm3: float
    frontal_area_m2_by_config: Dict[str, float]
    fd_speeds_ms: List[float]
    baseline_config: str
    config_ranges_by_run_id: Dict[str, Tuple[int, int]]


def load_constants(path: Path) -> VehicleConstants:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    vehicle = raw.get("vehicle", {})
    analysis = raw.get("analysis", {})

    mass = float(vehicle.get("mass_kg", 280.0))
    rho = float(vehicle.get("rho_kgm3", 1.225))
    area_map = dict(vehicle.get("frontal_area_m2_by_config", {}))

    speeds = [float(v) for v in analysis.get("fd_speeds_ms", [10.0, 15.0, 20.0])]
    baseline = str(analysis.get("baseline_config", "no_front_rear"))

    ranges_raw = analysis.get("config_ranges_by_run_id", {})
    ranges: Dict[str, Tuple[int, int]] = {}
    for k, v in ranges_raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            ranges[str(k)] = (int(v[0]), int(v[1]))

    if not ranges:
        ranges = {
            "full": (1, 12),
            "no_rear": (13, 24),
            "no_front_rear": (25, 36),
            "no_front": (37, 46),
        }

    return VehicleConstants(
        mass_kg=mass,
        rho_kgm3=rho,
        frontal_area_m2_by_config=area_map,
        fd_speeds_ms=speeds,
        baseline_config=baseline,
        config_ranges_by_run_id=ranges,
    )


def assign_config(run_id: int, config_ranges_by_run_id: Dict[str, Tuple[int, int]]) -> str:
    for name, (start, end) in config_ranges_by_run_id.items():
        if start <= run_id <= end:
            return name
    return "unknown"


def robust_linear_fit(x: np.ndarray, y: np.ndarray, *, max_iter: int = 5, sigma_thresh: float = 2.0) -> Tuple[float, float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size < 10:
        raise ValueError("Not enough data points for regression")

    m, b = np.polyfit(x_fit, y_fit, 1)
    for _ in range(max_iter):
        y_pred = m * x_fit + b
        resid = y_fit - y_pred
        std = np.std(resid)
        if std == 0 or not np.isfinite(std):
            break
        new_mask = np.abs(resid) <= sigma_thresh * std
        if new_mask.sum() == new_mask.size:
            break
        x_fit = x_fit[new_mask]
        y_fit = y_fit[new_mask]
        if x_fit.size < 10:
            break
        m, b = np.polyfit(x_fit, y_fit, 1)

    y_pred = m * x_fit + b
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(m), float(b), float(r2), int(x_fit.size)


def robust_linear_fit_multi(X: np.ndarray, y: np.ndarray, *, max_iter: int = 5, sigma_thresh: float = 2.0) -> Tuple[np.ndarray, float, int]:
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_fit = X[mask]
    y_fit = y[mask]
    if y_fit.size < 10:
        raise ValueError("Not enough data points for regression")

    beta, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    for _ in range(max_iter):
        y_pred = X_fit @ beta
        resid = y_fit - y_pred
        std = np.std(resid)
        if std == 0 or not np.isfinite(std):
            break
        new_mask = np.abs(resid) <= sigma_thresh * std
        if new_mask.sum() == new_mask.size:
            break
        X_fit = X_fit[new_mask]
        y_fit = y_fit[new_mask]
        if y_fit.size < 10:
            break
        beta, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)

    y_pred = X_fit @ beta
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return beta.astype(float), float(r2), int(X_fit.shape[0])


def prepare_run_slice(
    run_csv: Path,
    t0: float,
    t1: float,
    *,
    min_v_ms: float,
    window_median: int = 7,
) -> pd.DataFrame:
    df = pd.read_csv(run_csv)
    if "t_s" not in df.columns:
        raise ValueError(f"Run file {run_csv} missing t_s column")
    if "GPS1_VEL_MAG_ms" not in df.columns or "IMU_ACCEL_LONG_ms2" not in df.columns:
        raise ValueError("Run CSV must contain GPS1_VEL_MAG_ms and IMU_ACCEL_LONG_ms2")

    sl = df[(df["t_s"] >= t0) & (df["t_s"] <= t1)].copy()
    if sl.empty:
        raise ValueError("Selected time window produced no rows")

    # Optional filters: throttle & brakes (best effort)
    if "ICU_Apps_pct" in sl.columns:
        sl = sl[sl["ICU_Apps_pct"] <= 1.0]
    if "ICU_Brakpresf_bar" in sl.columns:
        sl = sl[sl["ICU_Brakpresf_bar"] <= 0.1]
    if "ICU_Brakpresr_bar" in sl.columns:
        sl = sl[sl["ICU_Brakpresr_bar"] <= 0.1]

    sl = sl[sl["GPS1_VEL_MAG_ms"] >= float(min_v_ms)]

    sl["decel_ms2"] = (-sl["IMU_ACCEL_LONG_ms2"])  # positive decel
    if window_median and window_median > 1:
        sl["decel_ms2"] = sl["decel_ms2"].rolling(window=window_median, center=True).median().bfill().ffill()

    sl["v"] = sl["GPS1_VEL_MAG_ms"]
    sl["v2"] = sl["v"] ** 2
    return sl[["t_s", "GPS1_VEL_MAG_ms", "v", "v2", "decel_ms2"]]


def derive_drag_metrics(*, k2: float, c: float, mass_kg: float, rho_kgm3: float, area_m2: float | None) -> Dict[str, float]:
    g = 9.80665
    cdA_m2 = (2.0 * mass_kg * k2) / rho_kgm3
    cr = c / g
    cd = float(cdA_m2 / area_m2) if area_m2 and area_m2 > 0 else float("nan")

    def fd(v_ms: float) -> float:
        return 0.5 * rho_kgm3 * cdA_m2 * (v_ms ** 2)

    return {
        "CdA_m2": float(cdA_m2),
        "Cd": float(cd),
        "Cr": float(cr),
        "Fd_10ms_N": float(fd(10.0)),
        "Fd_15ms_N": float(fd(15.0)),
        "Fd_20ms_N": float(fd(20.0)),
    }


def fd_table_from_summary(summary_df: pd.DataFrame, *, rho_kgm3: float, speeds_ms: List[float]) -> pd.DataFrame:
    # Find a mean CdA column
    cdA_col = None
    for c in summary_df.columns:
        if c == "CdA_m2_mean" or c.endswith("CdA_m2_mean"):
            cdA_col = c
            break
    if cdA_col is None:
        # fallback: first column containing 'CdA' and 'mean'
        candidates = [c for c in summary_df.columns if ("CdA" in c and "mean" in c)]
        cdA_col = candidates[0] if candidates else None
    if cdA_col is None:
        raise ValueError("Could not find mean CdA column in summary")

    rows: List[Dict[str, float | str]] = []
    for _, row in summary_df.iterrows():
        cfg = str(row["config"])
        CdA_mean = float(row[cdA_col])
        for v in speeds_ms:
            Fd = 0.5 * rho_kgm3 * CdA_mean * float(v) * float(v)
            rows.append({"config": cfg, "v_ms": float(v), "Fd_N": float(Fd)})
    return pd.DataFrame(rows)


def summarize_by_config(
    coeffs_df: pd.DataFrame,
    *,
    baseline_config: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "config" not in coeffs_df.columns:
        raise ValueError("coeffs_df must contain a 'config' column")
    for c in ("CdA_m2", "Cr", "r2", "n_points"):
        if c not in coeffs_df.columns:
            raise ValueError(f"coeffs_df missing required column '{c}'")

    stats = (
        coeffs_df.groupby("config")[["CdA_m2", "Cr", "r2", "n_points"]]
        .agg(["mean", "median", "count", "std", "var"])
    )
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    stats = stats.reset_index()

    baseline = stats[stats["config"] == baseline_config].copy()
    if baseline.empty:
        raise ValueError(f"Baseline '{baseline_config}' missing in summary; check config mapping")
    cdA_base = float(baseline["CdA_m2_mean"].iloc[0])
    stats["delta_CdA_vs_base"] = stats["CdA_m2_mean"] - cdA_base

    idx = {str(r["config"]): r for _, r in stats.iterrows()}

    def cdA(cfg: str) -> float:
        return float(idx[cfg]["CdA_m2_mean"]) if cfg in idx else float("nan")

    contrib = pd.DataFrame(
        [
            {"component": "front_wing", "CdA_m2": cdA("no_rear") - cdA(baseline_config)},
            {"component": "rear_wing", "CdA_m2": cdA("no_front") - cdA(baseline_config)},
            {"component": "both_wings", "CdA_m2": cdA("full") - cdA(baseline_config)},
        ]
    )
    return stats, contrib
