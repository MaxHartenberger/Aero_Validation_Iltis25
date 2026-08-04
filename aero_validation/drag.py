from __future__ import annotations

"""Drag coefficient computation: regression, metrics, and configuration.

This module contains the core scientific logic for coastdown drag analysis:
- Vehicle constants loading from JSON
- Robust linear regression (single and multi-variable)
- Run data slicing and preparation
- Drag coefficient derivation (CdA, Cd, Cr, Fd)
- Summary table generation and per-config contribution analysis.

These functions were originally in Code/03_Drag_Calculation/common.py and have been
consolidated here so they can be imported cleanly from the aero_validation package
without sys.path hacks.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .io import repo_root

__all__ = [
    "VehicleConstants",
    "load_constants",
    "assign_config",
    "robust_linear_fit",
    "robust_linear_fit_multi",
    "prepare_run_slice",
    "derive_drag_metrics",
    "fd_table_from_summary",
    "summarize_by_config",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleConstants:
    """Immutable vehicle and analysis parameters for drag computation."""
    mass_kg: float
    rho_kgm3: float
    frontal_area_m2_by_config: Dict[str, float]
    fd_speeds_ms: List[float]
    baseline_config: str
    config_ranges_by_run_id: Dict[str, Tuple[int, int]]


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_constants(path: Path) -> VehicleConstants:
    """Load vehicle constants from a JSON file.

    The JSON must contain ``vehicle`` and ``analysis`` sections with the
    required fields. If ``config_ranges_by_run_id`` is missing or empty, an
    error is raised rather than silently applying defaults.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    vehicle = raw.get("vehicle", {})
    analysis = raw.get("analysis", {})

    mass = float(vehicle.get("mass_kg", 280.0))
    rho = float(vehicle.get("rho_kgm3", 1.225))
    area_map = dict(vehicle.get("frontal_area_m2_by_config", {}))

    speeds = [float(v) for v in analysis.get("fd_speeds_ms", [10.0, 15.0, 20.0])]
    baseline = str(analysis.get("baseline_config", "no_front_rear"))

    # Warn on defaults that may silently affect results
    _warn_if_default(vehicle, "mass_kg", 280.0, path)
    _warn_if_default(vehicle, "rho_kgm3", 1.225, path)
    _warn_if_default(vehicle, "frontal_area_m2_by_config", {}, path)
    _warn_if_default(analysis, "baseline_config", "no_front_rear", path)
    _warn_if_default(analysis, "fd_speeds_ms", [10.0, 15.0, 20.0], path)

    ranges_raw = analysis.get("config_ranges_by_run_id", {})
    ranges: Dict[str, Tuple[int, int]] = {}
    for k, v in ranges_raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            ranges[str(k)] = (int(v[0]), int(v[1]))

    if not ranges:
        raise ValueError(
            "config_ranges_by_run_id is missing or empty in the constants JSON. "
            "This field maps configuration names to run ID ranges and is required "
            "to assign each run to its aerodynamic configuration. "
            f"File: {path}"
        )

    return VehicleConstants(
        mass_kg=mass,
        rho_kgm3=rho,
        frontal_area_m2_by_config=area_map,
        fd_speeds_ms=speeds,
        baseline_config=baseline,
        config_ranges_by_run_id=ranges,
    )


def assign_config(run_id: int, config_ranges_by_run_id: Dict[str, Tuple[int, int]]) -> str:
    """Return the configuration name for a given run ID.

    Args:
        run_id: The numeric run identifier.
        config_ranges_by_run_id: Mapping from config name to (start_id, end_id)
            inclusive ranges.

    Returns:
        The configuration name, or ``"unknown"`` if no range contains the run ID.
    """
    for name, (start, end) in config_ranges_by_run_id.items():
        if start <= run_id <= end:
            return name
    return "unknown"


def _warn_if_default(data: dict, key: str, default: object, path: Path) -> None:
    """Emit a warning if *key* is missing from *data* and a default is used."""
    if key not in data:
        warnings.warn(
            f"'{key}' is missing from constants file {path}; "
            f"using default {default!r}. Results may not reflect the actual vehicle.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Robust regression
# ---------------------------------------------------------------------------

def robust_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 5,
    sigma_thresh: float = 2.0,
) -> Tuple[float, float, float, int]:
    """Fit ``y = m*x + b`` with iterative outlier rejection.

    After an initial OLS fit, points with residuals exceeding
    ``sigma_thresh * std(residuals)`` are removed and the fit is recomputed,
    up to ``max_iter`` times.

    Args:
        x: Independent variable (1-D array).
        y: Dependent variable (1-D array).
        max_iter: Maximum outlier-rejection iterations.
        sigma_thresh: Residual threshold in units of standard deviation.

    Returns:
        Tuple ``(slope, intercept, r_squared, n_points_used)``.
    """
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


def robust_linear_fit_multi(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 5,
    sigma_thresh: float = 2.0,
) -> Tuple[np.ndarray, float, int]:
    """Fit ``y = X @ beta`` with iterative outlier rejection.

    Multivariate version of :func:`robust_linear_fit`. Uses least-squares
    (``np.linalg.lstsq``) instead of ``np.polyfit``.

    Args:
        X: Design matrix (2-D array, shape ``(n_samples, n_features)``).
        y: Dependent variable (1-D array).
        max_iter: Maximum outlier-rejection iterations.
        sigma_thresh: Residual threshold in units of standard deviation.

    Returns:
        Tuple ``(beta, r_squared, n_points_used)``.
    """
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


# ---------------------------------------------------------------------------
# Run data preparation
# ---------------------------------------------------------------------------

def prepare_run_slice(
    run_csv: Path,
    t0: float,
    t1: float,
    *,
    min_v_ms: float,
    window_median: int = 7,
) -> pd.DataFrame:
    """Load and filter a single coastdown run slice.

    Reads a per-run CSV, filters to the time window ``[t0, t1]``, removes
    samples with throttle or brake activity, applies a minimum speed cut,
    computes positive deceleration (``-IMU_ACCEL_LONG_ms2``), optionally
    smooths with a rolling median, and returns ``v`` and ``v²`` columns.

    Args:
        run_csv: Path to the per-run CSV (must contain ``t_s``,
            ``GPS1_VEL_MAG_ms``, ``IMU_ACCEL_LONG_ms2``).
        t0: Start time in seconds.
        t1: End time in seconds.
        min_v_ms: Minimum speed threshold in m/s.
        window_median: Rolling median window size (samples). Set to 0 or 1 to
            disable.

    Returns:
        DataFrame with columns ``t_s``, ``GPS1_VEL_MAG_ms``, ``v``, ``v2``,
        ``decel_ms2``.
    """
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
        sl["decel_ms2"] = (
            sl["decel_ms2"]
            .rolling(window=window_median, center=True)
            .median()
            .bfill()
            .ffill()
        )

    sl["v"] = sl["GPS1_VEL_MAG_ms"]
    sl["v2"] = sl["v"] ** 2
    return sl[["t_s", "GPS1_VEL_MAG_ms", "v", "v2", "decel_ms2"]]


# ---------------------------------------------------------------------------
# Drag metrics derivation
# ---------------------------------------------------------------------------

def derive_drag_metrics(
    *,
    k2: float,
    c: float,
    mass_kg: float,
    rho_kgm3: float,
    area_m2: float | None,
) -> Dict[str, float]:
    """Derive physical drag metrics from the regression coefficients.

    The coastdown model is:  decel = k2 * v² + c

    Where:
    - ``k2 = (ρ * CdA) / (2 * m)``  →  ``CdA = 2 * m * k2 / ρ``
    - ``c = Cr * g``                →  ``Cr = c / g``
    - ``Cd = CdA / A_ref``

    Also computes drag force at reference speeds.

    Args:
        k2: Quadratic (v²) coefficient from regression [m⁻¹].
        c: Constant term from regression [m/s²].
        mass_kg: Vehicle mass [kg].
        rho_kgm3: Air density [kg/m³].
        area_m2: Reference frontal area [m²] (may be None, in which case Cd
            is NaN).

    Returns:
        Dict with keys ``CdA_m2``, ``Cd``, ``Cr``, ``Fd_10ms_N``,
        ``Fd_15ms_N``, ``Fd_20ms_N``.
    """
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


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def fd_table_from_summary(
    summary_df: pd.DataFrame,
    *,
    rho_kgm3: float,
    speeds_ms: List[float],
) -> pd.DataFrame:
    """Build a drag-force table from a per-config summary DataFrame.

    Finds the mean CdA column (``CdA_m2_mean``), then computes
    ``Fd = 0.5 * ρ * CdA * v²`` for each configuration and each speed in
    ``speeds_ms``.

    Args:
        summary_df: DataFrame with at least ``config`` and a ``CdA_m2_mean``
            column.
        rho_kgm3: Air density [kg/m³].
        speeds_ms: List of reference speeds [m/s].

    Returns:
        DataFrame with columns ``config``, ``v_ms``, ``Fd_N``.
    """
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
    """Compute per-configuration summary statistics and wing-contribution table.

    Groups per-run coefficients by configuration, computes mean/median/count/
    std/var for CdA and Cr, computes delta vs baseline, and derives wing
    contributions (front wing, rear wing, both wings).

    Args:
        coeffs_df: DataFrame with per-run columns ``config``, ``CdA_m2``,
            ``Cr``, ``r2``, ``n_points``.
        baseline_config: Name of the baseline (no-wing) configuration.

    Returns:
        Tuple ``(stats_df, contributions_df)`` where:
        - ``stats_df`` has per-config mean/median/count/std/var and
          ``delta_CdA_vs_base``.
        - ``contributions_df`` has wing-component CdA contributions.
    """
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
        raise ValueError(
            f"Baseline '{baseline_config}' missing in summary; check config mapping"
        )
    cdA_base = float(baseline["CdA_m2_mean"].iloc[0])
    stats["delta_CdA_vs_base"] = stats["CdA_m2_mean"] - cdA_base

    idx = {str(r["config"]): r for _, r in stats.iterrows()}

    def cdA(cfg: str) -> float:
        return float(idx[cfg]["CdA_m2_mean"]) if cfg in idx else float("nan")

    contrib = pd.DataFrame(
        [
            {
                "component": "front_wing",
                "CdA_m2": cdA("no_rear") - cdA(baseline_config),
            },
            {
                "component": "rear_wing",
                "CdA_m2": cdA("no_front") - cdA(baseline_config),
            },
            {
                "component": "both_wings",
                "CdA_m2": cdA("full") - cdA(baseline_config),
            },
        ]
    )
    return stats, contrib
