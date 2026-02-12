from __future__ import annotations

"""Validation routines comparing IMU longitudinal acceleration to dV/dt.

Implements three tasks:
- vs GPS velocity
- vs wheel speed
- longitudinal combined evaluation writing metrics and plots for both sources
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .signals import smooth


def validate_accel_vs_velocity(df: pd.DataFrame, out_dir: Path) -> None:
    """Validate IMU longitudinal acceleration against dV/dt from GPS velocity.

    Produces a metrics CSV and a plot overlay plus scatter regression in
    `out_dir/accel_velocity_consistency_metrics.csv` and
    `out_dir/accel_velocity_consistency.png`.

    Args:
        df: DataFrame containing `t_s`, `GPS1_VEL_MAG_ms`, IMU axes, and
            optionally `ICU_SteeringAngle_deg`.
        out_dir: Output directory to write metrics and plots.
    """
    required = ["t_s", "GPS1_VEL_MAG_ms", "IMU_ACCEL_X_ms2", "IMU_ACCEL_Y_ms2", "IMU_ACCEL_Z_ms2"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    steer = df["ICU_SteeringAngle_deg"].to_numpy(dtype=float) if "ICU_SteeringAngle_deg" in df.columns else None
    d = df.loc[:, required + (["ICU_SteeringAngle_deg"] if steer is not None else [])].dropna().copy().sort_values("t_s")

    t = d["t_s"].to_numpy(dtype=float)
    v = d["GPS1_VEL_MAG_ms"].to_numpy(dtype=float)
    ax = d["IMU_ACCEL_X_ms2"].to_numpy(dtype=float)
    ay = d["IMU_ACCEL_Y_ms2"].to_numpy(dtype=float)
    az = d["IMU_ACCEL_Z_ms2"].to_numpy(dtype=float)
    steer_d = d["ICU_SteeringAngle_deg"].to_numpy(dtype=float) if steer is not None else None

    if len(t) < 5:
        raise RuntimeError("Not enough samples to validate acceleration vs velocity")

    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.05
    win = max(3, int(round(0.5 / dt_med)))

    v_s = smooth(v, win)

    v_th_zero = 0.05
    zero_mask_all = v_s < v_th_zero
    if zero_mask_all.any():
        zero_mask = pd.Series(zero_mask_all).rolling(window=max(3, win), center=True, min_periods=1).sum().to_numpy() >= max(3, win) * 0.8
    else:
        zero_mask = zero_mask_all
    if zero_mask.sum() < 50:
        zero_mask = v_s < 0.1

    if zero_mask.sum() >= 20:
        gx = float(np.nanmedian(ax[zero_mask])); gy = float(np.nanmedian(ay[zero_mask])); gz = float(np.nanmedian(az[zero_mask]))
    else:
        gx = float(np.nanmedian(ax)); gy = float(np.nanmedian(ay)); gz = float(np.nanmedian(az))

    ax_cal = ax - gx; ay_cal = ay - gy; az_cal = az - gz
    a_long = ax_cal
    a_s = smooth(a_long, win)
    dvdt = np.gradient(v_s, t)

    mask = np.isfinite(t) & np.isfinite(dvdt) & np.isfinite(a_s)
    t_m = t[mask]; dvdt_m = dvdt[mask]; a_m = a_s[mask]; v_m = v_s[mask]

    move_mask = v_m > 0.5
    straight_mask = np.ones_like(move_mask, dtype=bool)
    if steer_d is not None:
        steer_m = steer_d[mask]
        straight_mask = np.abs(steer_m) <= 5.0
    sel = move_mask & straight_mask
    if sel.sum() >= 50:
        t_m = t_m[sel]; dvdt_m = dvdt_m[sel]; a_m = a_m[sel]

    if len(dvdt_m) < 5:
        raise RuntimeError("Insufficient valid samples after masking for comparison")

    corr = float(np.corrcoef(dvdt_m, a_m)[0, 1])
    A = np.vstack([a_m, np.ones_like(a_m)])
    slope, intercept = np.linalg.lstsq(A.T, dvdt_m, rcond=None)[0]
    rmse = float(np.sqrt(np.mean((dvdt_m - a_m) ** 2)))
    rmse_rel = float(rmse / max(1e-6, np.nanstd(a_m)))
    bias = float(np.mean(dvdt_m - a_m))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        dict(
            samples=int(len(dvdt_m)),
            median_dt_s=dt_med,
            smooth_window_samples=win,
            zero_vel_thresh_ms=v_th_zero,
            bias_ax=gx,
            bias_ay=gy,
            bias_az=gz,
            corr=corr,
            slope=slope,
            intercept=intercept,
            rmse_abs=rmse,
            rmse_rel=rmse_rel,
            bias=bias,
        )
    ]).to_csv(out_dir / "accel_velocity_consistency_metrics.csv", index=False)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"hspace": 0.25})
    ax1.plot(t_m, dvdt_m, label="dV/dt [m/s²]", color="tab:blue", linewidth=1.2)
    ax1.plot(t_m, a_m, label="a_x IMU (cal) [m/s²]", color="tab:orange", linewidth=1.0, alpha=0.9)
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("Acceleration [m/s²]"); ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right")

    min_a = float(np.nanmin([a_m.min(), dvdt_m.min()])); max_a = float(np.nanmax([a_m.max(), dvdt_m.max()]))
    xline = np.linspace(min_a, max_a, 100)
    ax2.scatter(a_m, dvdt_m, s=6, alpha=0.5, color="tab:blue", label="samples")
    ax2.plot(xline, xline, color="tab:green", linestyle="--", linewidth=1.0, label="y = x")
    ax2.plot(xline, slope * xline + intercept, color="tab:red", linewidth=1.0, label=f"fit: y={slope:.3f}x+{intercept:.3f}")
    ax2.set_xlabel("a_x IMU (cal) [m/s²]"); ax2.set_ylabel("dV/dt [m/s²]"); ax2.grid(True, alpha=0.3); ax2.legend(loc="upper left")
    ax2.set_title(f"corr={corr:.3f}, rmse={rmse:.3f} m/s², bias={bias:.3f} m/s²")

    fig.tight_layout(); fig.savefig(out_dir / "accel_velocity_consistency.png", dpi=200); plt.close(fig)


def validate_accel_vs_wheelspeed(df: pd.DataFrame, out_dir: Path) -> None:
    """Validate IMU longitudinal acceleration against dV/dt from wheel speed.

    Writes metrics to `accel_wheelspeed_consistency_metrics.csv` and a plot
    `accel_wheelspeed_consistency.png` in `out_dir`.

    Args:
        df: DataFrame with `t_s`, `WHEEL_SPEED_kmh`, IMU axes, and optional
            `ICU_SteeringAngle_deg`.
        out_dir: Output directory to write metrics and plots.
    """
    required = ["t_s", "WHEEL_SPEED_kmh", "IMU_ACCEL_X_ms2", "IMU_ACCEL_Y_ms2", "IMU_ACCEL_Z_ms2"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    steer = df["ICU_SteeringAngle_deg"].to_numpy(dtype=float) if "ICU_SteeringAngle_deg" in df.columns else None
    d = df.loc[:, required + (["ICU_SteeringAngle_deg"] if steer is not None else [])].dropna().copy().sort_values("t_s")

    t = d["t_s"].to_numpy(dtype=float)
    w_kmh = d["WHEEL_SPEED_kmh"].to_numpy(dtype=float)
    v = w_kmh * (1000.0 / 3600.0)
    ax = d["IMU_ACCEL_X_ms2"].to_numpy(dtype=float)
    ay = d["IMU_ACCEL_Y_ms2"].to_numpy(dtype=float)
    az = d["IMU_ACCEL_Z_ms2"].to_numpy(dtype=float)
    steer_d = d["ICU_SteeringAngle_deg"].to_numpy(dtype=float) if steer is not None else None

    if len(t) < 5:
        raise RuntimeError("Not enough samples to validate acceleration vs wheelspeed")

    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.05
    win = max(3, int(round(0.5 / dt_med)))

    v_s = smooth(v, win)

    v_th_zero = 0.05
    zero_mask_all = v_s < v_th_zero
    if zero_mask_all.any():
        zero_mask = pd.Series(zero_mask_all).rolling(window=max(3, win), center=True, min_periods=1).sum().to_numpy() >= max(3, win) * 0.8
    else:
        zero_mask = zero_mask_all
    if zero_mask.sum() < 50:
        zero_mask = v_s < 0.3

    if zero_mask.sum() >= 20:
        gx = float(np.nanmedian(ax[zero_mask])); gy = float(np.nanmedian(ay[zero_mask])); gz = float(np.nanmedian(az[zero_mask]))
    else:
        gx = float(np.nanmedian(ax)); gy = float(np.nanmedian(ay)); gz = float(np.nanmedian(az))

    ax_cal = ax - gx; ay_cal = ay - gy; az_cal = az - gz
    a_long = ax_cal
    a_s = smooth(a_long, win)
    dvdt = np.gradient(v_s, t)

    mask = np.isfinite(t) & np.isfinite(dvdt) & np.isfinite(a_s)
    t_m = t[mask]; dvdt_m = dvdt[mask]; a_m = a_s[mask]
    v_m = v_s[mask]

    move_mask = v_m > 0.5
    straight_mask = np.ones_like(move_mask, dtype=bool)
    if steer_d is not None:
        steer_m = steer_d[mask]
        straight_mask = np.abs(steer_m) <= 5.0
    sel = move_mask & straight_mask
    if sel.sum() >= 50:
        t_m = t_m[sel]; dvdt_m = dvdt_m[sel]; a_m = a_m[sel]

    if len(dvdt_m) < 5:
        raise RuntimeError("Insufficient valid samples after masking for wheelspeed comparison")

    corr = float(np.corrcoef(dvdt_m, a_m)[0, 1])
    A = np.vstack([a_m, np.ones_like(a_m)])
    slope, intercept = np.linalg.lstsq(A.T, dvdt_m, rcond=None)[0]
    rmse = float(np.sqrt(np.mean((dvdt_m - a_m) ** 2)))
    rmse_rel = float(rmse / max(1e-6, np.nanstd(a_m)))
    bias = float(np.mean(dvdt_m - a_m))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        dict(
            samples=int(len(dvdt_m)),
            median_dt_s=dt_med,
            smooth_window_samples=win,
            zero_vel_thresh_ms=v_th_zero,
            bias_ax=gx,
            bias_ay=gy,
            bias_az=gz,
            corr=corr,
            slope=slope,
            intercept=intercept,
            rmse_abs=rmse,
            rmse_rel=rmse_rel,
            bias=bias,
        )
    ]).to_csv(out_dir / "accel_wheelspeed_consistency_metrics.csv", index=False)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"hspace": 0.25})
    ax1.plot(t_m, dvdt_m, label="dV/dt (wheel) [m/s²]", color="tab:blue", linewidth=1.2)
    ax1.plot(t_m, a_m, label="a_x IMU (cal) [m/s²]", color="tab:orange", linewidth=1.0, alpha=0.9)
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("Acceleration [m/s²]"); ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right")

    min_a = float(np.nanmin([a_m.min(), dvdt_m.min()])); max_a = float(np.nanmax([a_m.max(), dvdt_m.max()]))
    xline = np.linspace(min_a, max_a, 100)
    ax2.scatter(a_m, dvdt_m, s=6, alpha=0.5, color="tab:blue", label="samples")
    ax2.plot(xline, xline, color="tab:green", linestyle="--", linewidth=1.0, label="y = x")
    ax2.plot(xline, slope * xline + intercept, color="tab:red", linewidth=1.0, label=f"fit: y={slope:.3f}x+{intercept:.3f}")
    ax2.set_xlabel("a_x IMU (cal) [m/s²]"); ax2.set_ylabel("dV/dt (wheel) [m/s²]"); ax2.grid(True, alpha=0.3); ax2.legend(loc="upper left")
    ax2.set_title(f"corr={corr:.3f}, rmse={rmse:.3f} m/s², bias={bias:.3f} m/s²")

    fig.tight_layout(); fig.savefig(out_dir / "accel_wheelspeed_consistency.png", dpi=200); plt.close(fig)


def validate_accel_longitudinal(df: pd.DataFrame, out_dir: Path) -> None:
    """Validate longitudinal consistency vs GPS and wheel speed concurrently.

    Produces metrics+plots for both sources using the same smoothing and
    calibration window.

    Args:
        df: DataFrame with required columns for both sources.
        out_dir: Output directory path (Outputs/03_Drag_Calculation/plots).
    """
    validate_accel_vs_velocity(df, out_dir)
    validate_accel_vs_wheelspeed(df, out_dir)
