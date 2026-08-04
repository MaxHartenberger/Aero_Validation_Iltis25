from __future__ import annotations

"""Plotting utilities for runs, overviews, and peak markers.

Functions save figures to provided `out_path` and do not return values.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

__all__ = [
    "plot_run",
    "plot_overview",
    "plot_peaks_overview",
    "plot_overview_runs_subset",
]


def plot_run(
    t,
    v,
    steer,
    brake_f,
    brake_r,
    out_path: Path,
    run_start: Optional[Tuple[float, float]] = None,
    run_end: Optional[Tuple[float, float]] = None,
    steer_thresh: Optional[float] = None,
    w_kmh: Optional[np.ndarray] = None,
    acc_ms2: Optional[np.ndarray] = None,
    acc_series: Optional[List[Tuple[str, np.ndarray]]] = None,
    acc_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot a single run segment: speed, steering, brakes, and optional accel.

    Args:
        t: Time array in seconds.
        v: Speed array in m/s.
        steer: Steering angle array in degrees or None.
        brake_f: Front brake pressure array or None.
        brake_r: Rear brake pressure array or None.
        out_path: Output PNG path.
        run_start: Optional (t, v) marker for start of the run.
        run_end: Optional (t, v) marker for end of the run.
        steer_thresh: Optional steering axis cap for readability.
        w_kmh: Optional wheel speed in km/h to overlay (converted to m/s).
        acc_ms2: Optional longitudinal acceleration to plot.
        acc_series: Optional list of (label, array) acceleration series to plot.
        acc_ylim: Optional common y-axis limits (min, max) for all accel subplots.
    """
    # Build acceleration series list and de-duplicate by label
    series_list: List[Tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    if acc_series:
        for label, arr in acc_series:
            if label not in seen:
                series_list.append((label, np.asarray(arr, dtype=float)))
                seen.add(label)
    if acc_ms2 is not None and "IMU_ACCEL_LONG_ms2" not in seen:
        series_list.append(("IMU_ACCEL_LONG_ms2", np.asarray(acc_ms2, dtype=float)))
        seen.add("IMU_ACCEL_LONG_ms2")

    has_any_accel = len(series_list) > 0
    has_brake = (brake_f is not None) or (brake_r is not None)
    base_rows = 2 + (1 if has_brake else 0)
    nrows = base_rows + (len(series_list) if has_any_accel else 0)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 6 + 2 * (nrows - 2)), sharex=True)
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if has_brake else None

    ax1.plot(t, v, linewidth=1.2, label="GPS1_VEL_MAG [m/s]")
    if w_kmh is not None:
        w_ms = np.asarray(w_kmh, dtype=float) * (1000.0 / 3600.0)
        ax1.plot(t, w_ms, linewidth=1.0, color="tab:purple", alpha=0.9, label="WHEEL_SPEED [m/s]")
    if run_start is not None:
        ax1.scatter([run_start[0]], [run_start[1]], color="red", s=30, zorder=3, label="Run start")
    if run_end is not None:
        ax1.scatter([run_end[0]], [run_end[1]], color="red", s=30, zorder=3, label="Run end")
    ax1.set_ylabel("Speed [m/s]")
    ax1.grid(True, alpha=0.3)
    ax1.margins(x=0)
    ax1.legend()

    if steer is not None:
        ax2.plot(t, steer, linewidth=1.0, color="tab:orange", label="Steering [deg]")
        if steer_thresh is not None and steer_thresh > 0:
            ax2.set_ylim(-abs(steer_thresh), abs(steer_thresh))
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "ICU_SteeringAngle_deg not available", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_ylabel("Steering [deg]")
    ax2.grid(True, alpha=0.3)
    ax2.margins(x=0)

    if has_brake and ax3 is not None:
        if brake_f is not None:
            ax3.plot(t, brake_f, linewidth=1.0, color="tab:red", label="Brake Front [bar]")
        if brake_r is not None:
            ax3.plot(t, brake_r, linewidth=1.0, color="tab:purple", label="Brake Rear [bar]")
        ax3.legend()
        ax3.set_ylabel("Brake [bar]")
        ax3.grid(True, alpha=0.3)
        ax3.margins(x=0)

    accel_start = 3 if has_brake else 2
    if has_any_accel:
        color_cycle = ["tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan", "tab:olive"]
        for idx, (label, arr) in enumerate(series_list):
            ax_a = axes[accel_start + idx]
            arr_f = np.asarray(arr, dtype=float)
            ax_a.plot(t, arr_f, linewidth=1.0, color=color_cycle[idx % len(color_cycle)], label=label)
            finite_mask = np.isfinite(arr_f)
            mean_val = float(np.nanmean(arr_f[finite_mask])) if finite_mask.any() else float("nan")
            ax_a.axhline(mean_val, color=color_cycle[idx % len(color_cycle)], linestyle="--", linewidth=1.0, alpha=0.8, label=f"Mean = {mean_val:.3f} m/s^2")
            if acc_ylim is not None and np.isfinite(acc_ylim[0]) and np.isfinite(acc_ylim[1]):
                ax_a.set_ylim(acc_ylim)
            ax_a.legend(loc="upper right")
            ax_a.set_ylabel("a_long [m/s^2]")
            ax_a.grid(True, alpha=0.3)
            ax_a.margins(x=0)
        axes[-1].set_xlabel("t [s]")
    else:
        (ax3 if has_brake and ax3 is not None else ax2).set_xlabel("t [s]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overview(
    t,
    v,
    intervals: List[Tuple[int, int]],
    out_path: Path,
    steer: Optional[np.ndarray] = None,
    wheel_kmh: Optional[np.ndarray] = None,
    run_ids: Optional[List[int]] = None,
    acc_ms2: Optional[np.ndarray] = None,
    acc_series: Optional[List[Tuple[str, np.ndarray]]] = None,
) -> None:
    """Plot an overview of speed, steering, and acceleration with run shading.

    Shades provided run intervals and annotates them with optional run IDs.

    Args:
        t: Time array.
        v: Speed array (m/s).
        intervals: List of (start, end) index pairs.
        out_path: Output PNG path.
        steer: Optional steering angle array.
        wheel_kmh: Optional wheel speed array (km/h).
        run_ids: Optional list of numeric IDs per interval.
        acc_ms2: Optional IMU longitudinal acceleration array (m/s^2). If provided,
            this is used instead of computing acceleration from the velocity gradient.
    """
    # Build acceleration series list and de-duplicate by label
    series_list: List[Tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    if acc_series:
        for label, arr in acc_series:
            if label not in seen:
                series_list.append((label, np.asarray(arr, dtype=float)))
                seen.add(label)
    if acc_ms2 is not None and "IMU_ACCEL_LONG_ms2" not in seen:
        series_list.append(("IMU_ACCEL_LONG_ms2", np.asarray(acc_ms2, dtype=float)))
        seen.add("IMU_ACCEL_LONG_ms2")

    # Fallback: if no provided acceleration series, compute from velocity
    use_fallback_acc = len(series_list) == 0
    nrows = 2 + (len(series_list) if not use_fallback_acc else 1)
    fig, axes = plt.subplots(nrows, 1, figsize=(60, 8 + 3 * (nrows - 2)), sharex=True, gridspec_kw={"hspace": 0.15})
    ax_v, ax_s = axes[0], axes[1]

    line_v, = ax_v.plot(t, v, linewidth=1.2, color="tab:blue", label="GPS1_VEL_MAG [m/s]")

    if steer is not None:
        line_s, = ax_s.plot(t, steer, linewidth=1.0, color="tab:orange", label="Steering [deg]")
    else:
        line_s = None
        ax_s.text(0.5, 0.5, "ICU_SteeringAngle_deg not available", ha="center", va="center", transform=ax_s.transAxes)

    # Acceleration subplots: each series gets its own axis with mean in legend
    color_cycle = ["tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan", "tab:olive"]
    span_axes = [ax_v, ax_s]
    if not use_fallback_acc:
        for idx, (label, arr) in enumerate(series_list):
            ax_a = axes[2 + idx]
            arr_f = np.asarray(arr, dtype=float)
            ax_a.plot(t, arr_f, linewidth=1.0, color=color_cycle[idx % len(color_cycle)], label=label)
            finite_mask = np.isfinite(arr_f)
            mean_val = float(np.nanmean(arr_f[finite_mask])) if finite_mask.any() else float("nan")
            ax_a.axhline(mean_val, color=color_cycle[idx % len(color_cycle)], linestyle="--", linewidth=1.0,
                         alpha=0.8, label=f"Mean = {mean_val:.3f} m/s^2")
            ax_a.legend(loc="upper right")
            ax_a.set_ylabel("Accel [m/s^2]")
            ax_a.grid(True, alpha=0.3)
            span_axes.append(ax_a)
        axes[-1].set_xlabel("t [s]")
    else:
        ax_a = axes[2]
        try:
            acc = np.gradient(np.asarray(v, dtype=float), np.asarray(t, dtype=float))
        except Exception:
            acc = np.full_like(v, np.nan, dtype=float)
        ax_a.plot(t, acc, linewidth=1.0, color="tab:green", label="Accel [m/s^2]")
        finite_mask = np.isfinite(acc)
        mean_val = float(np.nanmean(acc[finite_mask])) if finite_mask.any() else float("nan")
        ax_a.axhline(mean_val, color="tab:green", linestyle="--", linewidth=1.0, alpha=0.8, label=f"Mean = {mean_val:.3f} m/s^2")
        ax_a.legend(loc="upper right")
        ax_a.set_ylabel("Accel [m/s^2]")
        ax_a.grid(True, alpha=0.3)
        axes[-1].set_xlabel("t [s]")
        span_axes.append(ax_a)

    has_intervals = len(intervals) > 0
    for (i0, i1) in intervals:
        t0 = float(t[i0])
        t1 = float(t[i1 - 1]) if i1 - 1 < len(t) else float(t[-1])
        for ax in span_axes:
            ax.axvspan(t0, t1, color="tab:green", alpha=0.15, linewidth=0)

    if has_intervals:
        ymin, ymax = ax_v.get_ylim()
        ytext = ymax - 0.03 * (ymax - ymin)
        for idx, (i0, i1) in enumerate(intervals, start=1):
            t0 = float(t[i0])
            t1 = float(t[i1 - 1]) if i1 - 1 < len(t) else float(t[-1])
            tmid = 0.5 * (t0 + t1)
            label_id = run_ids[idx - 1] if run_ids is not None and idx - 1 < len(run_ids) else idx
            ax_v.text(tmid, ytext, f"{label_id}", color="tab:green", fontsize=8, ha="center", va="top", alpha=0.9, clip_on=True)

    span_handle = Patch(facecolor="tab:green", alpha=0.15, label="Run intervals")
    if has_intervals:
        ax_v.legend(handles=[h for h in [line_v, span_handle] if h is not None], loc="upper right")
        if line_s is not None:
            ax_s.legend(handles=[line_s, span_handle], loc="upper right")

    ax_v.set_ylabel("Speed [m/s]")
    ax_s.set_ylabel("Steering [deg]")
    for ax in (ax_v, ax_s):
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_peaks_overview(
    t,
    v,
    out_path: Path,
    steer: Optional[np.ndarray] = None,
    brake_f: Optional[np.ndarray] = None,
    brake_r: Optional[np.ndarray] = None,
    peaks: Optional[List[int]] = None,
) -> None:
    """Plot velocity, steering, and braking over time; mark velocity peaks.

    Args:
        t: Time array.
        v: Speed array.
        out_path: Output PNG path.
        steer: Optional steering angle array.
        brake_f: Optional front brake pressure.
        brake_r: Optional rear brake pressure.
        peaks: Optional list of indices in `t`/`v` marking peaks.
    """
    fig, (ax_v, ax_s, ax_b) = plt.subplots(3, 1, figsize=(16, 9), sharex=True, gridspec_kw={"hspace": 0.15})

    ax_v.plot(t, v, linewidth=1.2, color="tab:blue", label="GPS1_VEL_MAG [m/s]")
    if peaks:
        tp = np.array([t[i] for i in peaks if 0 <= i < len(t)], dtype=float)
        vp = np.array([v[i] for i in peaks if 0 <= i < len(v)], dtype=float)
        ax_v.scatter(tp, vp, color="tab:red", s=30, zorder=3, label="Speed peaks")
    ax_v.set_ylabel("Speed [m/s]")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="upper right")

    if steer is not None:
        ax_s.plot(t, steer, linewidth=1.0, color="tab:orange", label="Steering [deg]")
        ax_s.legend(loc="upper right")
    else:
        ax_s.text(0.5, 0.5, "ICU_SteeringAngle_deg not available", ha="center", va="center", transform=ax_s.transAxes)
    ax_s.set_ylabel("Steering [deg]")
    ax_s.grid(True, alpha=0.3)

    has_brake = False
    if brake_f is not None:
        ax_b.plot(t, brake_f, linewidth=1.0, color="tab:red", label="Brake Front [bar]")
        has_brake = True
    if brake_r is not None:
        ax_b.plot(t, brake_r, linewidth=1.0, color="tab:purple", label="Brake Rear [bar]")
        has_brake = True
    if has_brake:
        ax_b.legend(loc="upper right")
    else:
        ax_b.text(0.5, 0.5, "Brake pressure not available", ha="center", va="center", transform=ax_b.transAxes)
    ax_b.set_xlabel("t [s]")
    ax_b.set_ylabel("Brake [bar]")
    ax_b.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overview_runs_subset(
    t,
    v,
    intervals: List[Tuple[int, int]],
    run_ids: List[int],
    out_path: Path,
    steer: Optional[np.ndarray] = None,
    run_ids_all: Optional[List[int]] = None,
) -> None:
    """Plot overview for a subset of runs defined by `run_ids`.

    Crops time axis to min/max of selected intervals and shades them.

    Args:
        t: Time array.
        v: Speed array (m/s).
        intervals: All intervals.
        run_ids: IDs to include in the plot.
        out_path: Output PNG path.
        steer: Optional steering angle array.
        run_ids_all: Optional full list of IDs aligned with `intervals`.
    """
    selected: List[Tuple[int, int]] = []
    if run_ids_all is not None:
        for (i0, i1), rid in zip(intervals, run_ids_all):
            if rid in run_ids:
                selected.append((i0, i1))
    else:
        for rid, (i0, i1) in enumerate(intervals, start=1):
            if rid in run_ids:
                selected.append((i0, i1))

    if not selected:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.text(0.5, 0.5, "No intervals for selected run IDs", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    t0s = [float(t[i0]) for (i0, _) in selected]
    t1s = [float(t[i1 - 1]) if i1 - 1 < len(t) else float(t[-1]) for (_, i1) in selected]
    tmin = min(t0s)
    tmax = max(t1s)
    pad = 0.5 * (tmax - tmin) * 0.05

    fig, (ax_v, ax_s) = plt.subplots(2, 1, figsize=(16, 7), sharex=True, gridspec_kw={"hspace": 0.12})

    ax_v.plot(t, v, linewidth=1.2, color="tab:blue", label="GPS1_VEL_MAG [m/s]")
    ax_v.set_ylabel("Speed [m/s]")
    ax_v.grid(True, alpha=0.3)

    if steer is not None:
        ax_s.plot(t, steer, linewidth=1.0, color="tab:orange", label="Steering [deg]")
        ax_s.set_ylabel("Steering [deg]")
        ax_s.grid(True, alpha=0.3)
    else:
        ax_s.text(0.5, 0.5, "ICU_SteeringAngle_deg not available", ha="center", va="center", transform=ax_s.transAxes)
        ax_s.set_ylabel("Steering [deg]")
        ax_s.grid(True, alpha=0.3)
    ax_s.set_xlabel("t [s]")

    for idx, (i0, i1) in enumerate(intervals, start=1):
        label_id = run_ids_all[idx - 1] if run_ids_all is not None and idx - 1 < len(run_ids_all) else idx
        if label_id not in run_ids:
            continue
        t0 = float(t[i0])
        t1 = float(t[i1 - 1]) if i1 - 1 < len(t) else float(t[-1])
        for ax in (ax_v, ax_s):
            ax.axvspan(t0, t1, color="tab:green", alpha=0.15, linewidth=0)
        ymin, ymax = ax_v.get_ylim()
        ytext = ymax - 0.03 * (ymax - ymin)
        tmid = 0.5 * (t0 + t1)
        ax_v.text(tmid, ytext, f"{label_id}", color="tab:green", fontsize=8, ha="center", va="top", alpha=0.9, clip_on=True)

    ax_s.set_xlim(tmin - pad, tmax + pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
