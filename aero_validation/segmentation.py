from __future__ import annotations

"""Run segmentation utilities for drag and general movement.

Implements speed peak detection and straight-line drag-run segmentation with
thresholds for steering, braking, re-acceleration, and stop conditions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "DragRunConfig",
    "segment_runs",
    "detect_drag_peaks",
    "segment_drag_runs",
    "segment_drag_runs_with_debug",
]


@dataclass
class DragRunConfig:
    """Configuration for drag-run segmentation.

    All fields have sensible defaults and can be overridden individually.
    """
    steer_thresh: float = 12.0
    straight_ratio: float = 0.70
    min_peak_speed: float = 1.0
    peak_window_s: float = 0.5
    peak_prominence: float = 0.1
    reaccel_eps: float = 0.2
    accel_window_s: float = 0.5
    brake_thresh: float = 0.05
    brake_spike_thresh: float = 0.1
    turn_window_s: float = 0.2
    min_dur_s: float = 2.0
    max_dur_s: float = 10.0
    stop_speed_th: float = 0.2
    smooth: bool = True

    def to_dict(self) -> dict:
        """Convert to a plain dict for backward-compatible config passing."""
        return {
            "steer_thresh": self.steer_thresh,
            "straight_ratio": self.straight_ratio,
            "min_peak_speed": self.min_peak_speed,
            "peak_window_s": self.peak_window_s,
            "peak_prominence": self.peak_prominence,
            "reaccel_eps": self.reaccel_eps,
            "accel_window_s": self.accel_window_s,
            "brake_thresh": self.brake_thresh,
            "brake_spike_thresh": self.brake_spike_thresh,
            "turn_window_s": self.turn_window_s,
            "min_dur_s": self.min_dur_s,
            "max_dur_s": self.max_dur_s,
            "stop_speed_th": self.stop_speed_th,
            "smooth": self.smooth,
        }


def segment_runs(t, v) -> List[Tuple[int, int]]:
    """Segment contiguous moving intervals separated by sufficiently long stops.

    Uses light smoothing of velocity to detect moving and stopped phases,
    returns half-open index intervals `[start, end)` for runs longer than 5 s.

    Args:
        t: Time vector.
        v: Speed vector.

    Returns:
        List of `(start_idx, end_idx)` intervals with `end_idx` exclusive.
    """
    v_series = pd.Series(v)
    v_smooth = v_series.rolling(window=21, min_periods=1, center=True).mean().to_numpy()

    vmax = float(np.nanmax(v_smooth)) if np.isfinite(v_smooth).any() else 0.0
    move_th = max(0.05, 0.01 * vmax)
    stop_th = max(0.05, 0.01 * vmax)
    stop_min_dur = 1.0
    run_min_dur = 5.0

    moving = v_smooth > move_th
    stopped = v_smooth < stop_th

    n = len(t)
    stop_edges: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if stopped[i]:
            j = i + 1
            while j < n and stopped[j]:
                j += 1
            if (t[j - 1] - t[i]) >= stop_min_dur:
                stop_edges.append((i, j))
            i = j
        else:
            i += 1

    intervals: List[Tuple[int, int]] = []
    start = 0
    for si, sj in stop_edges:
        if si > start:
            if (t[si - 1] - t[start]) >= run_min_dur and moving[start:si].any():
                intervals.append((start, si))
        start = sj
    if start < n - 1:
        if (t[-1] - t[start]) >= run_min_dur and moving[start:].any():
            intervals.append((start, n))

    return intervals


def detect_drag_peaks(t, v, config: Optional[dict] = None) -> List[int]:
    """Identify speed peaks relevant for drag-run segmentation.

    Applies optional smoothing and adaptive prominence and minimum speed
    thresholds. Combines local maxima with derivative sign-change points.

    Args:
        t: Time vector.
        v: Speed vector.
        config: Optional dict with keys like `peak_window_s`, `min_peak_speed`,
            `peak_prominence`, and `smooth`.

    Returns:
        Sorted unique indices of candidate peaks.
    """
    if config is None:
        config = {}

    peak_window_s = float(config.get("peak_window_s", 0.3))
    min_peak_speed_cfg = config.get("min_peak_speed")
    peak_prominence_cfg = config.get("peak_prominence")
    smooth = bool(config.get("smooth", True))
    dv_pos_th = float(config.get("dv_pos_th", 0.02))
    dv_neg_th = float(config.get("dv_neg_th", 0.0))

    t = np.asarray(t)
    v = np.asarray(v)
    n = len(t)
    if n < 3:
        return []

    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.05

    def sec_to_samples(sec: float) -> int:
        return max(1, int(round(sec / dt_med)))

    peak_win = sec_to_samples(peak_window_s)
    v_s = pd.Series(v).rolling(window=max(3, peak_win + 1), center=True, min_periods=1).mean().to_numpy() if smooth else np.asarray(v)

    vmax = float(np.nanmax(v_s)) if np.isfinite(v_s).any() else 0.0
    vmin = float(np.nanmin(v_s)) if np.isfinite(v_s).any() else 0.0
    vrange = max(0.0, vmax - vmin)
    min_peak_speed = float(min_peak_speed_cfg) if min_peak_speed_cfg is not None else max(0.005, 0.05 * vmax)
    peak_prominence = float(peak_prominence_cfg) if peak_prominence_cfg is not None else max(0.003, 0.15 * vrange)

    peaks: List[int] = []
    for i in range(1, n - 1):
        if v_s[i] >= v_s[i - 1] and v_s[i] >= v_s[i + 1]:
            i0 = max(0, i - peak_win)
            i1 = min(n, i + peak_win + 1)
            win = v_s[i0:i1]
            prom = float(v_s[i] - np.percentile(win, 60))
            if v_s[i] >= min_peak_speed and prom >= peak_prominence:
                peaks.append(i)

    dv = np.gradient(v_s, t)
    for i in range(1, n):
        if dv[i - 1] > dv_pos_th and dv[i] <= dv_neg_th and v_s[i] >= min_peak_speed:
            peaks.append(i)

    return sorted(set(peaks))


def segment_drag_runs(
    t,
    v,
    steer,
    brake_f=None,
    brake_r=None,
    config: DragRunConfig | dict | None = None,
) -> List[Tuple[int, int]]:
    """Detect straight-line drag runs as half-open intervals.

    Starts from peaks in speed and extends forward while steering remains under
    a threshold, braking is absent, and re-acceleration does not occur beyond
    an epsilon within a short window. Applies duration constraints and a
    straightness gate.

    Args:
        t: Time vector.
        v: Speed vector (m/s recommended).
        steer: Steering angle vector in degrees or None.
        brake_f: Front brake pressure or None.
        brake_r: Rear brake pressure or None.
        config: Optional :class:`DragRunConfig`, dict of thresholds, or None
            to use defaults.

    Returns:
        List of `(start_idx, end_idx)` intervals, `end_idx` exclusive.
    """
    if config is None:
        cfg = DragRunConfig()
    elif isinstance(config, DragRunConfig):
        cfg = config
    else:
        cfg = DragRunConfig(**{k: v for k, v in config.items() if k in DragRunConfig.__dataclass_fields__})

    smooth = cfg.smooth
    steer_thresh = cfg.steer_thresh
    straight_ratio = cfg.straight_ratio
    min_peak_speed = cfg.min_peak_speed
    peak_window_s = cfg.peak_window_s
    peak_prominence = cfg.peak_prominence
    accel_eps = cfg.reaccel_eps
    accel_window_s = cfg.accel_window_s
    brake_thresh = cfg.brake_thresh
    brake_spike_thresh = cfg.brake_spike_thresh
    turn_window_s = cfg.turn_window_s
    min_dur_s = cfg.min_dur_s
    max_dur_s = cfg.max_dur_s
    stop_speed_th = cfg.stop_speed_th

    n = len(t)
    if n < 3:
        return []

    t = np.asarray(t)
    v = np.asarray(v)
    steer = np.asarray(steer) if steer is not None else None
    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.05

    def sec_to_samples(sec: float) -> int:
        return max(1, int(round(sec / dt_med)))

    peak_win = sec_to_samples(peak_window_s)
    accel_win = sec_to_samples(accel_window_s)
    turn_win = sec_to_samples(turn_window_s)

    v_s = pd.Series(v).rolling(window=max(3, peak_win + 1), center=True, min_periods=1).mean().to_numpy() if smooth else np.asarray(v)
    steer_s = np.asarray(steer) if steer is not None else None

    brake_tot = None
    if brake_f is not None or brake_r is not None:
        bf = np.asarray(brake_f) if brake_f is not None else np.zeros(n)
        br = np.asarray(brake_r) if brake_r is not None else np.zeros(n)
        brake_tot = np.maximum(bf, br)
        if smooth:
            brake_tot = pd.Series(brake_tot).rolling(window=5, center=True, min_periods=1).mean().to_numpy()

    peaks: List[int] = detect_drag_peaks(t, v_s, config=dict(
        min_peak_speed=min_peak_speed,
        peak_window_s=peak_window_s,
        peak_prominence=peak_prominence,
        smooth=smooth,
    ))

    intervals: List[Tuple[int, int]] = []
    last_end = -1

    for p in peaks:
        if p <= last_end:
            continue
        start = p
        i = p + 1
        turn_count = 0

        while i < n:
            if v_s[i] <= stop_speed_th:
                i += 1
                break
            if steer_s is not None and abs(float(steer_s[i])) > steer_thresh:
                turn_count += 1
            else:
                turn_count = 0
            if turn_count >= turn_win:
                break
            if brake_tot is not None:
                if brake_tot[i] > brake_thresh:
                    break
                if i > 0:
                    db = (brake_tot[i] - brake_tot[i - 1]) / dt_med
                    if db > brake_spike_thresh:
                        break
            j = min(n - 1, i + accel_win)
            if (v_s[j] - v_s[i] > accel_eps):
                break
            i += 1

        end = max(i, start + 1)
        if steer_s is not None and end > start:
            straight_mask = np.abs(steer_s[start:end]) <= steer_thresh
            if (straight_mask.sum() / float(end - start)) < straight_ratio:
                continue

        dur = float(t[end - 1] - t[start])
        if dur < min_dur_s or dur > max_dur_s:
            continue

        intervals.append((start, end))
        last_end = end

    return intervals


def segment_drag_runs_with_debug(
    t,
    v,
    steer,
    brake_f=None,
    brake_r=None,
    config: DragRunConfig | dict | None = None,
) -> Tuple[List[Tuple[int, int]], List[dict]]:
    """Debug variant of `segment_drag_runs` returning intervals and decision log.

    For each candidate peak, records acceptance/rejection reasons, duration,
    straightness ratio, and end conditions (e.g., stop, turn, brake, reaccel).

    Args:
        t: Time vector.
        v: Speed vector.
        steer: Steering angle vector or None.
        brake_f: Front brake pressure or None.
        brake_r: Rear brake pressure or None.
        config: Optional :class:`DragRunConfig`, dict, or None for defaults.

    Returns:
        Tuple of `(intervals, debug_rows)`; `debug_rows` is a list of dicts per
        peak candidate with fields like `accepted`, `reason`, `start_idx`, etc.
    """
    if config is None:
        cfg = DragRunConfig()
    elif isinstance(config, DragRunConfig):
        cfg = config
    else:
        cfg = DragRunConfig(**{k: v for k, v in config.items() if k in DragRunConfig.__dataclass_fields__})

    smooth = cfg.smooth
    steer_thresh = cfg.steer_thresh
    straight_ratio_th = cfg.straight_ratio
    min_peak_speed = cfg.min_peak_speed
    peak_window_s = cfg.peak_window_s
    peak_prominence = cfg.peak_prominence
    accel_eps = cfg.reaccel_eps
    accel_window_s = cfg.accel_window_s
    brake_thresh = cfg.brake_thresh
    brake_spike_thresh = cfg.brake_spike_thresh
    turn_window_s = cfg.turn_window_s
    min_dur_s = cfg.min_dur_s
    max_dur_s = cfg.max_dur_s
    stop_speed_th = cfg.stop_speed_th

    n = len(t)
    if n < 3:
        return [], []

    t = np.asarray(t)
    v = np.asarray(v)
    steer = np.asarray(steer) if steer is not None else None
    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.05

    def sec_to_samples(sec: float) -> int:
        return max(1, int(round(sec / dt_med)))

    peak_win = sec_to_samples(peak_window_s)
    accel_win = sec_to_samples(accel_window_s)
    turn_win = sec_to_samples(turn_window_s)

    v_s = pd.Series(v).rolling(window=max(3, peak_win + 1), center=True, min_periods=1).mean().to_numpy() if smooth else np.asarray(v)
    steer_s = np.asarray(steer) if steer is not None else None

    brake_tot = None
    if brake_f is not None or brake_r is not None:
        bf = np.asarray(brake_f) if brake_f is not None else np.zeros(n)
        br = np.asarray(brake_r) if brake_r is not None else np.zeros(n)
        brake_tot = np.maximum(bf, br)
        if smooth:
            brake_tot = pd.Series(brake_tot).rolling(window=5, center=True, min_periods=1).mean().to_numpy()

    peaks: List[int] = detect_drag_peaks(t, v_s, config=dict(
        min_peak_speed=min_peak_speed,
        peak_window_s=peak_window_s,
        peak_prominence=peak_prominence,
        smooth=smooth,
    ))

    intervals: List[Tuple[int, int]] = []
    debug_rows: List[dict] = []
    last_end = -1

    for p in peaks:
        row = dict(peak_idx=int(p), t_peak=float(t[p]), accepted=False,
                   reason=None, start_idx=None, end_idx=None, t0=None, t1=None,
                   duration=None, steer_ratio=None, end_reason=None)
        if p <= last_end:
            row["reason"] = "overlap"
            debug_rows.append(row)
            continue

        start = p
        i = p + 1
        turn_count = 0
        end_reason = None

        while i < n:
            if v_s[i] <= stop_speed_th:
                i += 1
                end_reason = "stop"
                break
            if steer_s is not None and abs(float(steer_s[i])) > steer_thresh:
                turn_count += 1
            else:
                turn_count = 0
            if turn_count >= turn_win:
                end_reason = "turn"
                break
            if brake_tot is not None:
                if brake_tot[i] > brake_thresh:
                    end_reason = "brake"
                    break
                if i > 0:
                    db = (brake_tot[i] - brake_tot[i - 1]) / dt_med
                    if db > brake_spike_thresh:
                        end_reason = "brake_spike"
                        break
            j = min(n - 1, i + accel_win)
            if (v_s[j] - v_s[i] > accel_eps):
                end_reason = "reaccel"
                break
            i += 1

        end = max(i, start + 1)
        steer_ratio = None
        if steer_s is not None and end > start:
            straight_mask = np.abs(steer_s[start:end]) <= steer_thresh
            steer_ratio = float(straight_mask.sum()) / float(end - start)
            if steer_ratio < straight_ratio_th:
                row.update(reason="straight", start_idx=int(start), end_idx=int(end), t0=float(t[start]), t1=float(t[end - 1] if end - 1 < len(t) else t[-1]), duration=float(t[end - 1] - t[start]), steer_ratio=steer_ratio, end_reason=end_reason)
                debug_rows.append(row)
                continue

        dur = float(t[end - 1] - t[start])
        if dur < min_dur_s or dur > max_dur_s:
            row.update(reason="duration", start_idx=int(start), end_idx=int(end), t0=float(t[start]), t1=float(t[end - 1] if end - 1 < len(t) else t[-1]), duration=dur, steer_ratio=steer_ratio, end_reason=end_reason)
            debug_rows.append(row)
            continue

        intervals.append((start, end))
    
        last_end = end
        row.update(accepted=True, reason="accepted", start_idx=int(start), end_idx=int(end), t0=float(t[start]), t1=float(t[end - 1] if end - 1 < len(t) else t[-1]), duration=float(t[end - 1] - t[start]), steer_ratio=steer_ratio, end_reason=end_reason)
        debug_rows.append(row)

    return intervals, debug_rows
