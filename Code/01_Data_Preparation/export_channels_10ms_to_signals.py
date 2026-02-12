#!/usr/bin/env python3
"""Export selected MF4 channels to per-channel CSVs sampled at 10 ms.

This script is intentionally standalone (does not depend on other data-prep scripts).

It reads channel names from `channels_extraction.txt`, loads `Stint*.mf4` files,
concatenates them on a continuous time axis, resamples onto a fixed raster (default 10 ms),
and writes one CSV per channel:

- Output schema: two columns: `time` and `<channel_name>`
- Output location: `Outputs/01_Data_Preparation/Signals/*.csv`
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from asammdf import MDF


def list_stint_files(stints_dir: Path, pattern_prefix: str = "Stint", ext: str = ".mf4") -> List[Path]:
    files = [p for p in stints_dir.iterdir() if p.is_file() and p.name.startswith(pattern_prefix) and p.name.endswith(ext)]
    files.sort()
    return files


def read_channels_list(path: Path) -> List[str]:
    channels: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        channels.append(s)
    seen = set()
    out: List[str] = []
    for c in channels:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def get_best_occurrence_signal(m: MDF, name: str):
    """Return the occurrence of channel `name` with the most samples."""
    try:
        occs = m.channels_db.get(name)
        if not occs:
            return None
        best = None
        best_len = -1
        for gp, idx in occs:
            try:
                sig = m.get(name, group=gp, index=idx)
                n = len(sig.samples) if sig.samples is not None else 0
                if n > best_len:
                    best = sig
                    best_len = n
            except Exception:
                continue
        return best
    except Exception:
        return None


def _as_float_1d(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    except Exception:
        return None


def _interp_to_grid_nan(grid: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    if xp is None or fp is None or xp.size == 0 or fp.size == 0:
        return np.full(grid.shape, np.nan, dtype=float)

    mask_f = np.isfinite(xp) & np.isfinite(fp)
    xp = xp[mask_f]
    fp = fp[mask_f]
    if xp.size == 0:
        return np.full(grid.shape, np.nan, dtype=float)

    xp_u, idx = np.unique(xp, return_index=True)
    fp_u = fp[idx]

    out = np.full(grid.shape, np.nan, dtype=float)
    if xp_u.size == 1:
        mask_eq = np.isclose(grid, xp_u[0], rtol=0.0, atol=5e-4)
        out[mask_eq] = fp_u[0]
        return out

    inside = (grid >= xp_u[0]) & (grid <= xp_u[-1])
    if np.any(inside):
        out[inside] = np.interp(grid[inside], xp_u, fp_u)
    return out


def _safe_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def export_channels(
    stints_dir: Path,
    channels: List[str],
    out_dir: Path,
    dt_s: float,
    add_gap: bool,
) -> None:
    stints = list_stint_files(stints_dir)
    if not stints:
        raise FileNotFoundError(f"No Stint*.mf4 files found in: {stints_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    time_parts: List[np.ndarray] = []
    values_parts: Dict[str, List[np.ndarray]] = {c: [] for c in channels}
    offset = 0.0

    for mf4 in stints:
        m = MDF(str(mf4))
        m.configure(raise_on_multiple_occurrences=False)

        raw: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        seg_ends: List[float] = []
        for ch in channels:
            sig = get_best_occurrence_signal(m, ch)
            if sig is None:
                continue
            ts = _as_float_1d(sig.timestamps)
            ys = _as_float_1d(sig.samples)
            if ts is None or ys is None or ts.size == 0 or ys.size == 0:
                continue
            ts0 = float(ts[0])
            t_rel = ts - ts0
            raw[ch] = (t_rel, ys)
            seg_ends.append(float(t_rel[-1]))

        if not seg_ends:
            continue

        seg_end = float(np.nanmax(seg_ends))
        n = int(np.floor(seg_end / dt_s)) + 1
        grid_rel = np.arange(n, dtype=float) * dt_s
        grid_abs = grid_rel + offset
        time_parts.append(grid_abs)

        for ch in channels:
            if ch not in raw:
                values_parts[ch].append(np.full(grid_abs.shape, np.nan, dtype=float))
                continue
            xp, fp = raw[ch]
            values_parts[ch].append(_interp_to_grid_nan(grid_rel, xp, fp))

        offset = float(grid_abs[-1] + (dt_s if add_gap else 0.0))

    if not time_parts:
        raise RuntimeError("No data exported (no usable channels across stints)")

    t_all = np.concatenate(time_parts)

    for ch in channels:
        y_all = np.concatenate(values_parts[ch]) if values_parts.get(ch) else np.full(t_all.shape, np.nan, dtype=float)
        df = pd.DataFrame({"time": t_all, ch: y_all})
        out_path = out_dir / f"{_safe_filename(ch)}.csv"
        df.to_csv(out_path, index=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export only channels from channels_extraction.txt resampled to 10 ms into Outputs/01_Data_Preparation/Signals"
    )
    parser.add_argument(
        "--stints-dir",
        default=os.path.join("Data", "2025-10-22_17-44-15_Steißlingen_AeroValidation_Kibele"),
        help="Directory containing Stint*.mf4 files",
    )
    parser.add_argument(
        "--channels-file",
        default=None,
        help=(
            "Path to channels_extraction.txt (default: Data/channels_extraction.txt if present, else "
            "Code/01_Data_Preparation/channels_extraction.txt)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("Outputs", "01_Data_Preparation", "Signals"),
        help="Output directory for per-channel CSVs",
    )
    parser.add_argument(
        "--dt-ms",
        type=float,
        default=10.0,
        help="Sample raster in milliseconds (default: 10)",
    )
    parser.add_argument(
        "--no-gap",
        action="store_true",
        help="Do not insert a dt-sized gap between stints when concatenating",
    )

    args = parser.parse_args(argv)

    stints_dir = Path(args.stints_dir)
    if args.channels_file:
        channels_file = Path(args.channels_file)
    else:
        preferred = Path("Data") / "channels_extraction.txt"
        fallback = Path("Code") / "01_Data_Preparation" / "channels_extraction.txt"
        channels_file = preferred if preferred.exists() else fallback
    out_dir = Path(args.out_dir)

    if not channels_file.exists():
        raise FileNotFoundError(f"Channels file not found: {channels_file}")

    channels = read_channels_list(channels_file)
    if not channels:
        raise ValueError(f"No channels found in: {channels_file}")

    dt_s = float(args.dt_ms) / 1000.0
    if dt_s <= 0:
        raise ValueError("--dt-ms must be > 0")

    export_channels(
        stints_dir=stints_dir,
        channels=channels,
        out_dir=out_dir,
        dt_s=dt_s,
        add_gap=not args.no_gap,
    )

    print(f"Exported {len(channels)} channels to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
