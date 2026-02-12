#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import (
    assign_config,
    load_constants,
    prepare_run_slice,
    repo_root,
    robust_linear_fit,
    derive_drag_metrics,
    fd_table_from_summary,
)


def _ensure_dirs(out_root: Path) -> tuple[Path, Path]:
    csv_dir = out_root / "csv"
    plots_dir = out_root / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, plots_dir


def _contributions_from_cfg_df(cfg_df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    idx = {str(r["config"]): r for _, r in cfg_df.iterrows()}

    def cdA(cfg: str) -> float:
        return float(idx[cfg]["CdA_m2"]) if cfg in idx else float("nan")

    return pd.DataFrame(
        [
            {"component": "front_wing", "CdA_m2": cdA("no_rear") - cdA(baseline)},
            {"component": "rear_wing", "CdA_m2": cdA("no_front") - cdA(baseline)},
            {"component": "both_wings", "CdA_m2": cdA("full") - cdA(baseline)},
        ]
    )


def _plot_cfg_bar(cfg_df: pd.DataFrame, *, plots_dir: Path, config_order: list[str]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    order = [c for c in config_order if c in set(cfg_df["config"].astype(str))] if config_order else None
    plt.figure(figsize=(6, 4))
    sns.barplot(data=cfg_df, x="config", y="CdA_m2", order=order)
    plt.xlabel("Configuration")
    plt.ylabel("CdA [m$^2$]")
    plt.title("CdA by configuration (pooled fit)")
    plt.tight_layout()
    plt.savefig(plots_dir / "CdA_bar_by_config_pooled.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Approach 3: stack all coastdown samples per configuration and fit one CdA/Cr per config.")
    parser.add_argument("--constants", type=Path, default=None, help="Vehicle constants JSON (default: Data/vehicle_constants.json)")
    parser.add_argument("--intervals", type=Path, default=None, help="Intervals CSV (default: Outputs/02_Run_Extraction/csv/drag_intervals.csv)")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Runs directory (default: Outputs/02_Run_Extraction/csv)")
    parser.add_argument("--out-root", type=Path, default=None, help="Output root dir (default: Outputs/03_Drag_Calculation/Approach3_PooledByConfig)")
    parser.add_argument("--min-v-ms", type=float, default=5.0, help="Minimum speed threshold applied before stacking [m/s]")
    args = parser.parse_args()

    repo = repo_root()
    constants_path = args.constants if args.constants is not None else (repo / "Data" / "vehicle_constants.json")
    const = load_constants(constants_path)

    intervals_path = args.intervals if args.intervals is not None else (repo / "Outputs" / "02_Run_Extraction" / "csv" / "drag_intervals.csv")
    runs_dir = args.runs_dir if args.runs_dir is not None else (repo / "Outputs" / "02_Run_Extraction" / "csv")
    out_root = args.out_root if args.out_root is not None else (repo / "Outputs" / "03_Drag_Calculation" / "Approach3_PooledByConfig")
    csv_dir, plots_dir = _ensure_dirs(out_root)

    intervals = pd.read_csv(intervals_path)
    required = {"run_id", "t0_s", "t1_s"}
    if not required.issubset(intervals.columns):
        raise ValueError(f"Intervals CSV missing columns: {required - set(intervals.columns)}")

    config_order = list(const.config_ranges_by_run_id.keys())
    data: dict[str, dict[str, list[np.ndarray]]] = {cfg: {"v2": [], "decel": []} for cfg in config_order}

    for _, row in intervals.iterrows():
        run_id = int(row["run_id"])
        cfg = assign_config(run_id, const.config_ranges_by_run_id)
        if cfg not in data:
            continue
        run_csv = runs_dir / f"run_{run_id:03d}.csv"
        try:
            sl = prepare_run_slice(run_csv, float(row["t0_s"]), float(row["t1_s"]), min_v_ms=float(args.min_v_ms))
        except Exception as exc:
            print(f"Skipping run {run_id}: {exc}")
            continue
        v2 = sl["v2"].to_numpy()
        decel = sl["decel_ms2"].to_numpy()
        if v2.size:
            data[cfg]["v2"].append(v2)
            data[cfg]["decel"].append(decel)

    rows: list[dict] = []
    for cfg in config_order:
        v2_list = data[cfg]["v2"]
        decel_list = data[cfg]["decel"]
        if not v2_list:
            continue
        v2 = np.concatenate(v2_list)
        decel = np.concatenate(decel_list)
        mask = np.isfinite(v2) & np.isfinite(decel)
        v2 = v2[mask]
        decel = decel[mask]
        if v2.size < 50:
            print(f"Skipping config {cfg}: insufficient samples")
            continue
        k2, c, r2, n_points = robust_linear_fit(v2, decel)
        k1 = 0.0
        area = const.frontal_area_m2_by_config.get(cfg)
        metrics = derive_drag_metrics(k2=float(k2), c=float(c), mass_kg=const.mass_kg, rho_kgm3=const.rho_kgm3, area_m2=area)
        rows.append(
            {
                "config": cfg,
                "n_points": int(n_points),
                "k2_v2_term_ms2_per_ms2": float(k2),
                "c_const_ms2": float(c),
                "r2": float(r2),
                **metrics,
            }
        )

    cfg_df = pd.DataFrame(rows)
    out_cfg = csv_dir / "drag_config_pooled_fits.csv"
    cfg_df.to_csv(out_cfg, index=False)

    contrib = _contributions_from_cfg_df(cfg_df, baseline=const.baseline_config)
    out_contrib = csv_dir / "drag_config_pooled_contributions.csv"
    contrib.to_csv(out_contrib, index=False)

    # Create a "summary-like" table with CdA_m2_mean so fd_table_from_summary() can be reused.
    summary_like = pd.DataFrame(
        {
            "config": cfg_df["config"],
            "CdA_m2_mean": cfg_df["CdA_m2"],
        }
    )
    fd = fd_table_from_summary(summary_like, rho_kgm3=const.rho_kgm3, speeds_ms=const.fd_speeds_ms)
    out_fd = csv_dir / "drag_config_fd_table.csv"
    fd.to_csv(out_fd, index=False)

    _plot_cfg_bar(cfg_df, plots_dir=plots_dir, config_order=config_order)

    print(f"Wrote: {out_cfg}")
    print(f"Wrote: {out_contrib}")
    print(f"Wrote: {out_fd}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
