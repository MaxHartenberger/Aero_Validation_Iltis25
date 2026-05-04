#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Allow importing ../common.py when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import (
    assign_config,
    fd_table_from_summary,
    load_constants,
    prepare_run_slice,
    repo_root,
    robust_linear_fit,
    robust_linear_fit_multi,
    derive_drag_metrics,
    summarize_by_config,
)


def _ensure_dirs(out_root: Path) -> tuple[Path, Path]:
    csv_dir = out_root / "csv"
    plots_dir = out_root / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, plots_dir


def _plot_summary(coeffs: pd.DataFrame, *, plots_dir: Path, config_order: list[str]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")

    # Scatter of CdA per run colored by config
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=coeffs, x="run_id", y="CdA_m2", hue="config", palette="tab10")
    plt.xlabel("Run ID")
    plt.ylabel("CdA [m$^2$]")
    plt.title("CdA per run by configuration")
    plt.tight_layout()
    plt.savefig(plots_dir / "CdA_per_run_by_config.png", dpi=160)
    plt.close()

    # Boxplot of CdA by configuration
    if config_order:
        order = [c for c in config_order if c in set(coeffs["config"].astype(str))]
    else:
        order = None
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=coeffs, x="config", y="CdA_m2", order=order)
    plt.xlabel("Configuration")
    plt.ylabel("CdA [m$^2$]")
    plt.title("CdA distribution by configuration")
    plt.tight_layout()
    plt.savefig(plots_dir / "CdA_box_by_config.png", dpi=160)
    plt.close()

    # Boxplot of Cr by configuration
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=coeffs, x="config", y="Cr", order=order)
    plt.xlabel("Configuration")
    plt.ylabel("Cr [-]")
    plt.title("Cr distribution by configuration")
    plt.tight_layout()
    plt.savefig(plots_dir / "Cr_box_by_config.png", dpi=160)
    plt.close()


def _plot_run_fit(run_id: int, sl: pd.DataFrame, *, k2: float, k1: float, c: float, plots_dir: Path) -> None:
    import matplotlib.pyplot as plt

    v = sl["v"].to_numpy(dtype=float)
    v2 = sl["v2"].to_numpy(dtype=float)
    y = sl["decel_ms2"].to_numpy(dtype=float)

    mask = np.isfinite(v2) & np.isfinite(v) & np.isfinite(y)
    v = v[mask]
    v2 = v2[mask]
    y = y[mask]
    if y.size < 10:
        return

    y_pred = (k2 * v2) + (k1 * v) + c
    resid = y - y_pred
    std = float(np.std(resid)) if resid.size else float("nan")
    inlier = np.abs(resid) <= (2.0 * std) if (std > 0 and np.isfinite(std)) else np.ones_like(resid, dtype=bool)

    idx = np.argsort(v2)
    v2s = v2[idx]
    vs = v[idx]
    y_line = (k2 * v2s) + (k1 * vs) + c

    plt.figure(figsize=(7, 5))
    plt.scatter(v2[~inlier], y[~inlier], s=12, c="#bbbbbb", label="outliers")
    plt.scatter(v2[inlier], y[inlier], s=14, c="#1f77b4", label="inliers")
    plt.plot(v2s, y_line, c="#d62728", lw=2, label="fit")
    plt.xlabel(r"$v^2$ [m$^2$/s$^2$]")
    plt.ylabel(r"decel $a$ [m/s$^2$]")
    plt.title(f"Coastdown fit: run {run_id:03d}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plots_dir / f"coastdown_run_{run_id:03d}.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Approach 1: per-run coastdown regression (CdA/Cr per run + per-config statistics).")
    parser.add_argument("--constants", type=Path, default=None, help="Vehicle constants JSON (default: Data/vehicle_constants.json)")
    parser.add_argument("--intervals", type=Path, default=None, help="Intervals CSV (default: Outputs/02_Run_Extraction/csv/drag_intervals.csv)")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Runs directory (default: Outputs/02_Run_Extraction/csv)")
    parser.add_argument("--out-root", type=Path, default=None, help="Output root dir (default: Outputs/03_Drag_Calculation/Approach1_PerRun)")
    parser.add_argument("--min-v-ms", type=float, default=5.0, help="Minimum speed to include in fit [m/s]")
    args = parser.parse_args()

    repo = repo_root()
    constants_path = args.constants if args.constants is not None else (repo / "Data" / "vehicle_constants.json")
    const = load_constants(constants_path)

    intervals_path = args.intervals if args.intervals is not None else (repo / "Outputs" / "02_Run_Extraction" / "csv" / "drag_intervals.csv")
    runs_dir = args.runs_dir if args.runs_dir is not None else (repo / "Outputs" / "02_Run_Extraction" / "csv")
    out_root = args.out_root if args.out_root is not None else (repo / "Outputs" / "03_Drag_Calculation" / "Approach1_PerRun")
    csv_dir, plots_dir = _ensure_dirs(out_root)

    intervals = pd.read_csv(intervals_path)
    required = {"run_id", "t0_s", "t1_s"}
    if not required.issubset(intervals.columns):
        raise ValueError(f"Intervals CSV missing columns: {required - set(intervals.columns)}")

    rows: list[dict] = []
    config_order = list(const.config_ranges_by_run_id.keys())

    for _, row in intervals.iterrows():
        run_id = int(row["run_id"])
        cfg = assign_config(run_id, const.config_ranges_by_run_id)
        run_csv = runs_dir / f"run_{run_id:03d}.csv"
        sl = prepare_run_slice(run_csv, float(row["t0_s"]), float(row["t1_s"]), min_v_ms=float(args.min_v_ms))

        k2, c, r2, n_points = robust_linear_fit(sl["v2"].to_numpy(), sl["decel_ms2"].to_numpy())
        k1 = 0.0

        area = const.frontal_area_m2_by_config.get(cfg)
        metrics = derive_drag_metrics(k2=k2, c=c, mass_kg=const.mass_kg, rho_kgm3=const.rho_kgm3, area_m2=area)

        rows.append(
            {
                "run_id": run_id,
                "config": cfg,
                "t0_s": float(row["t0_s"]),
                "t1_s": float(row["t1_s"]),
                "n_points": int(n_points),
                "k2_v2_term_ms2_per_ms2": float(k2),
                "k1_v_term_ms2_per_ms": float(k1),
                "c_const_ms2": float(c),
                "r2": float(r2),
                **metrics,
            }
        )

        # Always write per-run diagnostic fit plots for this approach
        _plot_run_fit(run_id, sl, k2=float(k2), k1=float(k1), c=float(c), plots_dir=plots_dir)

    coeffs = pd.DataFrame(rows).sort_values("run_id")
    coeffs_path = csv_dir / "drag_coefficients.csv"
    coeffs.to_csv(coeffs_path, index=False)

    stats, contrib = summarize_by_config(coeffs, baseline_config=const.baseline_config)
    summary_path = csv_dir / "drag_config_summary.csv"
    contrib_path = csv_dir / "drag_config_summary_contributions.csv"
    stats.to_csv(summary_path, index=False)
    contrib.to_csv(contrib_path, index=False)

    fd = fd_table_from_summary(stats, rho_kgm3=const.rho_kgm3, speeds_ms=const.fd_speeds_ms)
    fd_path = csv_dir / "drag_config_fd_table.csv"
    fd.to_csv(fd_path, index=False)

    # Write summary plots into the mode-specific plots directory
    _plot_summary(coeffs, plots_dir=plots_dir, config_order=config_order)

    print(f"Wrote: {coeffs_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {contrib_path}")
    print(f"Wrote: {fd_path}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
