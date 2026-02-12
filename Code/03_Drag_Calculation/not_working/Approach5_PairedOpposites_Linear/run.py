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
    fd_table_from_summary,
    load_constants,
    prepare_run_slice,
    repo_root,
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


def _plot_paired_summary(pairs: pd.DataFrame, *, plots_dir: Path, config_order: list[str]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=pairs, x="run_id_low", y="CdA_m2", hue="config", palette="tab10")
    plt.xlabel("Pair start run ID")
    plt.ylabel("CdA [m$^2$]")
    plt.title("CdA per paired opposite-direction run (linear v-term)")
    plt.tight_layout()
    plt.savefig(plots_dir / "CdA_per_pair_by_config.png", dpi=160)
    plt.close()

    order = [c for c in config_order if c in set(pairs["config"].astype(str))] if config_order else None
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=pairs, x="config", y="CdA_m2", order=order)
    plt.xlabel("Configuration")
    plt.ylabel("CdA [m$^2$]")
    plt.title("CdA distribution by configuration (paired, linear v-term)")
    plt.tight_layout()
    plt.savefig(plots_dir / "CdA_box_by_config_paired.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Approach 5: per-run regression with linear v-term, then pair opposite-direction runs.")
    parser.add_argument("--constants", type=Path, default=None, help="Vehicle constants JSON (default: Data/vehicle_constants.json)")
    parser.add_argument("--intervals", type=Path, default=None, help="Intervals CSV (default: Outputs/02_Run_Extraction/csv/drag_intervals.csv)")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Runs directory (default: Outputs/02_Run_Extraction/csv)")
    parser.add_argument("--out-root", type=Path, default=None, help="Output root dir (default: Outputs/03_Drag_Calculation/Approach5_PairedOpposites_Linear)")
    parser.add_argument("--min-v-ms", type=float, default=5.0, help="Minimum speed to include in fit [m/s]")
    args = parser.parse_args()

    repo = repo_root()
    constants_path = args.constants if args.constants is not None else (repo / "Data" / "vehicle_constants.json")
    const = load_constants(constants_path)

    intervals_path = args.intervals if args.intervals is not None else (repo / "Outputs" / "02_Run_Extraction" / "csv" / "drag_intervals.csv")
    runs_dir = args.runs_dir if args.runs_dir is not None else (repo / "Outputs" / "02_Run_Extraction" / "csv")
    out_root = args.out_root if args.out_root is not None else (repo / "Outputs" / "03_Drag_Calculation" / "Approach5_PairedOpposites_Linear")
    csv_dir, plots_dir = _ensure_dirs(out_root)

    intervals = pd.read_csv(intervals_path)
    required = {"run_id", "t0_s", "t1_s"}
    if not required.issubset(intervals.columns):
        raise ValueError(f"Intervals CSV missing columns: {required - set(intervals.columns)}")

    run_rows: list[dict] = []
    for _, row in intervals.iterrows():
        run_id = int(row["run_id"])
        cfg = assign_config(run_id, const.config_ranges_by_run_id)
        run_csv = runs_dir / f"run_{run_id:03d}.csv"
        sl = prepare_run_slice(run_csv, float(row["t0_s"]), float(row["t1_s"]), min_v_ms=float(args.min_v_ms))

        X = np.column_stack([sl["v2"].to_numpy(), sl["v"].to_numpy(), np.ones(len(sl))])
        beta, r2, n_points = robust_linear_fit_multi(X, sl["decel_ms2"].to_numpy())
        k2, k1, c = beta.tolist()

        area = const.frontal_area_m2_by_config.get(cfg)
        metrics = derive_drag_metrics(k2=k2, c=c, mass_kg=const.mass_kg, rho_kgm3=const.rho_kgm3, area_m2=area)

        run_rows.append(
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

    run_coeffs = pd.DataFrame(run_rows).sort_values("run_id")
    run_coeffs_path = csv_dir / "drag_coefficients_raw.csv"
    run_coeffs.to_csv(run_coeffs_path, index=False)

    by_run = {int(r["run_id"]): r for _, r in run_coeffs.iterrows()}
    used: set[int] = set()
    pair_rows: list[dict] = []
    for run_id in sorted(by_run):
        if run_id in used:
            continue
        mate_id = run_id + 1
        if mate_id not in by_run or mate_id in used:
            continue
        r1 = by_run[run_id]
        r2 = by_run[mate_id]
        if str(r1["config"]) != str(r2["config"]) or str(r1["config"]) == "unknown":
            continue
        cfg = str(r1["config"])
        used.add(run_id)
        used.add(mate_id)

        k2_avg = 0.5 * (float(r1["k2_v2_term_ms2_per_ms2"]) + float(r2["k2_v2_term_ms2_per_ms2"]))
        k1_avg = 0.5 * (float(r1["k1_v_term_ms2_per_ms"]) + float(r2["k1_v_term_ms2_per_ms"]))
        c_avg = 0.5 * (float(r1["c_const_ms2"]) + float(r2["c_const_ms2"]))

        area = const.frontal_area_m2_by_config.get(cfg)
        metrics = derive_drag_metrics(k2=k2_avg, c=c_avg, mass_kg=const.mass_kg, rho_kgm3=const.rho_kgm3, area_m2=area)

        pair_rows.append(
            {
                "pair_label": f"{cfg}_pair_{run_id:03d}-{mate_id:03d}",
                "config": cfg,
                "run_ids": f"{run_id},{mate_id}",
                "run_id_low": int(run_id),
                "run_id_high": int(mate_id),
                "n_runs": 2,
                "n_points": int(r1["n_points"]) + int(r2["n_points"]),
                "k2_v2_term_ms2_per_ms2": float(k2_avg),
                "k1_v_term_ms2_per_ms": float(k1_avg),
                "c_const_ms2": float(c_avg),
                "r2_mean": 0.5 * (float(r1["r2"]) + float(r2["r2"]))
                if np.isfinite(float(r1["r2"])) and np.isfinite(float(r2["r2"]))
                else float("nan"),
                **metrics,
            }
        )

    pairs = pd.DataFrame(pair_rows).sort_values(["config", "run_id_low"]) if pair_rows else pd.DataFrame()
    pairs_path = csv_dir / "drag_coefficients_paired.csv"
    pairs.to_csv(pairs_path, index=False)
    if pairs.empty:
        raise RuntimeError("No pairs created. Check run IDs and config mapping in Data/vehicle_constants.json")

    stats, contrib = summarize_by_config(pairs.rename(columns={"r2_mean": "r2"}), baseline_config=const.baseline_config)
    summary_path = csv_dir / "drag_config_summary.csv"
    contrib_path = csv_dir / "drag_config_summary_contributions.csv"
    stats.to_csv(summary_path, index=False)
    contrib.to_csv(contrib_path, index=False)

    fd = fd_table_from_summary(stats, rho_kgm3=const.rho_kgm3, speeds_ms=const.fd_speeds_ms)
    fd_path = csv_dir / "drag_config_fd_table.csv"
    fd.to_csv(fd_path, index=False)

    config_order = list(const.config_ranges_by_run_id.keys())
    _plot_paired_summary(pairs, plots_dir=plots_dir, config_order=config_order)

    print(f"Wrote: {run_coeffs_path}")
    print(f"Wrote: {pairs_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {contrib_path}")
    print(f"Wrote: {fd_path}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
