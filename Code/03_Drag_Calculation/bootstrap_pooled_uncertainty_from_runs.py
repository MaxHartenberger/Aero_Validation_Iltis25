"""Bootstrap uncertainty estimates for pooled-by-config reporting.

This script derives an uncertainty estimate for pooled results (Approach 3 / 6)
from the per-run fit outputs (Approach 1 / 4).

It bootstraps the *mean across runs* per configuration (resampling run-level
CdA/Cr with replacement). The standard deviation of bootstrap means is reported
as an uncertainty estimate.

Outputs:
- Outputs/03_Drag_Calculation/bootstrap_pooled_uncertainty.csv

Notes:
- This does not re-fit the pooled regression. It provides a reproducible,
  run-level repeatability / mean-uncertainty proxy that can be computed from
  existing outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_csv: Path


def _bootstrap_means(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[indices]
    return samples.mean(axis=1)


def compute_bootstrap_table(
    per_run_csv: Path,
    model_name: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(per_run_csv)

    required = {"run_id", "config", "CdA_m2", "Cr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{per_run_csv} missing columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)

    rows: list[dict[str, object]] = []
    for config, grp in df.groupby("config", sort=True):
        cda = grp["CdA_m2"].to_numpy(dtype=float)
        cr = grp["Cr"].to_numpy(dtype=float)

        cda_boot = _bootstrap_means(cda, n_boot=n_boot, rng=rng)
        cr_boot = _bootstrap_means(cr, n_boot=n_boot, rng=rng)

        rows.append(
            {
                "model": model_name,
                "config": config,
                "n_runs": int(len(grp)),
                "CdA_m2_boot_mean": float(np.mean(cda_boot)),
                "CdA_m2_boot_std": float(np.std(cda_boot, ddof=1)) if len(cda_boot) > 1 else float("nan"),
                "Cr_boot_mean": float(np.mean(cr_boot)),
                "Cr_boot_std": float(np.std(cr_boot, ddof=1)) if len(cr_boot) > 1 else float("nan"),
                "n_boot": int(n_boot),
                "seed": int(seed),
                "source_csv": str(per_run_csv.as_posix()),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("Outputs/03_Drag_Calculation"),
        help="Root output folder where bootstrap CSV will be written.",
    )
    args = parser.parse_args()

    specs = [
        ModelSpec(
            name="baseline",
            input_csv=Path("Outputs/03_Drag_Calculation/Approach1_PerRun/csv/drag_coefficients.csv"),
        ),
    ]

    out_root: Path = args.outputs_root
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "bootstrap_pooled_uncertainty.csv"

    frames = [
        compute_bootstrap_table(spec.input_csv, spec.name, n_boot=args.n_boot, seed=args.seed)
        for spec in specs
    ]
    out = pd.concat(frames, ignore_index=True)

    # Stable ordering for readability
    out = out.sort_values(["model", "config"], kind="mergesort").reset_index(drop=True)

    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
