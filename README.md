# Aero Validation

Formula Student aerodynamics validation pipeline: process telemetry logs, extract
coastdown runs, and estimate drag ($C_dA$, $C_d$) and rolling resistance ($C_r$)
coefficients using robust regression.

## Quickstart

```bash
# 1. Install the package (editable mode)
pip install -e .

# 2. Export MF4 channels to per-channel CSVs at 10 ms
python Code/01_Data_Preparation/export_channels_10ms_to_signals.py \
    --stints-dir Data/<session> \
    --channels-file Data/channels_extraction.txt \
    --out-dir Outputs/01_Data_Preparation/Signals \
    --dt-ms 10

# 3. Combine into the canonical cleaned dataset with derived signals
python Code/01_Data_Preparation/build_cleaned_signals_10ms.py \
    --channels-dir Outputs/01_Data_Preparation/Signals \
    --output Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned_10ms.csv

# 4. Extract coastdown intervals
python Code/02_Run_Extraction/export_drag_runs.py \
    --csv Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned_10ms.csv \
    --out Outputs/02_Run_Extraction/csv/drag_intervals.csv

python Code/02_Run_Extraction/generate_run_outputs.py \
    --csv Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned_10ms.csv \
    --intervals Outputs/02_Run_Extraction/csv/drag_intervals.csv \
    --outdir Outputs/02_Run_Extraction/csv

# 5. Compute drag coefficients (three complementary approaches)
python Code/03_Drag_Calculation/Approach1_PerRun/run.py          # per-run fit
python Code/03_Drag_Calculation/Approach2_PairedOpposites/run.py  # paired opposite-direction
python Code/03_Drag_Calculation/Approach3_PooledByConfig/run.py   # pooled per-config

# 6. (Optional) Bootstrap uncertainty estimates
python Code/03_Drag_Calculation/bootstrap_pooled_uncertainty_from_runs.py
```

## Project Structure

```
aero_validation/         # Reusable library package
├── io.py                #   Data loading, path resolution
├── signals.py           #   Column sanitization, derived signals, smoothing
├── segmentation.py      #   Coastdown run detection (DragRunConfig, peak detection)
├── drag.py              #   Drag computation (VehicleConstants, regression, metrics)
├── plots.py             #   Run overview, per-run QC, peak plots
├── intervals.py         #   CSV interval I/O, range parsing
├── validation.py        #   IMU acceleration validation (GPS & wheel speed)
└── py.typed             #   PEP 561 marker for type-checker support

Code/                    # Pipeline entry-point scripts
├── 01_Data_Preparation/
├── 02_Run_Extraction/
└── 03_Drag_Calculation/
    ├── common.py                 # Backward-compat shim (re-exports from aero_validation.drag)
    ├── Approach1_PerRun/
    ├── Approach2_PairedOpposites/
    ├── Approach3_PooledByConfig/
    └── bootstrap_pooled_uncertainty_from_runs.py

Data/                    # Source data (gitignored except vehicle_constants.json)
├── vehicle_constants.json       # Mass, air density, aero config mapping
├── channels_extraction.txt      # Channel names for MF4 export
└── <session>/Stint*.mf4         # Raw telemetry logs

Docs/                    # LaTeX documentation (compile with pdflatex)
├── 01_Data_Preparation/
├── 02_Run_Extraction/
├── 03_Drag_Calculation/
└── legacy/                     # Historical/archived docs

Outputs/                 # Generated artifacts (gitignored)
Tests/                   # pytest test suite
Tools/                   # Ad-hoc calibration, debug, and plotting utilities
```

## Installation

Requires Python ≥ 3.10. Install in editable mode:

```bash
pip install -e .
```

### Conda environment

```bash
conda activate formula_student
python -m pip install -e .
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `seaborn`, `asammdf`.

## Running Tests

```bash
pip install -e ".[dev]"    # first time only (installs pytest)
pytest Tests/ -v
```

## Three Analysis Approaches

| Approach | Method | Motivation |
|---|---|---|
| **1 — Per-run** | Fit $a = k_2 v^2 + c$ per coastdown interval | Baseline; gives run-to-run scatter |
| **2 — Paired opposites** | Average coefficients of consecutive runs within a config | Cancels steady biases (wind, grade) in out-and-back pairs |
| **3 — Pooled by config** | Stack all runs of a config and fit once | Maximizes sample count; reduces sensitivity to noisy runs |

All three use iterative $2\sigma$ outlier rejection (up to 5 iterations). $C_dA$ and $C_r$ are derived from $k_2$ and $c$ via:

$$C_dA = \frac{2 m k_2}{\rho}, \qquad C_r = \frac{c}{g}$$

## Wing Contribution Decomposition

Four aero configurations isolate individual wing elements:

| Config | State |
|---|---|
| `full` | Both wings |
| `no_rear` | Rear wing removed |
| `no_front_rear` | Both removed (baseline) |
| `no_front` | Front wing removed |

Contributions: $C_dA_\text{wing} = C_dA_\text{config} - C_dA_\text{baseline}$.

## Key Parameters

Vehicle constants are in `Data/vehicle_constants.json` — edit mass, air density,
frontal area, and run-ID-to-config mapping there. Missing fields emit warnings;
`config_ranges_by_run_id` is required.

Segmentation thresholds are defined in `DragRunConfig`
(`aero_validation/segmentation.py`) and can be overridden via CLI flags of
`export_drag_runs.py`.

## Documentation

Each pipeline stage has a PDF report in `Docs/`:

| Document | Content |
|---|---|
| `01_Data_Preparation.pdf` | MF4 → per-channel CSVs → cleaned dataset → derived signals |
| `02_Run_Extraction.pdf` | Peak detection, interval extension, quality gates, debug log |
| `03_Drag_Calculation.pdf` | Coastdown model, regression, three approaches, wing contributions, results |

Compile with `pdflatex` from each doc's directory:

```bash
cd Docs/01_Data_Preparation && pdflatex 01_Data_Preparation.tex
cd Docs/02_Run_Extraction   && pdflatex 02_Run_Extraction.tex
cd Docs/03_Drag_Calculation && pdflatex 03_Drag_Calculation.tex
```

## License

MIT
