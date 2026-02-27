# Validation_Aero_Test

Project to prepare telemetry signals, extract coastdown runs, and compute drag coefficients.

Documentation (LaTeX) lives in:
- Docs/01_Data_Preparation/01_Data_Preparation.tex
- Docs/02_Run_Extraction/02_Run_Extraction.tex
- Docs/03_Drag_Calculation/03_Drag_Calculation.tex
- Docs/main.tex (index)

## Project structure
- Part 1 (data preparation): Code/01_Data_Preparation/
- Part 2 (run extraction): Code/02_Run_Extraction/
- Part 3 (drag calculation): Code/03_Drag_Calculation/
- Outputs (generated artifacts): Outputs/

## Part 1 — Data preparation (quickstart)

```bash
# Export selected channels from Stint*.mf4 into one CSV per channel,
# resampled onto a fixed raster (default: 10 ms).
python Code/01_Data_Preparation/export_channels_10ms_to_signals.py \
	--stints-dir Data/2025-10-22_17-44-15_Steißlingen_AeroValidation_Kibele \
	--channels-file Data/channels_extraction.txt \
	--out-dir Outputs/01_Data_Preparation/Signals \
	--dt-ms 10

# Combine per-channel exports into the canonical cleaned dataset and add derived signals
# (GPS1_VEL_MAG_ms, WHEEL_SPEED_kmh, IMU_ACCEL_LONG_ms2).
python Code/01_Data_Preparation/build_cleaned_signals_10ms.py \
	--channels-dir Outputs/01_Data_Preparation/Signals \
	--output Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned_10ms.csv
```

## Part 2 + Part 3 — Run extraction and drag calculation

```bash
# Part 2: extract intervals and per-run outputs
python Code/02_Run_Extraction/export_drag_runs.py \
	--csv Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned_10ms.csv \
	--out Outputs/02_Run_Extraction/csv/drag_intervals.csv

python Code/02_Run_Extraction/generate_run_outputs.py \
	--csv Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned_10ms.csv \
	--intervals Outputs/02_Run_Extraction/csv/drag_intervals.csv \
	--outdir Outputs/02_Run_Extraction/csv

# Part 3: compute coefficients
python Code/03_Drag_Calculation/Approach1_PerRun/run.py

# Optional alternatives:
# - Approach2_PairedOpposites/run.py (pair consecutive opposite-direction runs)
# - Approach3_PooledByConfig/run.py (pooled regression per configuration)
# - Approach4/5/6 *_Linear/run.py (include linear-in-v term - not working)
```

## Data & outputs
- Raw input: Data/<session>/Stint*.mf4
- Per-channel exports: Outputs/01_Data_Preparation/Signals/*.csv
- Canonical cleaned CSV: Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned_10ms.csv
- Run intervals + per-run CSVs: Outputs/02_Run_Extraction/csv/
- Drag results per approach: Outputs/03_Drag_Calculation/<ApproachX_*/>

## Notes
- Scripts use reusable functions in the aero_validation package (e.g., aero_validation.io for paths/data loading and aero_validation.segmentation for run detection).
- Columns are sanitized (degree symbols, names) and derived signals are added:
	- WHEEL_SPEED_kmh: mean of 4 wheel speeds
	- GPS1_VEL_MAG_ms: magnitude of GPS D/E/N velocities

Adjust thresholds in the CLI scripts under Code/02_Run_Extraction/ and Code/03_Drag_Calculation/ if needed.
