"""Top-level package for Aero Validation utilities.

Provides cohesive modules for data I/O, signal processing, segmentation,
plotting, intervals handling, drag computation, and validation routines.
All functions used by scripts under the Code/ folder are exposed here.
"""

from .io import repo_root, data_path, ensure_global_plots_dir, load_data, ensure_columns
from .signals import sanitize_columns, add_derived_signals, smooth
from .segmentation import (
    DragRunConfig,
    segment_drag_runs,
    segment_drag_runs_with_debug,
    detect_drag_peaks,
)
from .plots import (
    plot_run,
    plot_overview,
    plot_peaks_overview,
    plot_overview_runs_subset,
)
from .intervals import (
    load_intervals_csv,
    parse_ranges,
)
from .validation import (
    validate_accel_vs_velocity,
    validate_accel_vs_wheelspeed,
    validate_accel_longitudinal,
)
from .drag import (
    VehicleConstants,
    load_constants,
    assign_config,
    robust_linear_fit,
    robust_linear_fit_multi,
    prepare_run_slice,
    derive_drag_metrics,
    fd_table_from_summary,
    summarize_by_config,
)
