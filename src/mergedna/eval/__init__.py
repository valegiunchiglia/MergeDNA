"""
Evaluation helpers for downstream genomics tasks.
"""

from .data import (
    GENOMICS_TASK_GROUPS,
    infer_num_classes_from_labels,
    load_task_regression_raw,
    load_task_regression_synthetic,
    load_task_raw,
    load_task_synthetic,
    make_loaders,
    make_regression_loaders,
    remap_labels_to_contiguous,
)
from .models import validate_alphabet_compatibility
from .train_eval import average_metrics, parse_float_grid, run_linear_probe_once, select_best_setting, set_seed

__all__ = [
    "GENOMICS_TASK_GROUPS",
    "infer_num_classes_from_labels",
    "load_task_raw",
    "load_task_synthetic",
    "make_loaders",
    "remap_labels_to_contiguous",
    "parse_float_grid",
    "select_best_setting",
    "set_seed",
    "average_metrics",
    "load_task_regression_raw",
    "load_task_regression_synthetic",
    "make_regression_loaders",
    "run_linear_probe_once",
    "validate_alphabet_compatibility",
]
