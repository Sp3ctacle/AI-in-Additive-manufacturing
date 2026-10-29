"""
ti64 - Shared research package for Ti-6Al-4V LPBF Microstructure and Porosity Modeling
Adheres strictly to Global Rules A1-A15.
"""

from .common import (
    set_seed,
    load_config,
    write_run_info,
    preprocess,
    load_folds,
    assert_disjoint,
    metrics,
    bootstrap_ci_by_specimen,
    PIXEL_SIZE_UM,
)
from .model import build_model, get_gradcam_layer
from .train import train_one_fold, run_kfold_training

__all__ = [
    "set_seed",
    "load_config",
    "write_run_info",
    "preprocess",
    "load_folds",
    "assert_disjoint",
    "metrics",
    "bootstrap_ci_by_specimen",
    "PIXEL_SIZE_UM",
    "build_model",
    "get_gradcam_layer",
    "train_one_fold",
    "run_kfold_training",
]
