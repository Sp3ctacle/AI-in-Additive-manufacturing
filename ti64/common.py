"""
ti64.common - Global utilities and unified operations for Ti-6Al-4V research pipeline.

Adheres strictly to:
- A1: Specimen grouping is absolute (assert_disjoint).
- A4: Unified preprocessing function for everything (cv2.INTER_AREA, BGR->RGB, ImageNet stats).
- A5: No geometric distortion.
- A7: No hardcoded result numbers.
- A8: Physical units (PIXEL_SIZE_UM = 0.55812 um/px).
- A9: Specimen-level bootstrap CI (2000 resamples), paired stats.
- A11: Reproducibility (seed 42, write_run_info).
- A14: Fresh codebase only.
"""

import os
import sys
import json
import time
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import cv2
import torch

# Rule A8: Physical scale at native resolution established in Phase 0
PIXEL_SIZE_UM = 0.55812

# Rule B10: Dataset license and scholarly citation
DATASET_CITATION = (
    "Meluhn, L. A., Müller, S., Altmann, M. L., Benthien, T., & Toenjes, A. (2023). "
    "PBF-LB/M Ti6Al4V Micrographs and Labeled Manufacturing Defects (Version 1.0) [Data set]. "
    "Zenodo. https://doi.org/10.5281/zenodo.8303011"
)
DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
DATASET_DOI = "10.5281/zenodo.8303011"

# Standard ImageNet normalization parameters
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(seed: int = 42) -> None:
    """Enforce complete reproducibility across random, numpy, and torch (Rule A11)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from JSON file or return default pipeline hyperparameters."""
    defaults = {
        "seed": 42,
        "pixel_size_um": PIXEL_SIZE_UM,
        "backbone": "tf_efficientnetv2_s",
        "input_size": 256,
        "native_tile_size_px": 512,
        "effective_batch_size": 32,
        "mini_batch_size": 16,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 20,
        "checkpoint_mode": "last",
        "num_folds": 5,
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        defaults.update(user_cfg)
    return defaults


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown_or_no_git"


def get_gpu_info() -> str:
    """Detect available GPU accelerator."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "None (CPU)"


def write_run_info(out_dir: Union[str, Path], config: Optional[Dict[str, Any]] = None, **kwargs) -> Path:
    """
    Save complete execution metadata into run_info.json (Rule A11).
    Records config, commit hash, package versions, GPU model, and timestamp.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "git_commit": get_git_commit(),
        "gpu_model": get_gpu_info(),
        "license": DATASET_LICENSE,
        "dataset_doi": DATASET_DOI,
        "dataset_citation": DATASET_CITATION,
        "package_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cv2": cv2.__version__,
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "sklearn": sys.modules["sklearn"].__version__ if "sklearn" in sys.modules else "N/A",
        },
        "config": config or {},
    }
    info.update(kwargs)
    path = out_dir / "run_info.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path


def preprocess(tile_bgr: np.ndarray, size: int) -> torch.Tensor:
    """
    THE ONLY preprocessing function used anywhere in the codebase (Rule A4).
    Accepts an input BGR image (OpenCV format), resizes it using cv2.INTER_AREA
    to (size, size), converts BGR to RGB, converts to float in [0, 1], and
    normalizes with standard ImageNet mean and standard deviation.

    Args:
        tile_bgr: uint8 BGR image numpy array of shape (H, W, 3).
        size: Target integer height and width (e.g. 256).

    Returns:
        torch.FloatTensor of shape (3, size, size) normalized by ImageNet statistics.
    """
    if not isinstance(tile_bgr, np.ndarray):
        raise TypeError(f"tile_bgr must be a numpy ndarray, got {type(tile_bgr)}")
    if tile_bgr.ndim != 3 or tile_bgr.shape[2] != 3:
        raise ValueError(f"tile_bgr must have shape (H, W, 3), got {tile_bgr.shape}")

    # Resize using cv2.INTER_AREA for anti-aliased downsampling (Rule A4)
    h, w = tile_bgr.shape[:2]
    if h != size or w != size:
        resized = cv2.resize(tile_bgr, (size, size), interpolation=cv2.INTER_AREA)
    else:
        resized = tile_bgr

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Convert to PyTorch float tensor in range [0, 1]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    # Apply ImageNet normalization: (x - mean) / std
    normalized = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return normalized


def load_folds(folds_path: Union[str, Path] = "outputs/phase2/folds.json") -> Dict[int, int]:
    """
    Load cross-validation fold assignments mapping specimen_id -> fold index.
    Ensures absolute specimen grouping (Rule A1).
    """
    path = Path(folds_path)
    if not path.exists():
        raise FileNotFoundError(f"Folds definition not found at {path}. Run Phase 2 first.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert string keys to int specimen_id
    return {int(k): int(v) for k, v in data.items()}


def assert_disjoint(train_ids: Union[List[int], set, np.ndarray], val_ids: Union[List[int], set, np.ndarray]) -> None:
    """
    Assert in code, for every fold, that the set of specimen_ids in train
    and in validation is strictly disjoint (Rule A1).
    """
    set_train = set(int(x) for x in train_ids)
    set_val = set(int(x) for x in val_ids)
    intersection = set_train.intersection(set_val)
    if intersection:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED (Rule A1 violated)! Specimen IDs appear in both train and val: {sorted(list(intersection))}"
        )


def metrics(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> Dict[str, float]:
    """
    Compute rigorous statistical regression metrics (Rule A9).
    Returns R2, MAE, RMSE, Pearson r, and Mean Bias.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) == 0:
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "Pearson_r": np.nan, "Mean_bias": np.nan}

    r2 = float(r2_score(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mean_bias = float(np.mean(y_p - y_t))

    if len(y_t) > 1 and np.std(y_t) > 1e-12 and np.std(y_p) > 1e-12:
        r_val, _ = scipy.stats.pearsonr(y_t, y_p)
        pearson_r = float(r_val)
    else:
        pearson_r = 0.0

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Pearson_r": pearson_r,
        "Mean_bias": mean_bias,
    }


def bootstrap_ci_by_specimen(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    specimen_col: str = "specimen_id",
    n: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Compute 95% confidence intervals by bootstrap resampling over SPECIMENS,
    never over individual tiles or images (Rule A9).

    Args:
        df: DataFrame containing predictions, targets, and specimen_id.
        metric_fn: Function that accepts a DataFrame and returns a scalar metric float.
        specimen_col: Column name identifying the specimen.
        n: Number of bootstrap iterations (default 2000).
        seed: Random seed.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (ci_lower, ci_upper, point_estimate).
    """
    rng = np.random.default_rng(seed)
    unique_specimens = np.array(sorted(df[specimen_col].unique()))
    n_specimens = len(unique_specimens)

    point_estimate = float(metric_fn(df))

    # Pre-index DataFrame by specimen for speed
    grouped = {spec_id: sub_df for spec_id, sub_df in df.groupby(specimen_col)}

    bootstrap_metrics = []
    for _ in range(n):
        sampled_specimens = rng.choice(unique_specimens, size=n_specimens, replace=True)
        # Assemble resampled dataframe
        sampled_df = pd.concat([grouped[s] for s in sampled_specimens], ignore_index=True)
        val = metric_fn(sampled_df)
        bootstrap_metrics.append(val)

    bootstrap_metrics = np.array(bootstrap_metrics)
    ci_lower = float(np.percentile(bootstrap_metrics, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_metrics, 100 * (1.0 - alpha / 2.0)))

    return ci_lower, ci_upper, point_estimate

