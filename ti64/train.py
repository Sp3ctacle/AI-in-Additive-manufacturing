"""
ti64.train - Standardized training loop for Ti-6Al-4V LPBF Microstructure Models.

Follows Global Rules:
- A1: Specimen grouping is absolute (assert_disjoint enforced at fold start).
- A2: checkpoint_mode = 'last' (weights after final epoch, never selected on val fold).
- A3: Target scalers / transformations fit on training fold only, applied to val.
- A7: No hard-coded numbers; metrics computed dynamically.
- A9: Pooled OOF metrics + per-fold values reported.
- A11: Reproducibility via set_seed.
- A15: Support resuming interrupted runs without restarting from scratch.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .common import (
    set_seed,
    assert_disjoint,
    metrics,
    preprocess,
)
from .model import build_model


def augment_tile(tile_bgr: np.ndarray) -> np.ndarray:
    """
    Data augmentation satisfying Rule B1 and Phase 3/4 standards:
    - Horizontal / Vertical flips (p=0.5)
    - 90-degree rotations (0, 90, 180, 270 deg)
    - Brightness & contrast jitter +/-30%
    - Mild Gaussian blur (p=0.2)
    - JPEG re-compression (quality 70-95, p=0.3)
    - No diagonal crop or non-90 deg rotations that create empty corners (Rule A5).
    """
    import random
    import cv2
    if random.random() < 0.5:
        tile_bgr = cv2.flip(tile_bgr, 1)
    if random.random() < 0.5:
        tile_bgr = cv2.flip(tile_bgr, 0)
    k = random.choice([0, 1, 2, 3])
    if k > 0:
        tile_bgr = np.rot90(tile_bgr, k).copy()

    alpha = 1.0 + random.uniform(-0.3, 0.3)
    beta = random.uniform(-30, 30)
    tile_bgr = np.clip(alpha * tile_bgr.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Rule B1: Mild Gaussian blur & JPEG re-compression
    if random.random() < 0.2:
        tile_bgr = cv2.GaussianBlur(tile_bgr, (3, 3), sigmaX=0.5)

    if random.random() < 0.3:
        q = random.randint(70, 95)
        _, enc = cv2.imencode(".jpg", tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        tile_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return tile_bgr


class TileDataset(Dataset):
    """PyTorch Dataset loading micrograph tiles from disk with unified preprocessing."""
    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        input_size: int = 256,
        scaler: Optional[Any] = None,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.input_size = input_size
        self.scaler = scaler
        self.augment = augment

        self.paths = self.df["tile_path"].tolist()
        self.specimen_ids = self.df["specimen_id"].tolist()

        raw_targets = self.df[self.target_cols].values.astype(np.float32)
        if self.scaler is not None:
            self.targets = self.scaler.transform(raw_targets).astype(np.float32)
        else:
            self.targets = raw_targets

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        path = self.paths[idx]
        spec_id = self.specimen_ids[idx]
        target = torch.from_numpy(self.targets[idx])

        # Load tile using cv2 (BGR)
        import cv2
        tile_bgr = cv2.imread(path)
        if tile_bgr is None:
            raise FileNotFoundError(f"Failed to read tile at {path}")

        if self.augment:
            tile_bgr = augment_tile(tile_bgr)

        # Unified Rule A4 preprocessing
        tensor = preprocess(tile_bgr, size=self.input_size)
        return tensor, target, spec_id, path


def train_one_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_cols: List[str],
    out_dir: Path,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Train a single cross-validation fold with strict Rule A1-A15 compliance.
    """
    fold_dir = out_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = fold_dir / "checkpoint_last.pt"
    val_preds_path = fold_dir / "val_predictions.csv"

    # Rule A1: Assert in code that specimen_ids in train and val are strictly disjoint
    train_specimens = train_df["specimen_id"].unique()
    val_specimens = val_df["specimen_id"].unique()
    assert_disjoint(train_specimens, val_specimens)

    # Check for existing fold completion (Rule A15: per-fold resume)
    if last_ckpt_path.exists() and val_preds_path.exists():
        print(f"[Fold {fold_idx}] Completed checkpoint and predictions found. Loading saved results...")
        saved_preds = pd.read_csv(val_preds_path)
        fold_metrics = metrics(saved_preds["target_true"].values, saved_preds["target_pred"].values)
        return saved_preds, fold_metrics

    # Rule A3: Fit target scaler on training fold only, apply to val
    from sklearn.preprocessing import StandardScaler
    scaler = None
    if config.get("scale_targets", False):
        scaler = StandardScaler()
        scaler.fit(train_df[target_cols].values)

    input_size = config.get("input_size", 256)
    train_dataset = TileDataset(train_df, target_cols, input_size=input_size, scaler=scaler, augment=True)
    val_dataset = TileDataset(val_df, target_cols, input_size=input_size, scaler=scaler, augment=False)

    mini_batch_size = config.get("mini_batch_size", 16)
    effective_batch_size = config.get("effective_batch_size", 32)
    accum_steps = max(1, effective_batch_size // mini_batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=mini_batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=mini_batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=True if device.type == "cuda" else False,
    )

    # Build model architecture
    backbone = config.get("backbone", "tf_efficientnetv2_s")
    model = build_model(
        n_outputs=len(target_cols),
        backbone=backbone,
        pretrained=config.get("pretrained", True),
    ).to(device)

    # Optimizer & LR Scheduler (AdamW + Cosine)
    lr = float(config.get("lr", 3e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = config.get("epochs", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion = nn.MSELoss()
    use_amp = (device.type == "cuda" and config.get("amp", True))
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[Fold {fold_idx}] Starting training: {epochs} epochs, {len(train_dataset)} train tiles, {len(val_dataset)} val tiles...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, (inputs, targets, _, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(inputs)
                loss = criterion(preds, targets)
                loss_scaled = loss / accum_steps

            scaler_amp.scale(loss_scaled).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / max(1, n_batches)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  [Fold {fold_idx} | Epoch {epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Rule A2: checkpoint_mode = "last" (weights strictly after final epoch, never selected on val fold)
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "config": config,
            "fold": fold_idx,
            "scaler": scaler,
        },
        last_ckpt_path,
    )
    print(f"[Fold {fold_idx}] Saved final epoch checkpoint to {last_ckpt_path} (Rule A2 compliant).")

    # Evaluate validation fold using final checkpoint
    model.eval()
    val_preds_list = []
    val_targets_list = []
    val_specimens_list = []
    val_paths_list = []

    with torch.no_grad():
        for inputs, targets, spec_ids, paths in val_loader:
            inputs = inputs.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(inputs)
            val_preds_list.append(preds.cpu().numpy())
            val_targets_list.append(targets.numpy())
            val_specimens_list.extend(spec_ids.tolist())
            val_paths_list.extend(paths)

    raw_preds = np.vstack(val_preds_list)
    raw_targets = np.vstack(val_targets_list)

    # Invert target scaling if applied (Rule A3)
    if scaler is not None:
        inv_preds = scaler.inverse_transform(raw_preds)
        inv_targets = scaler.inverse_transform(raw_targets)
    else:
        inv_preds = raw_preds
        inv_targets = raw_targets

    # Save out-of-fold validation predictions
    val_df_out = pd.DataFrame({
        "tile_path": val_paths_list,
        "specimen_id": val_specimens_list,
        "fold": fold_idx,
    })
    for i, col in enumerate(target_cols):
        val_df_out[f"{col}_true"] = inv_targets[:, i]
        val_df_out[f"{col}_pred"] = inv_preds[:, i]

    # For primary target
    primary_col = target_cols[0]
    val_df_out["target_true"] = val_df_out[f"{primary_col}_true"]
    val_df_out["target_pred"] = val_df_out[f"{primary_col}_pred"]
    val_df_out.to_csv(val_preds_path, index=False)

    fold_metrics = metrics(val_df_out["target_true"].values, val_df_out["target_pred"].values)
    print(f"[Fold {fold_idx}] Validation: R2={fold_metrics['R2']:.4f}, MAE={fold_metrics['MAE']:.4f}, RMSE={fold_metrics['RMSE']:.4f}")
    return val_df_out, fold_metrics


def run_kfold_training(
    df: pd.DataFrame,
    target_cols: List[str],
    folds_dict: Dict[int, int],
    out_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run full 5-fold cross-validation with pooled OOF metric evaluation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or {}
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} ({get_gpu_info() if device.type == 'cuda' else 'CPU'})")

    df_tiles = df.copy()
    df_tiles["fold"] = df_tiles["specimen_id"].map(folds_dict)
    if df_tiles["fold"].isna().any():
        unmapped = df_tiles[df_tiles["fold"].isna()]["specimen_id"].unique()
        raise ValueError(f"Unmapped specimen_ids found in fold dictionary: {unmapped}")

    all_folds = sorted(df_tiles["fold"].unique())
    oof_predictions = []
    per_fold_metrics = {}

    for f_idx in all_folds:
        train_df = df_tiles[df_tiles["fold"] != f_idx]
        val_df = df_tiles[df_tiles["fold"] == f_idx]
        val_preds_df, f_metrics = train_one_fold(
            fold_idx=int(f_idx),
            train_df=train_df,
            val_df=val_df,
            target_cols=target_cols,
            out_dir=out_dir,
            config=cfg,
            device=device,
        )
        oof_predictions.append(val_preds_df)
        per_fold_metrics[int(f_idx)] = f_metrics

    # Concatenate all OOF predictions (Rule A9)
    df_oof = pd.concat(oof_predictions, ignore_index=True)
    df_oof.to_csv(out_dir / "oof_predictions.csv", index=False)

    # Compute pooled OOF metrics
    pooled_metrics = metrics(df_oof["target_true"].values, df_oof["target_pred"].values)
    print("\n" + "=" * 50)
    print(f"POOLED OUT-OF-FOLD METRICS: R2={pooled_metrics['R2']:.4f}, MAE={pooled_metrics['MAE']:.4f}, RMSE={pooled_metrics['RMSE']:.4f}, r={pooled_metrics['Pearson_r']:.4f}")
    print("=" * 50)

    summary = {
        "pooled_metrics": pooled_metrics,
        "per_fold_metrics": per_fold_metrics,
        "config": cfg,
    }
    with open(out_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df_oof, summary

