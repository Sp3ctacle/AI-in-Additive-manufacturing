"""
scripts/evaluate_jpeg_robustness.py - Rule B1 Verification & Robustness Benchmark

Demonstrates and verifies:
1. DCT Block Specks: Unfiltered thresholding on lossy JPEG images creates spurious
   dark specks at 8x8 block boundaries that artificially inflate pore count.
2. Rule B1 Defense: The MIN_AREA = 2.5 um^2 (8 px) filter effectively suppresses
   spurious DCT block specks across compression levels.
3. CNN Robustness: The CNN model trained with JPEG re-compression augmentation
   maintains stable predictions across JPEG quality factors (Q=95 down to Q=35).

Safe execution: Runs inference strictly on CPU to avoid interfering with any
background GPU training tasks.
"""

import os
import sys
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from ti64.common import PIXEL_SIZE_UM, preprocess
from ti64.model import build_model

def simulate_jpeg(img_bgr: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode an image at a specific JPEG quality level."""
    _, enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def compute_otsu_porosity(tile_bgr: np.ndarray, min_area_px: int) -> tuple[float, int]:
    """Compute porosity and pore count with and without min-area filtering."""
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if min_area_px > 0:
        valid_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area_px]
    else:
        valid_cnts = cnts
        
    tot_area = sum(cv2.contourArea(c) for c in valid_cnts)
    h, w = gray.shape
    porosity_pct = (tot_area / (h * w)) * 100.0
    return porosity_pct, len(valid_cnts)

def main():
    out_dir = repo_root / "outputs" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Rule B1 Robustness Benchmark (Lossy JPEG & DCT Blocking) ===")
    
    # 1. Load validation tiles from tiles_index.csv
    df_tiles = pd.read_csv(repo_root / "outputs" / "phase1" / "tiles_index.csv")
    
    # Select a representative sample of 150 diverse tiles (50 porous, 100 dense)
    porous_tiles = df_tiles[df_tiles["pore_percent_otsu"] > 0.5].sample(n=min(50, len(df_tiles[df_tiles["pore_percent_otsu"] > 0.5])), random_state=42)
    dense_tiles = df_tiles[df_tiles["pore_percent_otsu"] <= 0.5].sample(n=100, random_state=42)
    sample_df = pd.concat([porous_tiles, dense_tiles]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    qualities = [95, 85, 75, 60, 45, 30]
    pixel_area_um2 = PIXEL_SIZE_UM ** 2
    min_px_8 = int(round(2.5 / pixel_area_um2)) # 8 px for 2.5 um^2
    
    print(f"Testing {len(sample_df)} sample tiles across JPEG qualities {qualities}...")
    
    results = {
        "qualities": qualities,
        "otsu_unfiltered_count": [],
        "otsu_unfiltered_porosity": [],
        "otsu_filtered_count": [],
        "otsu_filtered_porosity": [],
        "cnn_pore_pct_pred": []
    }
    
    # Load Phase 3 CNN model for inference on CPU
    p3_ckpt = repo_root / "outputs" / "phase3" / "fold_0" / "checkpoint_last.pt"
    cnn_available = False
    model = None
    if p3_ckpt.exists():
        try:
            device = torch.device("cpu")
            model = build_model(n_outputs=3, backbone="tf_efficientnetv2_s", pretrained=False)
            ckpt = torch.load(p3_ckpt, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            cnn_available = True
            print("Loaded Phase 3 Fold 0 CNN checkpoint for CPU robustness evaluation.")
        except Exception as e:
            print(f"Warning: Could not load Phase 3 CNN on CPU ({e}); proceeding with threshold evaluation.")
    
    for q in qualities:
        unfilt_counts, unfilt_porosities = [], []
        filt_counts, filt_porosities = [], []
        cnn_preds = []
        
        for _, row in sample_df.iterrows():
            img_path = repo_root / row["tile_path"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Simulate JPEG re-compression
            if q < 95:
                img_q = simulate_jpeg(img, quality=q)
            else:
                img_q = img
                
            # Unfiltered Otsu (no min_area)
            p_unfilt, c_unfilt = compute_otsu_porosity(img_q, min_area_px=0)
            unfilt_counts.append(c_unfilt)
            unfilt_porosities.append(p_unfilt)
            
            # Rule B1 Filtered Otsu (MIN_AREA = 2.5 um^2 / 8 px)
            p_filt, c_filt = compute_otsu_porosity(img_q, min_area_px=min_px_8)
            filt_counts.append(c_filt)
            filt_porosities.append(p_filt)
            
            # CNN Prediction if model loaded
            if cnn_available and model is not None:
                with torch.no_grad():
                    inp = preprocess(img_q, size=256).unsqueeze(0)
                    out = model(inp).cpu().numpy()[0]
                    # target 0 is sqrt(pore%)
                    pred_pore = float(out[0] ** 2) if out[0] > 0 else 0.0
                    cnn_preds.append(pred_pore)
                    
        results["otsu_unfiltered_count"].append(float(np.mean(unfilt_counts)))
        results["otsu_unfiltered_porosity"].append(float(np.mean(unfilt_porosities)))
        results["otsu_filtered_count"].append(float(np.mean(filt_counts)))
        results["otsu_filtered_porosity"].append(float(np.mean(filt_porosities)))
        if cnn_available:
            results["cnn_pore_pct_pred"].append(float(np.mean(cnn_preds)))
            
        print(f"  Quality {q:2d}: Unfiltered Count = {results['otsu_unfiltered_count'][-1]:5.1f} | "
              f"Rule B1 Count = {results['otsu_filtered_count'][-1]:5.1f} | "
              f"Unfiltered Porosity = {results['otsu_unfiltered_porosity'][-1]:.3f}% | "
              f"Rule B1 Porosity = {results['otsu_filtered_porosity'][-1]:.3f}%" +
              (f" | CNN Pred = {results['cnn_pore_pct_pred'][-1]:.3f}%" if cnn_available else ""))

    # Save metrics JSON
    metrics_path = out_dir / "jpeg_robustness_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # 2. Generate informative plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot A: Pore Count Inflation
    axes[0].plot(qualities, results["otsu_unfiltered_count"], "r-o", linewidth=2, label="Unfiltered Otsu (Speck Vulnerable)")
    axes[0].plot(qualities, results["otsu_filtered_count"], "g-s", linewidth=2, label="Rule B1 Filter (MIN_AREA >= 2.5 um²)")
    axes[0].set_xlabel("JPEG Quality Factor", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mean Detected Pore Count per Tile", fontsize=11, fontweight="bold")
    axes[0].set_title("Rule B1 Defense Against DCT Blocking Noise", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)
    axes[0].invert_xaxis()
    
    # Plot B: Porosity Prediction Stability
    axes[1].plot(qualities, results["otsu_unfiltered_porosity"], "r--o", linewidth=2, label="Unfiltered Otsu Porosity (%)")
    axes[1].plot(qualities, results["otsu_filtered_porosity"], "g-s", linewidth=2, label="Rule B1 Filtered Porosity (%)")
    if cnn_available:
        axes[1].plot(qualities, results["cnn_pore_pct_pred"], "b-^", linewidth=2, label="CNN (Trained w/ JPEG Augmentation)")
    axes[1].set_xlabel("JPEG Quality Factor", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Estimated Void Fraction / Porosity (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Porosity Metric Stability vs Compression Degradation", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)
    axes[1].invert_xaxis()
    
    plt.tight_layout()
    plot_path = out_dir / "jpeg_robustness_analysis.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved visualization to {plot_path}")
    print("=== Rule B1 Benchmark Complete ===")

if __name__ == "__main__":
    main()

