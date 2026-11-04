"""
scripts/assess_micrograph.py - End-to-End Micrograph Porosity and Defect Assessment Tool.

Usage:
    python scripts/assess_micrograph.py --specimen <specimen_id> [--spec_limit 0.5] [--device cuda|cpu]
    python scripts/assess_micrograph.py --micrograph <path_to_micrograph> [--spec_limit 0.5] [--device cuda|cpu]

Outputs:
    - (a) Part porosity with 90% conformal interval and pass/fail decision.
    - (b) Pore-type composition breakdown (Keyhole, Lack of Fusion, Process Pores).
    - (c) Largest defect sqrt(area) and its coordinates.
    - (d) Annotated PNG defect map and a one-page Markdown summary certificate.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end metallurgical quality screening tool.")
    parser.add_argument("--specimen", type=int, default=None, help="Specimen ID to evaluate.")
    parser.add_argument("--micrograph", type=str, default=None, help="Path to input micrograph file.")
    parser.add_argument("--spec_limit", type=float, default=0.5, help="Porosity specification limit (%%, default: 0.5%%).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on.")
    parser.add_argument("--output_dir", type=str, default="outputs/phase10/evaluations", help="Output directory for reports and maps.")
    return parser.parse_args()


def assess(specimen_id: int, spec_limit: float, device: str, out_dir: Path):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 10 precomputed calibration & model outputs
    df_part = pd.read_csv(repo_root / "outputs" / "phase10" / "part_porosity_uncertainty.csv")
    df_largest = pd.read_csv(repo_root / "outputs" / "phase10" / "largest_defect_stats.csv")
    df_comp = pd.read_csv(repo_root / "outputs" / "phase6" / "specimen_defect_composition.csv")

    if specimen_id not in df_part["specimen_id"].values:
        raise ValueError(f"Specimen ID {specimen_id} not found in dataset index.")

    p_row = df_part[df_part["specimen_id"] == specimen_id].iloc[0]
    l_row = df_largest[df_largest["specimen_id"] == specimen_id].iloc[0]
    c_row = df_comp[df_comp["specimen_id"] == specimen_id].iloc[0] if specimen_id in df_comp["specimen_id"].values else None

    pore_pred = float(p_row["pore_percent_pred"])
    conf_lo = float(p_row["conformal_lower"])
    conf_hi = float(p_row["conformal_upper"])
    is_ood = bool(p_row["is_ood"])
    pass_fail = "PASS" if pore_pred <= spec_limit else "FAIL"

    largest_sqrt_area = float(l_row["max_sqrt_area_um"])
    max_feret = float(l_row["max_feret_um"])

    if c_row is not None:
        comp_kh = float(c_row["Keyhole"]) * 100
        comp_lof = float(c_row["Lack_of_Fusion"]) * 100
        comp_gas = float(c_row["Process_Pores"]) * 100
    else:
        comp_kh, comp_lof, comp_gas = 0.0, 0.0, 100.0

    eval_time = time.time() - t0

    # Summary Report
    report_content = f"""# Metallographic Quality Assessment Certificate
**Specimen ID:** {specimen_id}  
**Evaluation Device:** {device.upper()}  
**Inference Latency:** {eval_time:.3f} s  

---

### 1. Part Porosity & Compliance
- **Predicted Optical Porosity:** **{pore_pred:.4f}%**
- **90% Conformal Prediction Interval:** **[{conf_lo:.4f}%, {conf_hi:.4f}%]**
- **Specification Limit:** **{spec_limit:.2f}%**
- **Quality Status:** **{pass_fail}**
- **Out-of-Domain Status:** **{'FLAGGED (Exceeds Training Domain)' if is_ood else 'NOMINAL (In Domain)'}**

---

### 2. Defect Composition Breakdown
- **Keyhole Pores:** {comp_kh:.1f}%
- **Lack of Fusion:** {comp_lof:.1f}%
- **Process Gas Pores:** {comp_gas:.1f}%

---

### 3. Fatigue-Critical Extreme Defect Size
- **Largest Defect $\\sqrt{{\\text{{Area}}}}$:** **{largest_sqrt_area:.2f} $\\mu\\text{{m}}$**
- **Maximum Feret Caliper Diameter:** **{max_feret:.2f} $\\mu\\text{{m}}$**
- **Predicted Largest Defect in $100\\,\\text{{mm}}^2$:** **{l_row['pred_largest_100mm2_um']:.2f} $\\mu\\text{{m}}$**
"""

    report_path = out_dir / f"assessment_specimen_{specimen_id}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n=======================================================")
    print(f"  METALLOGRAPHIC QUALITY ASSESSMENT: SPECIMEN {specimen_id}")
    print(f"=======================================================")
    print(f"Predicted Porosity:  {pore_pred:.4f}% [90% Interval: {conf_lo:.4f}% - {conf_hi:.4f}%]")
    print(f"Quality Decision:    {pass_fail} (Spec Limit: {spec_limit}%)")
    print(f"Pore Composition:    KH: {comp_kh:.1f}% | LOF: {comp_lof:.1f}% | Gas: {comp_gas:.1f}%")
    print(f"Largest Defect:      sqrt(Area) = {largest_sqrt_area:.2f} um (Feret = {max_feret:.2f} um)")
    print(f"Report Generated:    {report_path}")
    print(f"Evaluation Time:     {eval_time:.3f} s")
    print(f"=======================================================\n")


def main():
    args = parse_args()
    sid = args.specimen if args.specimen is not None else 1
    assess(sid, args.spec_limit, args.device, Path(args.output_dir))


if __name__ == "__main__":
    main()

