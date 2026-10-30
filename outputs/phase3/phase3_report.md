# Phase 3: Forward Model Report
**Ti-6Al-4V LPBF Micrographs Pipeline**
**Date:** 2026-10-30 11:45:24
**Global Rules Enforced:** A1-A15

---

## Executive Summary & Gate 3 Verdict

| Gate 3 Requirement | Finding / Result | Status |
| :--- | :--- | :--- |
| **1. Multi-Target CNN** | EfficientNetV2-S predicting [pore%, log N, roundness] with sqrt-transform | **PASS** |
| **2. 5-Fold Stratified CV** | Strictly disjoint specimens across 5 folds from `folds.json`; 40 epochs | **PASS** |
| **3. Specimen-Level Porosity** | $R^2 = 0.9969$ (95% CI: [0.9957, 0.9979]), MAE = 0.211% | **PASS** |
| **4. Comparison vs. Calibrated Thresh** | Delta MAE = +0.421% [95% CI: +0.331%, +0.520%] | **BEATS** |
| **5. Comparison vs. Pixel Fraction** | Delta MAE = +0.988% [95% CI: +0.899%, +1.080%] | **BEATS** |

### **GATE 3 VERDICT: PASS**
- **Plain-Language Summary:**
  The CNN forward model achieves an area-weighted specimen-level $R^2$ of **0.9969** and MAE of **0.211%** for porosity, and $R^2 = 0.9974$ for pore count.
  Compared to the calibrated threshold baseline (specimen MAE +0.421%), the CNN demonstrates robust morphology awareness while jointly predicting spatial defect count and shape roundness.

---

## 1. Multi-Target Forward Performance

All metrics evaluated Out-Of-Fold (OOF) across the 5 cross-validation folds. Confidence intervals computed via 2,000 bootstrap resamples over **specimens** (Rule A9).

### Tile-Level Performance (50,494 Tiles)

| Target | $R^2$ (95% CI) | MAE (95% CI) | RMSE (95% CI) | Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: |
| **Pore% (%)** | **0.9181** [0.8928, 0.9415] | **0.344%** [0.301, 0.392] | **3.052%** [2.569, 3.518] | **0.9586** |
| **Log Pore Count ($\log(1+N)$)** | **0.9712** [0.9662, 0.9750] | **0.094** | **0.149** | **0.9855** |
| **Convex Hull Roundness** | **0.9150** [0.8948, 0.9304] | **0.0221** | **0.0376** | **0.9566** |

### Specimen-Level Performance (734 Retained Specimens)

| Target | Specimen $R^2$ (95% CI) | Specimen MAE (95% CI) | Specimen RMSE (95% CI) | Specimen $r$ |
| :--- | :---: | :---: | :---: | :---: |
| **Pore% (%)** | **0.9969** [0.9957, 0.9979] | **0.211%** [0.174, 0.248] | **0.551%** [0.447, 0.648] | **0.9984** |
| **Log Pore Count ($\log(1+N)$)** | **0.9974** | **0.024** | **0.037** | **0.9989** |
| **Convex Hull Roundness** | **0.9945** | **0.0050** | **0.0080** | **0.9973** |

---

## 2. Comparison to Phase 2 Baselines (Rule A7/A9/A10)

Baseline metrics loaded directly from [`outputs/phase2/baselines.json`](file:///outputs/phase2/baselines.json).

| Baseline Model | Specimen Baseline MAE (%) | CNN Specimen MAE (%) | Delta MAE (%) (95% CI) | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Calibrated Threshold** | 0.631% | **0.211%** | +0.421% [+0.331%, +0.520%] | BEATS (p < 0.05) |
| **Pixel Fraction (Linear)** | 1.198% | **0.211%** | +0.988% [+0.899%, +1.080%] | BEATS (p < 0.05) |

---

## 3. Deliverables Registry

1. [`outputs/phase3/oof_predictions.csv`](file:///outputs/phase3/oof_predictions.csv): Full tile-level out-of-fold multi-target predictions.
2. [`outputs/phase3/oof_predictions_specimen.csv`](file:///outputs/phase3/oof_predictions_specimen.csv): Specimen-level aggregated predictions.
3. [`outputs/phase3/forward_summary.json`](file:///outputs/phase3/forward_summary.json): Complete machine-readable metrics with 95% bootstrap CIs.
4. [`outputs/phase3/phase3_report.md`](file:///outputs/phase3/phase3_report.md): Master Phase 3 report.
5. [`outputs/phase3/run_info.json`](file:///outputs/phase3/run_info.json): Execution reproducibility record.
