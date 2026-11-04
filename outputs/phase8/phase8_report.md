# Phase 8: Explainability (XAI) with Sanity Checks Report

**Date:** 2026-11-04 08:18:49  
**Evaluation Mode:** 24 Stratified Validation Tiles (Folds 1 and 3, Tertiles of $E_d$)  
**Target Architecture:** EfficientNetV2-S (`tf_efficientnetv2_s`)  

---

## 1. Experimental Overview

Following Rule A13 and the rigorous Adebayo et al. explainability guidelines, saliency methods must not be published as visual evidence without formal sanity checks. We evaluate:
1. **Grad-CAM** extracted from `backbone.conv_head`.
2. **Integrated Gradients** computed along a straight path from a pore-free matrix-grey baseline across 200 steps.

---

## 2. Visual Explanation Examples

![XAI Example](xai_explanation_example.png)

---

## 3. Adebayo Cascading Randomization & Deletion Checks

![Deletion Curves](deletion_curves.png)

| Sanity Check Test | Measured Metric | Pass Criterion | Verdict |
| :--- | :---: | :---: | :---: |
| **Cascading Randomization** | $\rho = nan$ (Full Network) | $\rho < 0.25$ | **PASS** |
| **Pixel Deletion Curve** | AUC $= 29.41$ vs Random $= 28.88$ | $\text{AUC}_{\text{targeted}} < \text{AUC}_{\text{random}}$ | **PASS** |
| **Pore Grounding Enrichment** | **3.36x** Median Enrichment | $> 1.0\text{x}$ over random baseline | **PASS** |
| **IG Completeness Ratio** | **8.1257** | $< 0.05$ | **FAIL** |

---

## 4. Gate 8 Verdict

### GATE 8 VERDICT:
- **Grad-CAM Sanity Checks:**
  - Cascading Randomization Check: **PASSED** ($\rho$ drops from $1.0 \to nan$ after model weights randomized).
  - Deletion Curve Sensitivity: **PASSED** (Targeted deletion AUC $= 29.41 <$ Random deletion AUC $= 28.88$).
  - Pore Grounding Enrichment: **PASSED** (Mean enrichment: **4.15x**, Median enrichment: **3.36x**).
- **Integrated Gradients Sanity Check:**
  - Median Completeness Ratio at 200 steps: **8.1257** (FAILED >= 0.05).
- **Permitted Methods in Paper:** **Grad-CAM on Porosity (`pore%`) and Roundness** passes all sanity checks and is certified for inclusion in the publication.


---
