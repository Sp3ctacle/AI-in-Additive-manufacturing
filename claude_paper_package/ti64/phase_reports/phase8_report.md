# Phase 8: Explainability (XAI) with Rigorous Sanity Checks Report (Updated)

**Evaluation Date:** 2026-11-04 12:25:40  
**Evaluation Mode:** 24 Stratified Validation Tiles (Folds 1 and 3, Tertiles of $E_d$)  
**Target Architecture:** EfficientNetV2-S (`tf_efficientnetv2_s`)  
**Global Rules Enforced:** Rule A10 (Honest Reporting), Rule A13 (XAI Sanity Checks)  

---

## 1. Experimental Overview & Metric Directions

Following Rule A13 and the Adebayo et al. explainability guidelines, saliency methods cannot be published as valid representations without formal falsification tests:
1. **Pore Grounding Enrichment:** Ratio of saliency inside true physical defect masks vs. solid matrix (ratio $> 1.0\times$ indicates true defect grounding).
2. **Adebayo Cascading Randomization:** Progressively randomizing model weights from output head back to convolutional stem. Spearman rank correlation $\rho$ must drop below $0.25$ to confirm sensitivity to learned weights rather than edge priors.
3. **Pixel Deletion Faithfulness:**
   - **Metric Direction:** Prediction change $|f(x_k) - f(x)|$, where larger change indicates that deleted pixels had higher true functional impact on the prediction (**larger = more faithful**).
   - Deletion curve from $k = 0\%$ to $20\%$ at step $2\%$, replacing pixels with a pore-free matrix grey baseline ($I = 140$).
   - Evaluated across all 24 validation tiles and 5 random seeds with paired bootstrap confidence intervals.
4. **Integrated Gradients Completeness:** Completeness axiom test evaluated across 200 interpolation steps from a pore-free solid matrix grey baseline ($I = 140$).

---

## 2. Sanity Check Results

| Sanity Check Test | Measured Metric | Pass Criterion | Statistical Verdict |
| :--- | :---: | :---: | :---: |
| **Pore Grounding Enrichment** | **3.36x** Median (**4.15x** Mean) | $> 1.0\text{x}$ over random baseline | **PASS** |
| **Cascading Randomization** | $\rho = 0.1842$ (Full Network) | $\rho < 0.25$ | **PASS** |
| **Pixel Deletion Faithfulness** | Paired $\Delta\text{AUC} = -4.4573$ [-5.9857, -2.9267] | $\text{AUC}_{\text{targeted}} > \text{AUC}_{\text{random}}$ with CI $> 0$ | **FAILED** |
| **Integrated Gradients Completeness** | Completeness Ratio $= 8.1257$ | $< 0.05$ | **FAILED** |

---

## 3. Detailed Physical & Methodological Interpretation

1. **Pore Grounding vs. Faithfulness Disconnect:**
   - Grad-CAM demonstrates strong spatial grounding on physical voids, focusing **3.36x** more saliency on genuine pore contours than on the surrounding solid metal matrix.
   - However, in the pixel deletion test, targeted deletion of the top salient pixels produced an AUC of **12.13**, which was lower than random pixel deletion (**16.59**), yielding a paired difference of **-4.4573 [95% CI: -5.9857, -2.9267]**.
   - Because the 95% bootstrap confidence interval strictly excludes zero and is negative, **Grad-CAM is pore-grounded, but is not shown to be functionally faithful by the pixel deletion sanity check**.
2. **Integrated Gradients Completeness Axiom Failure:**
   - When integrating gradients along 200 steps from the pore-free matrix baseline, the median completeness ratio was **8.1257**, violating the $< 0.05$ threshold.
   - Deep residual networks with Swish/SiLU activations suffer from extreme gradient saturation along the interpolation trajectory, causing integrated gradients to diverge from the total function change.
3. **Publication Claim Boundaries (Rule A10 & Rule A13):**
   - Neither Integrated Gradients nor Grad-CAM can be claimed as provably faithful explanations of individual model decisions.
   - Grad-CAM maps may be presented in publication figures **solely as qualitative visual evidence of defect co-localization**, with explicit text stating that deletion faithfulness was not verified.

---

## 4. Gate 8 Verdict

### GATE 8 VERDICT:
- **Test Executed Correctly:** **YES**. All 4 sanity checks executed across 24 tiles, 5 random seeds, and 2,000 bootstrap resamples.
- **Hypothesis Supported:** **NO (HYPOTHESIS OF FAITHFUL SALIENCY FALSIFIED)**.
  - Grad-CAM passes pore grounding and weight randomization.
  - Grad-CAM fails deletion faithfulness.
  - Integrated Gradients fails the completeness axiom.
  - **Verdict:** Publication claims must explicitly reflect that saliency maps are pore-co-localized but not faithful decision explanations.
