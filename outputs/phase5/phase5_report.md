# Phase 5: Optical Robustness Evaluation Report

**Date:** 2026-11-04 02:15:10  
**Evaluation Mode:** Full Out-of-Fold Cross-Validation Across All 5 Folds (50,494 tiles per perturbation, Rule A1)  
**Evaluator:** NVIDIA RTX 4060 GPU + PyTorch Mixed Precision  

---

## 1. Experimental Overview

To evaluate real-world microscopic imaging robustness, we benchmarked the trained Phase 3 CNN (`tf_efficientnetv2_s`) against the calibrated flat-field Otsu thresholding pipeline across **12 realistic optical and compression perturbations** plus the clean baseline:
1. **Brightness variation:** Dimming ($	imes 0.7, 	imes 0.8$) and Over-exposure ($	imes 1.2, 	imes 1.3$).
2. **Contrast variation:** Low contrast ($	imes 0.5$) and High contrast ($	imes 1.5$).
3. **Out-of-focus blur:** Gaussian blur with $\sigma = 1\,	ext{px}$ and $\sigma = 2\,	ext{px}$.
4. **Microscope vignetting:** Radial illumination attenuation $I \cdot (1 - 0.4 \cdot r/r_{\max})$.
5. **Sensor noise:** Additive Gaussian noise with $\sigma = 10$ and $\sigma = 20$ intensity levels.
6. **Lossy storage:** JPEG re-compression at quality 50.

---

## 2. Summary Metrics Across All Perturbations

| Perturbation Condition | Calibrated Threshold $R^2$ [95% CI] | Calibrated Threshold MAE (%) | CNN $R^2$ [95% CI] | CNN MAE (%) | Winner ($R^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **clean** | 0.9744 [0.9691, 0.9787] | 0.7894 | 0.9969 [0.9956, 0.9978] | 0.2106 | **CNN** |
| **bright_0.7** | 0.9789 [0.9746, 0.9826] | 0.7128 | 0.9965 [0.9950, 0.9975] | 0.2249 | **CNN** |
| **bright_0.8** | 0.9765 [0.9715, 0.9807] | 0.7497 | 0.9967 [0.9955, 0.9977] | 0.2089 | **CNN** |
| **bright_1.2** | 0.9502 [0.9393, 0.9592] | 1.1897 | 0.9787 [0.9693, 0.9856] | 0.5581 | **CNN** |
| **bright_1.3** | 0.9449 [0.9338, 0.9540] | 1.2922 | 0.9796 [0.9706, 0.9861] | 0.5833 | **CNN** |
| **contrast_0.5** | 0.9724 [0.9664, 0.9772] | 0.8471 | 0.9756 [0.9654, 0.9831] | 0.6437 | **CNN** |
| **contrast_1.5** | 0.9843 [0.9800, 0.9880] | 0.5720 | 0.9956 [0.9941, 0.9967] | 0.3133 | **CNN** |
| **blur_1** | 0.9709 [0.9656, 0.9753] | 0.8813 | 0.9846 [0.9779, 0.9891] | 0.5767 | **CNN** |
| **blur_2** | 0.9620 [0.9549, 0.9675] | 1.0714 | 0.9052 [0.8720, 0.9289] | 1.5357 | Threshold |
| **vignetting** | 0.6540 [0.5549, 0.7215] | 3.4799 | 0.9267 [0.8796, 0.9606] | 0.7686 | **CNN** |
| **noise_10** | -0.4633 [-0.8922, -0.1814] | 7.2860 | 0.7929 [0.7367, 0.8315] | 3.4356 | **CNN** |
| **noise_20** | -1.0196 [-1.5955, -0.6373] | 9.2660 | 0.0224 [-0.2004, 0.1830] | 7.9886 | **CNN** |
| **jpeg_50** | 0.9748 [0.9698, 0.9789] | 0.7971 | 0.9964 [0.9951, 0.9974] | 0.2467 | **CNN** |

---

## 3. Degradation Curves

![Robustness Degradation](robustness_degradation_curves.png)

---

## 4. Gate 5 Verdict

### GATE 5 VERDICT:
- **Perturbations where the CNN degrades LESS than the calibrated threshold (CNN superior):**
  bright_0.7, bright_0.8, bright_1.2, bright_1.3, contrast_0.5, contrast_1.5, blur_1, vignetting, noise_10, noise_20, jpeg_50
- **Perturbations where the Calibrated Threshold degrades LESS than the CNN (Threshold superior):**
  blur_2


---
