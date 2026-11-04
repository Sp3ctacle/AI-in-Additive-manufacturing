# Master Synthesis & Camera-Ready Consolidated Report

**Document Version:** Final Consolidated v5  
**Compilation Date:** 2026-11-04 13:34:38  
**Hardware Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)  
**Dataset Grounding:** Ti-6Al-4V LPBF Micrographs (SLM 125 HL, Zenodo 8303011) & Ackermann et al. 316L Stainless Steel (EOS M 290)  
**Core Governance:** Strict adherence to Global Rules A1–A16 and Gate Fix-Ups F1–F8.

---

## 1. Master Two-Column Gate Evaluation Matrix

As mandated by Fixup F8, every phase gate is evaluated across two distinct, independent columns:
1. **Test Executed Correctly?** Validates experimental protocol, code correctness, data partitioning, leakage prevention, and cluster bootstrap CI computation.
2. **Hypothesis Supported?** States whether the empirical metallurgical or machine learning hypothesis was confirmed by the evidence (or whether the null / falsification was upheld).

| Gate | Phase Name | Primary Hypothesis Evaluated | Test Executed Correctly? | Hypothesis Supported? | Decisive Finding / Statistical Evidence |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Gate 1** | Phase 1: Tile & Specimen Pipeline | Multi-scale panoramic tiling preserves true void fraction without resin contamination | **YES** | **YES** | 50,494 tiles extracted across 734 specimens; 0 contaminated tiles; scale confirmed at 2.2325 $\mu$m/px. |
| **Gate 1b** | Phase 1b: Dataset Quality Assurance | Automated QC flags stitch defects, contrast drift, and mosaic seams | **YES** | **YES** | All QC checks pass; zero resin leakage into specimen bulk porosity masks. |
| **Gate 2** | Phase 2: Baselines & Data Splits | Classical thresholding baseline establishes benchmark for void segmentation | **YES** | **YES** | Global Otsu achieves Specimen $R^2 = 0.9942$, MAE $= 0.6310\%$; specimen grouping strictly enforced. |
| **Gate 3** | Phase 3: Forward Porosity CNN | Deep vision backbones predict optical porosity from unsegmented grayscale tiles | **YES** | **YES** | EfficientNetV2-S achieves Specimen $R^2 = 0.9969$, MAE $= 0.2106\%$ across 5-fold cross-validation. |
| **Gate 3b** | Phase 3b: Fair Classical Baseline | CNN beats a fair per-tile Otsu thresholding baseline at specimen level | **YES** | **NO (STATISTICAL PARITY)** | Paired $\Delta\text{MAE} = -0.0054\%$ [95% CI: $-0.0415\%, +0.0280\%$]. CI strictly includes 0 $\implies$ statistical parity. |
| **Gate 4** | Phase 4: Inverse Physics Regression | Tile visual features identify laser power $P$, scan speed $v$, hatch $h$, and layer $t$ | **YES** | **YES** | Validated: $P$ ($R^2=0.8844$), $v$ ($R^2=0.8291$), $h$ ($R^2=0.7481$), $t$ ($R^2=0.7485$). |
| **Gate 4b** | Phase 4b: Physical Causality Diagnostics | Hatch spacing leaves spatial periodicity in lack-of-fusion voids | **YES** | **YES** | Autocorrelation detects hatch pitch in $19.75\%$ of LOF specimens ($p < 0.0001$ vs $12.03\%$ null). Section is horizontal $xy$. |
| **Gate 5** | Phase 5: Optical Sensor Robustness | CNN preserves calibration under sensor degradation better than thresholding | **YES** | **YES** | CNN beats Otsu in 10 of 12 perturbations; Otsu wins only under low contrast ($0.5\times$). Both fail under extreme noise. |
| **Gate 6** | Phase 6: Defect Morphology Classification | CNN classifies pore physical root causes (Keyhole, LOF, Gas pores) | **YES** | **YES** | CNN matches tabular shape classifiers ($97.75\%$ vs $97.25\%$, McNemar $p=0.3616$, n.s.). Random split leaks $+0.42\%$. |
| **Gate 7** | Phase 7: Cross-Alloy Domain Transfer | Steel pretraining transfers zero-shot and outperforms ImageNet initialization | **YES** | **NO (FALSIFIED)** | Zero-shot fails ($R^2=-6.44$) due to reflectance offset. Adaptation matches ImageNet ($R^2 \approx 0.87$ vs $0.85$, CI includes 0). |
| **Gate 8** | Phase 8: Explainability & Saliency Axioms | Saliency maps are faithful to CNN decision logic via pixel deletion | **YES** | **NO (FALSIFIED)** | Grad-CAM is pore-grounded (3.36x enrichment), but targeted deletion fails to drop prediction faster than random ($\Delta\text{AUC} = -4.46$). |
| **Gate 10** | Phase 10: Porosity & Defect Assessment | Conformal prediction delivers $\ge 90\%$ valid coverage and quality screening | **YES** | **YES** | Conformal coverage is **91.96%** (within $\pm 2\%$ window); accept/reject accuracy is **98.37%**; tool matches OOF. |
| **Gate 9** | Phase 9: Camera-Ready Consolidation | All metrics traceable to JSON; figures regenerated; 10 random checks pass | **YES** | **YES** | 10 of 10 consistency checks match to 4 decimal places; package assembled for submission. |

---

## 2. Decisive Scientific Insights & Physical Grounding

1. **Equivalence of Deep Vision and Classical Morphology on Clean Data:**
   On pristine brightfield micrographs, EfficientNetV2-S (MAE $0.2106\%$) and Fair 256px Otsu (MAE $0.2053\%$) achieve statistical parity ($\Delta\text{MAE} = -0.0054\%$, 95% CI includes zero). The true superiority of deep CNNs lies exclusively in **optical robustness under real-world sensor degradation** (vignetting, blur, illumination drift, noise).
2. **Cross-Alloy Transfer Falsification:**
   Pretraining on 316L stainless steel does not provide superior representations for Ti-6Al-4V porosity over standard ImageNet features. General edge and texture filters from ImageNet adapt equally or better ($R^2 = 0.874$ vs $0.853$ at 10% target data), disproving the assumption that alloy-specific pretraining is required.
3. **Pore-Grounded Saliency vs Deletion Faithfulness:**
   Grad-CAM heatmaps localize physical voids with 3.36x enrichment over matrix metal. However, causal pixel deletion demonstrates that CNN predictions do not collapse faster under void deletion than random masking ($\Delta\text{AUC} = -4.46$). Saliency maps must be presented as **visual feature localization**, not faithful causal decision paths.
4. **Decoupling of Optical and Archimedes Porosity in Dense Parts:**
   In parts with $>98\%$ bulk density, optical porosity and Archimedes fluid immersion density decouple completely (Spearman $\rho = 0.1259$). Surface roughness and open surface pores introduce Archimedes buoyancy artifacts, proving that direct optical image assessment is mandatory for near-dense qualification.
5. **Split Conformal Uncertainty for Industrial Quality Assurance:**
   The implemented split conformal prediction framework guarantees **91.96%** empirical coverage at a nominal 90% target. Quality screening at a $0.5\%$ porosity specification limit achieves **98.37% accuracy** and **99.48% specificity**, enabling automated metallographic lot release.
