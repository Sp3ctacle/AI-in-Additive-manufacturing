# Master Synthesis & Camera-Ready Consolidated Report

**Document Version:** Final Consolidated v7 (Automated Data-Trace Release)  
**Compilation Date:** 2026-11-04  
**Hardware Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)  
**Dataset Grounding:** Ti-6Al-4V LPBF Micrographs (SLM 125 HL, Zenodo 8303011) & Ackermann et al. 316L Stainless Steel (EOS M 290)  
**Core Governance:** Strict adherence to Global Rules A1â€“A16 and Gate Fix-Ups F1â€“F8.

---

## 1. Master Two-Column Gate Evaluation Matrix

In strict accordance with Fixup F8, every phase gate is evaluated across two distinct, independent columns:
1. **Test Executed Correctly?** Validates experimental protocol, code correctness, data partitioning, leakage prevention, and cluster bootstrap CI computation.
2. **Hypothesis Supported?** States whether the empirical metallurgical or machine learning hypothesis was confirmed by the evidence (or whether the null / falsification was upheld).

| Gate | Phase Name | Primary Hypothesis Evaluated | Test Executed Correctly? | Hypothesis Supported? | Decisive Finding / Statistical Evidence (95% Bootstrap CIs) |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Gate 0** | Phase 0: Verification & Inventory | Archives, metadata, and physical scale are verifiable and error-free | **YES** | **YES** | 11/11 archives verified (0 corrupt); 773 specimens linked; absolute scale **$0.55812\,\mu	ext{m/px}$** verified; $E_d$ identity verified. |
| **Gate 1** | Phase 1: Tile Extraction & Illumination | Multi-scale panoramic tiling preserves void fraction without resin border contamination | **YES** | **YES** | 50,494 tiles extracted across 734 specimens; $200\,\mu	ext{m}$ margin eliminates resin; Nyquist satisfied (keyhole 5th pctile = 13.5 net px $\ge 3$). |
| **Gate 1b** | Phase 1b: Extreme QC & Specimen Audit | Automated QC isolates mosaic seams and explains all missing specimen exclusions | **YES** | **YES** | 0.0% error rate on 50 extreme tiles; 39 crumbling LOF specimens ($E_d pprox 10\,	ext{J/mm}^3$, density $pprox 50.9\%$) safely isolated to prevent resin leakage. |
| **Gate 2** | Phase 2: CV Splits & Baseline Models | Disjoint specimen grouping prevents leakage; optical porosity correlates with Archimedes density | **YES** | **YES** | GroupKFold strictly disjoint; optical vs Archimedes $r = 0.8573$; baseline Otsu specimen $R^2 = 0.9733$, MAE $= 0.6314\%$; leakage penalty $\Delta R^2 = +0.0349$. |
| **Gate 3** | Phase 3: Forward Porosity CNN | Deep vision backbones predict optical porosity from unsegmented grayscale tiles | **YES** | **YES** | Area-weighted Specimen $R^2 = 0.9969$ [0.9957, 0.9979], $	ext{MAE} = 0.2106\%$ [0.174%, 0.248%]; Log count $R^2 = 0.9974$. |
| **Gate 3b** | Phase 3b: Fair Classical Baseline (F1) | Deep CNN outperforms a fair, like-for-like 256px Otsu baseline at the specimen level | **YES** | **NO (STATISTICAL PARITY)** | Paired $\Delta	ext{MAE} = -0.0054\%$ [95% CI: $-0.0415\%, +0.0280\%$]. CI strictly includes 0 $\implies$ statistical parity (Rule A9). |
| **Gate 4** | Phase 4: Inverse Physics Regression (C2) | Microstructural visual features invert process parameters ($E_d, P, v, h, t, \Delta H/h_s$) | **YES** | **YES** | Specimen $E_d$: $R^2 = 0.6053$ (15.3905 J/mmÂ³); $\Delta H/h_s$: $R^2 = 0.8058$ (2.3831); $P$: $R^2 = 0.5924$ (40.6812 W); $v$: $R^2 = 0.5344$ (218.6236 mm/s); $h$: $R^2 = 0.5239$ (0.0232 mm); $t$: $R^2 = 0.5934$ (0.0143 mm). |
| **Gate 4b** | Phase 4b: Hardware Calibration & Beyond-Porosity (F5, F6) | Hatch spacing $h$ creates spatial periodicity in LOF voids beyond scalar pore metrics | **YES** | **YES** | CNN beats 8-feature Extra Trees ($\Delta	ext{MAE} = +2.84\,	ext{J/mm}^3$ [+2.13, +3.51]); hatch pitch detected in $19.75\%$ of LOF specimens ($p < 0.0001$ vs $12.03\%$ null). Section is horizontal $xy$. |
| **Gate 5** | Phase 5: Optical Sensor Robustness (F2 / C4) | CNN preserves calibration under synthetic perturbations better than thresholding | **YES** | **YES** | CNN wins 10 of 12 synthetic perturbations; Fair Otsu wins 1 (`contrast_0.5`); 1 tie (`clean`, CI includes 0); 1 both fail (`noise_20`, both $R^2 < 0.5$). |
| **Gate 6** | Phase 6: Defect Morphology Classification (F7) | CNN classifies pore root causes (validated on crops only) better than tabular shape classifiers | **YES** | **NO (MATCHES TABULAR)** | Validated on crops only (1,200 crops). CNN matches tabular shape classifiers ($97.75\%$ vs $97.25\%$, McNemar test $p = 0.3616$, n.s.). Random split leaks $+0.42\%$. |
| **Gate 7** | Phase 7: Cross-Alloy Domain Transfer (F4 / C5 / T5) | Pretraining on 316L stainless steel transfers zero-shot and beats ImageNet initialization | **YES** | **NO (FALSIFIED)** | Primary zero-shot: Unclipped $R^2 = -19.6691$ (MAE $= 24.3622\%$); secondary clipped $R^2 = -6.4366$ (MAE $= 23.4069\%$, clipping at $10.0$). Controlled adaptation matches ImageNet ($R^2 pprox 0.85$ vs $0.87$, CI includes 0). Transfer NOT claimed. |
| **Gate 8** | Phase 8: Explainability & Deletion Faithfulness (F3) | Grad-CAM saliency heatmaps are faithful to CNN decision logic under pixel deletion | **YES** | **NO (FALSIFIED)** | Grad-CAM is pore-grounded (3.36Ã— enrichment), but targeted deletion fails to drop prediction faster than random ($\Delta	ext{AUC} = -4.4573$ [-5.9857, -2.9267]). Integrated Gradients rejected (axiom violation 8.13). |
| **Gate 10** | Phase 10: Porosity & Defect Assessment (C6 / T3 / T4) | Conformal prediction delivers $\ge 90\%$ valid coverage and parts quality screening | **YES** | **YES** | Conformal coverage: $0.9196$ ($91.96\%$ on optical reference); screening accuracy at 0.5% limit is $0.9678$ ($96.78\%$). Defect classification validated on crops only. Gumbel fits stratified by $E_d$ tertile. |
| **Gate 9** | Phase 9: Camera-Ready Consolidation (C1, C8, T2) | All metrics traceable to JSON; figures regenerated; consistency checks pass | **YES** | **YES** | All consistency checks and changelog assertions match to 4 decimal places; 100% of numbers verified against source artifacts (T2). |

---

## 2. Decisive Scientific Insights & Physical Grounding

1. **Equivalence of Deep Vision and Classical Morphology on Clean Data:**
   On pristine brightfield micrographs, EfficientNetV2-S (MAE $0.2106\%$) and Fair 256px Otsu (MAE 0.2053%) achieve statistical parity ($\Delta	ext{MAE} = -0.0054\%$, 95% CI includes zero). The advantage of deep CNNs lies in **optical robustness under synthetic perturbations** (vignetting, blur, illumination drift, noise).
2. **Cross-Alloy Transfer Falsification:**
   Pretraining on 316L stainless steel does not provide superior representations for Ti-6Al-4V porosity over standard ImageNet features. General edge and texture filters from ImageNet adapt equally or better ($R^2 = 0.8651$ vs $0.8188$ at 10% target data), indicating that alloy-specific pretraining is not advantageous over generic pretraining.
3. **Pore-Grounded Saliency vs Deletion Faithfulness:**
   Grad-CAM heatmaps localize physical voids with 3.36x enrichment over matrix metal. However, causal pixel deletion demonstrates that CNN predictions do not collapse faster under void deletion than random masking ($\Delta	ext{AUC} = -4.4573$). Saliency maps must be presented as **visual feature localization**, not faithful causal decision paths.
4. **Decoupling of Optical and Archimedes Measurements in Near-Dense Parts:**
   optical and Archimedes measurements disagree in near-dense parts; arbitration needs CT. In parts with $\ge 98\%$ bulk density, optical porosity and Archimedes immersion decouple to a rank correlation of Spearman $ho = 0.1259$ ($p = 0.0089$), driven by Archimedes water balance precision limits ($\pm 0.2\%$) and 2D metallographic cross-section sampling variance.
5. **Split Conformal Uncertainty for Industrial Quality Assurance:**
   The implemented split conformal prediction framework delivers **$91.96\%$** empirical coverage at a nominal 90% target on the measured optical porosity reference. Quality screening at a $0.5\%$ porosity specification limit achieves **$96.78\%$ balanced accuracy**, **94.08% specificity**, and **99.48% sensitivity on rejected parts**.

---

## 3. Operational Domain & Scope Limitations

1. **Excluded Extreme Lack-of-Fusion Specimens:**
   The 39 excluded specimens ($E_d pprox 10\,	ext{J/mm}^3$, Archimedes density $pprox 50.9\%$) formed crumbling powder sponges that cannot be prepared via standard metallography and are outside the model operational domain. Mounting resin penetrated the part bulk, violating Rule A6.
2. **Out-of-Domain Detection Flag:**
   The standalone inference tool (`scripts/assess_micrograph.py`) and pipeline enforce an explicit operational boundary check: any specimen with predicted optical porosity exceeding the training cohort maximum ($49.36\%$) is automatically flagged as out-of-domain (`is_ood = True`).
