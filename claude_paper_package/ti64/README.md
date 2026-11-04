# Claude Paper Package: LPBF Ti-6Al-4V Microstructural Diagnostics & Quality Assurance

**Release Version:** Final Consolidated v7 (Automated Trace Release)  
**Compilation Date:** 2026-11-04  
**Primary Material:** Ti-6Al-4V Grade 5 (Laser Powder Bed Fusion, SLM 125 HL)  
**Primary Dataset:** Zenodo Record [10.5281/zenodo.8303011](https://doi.org/10.5281/zenodo.8303011)  
**Transfer Source:** AISI 316L Stainless Steel (EOS M 290, Ackermann et al.)  
**Compute Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) | PyTorch 2.5.1 + CUDA 13.0  
**Experimental Governance:** Strictly governed by Global Rules A1–A16 and Gate Fix-Ups F1–F8 (GroupKFold disjoint partitioning, `checkpoint_mode='last'`, train-only scalers, physical units, 2,000 cluster-bootstrap CIs, honest negative result reporting).

---

## 1. Package Contents & Directory Layout

This package contains the complete, self-contained set of camera-ready artifacts, raw metrics, individual phase reports, key datasets, and diagnostic figures:

```
claude_paper_package/ti64/
├── README.md                           <- This comprehensive guide
├── final_summary.md                    <- Camera-ready executive summary & headline findings
├── all_gates_consolidated_report.md    <- Master consolidated gate report across all 12 gates
├── changelog.csv                       <- Metric audit log documenting all metric origins & fixes
├── verification_master.json            <- Unified JSON containing all gate results, CIs, and metadata
│
├── tables/                             <- Camera-ready paper tables (CSV + JSON provenance sidecars)
│   ├── table1_forward_models.csv (.json)             (Forward porosity CNN vs classical baselines)
│   ├── table2_inverse_physics.csv (.json)            (Inverse process parameter regression: Ed, P, v, h, t, dH/hs)
│   ├── table3_robustness_perturbations.csv (.json)   (Optical sensor robustness over 12 synthetic perturbations)
│   ├── table4_defect_classification.csv (.json)      (Defect morphology classification on crops)
│   ├── table5_conformal_uncertainty.csv (.json)      (Split conformal coverage stratified by density & porosity)
│   ├── table5b_quality_screening.csv (.json)         (Quality assurance screening at 0.5%, 1.0%, 2.0% limits)
│   └── table6_cross_alloy_adaptation.csv (.json)     (Steel-init vs ImageNet-init controlled transfer benchmarks)
│
├── figures/                            <- Camera-ready figures at 300 DPI
│   ├── figure1_sensor_robustness.png                 (Robustness comparison: CNN vs Fair Otsu across perturbations)
│   ├── figure2_conformal_calibration.png             (90% split conformal prediction calibration curve)
│   └── figure3_defect_confusion_matrix.png           (Defect classification confusion matrix on crops)
│
├── phase_reports/                      <- Full technical reports for each phase
│   ├── phase0_report.md                              (Raw inventory, archive verification, Ed identity)
│   ├── phase1_report.md                              (Image tiling, 200 um boundary margin, Nyquist criteria)
│   ├── phase1b_report.md                             (Flat-field correction, extreme QC, 39 missing LOF specimens)
│   ├── phase2_report.md                              (GroupKFold CV splits, optical vs Archimedes correlation)
│   ├── phase3_report.md                              (Forward porosity CNN architecture, multi-target head)
│   ├── phase3b_report.md                             (Fair 256px Otsu baseline comparison & statistical parity)
│   ├── phase4_report.md                              (Inverse process parameter regression & identifiability)
│   ├── phase4b_report.md                             (Hatch pitch autocorrelation, 100-permutation null, xy orientation)
│   ├── phase5_report.md                              (12 synthetic optical perturbations & paired bootstrap CIs)
│   ├── phase6_report.md                              (Morphological defect classification, crop-only scope)
│   ├── phase7_report.md                              (316L Stainless Steel transfer falsification & diagnostics)
│   ├── phase8_report.md                              (Grad-CAM void grounding & pixel deletion faithfulness check)
│   └── phase10_report.md                             (Split conformal prediction & Gumbel extreme value analysis)
│
├── phase_metrics/                      <- Raw canonical JSON metrics files
│   ├── phase0_run_info.json, phase1_run_info.json, phase1b_run_info.json
│   ├── phase2_baselines.json, phase2_folds.json (SHA-256 verified)
│   ├── phase3_forward_summary.json, phase3b_summary.json, phase3b_ge10um_metrics.json
│   ├── phase4_inverse_summary.json, phase4b_metrics.json
│   ├── phase5_robustness_metrics.json, phase6_poretype_metrics.json
│   ├── phase7_transfer_metrics.json, phase7_adaptation_comparison_summary.json, phase7_zero_shot_diagnostics.json
│   ├── phase8_xai_metrics.json
│   └── phase10_uncertainty_metrics.json, phase10_gumbel_specimen_tertiles.json
│
├── diagnostic_figures/                 <- Intermediate diagnostic figures
│   ├── phase1_qc_overlays.png
│   ├── phase1b_bland_altman_flatfield.png, phase1b_extreme_tiles_qc.png
│   ├── phase2_label_validation_scatter.png
│   ├── phase3b_bland_altman_stratified.png, phase3b_density_disagreement_top10.png
│   ├── phase4_ambiguity_analysis.png
│   ├── phase4b_ed_log_residual_analysis.png, phase4b_fair_baselines_comparison.png, phase4b_spatial_autocorrelation_lof.png
│   ├── phase5_robustness_degradation_curves.png
│   ├── phase6_confusion_matrix_cnn.png
│   ├── phase7_transfer_learning_curve.png
│   └── phase8_deletion_curves.png, phase8_xai_explanation_example.png
│
└── key_data/                           <- Core tabular CSV datasets (< 5 MB)
    ├── phase0_checksums.csv, phase0_inventory.csv, phase0_manifest.csv
    ├── phase1_specimen_density.csv, phase1_specimen_porosity.csv
    ├── phase1b_missing_specimens.csv
    ├── phase4b_equivalence_pairs.csv, phase4b_parameter_correlations.csv
    ├── phase5_robustness_metrics_summary_v2.csv
    ├── phase7_adaptation_control_results.csv
    └── phase10_gumbel_specimen_tertiles.csv
```

---

## 2. Decisive Scientific Findings

1. **Statistical Parity on Clean Micrographs:** On pristine brightfield micrographs, EfficientNetV2-S (MAE 0.2106%) and Fair 256px Otsu (MAE 0.2053%) achieve statistical parity (paired dMAE = -0.0054%, 95% CI includes zero). The advantage of deep vision backbones lies in **robustness under optical degradation** (vignetting, blur, illumination drift, noise), where the CNN wins 10 of 12 perturbations.
2. **Cross-Alloy Domain Transfer Falsification:** Pretraining on 316L stainless steel does not benefit Ti-6Al-4V porosity over standard ImageNet features. ImageNet-initialized backbones adapt equally or better (R2 = 0.8651 vs 0.8188 at 10% data), disproving the hypothesis that metallurgical pretraining on a different alloy transfers zero-shot or enhances sample efficiency.
3. **Pore-Grounded Saliency vs Deletion Faithfulness:** Grad-CAM saliency heatmaps localize true void masks with 3.36x enrichment over matrix metal. However, progressive pixel deletion demonstrates that CNN predictions do not collapse faster under void deletion than random deletion (dAUC = -4.4573). Saliency heatmaps represent visual feature localization, not faithful causal decision pathways.
4. **Optical vs. Archimedes Decoupling in Dense Regimes:** In parts with bulk density >= 98%, optical porosity and Archimedes immersion decouple to a rank correlation of Spearman rho = 0.1259 (p = 0.0089). This divergence is driven by Archimedes balance precision limits (+/- 0.2%) and 2D cross-section sampling variance; CT arbitration is mandatory for high-density lot release.
5. **Split Conformal Industrial Quality Assurance:** Split conformal prediction achieves 91.96% empirical coverage at a nominal 90% target on measured optical porosity (mean width 1.3930%). Quality screening at a 0.5% porosity limit achieves 96.78% balanced accuracy, 94.08% specificity, and 99.48% sensitivity on rejected parts.
6. **Defect Classification Scope:** Defect classification is validated on crops only (1,200 curated defect crops across 114 specimens: 97.75% accuracy, Macro-F1 0.9775, McNemar p = 0.3616 vs tabular shape descriptors). Full parent micrograph defect localization is unvalidated against parent annotations.

---

## 3. Data Integrity & Verification

All numbers in this package are generated strictly from code and verified against source data:
- Folds file hash: `b8cb074373f890b7a024dfa0cc8ad155f30d23f43663ce79af6967f5b5762792`
- All 16 core consistency checks pass with 4-decimal exact match.
- 100% of decimal numbers across reports and tables match source artifacts with tolerance < 1e-4.
