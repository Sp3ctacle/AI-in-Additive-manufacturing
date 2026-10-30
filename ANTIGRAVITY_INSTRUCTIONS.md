ANTIGRAVITY PROMPT PACKAGE: FINAL MODEL ON THE Ti-6Al-4V DATASET (Zenodo 8303011)
Author: Siddharth | Version 3, 29 Oct 2026 (fresh-folder edition)
======================================================================
 
HOW TO USE
- This package runs in a NEW, EMPTY project folder. The old project folder is
  kept only as an archive: no script in this package imports, calls or reads
  code from it. Only raw data and Phase 0 outputs are copied in (Part 0).
- Paste PART A (global rules) first, ideally as a workspace rule, otherwise at
  the start of every new conversation. Then do PART 0 (setup) once.
- Then paste the phases one at a time, in order. Each phase ends with a GATE.
  If a gate fails, Antigravity must stop and report; do not let it continue.
- Every phase writes only to its own folder under outputs/phaseN/.
- After each phase, upload the phase's JSON/CSV/PNG outputs to
  claude_paper_package/ti64/ in the Claude project so the paper can be updated.
- STRICT GATE APPROVAL RULE: DO NOT RUN any future phases before first
  sending the user their outputs and asking for approval on the gate passes.
 
Dataset: Zenodo 8303011, "PBF-LB/M Ti6Al4V micrographs and labelled
manufacturing defects" (CC BY 4.0). Verified facts are recorded under PHASE 0.
 
 
######################################################################
PART A: GLOBAL RULES (paste first; applies to every phase)
######################################################################
 
You are working on a research codebase. A previous version produced several
silent errors. Follow every rule below in every script you write. If a rule
cannot be followed, stop and say so instead of working around it.
 
A1. Specimen grouping is absolute.
    - Every image, tile and crop carries a specimen_id. All cross-validation
      uses GroupKFold or StratifiedGroupKFold on specimen_id.
    - Assert in code, for every fold, that the set of specimen_ids in train
      and in validation is disjoint. Assert the same for tiles and crops.
    - Never use random KFold except in a run explicitly labelled
      "leakage demonstration".
 
A2. Never select anything on the reported validation fold.
    - checkpoint_mode = "last" (weights after the final epoch).
    - If early stopping or hyper-parameter tuning is needed, carve an inner
      validation split from the TRAINING specimens only.
 
A3. Fit every data-dependent transform on training folds only.
    - Target scalers, log transforms' constants, threshold calibrations,
      field-area constants, normalisation statistics: fit on train, apply to val.
 
A4. One preprocessing function for everything.
    - The same function in ti64/common.py (cv2.INTER_AREA resizing, same crop,
      same colour conversion) is used for training, validation, inference,
      robustness and XAI. No PIL in one place and OpenCV in another.
 
A5. No geometric distortion.
    - Never resize a non-square image to a square. Use square tiles cut from
      native resolution, or aspect-preserving resize with padding.
 
A6. The label must describe exactly the pixels the network sees.
    - If a target is computed on a region, the input is that same region.
      No centre crop after labelling; no input wider than the labelled field.
 
A7. No hard-coded result numbers.
    - Every number in any output table or JSON is computed in the script that
      writes it, from saved predictions. Baselines from other runs are loaded
      from their JSON files by path, never typed in. Log the source path.
 
A8. Physical units.
    - Pixel size is 0.55812 um/px at native resolution (Phase 0). Report
      areas in um^2, lengths in um, porosity in %, energy density in J/mm^3.
 
A9. Statistics.
    - Report pooled OOF metrics AND every per-fold value.
    - Use paired t-tests across folds; also report the Wilcoxon p, noting that
      with 5 folds its minimum possible two-sided value is 0.0625.
    - 95% confidence intervals by bootstrap over SPECIMENS (2,000 resamples),
      never over images or tiles.
    - Never write "beats" or "outperforms" unless the CI of the difference
      excludes zero. Report ties as ties.
 
A10. Report every result whichever way it goes. No result is dropped because
     it is negative. Every phase ends with a plain-language verdict.
 
A11. Reproducibility.
     - Seed 42 for python, numpy, torch (and cudnn.deterministic where
       feasible). Save the full config, git commit hash, package versions,
       GPU model and run time into each output folder as run_info.json.
     - Save a manifest (file path, SHA-256, specimen_id) of every image used.
 
A12. Process-derived labels only for regimes.
     - Energy regimes come from process parameters only, never from measured
       or predicted porosity.
 
A13. Explainability rules.
     - Grad-CAM is the primary method. Integrated Gradients only with an
       in-distribution baseline (a pore-free matrix image), and only if its
       completeness ratio is < 0.05.
     - Deletion tests fill pixels with the matrix grey value converted into
       normalised input space.
     - Masks used for grounding are cropped and resized exactly like the input.
 
A14. Fresh codebase only.
     - Never import, copy or call code from the old project folder. All shared
       code lives in ti64/ in this project. If something from the old project
       is needed, re-implement it here and say so.
     - Raw data copied from the old folder is read-only input.
 
A15. Long jobs run in the background.
     - Any job longer than about 10 minutes is launched as a background
       process writing to logs/<phase>_<name>.log, with progress printed at
       least every few minutes. Resume from saved per-fold outputs if a run
       is interrupted; never silently restart from scratch.
 
 
######################################################################
PART 0: FRESH PROJECT SETUP (run once, before Phase 1)
######################################################################
 
Context: This is a new, empty project folder. The old project folder is
archived at OLD_PROJECT_PATH (fill in the real path). Nothing from its code is
reused (rule A14).
 
Task:
1. Create this layout and initialise git:
     data/ti64_raw/          the 11 Zenodo files, unmodified
     data/ti64/              extracted micrographs and defect crops
     data/ackermann/         steel data for Phase 7 (step 3 below)
     ti64/                   shared Python package (step 4)
     scripts/                one script per phase
     outputs/phase0..phase9/ outputs, one folder per phase
     logs/
     requirements.txt        pinned versions (torch, timm, opencv-python,
                             scikit-learn, scipy, pandas, numpy, matplotlib,
                             pillow, imagehash)
2. Ti-6Al-4V data: move or link the 11 Zenodo files into data/ti64_raw/
   (do not re-download if they already exist; verify MD5 against
   outputs/phase0/checksums.csv after copying). Copy the Phase 0 outputs
   (checksums.csv, inventory.csv, manifest.csv, id_mapping.csv,
   phase0_report.md, previews/) and scripts/ti64_phase0_inventory.py from the
   old folder into outputs/phase0/ and scripts/. These are data and a
   finished script, not code to import.
3. Steel data (for Phase 7 only): copy, as read-only data, the raw Ackermann
   micrographs (reference-cropped-step3 folder), df_trainset_midi.csv and
   raw_registration.csv into data/ackermann/. Do NOT copy any model,
   checkpoint, src/ module or script.
4. Write the shared package ti64/ with:
   - common.py:
       set_seed(seed); load_config(); write_run_info(out_dir)
       preprocess(tile_bgr, size) -> normalised float tensor
         (cv2.INTER_AREA resize, BGR->RGB, ImageNet mean/std); the ONLY
         preprocessing function used anywhere (A4)
       load_folds() -> dict specimen_id -> fold (from outputs/phase2/folds.json)
       assert_disjoint(train_ids, val_ids)
       metrics(y_true, y_pred) -> R2, MAE, RMSE, Pearson r, mean bias
       bootstrap_ci_by_specimen(df, metric_fn, n=2000, seed=42)
   - model.py: build_model(n_outputs, backbone="tf_efficientnetv2_s",
       pretrained=True). Backbone from timm with num_classes=0, then head:
       Dropout(0.2) -> Linear(F,256) -> SiLU -> Dropout(0.15) ->
       Linear(256,128) -> SiLU -> Dropout(0.1) -> Linear(128, n_outputs).
       Also get_gradcam_layer(model) returning the last feature stage and its
       name.
   - train.py: one training loop used by every phase: AdamW, cosine schedule,
       gradient accumulation to an effective batch size, mixed precision,
       checkpoint_mode "last", per-fold resume, OOF prediction saving.
5. Write tests/test_common.py: assert preprocess is deterministic, folds are
   disjoint on a toy example, metrics match sklearn, and the model returns the
   right output shape. Run it.
 
GATE P0: tests pass; MD5 checks pass after the copy; git commit created.
 
 
######################################################################
PHASE 0: DATASET VERIFICATION  (COMPLETED; do not rerun)
######################################################################
 
Recorded facts (from outputs/phase0/phase0_report.md). Use them; do not
re-derive them:
- All 11 files match the Zenodo MD5 checksums; 0 of 774 micrographs corrupt.
- Parameter_Combinations.csv: 816 rows, 816 distinct parameter combinations,
  no replicates; columns include P, v, h (0.0501-0.1994 mm), t (0.025-0.1 mm),
  ED (matches P/(v*h*t)), TE (P/v) and Density (36 NaN).
- ID 607-696 micrographs are inside 571-606.zip (mislabelled archive).
- specimen_id = int(Index). 772 specimens have one micrograph; specimen 325 has
  two (325.jpg, 325_2.jpg); 43 IDs have none (36 of them Density = NaN).
- Micrographs: Zeiss stitched mosaics, ~13,635 x 13,284 px, 24-bit RGB JPEG,
  unetched, polished; dark mounting resin at the border; no burned-in scale
  bars; visible stitching seams (27 x 27 tiles) and tile vignetting.
- Pixel size: 0.55812 um/px, inferred from defect crops vs the feature CSV
  and the Altmann/Meluhn et al. 2023 description (360 px = 0.2 mm). To be
  confirmed in Phase 1 pre-check P1.
- Labeled_Defects: 1,200 crops of 360 x 360 px, 400 each Keyhole,
  Lack_of_Fusion, Process_Pores, from 114 specimens; linked through the `img`
  column (1-based Index) while crop names use img-1. The direction of this
  offset must be verified (P1).
- Model_Application_Micrographs.zip: 10 exact duplicates of numbered-archive
  images. Excluded everywhere.
 
 
######################################################################
PHASE 1: PRE-CHECKS, SPECIMEN MASKING, TILING AND LABELS
######################################################################
 
Context: Micrographs are ~13,600 x 13,300 px at 0.558 um/px with dark resin
borders, stitching seams and vignetting. Labels are generated by us and must
describe exactly the tile the network sees (A6).
 
Task: Write scripts/phase1_tiles_labels.py using ti64/common.py. Output
outputs/phase1/. Start with the pre-checks and stop if any pre-check gate fails.
 
PRE-CHECKS (report in a section "Phase 1 pre-checks"):
P1. Crop-to-parent verification (off-by-one and pixel scale): for 10 crops per
    defect class (30 total), locate each crop inside candidate parents
    {img}.jpg AND {img-1}.jpg by normalised cross-correlation (coarse search
    at 1/4 scale, refine at full resolution). Report matched parent, peak
    correlation and location. Repeat the best match at scales 0.9, 1.0, 1.1.
    Gate: >= 90% of crops match one parent rule with peak > 0.9, and scale 1.0
    gives the highest peak. Write the verified rule into
    outputs/phase1/id_mapping_verified.csv. If the gate fails, stop.
P2. De-duplication: SHA-256 of every micrograph in data/ti64/; assert no hash
    appears twice in the pool (Model_Application duplicates excluded).
P3. Density: report units, range, missing count and the measurement method
    (search the Zenodo description and the Altmann/Meluhn et al. 2023 paper).
    Save specimen_density.csv. It is used as an independent check in Phase 2.
P4. Pore sizes: from Ti6Al4V_local_defect_features.csv, report equivalent-
    diameter percentiles (5th, 50th, 95th) per class in um.
 
MAIN STEPS:
1. Specimen mask: on a 1/8-scale copy, separate metal from resin (Otsu,
   largest connected component, hole filling), upscale, then erode by 200 um
   to avoid the specimen edge. Save mask previews for 20 micrographs.
2. Illumination correction: per micrograph or per tile, estimate the
   background (Gaussian blur with sigma >= 10 x the largest pore radius, or
   large-kernel morphological closing) and divide it out (flat-field). Show 5
   before/after examples that include a stitching seam.
3. Tile size: choose physical tile size and network input size so that the
   5th-percentile keyhole diameter from P4 spans >= 3 network pixels. Cut
   non-overlapping square tiles at native resolution; keep tiles >= 98% inside
   the mask. Report tile size (um and px), input size, effective um/px,
   number of tiles, and disk use.
4. Labels per tile, on the full-resolution, flat-fielded tile before resizing:
   - Pore mask: Gaussian blur sigma 1 px, global threshold (Otsu, and
     median - k with k chosen on 30 hand-checked tiles; report both),
     minimum pore area MIN_AREA um^2 chosen to exceed JPEG block artefacts
     (document the value).
   - Targets: pore%, pore count N, log N, mean pore area (um^2),
     area-weighted mean pore area sum(A^2)/sum(A), convex-hull roundness
     4*pi*A / P_hull^2 averaged per tile.
   - Save every tile's pore mask (compressed PNG) for XAI grounding.
5. QC: qc_overlays.png with 40 random tiles plus 10 tiles from the specimen
   boundary, pore mask overlaid. Report scratches, stains, seams or resin
   labelled as pores, and pores missed.
6. Specimen labels: area-weighted porosity over all tiles of a specimen.
7. tiles_index.csv: tile_path, micrograph_id, specimen_id, x, y, P, v, h, t,
   E_d, TE, density, all labels.
 
GATE 1:
- P1-P4 reported; P1 and P2 gates passed.
- Every tile has a specimen_id; tile counts per specimen reported.
- QC: if more than 10% of the 50 QC tiles look wrong, or any seam or resin
  edge is labelled as pore, stop and propose a fix.
- Report the distribution of tile pore% (expect skewed and zero-inflated).
 
 
######################################################################
PHASE 2: SPLITS, LABEL VALIDATION AND BASELINES (no CNN yet)
######################################################################
 
Task: Write scripts/phase2_splits_baselines.py. Output outputs/phase2/.
1. Folds: StratifiedGroupKFold(n_splits=5) on specimen_id, stratified by
   E_d tertile, seed 42. Save folds.json (specimen_id -> fold). Assert
   disjointness (A1). Every later phase loads this file via
   ti64.common.load_folds(); no phase creates its own folds.
2. Independent label check: specimen optical porosity vs (100 - relative
   density) from specimen_density.csv: Pearson r, Spearman rho, Bland-Altman
   bias and 95% limits, scatter plot. State whether the optical labels agree
   with the density measurement.
3. Baselines for pore%, per tile and per specimen, OOF:
   - Mean predictor (training-fold mean).
   - Threshold pipeline with linear calibration fitted on training folds.
   - Black-pixel fraction of the resized network-input tile, calibrated on
     training folds (what the CNN could trivially learn).
4. Baselines for inverse regression of E_d, P, v, h, t (per specimen):
   - Training-fold mean.
   - Ridge and Extra Trees on tile features (pore%, log N, area-weighted mean
     area, roundness) aggregated per specimen.
   - Porosity-only model: E_d from specimen pore% alone (isotonic or
     exponential fit on training folds).
5. Leakage demonstration: Extra Trees inverse baseline with random KFold over
   tiles; report the random-minus-grouped R^2 gap.
 
GATE 2: folds.json passes the disjointness assertion; baselines.json written
with per-fold values; label-check verdict stated.
 
 
######################################################################
PHASE 3: FORWARD MODEL (porosity and morphology from raw tiles)
######################################################################
 
Task: Write scripts/phase3_forward.py. Output outputs/phase3/.
1. Model: ti64.model.build_model(n_outputs=3); targets [pore%, log N,
   Roundness_avg]. Transform pore% with sqrt or log1p before standardisation
   (zero inflation); report metrics in original units after inverting.
2. Training with ti64.train: 5 folds from folds.json; 40 epochs; AdamW lr 1e-4
   wd 1e-4; cosine schedule; effective batch 16 via gradient accumulation;
   mixed precision; checkpoint_mode last; seed 42.
3. Augmentation: flips, 90-degree rotations, brightness and contrast jitter
   +/-30%, mild Gaussian blur (p=0.2), JPEG re-compression (quality 70-95,
   p=0.3). No arbitrary-angle rotation that creates empty corners.
4. Sampling: oversample tiles with pores so zero-pore tiles are at most 50% of
   each batch. Evaluate on ALL validation tiles without resampling.
5. Report per tile and per specimen (area-weighted): R^2, MAE, RMSE, Pearson r,
   bootstrap CI over specimens, per-fold values. Compare to baselines.json
   with paired tests (A7, A9).
6. Resolution check: repeat at two other input sizes (one smaller, one larger)
   with the same tiles and effective batch size. Report the trend per target.
 
GATE 3: plain verdict on whether the CNN beats the calibrated threshold and the
pixel-fraction baseline on pore% at specimen level, with CIs.
 
 
######################################################################
PHASE 4: INVERSE PROCESS REGRESSION
######################################################################
 
Task: Write scripts/phase4_inverse.py. Output outputs/phase4/.
1. Targets: E_d, P, v, h, t (each standardised). If Ti-6Al-4V constants
   (absorptivity, diffusivity, beam radius, rho, c_p, T_m) can be sourced,
   also compute dH/hs = A*P / (rho*c_p*T_m*sqrt(pi*D*v*sigma^3)) and document
   every constant and source; otherwise skip it and say so.
2. Same backbone, folds and protocol as Phase 3 (ti64.model, ti64.train).
3. Report per tile and per specimen (mean of tile predictions), with CIs.
4. Compare against Phase 2 inverse baselines, especially the porosity-only
   model: does the CNN extract information beyond porosity?
5. Ambiguity analysis: specimens whose predicted E_d differs from the true E_d
   by more than 2 x MAE, with measured pore%, pore-type composition (Phase 6
   if available) and P, v, h, t.
6. Leakage demonstration: random KFold over tiles (labelled as such); report
   the gap.
 
GATE 4: plain verdict on whether micrographs predict process state in this
dataset, and whether this exceeds the porosity-only baseline.
 
 
######################################################################
PHASE 5: ROBUSTNESS (all folds)
######################################################################
 
Task: Write scripts/phase5_robustness.py. Output outputs/phase5/.
1. For every validation tile of every fold, apply: brightness x0.7, x0.8,
   x1.2, x1.3; contrast x0.5, x1.5; Gaussian blur sigma 1 and 2 px;
   vignetting 1 - 0.4*r/r_max; Gaussian noise sigma 10 and 20 grey levels;
   JPEG quality 50.
2. Predict with the fold model that never saw the specimen, and re-run the
   calibrated threshold pipeline (including flat-field correction) on the same
   perturbed tile, using ti64.common.preprocess (A4).
3. Report R^2, MAE and mean signed bias for pore% per perturbation per method,
   per tile and per specimen, with CIs, and the change vs clean.
4. Plot pore% R^2 by perturbation, CNN vs threshold, porosity only on its axis.
 
GATE 5: plain verdict: under which perturbations the CNN degrades less than the
calibrated threshold, and under which it degrades more.
 
 
######################################################################
PHASE 6: PORE-TYPE CLASSIFICATION (defect crops)
######################################################################
 
Task: Write scripts/phase6_poretype.py. Output outputs/phase6/.
1. Linking: use the verified crop-to-specimen rule from
   outputs/phase1/id_mapping_verified.csv and the fold of each specimen from
   folds.json. All 114 specimens with crops stay in their assigned folds.
2. Near-duplicate check: perceptual hash on all crops; report pairs with
   Hamming distance <= 4 and confirm duplicates fall in the same fold.
3. Models: (a) logistic regression and gradient boosting on the provided
   feature CSV; (b) ti64.model.build_model(n_outputs=3) fine-tuned on crops
   as a classifier (cross-entropy). Same protocol rules.
4. Report accuracy, macro-F1, per-class recall, confusion matrix, per fold.
5. Leakage demonstration: random split over crops; report the gap.
6. Apply the classifier to pores detected in Phase 1 tiles to estimate each
   specimen's pore-type composition; feed it to Phase 4, step 5.
 
GATE 6: plain verdict on pore-type accuracy under specimen grouping.
 
 
###########################################################
PHASE 10: POROSITY AND DEFECT ASSESSMENT SYSTEM. Follow Part A rules.
Reuse Phase 3 and Phase 6 models; no retraining of the backbone unless stated.
Output outputs/phase10/.

1. Part-level porosity with uncertainty:
   - Calibrate specimen CNN pore% to Archimedes porosity (100 - density) with a
     monotone fit (isotonic or linear) per fold on training specimens only.
   - Add prediction intervals with split conformal prediction (calibration set
     carved from training specimens; target 90% coverage). Report empirical
     coverage and mean interval width OOF, by density bin.
   - Accept/reject: for spec limits of 0.5%, 1% and 2% Archimedes porosity,
     report sensitivity, specificity, and precision of the calibrated
     prediction, plus the confusion matrix, OOF and per fold.

2. Full-micrograph defect detection and classification:
   - Use the Phase 1 pore masks to extract every pore (connected component >= 10
     um equivalent diameter) in each full micrograph.
   - Classify each pore with the Phase 6 classifier on a 360 x 360 px crop
     centred on it (same crop geometry as the labelled crops).
   - Validate on the 1,200 labelled crops by locating them in the full images
     (Phase 1 P1 coordinates) and checking that the full-image pipeline assigns
     the same class: report accuracy, macro-F1, confusion matrix, specimen-grouped.
   - Per specimen: count and area fraction of each pore type; defect map image
     for 20 example specimens (one colour per class).

3. Largest-defect statistics (fatigue-relevant):
   - Per specimen: largest pore sqrt(area) in um, 95th percentile pore size, and
     maximum Feret diameter.
   - Fit a Gumbel extreme-value distribution of max sqrt(area) per tile within
     each specimen and report the predicted largest defect in a reference
     volume/area (state the reference area used).
   - Train a regression head on frozen Phase 3 features to predict tile max
     sqrt(area) directly; report OOF R2 and MAE vs the label pipeline.
   - Report how largest-defect size relates to porosity, pore type and E_d.

4. End-to-end tool: scripts/assess_micrograph.py taking one micrograph path and
   returning:
   (a) part porosity with 90% interval and pass/fail for a given spec limit;
   (b) pore-type composition;
   (c) largest defect sqrt(area) and its location;
   (d) a PNG defect map and a one-page Markdown report.
   Use the fold model ensemble, or a model trained on all specimens and saved
   separately as final_model.pt, clearly labelled as not used for any reported
   metric. Report runtime per micrograph on GPU and CPU.

GATE 10: coverage within 2 points of the 90% target; full-image classification
validated against labelled crops; tool runs on 5 held-out micrographs and its
outputs match the OOF results for those specimens.
######################################################################
PHASE 7: CROSS-DATASET TRANSFER (Ackermann steel <-> Ti-6Al-4V)
######################################################################
 
Context: No old model or checkpoint is reused (A14). The steel raw-image model
is retrained here, in this codebase, from data/ackermann/.
 
Task: Write scripts/phase7_transfer.py. Output outputs/phase7/.
1. Steel model, retrained here: crop each raw Ackermann micrograph to its
   registered 600 x 600 field using data/ackermann/raw_registration.csv; use
   the published pore% labels from df_trainset_midi.csv; 5-fold GroupKFold by
   Sample (save steel_folds.json); same model, training loop and protocol as
   Phase 3. Report its OOF metrics as a sanity check against the earlier
   result (pore% R^2 around 0.80 at 384 px), loading that number from a JSON
   copied into data/ackermann/ if available, never typed in (A7).
2. Common physical scale: the Ackermann pixel size is not established. Report
   this, state the assumption used to resample both datasets to a common
   um/px, and run a sensitivity check at two alternative scales.
3. Zero-shot: steel model on Ti-6Al-4V tiles, and Ti-6Al-4V Phase 3 model on
   steel crops, for pore%.
4. Fine-tuning: head-only and full fine-tuning with 10%, 25% and 50% of the
   target-domain training specimens (grouped), evaluated on held-out target
   specimens. Report the learning curve.
5. Inverse transfer only for E_d normalised within each dataset, labelled
   exploratory.
 
GATE 7: plain verdict on whether porosity estimation transfers across alloy and
machine, and how many target specimens fine-tuning needs.
 
 
######################################################################
PHASE 8: EXPLAINABILITY (with sanity checks)
######################################################################
 
Task: Write scripts/phase8_xai.py. Output outputs/phase8/. Implement Grad-CAM
and Integrated Gradients in ti64/xai.py (new code, A14).
1. 24 validation tiles (8 per E_d tertile) from folds 1 and 3, both forward
   targets pore% and Roundness_avg, Phase 3 models.
2. Grad-CAM at ti64.model.get_gradcam_layer(); log the layer name.
3. Integrated Gradients only with a pore-free baseline: a flat-fielded tile of
   uniform matrix grey, in normalised space. Report f(baseline) and the
   completeness ratio at 50/100/200 steps (signed sums, float64). If the
   median ratio at 200 steps is >= 0.05, do not report IG maps or metrics.
4. Cascading randomisation (head; head + last 2 stages; whole network), mean
   and SD Spearman rho per method per target.
5. Deletion at k = 0..20% (step 2), matrix-grey fill in normalised space;
   random baseline averaged over 5 seeds; curves and normalised AUC.
6. Grounding: share of top-5% Grad-CAM pixels within 3 network pixels of a pore
   (Phase 1 masks, cropped and resized exactly like the input) divided by that
   band's area share; mean and median.
7. If Phase 6 succeeded: compare Grad-CAM attention on keyhole vs lack-of-fusion
   pores for the inverse model.
 
GATE 8: list which method/target combinations pass all checks; only those may
be described in the paper.
 
 
######################################################################
PHASE 9: CONSOLIDATION
######################################################################
 
Task: Write scripts/phase9_consolidate.py. Output outputs/final/ and upload to
claude_paper_package/ti64/ in the Claude project.
1. Build every paper table (CSV and Markdown) from the phase JSON files only
   (A7). Each table cell records its source file in a sidecar JSON.
2. Regenerate every figure at 300 DPI from saved predictions.
3. Write final_summary.md: one verdict line per gate, all key numbers with
   CIs, and every deviation from these rules with its reason.
4. Consistency check: recompute 10 random table numbers from raw prediction
   files and assert they match to 4 decimals.
 
GATE 9: consistency check passes; outputs uploaded.
 
 
######################################################################
PART B: RISKS SPECIFIC TO THIS DATASET
######################################################################
Paste this block together with Phase 1.
 
B1. Lossy JPEG: compression blocks can create dark specks that thresholds count
    as pores. Use a minimum pore area in um^2 above block size, and include JPEG
    re-compression in augmentation and robustness.
B2. Huge images (~181 MP each, ~545 MB in RAM decoded): process one micrograph
    at a time, tile once to disk, stream tiles. Report disk use after Phase 1.
B3. Resin, edges and seams counted as porosity: resin is dark like pores; the
    mosaic has seams and vignetting. Mask the resin, flat-field correct, and
    check boundary tiles and seams in QC.
B4. Zero-inflated, skewed targets: transform targets, balance batches, and judge
    models at specimen level.
B5. One micrograph per specimen (except 325): all tiles of a specimen come from
    one image and are highly correlated; grouping by specimen is mandatory.
B6. Hatch and layer thickness vary: E_d is not a function of speed alone. Never
    compute E_d with fixed h or t.
B7. Defect crops may overlap or repeat: near-duplicate check in Phase 6.
B8. Keyhole and lack-of-fusion pores look alike at low resolution: keep the
    smallest keyhole pores >= 3 network pixels wide (Phase 1, step 3).
B9. Domain shift between steel and Ti-6Al-4V: never pool datasets in one CV
    without a dataset indicator, never report pooled metrics as one source.
B10. Licence: CC BY 4.0; record the citation in run_info.json and in the
     paper's data statement.
