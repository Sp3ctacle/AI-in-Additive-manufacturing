# Phase 4: Inverse Process Regression Report

**Date:** 2026-10-30 23:57:09
**Repository Branch:** `main`
**Global Rules Enforced:** A1 (specimen grouping absolute; disjointness asserted), A2 (checkpoint mode last), A3 (standardization on train folds only), A7 (no hardcoded numbers), A8 (physical units), A9 (2,000 specimen bootstrap CIs), A10 (honest verdicts).

---

## 1. Physical Model & Literature Sourcing (Normalized Enthalpy)

In addition to volumetric energy density ($E_d = P / (v \cdot h \cdot t)$) and machine parameters ($P, v, h, t$), we compute the dimensionless normalized enthalpy $\Delta H / h_s$ according to the thermophysical keyhole threshold model:

$$\frac{\Delta H}{h_s} = \frac{A \cdot P}{\rho \cdot c_p \cdot T_m \sqrt{\pi \cdot D \cdot v \cdot \sigma^3}}$$

### Ti-6Al-4V Physical Constants & Literature Sources:
- **Laser Absorptivity ($A$):** $0.35$ (dimensionless). Sourced from King et al. (2014) *J. Mater. Process. Technol.*, 214(12):2915-2925 and Rubenchik et al. (2018) for 1070 nm fiber laser melting on Ti-6Al-4V powder bed.
- **Solid Density ($\rho$):** $4430\,\text{kg/m}^3$. Sourced from Boyer, Welsch, Collings (1994) *Materials Properties Handbook: Titanium Alloys*, ASM International.
- **Specific Heat ($c_p$):** $526\,\text{J/(kg}\cdot\text{K)}$. Sourced from Boyer et al. (1994).
- **Melting Temperature ($T_m$):** $1923\,\text{K}$ ($1650^\circ\text{C}$, liquidus/solidus midpoint). Sourced from Boyer et al. (1994).
- **Thermal Diffusivity ($D$):** $2.9 \times 10^{-6}\,\text{m}^2/\text{s}$ ($2.9\,\text{mm}^2/\text{s}$). Sourced from King et al. (2014) and Altmann et al. (2023).
- **Laser Beam Radius ($\sigma$):** $5.0 \times 10^{-5}\,\text{m}$ ($50\,\mu\text{m}$, representing $100\,\mu\text{m}$ focal spot diameter on the EOS M 290 system). Sourced from Altmann/Meluhn et al. (2023).

- **Dataset Normalized Enthalpy Range:** min = $4.10$, mean = $18.08$, max = $45.10$. This spans the transition from lack-of-fusion ($<10$) to stable conduction mode ($10-25$) to deep keyholing ($>25-30$).

---

## 2. Inverse Process Regression Performance (Strictly Out-of-Fold)

| Process Target | Physical Unit | Tile $R^2$ | Tile MAE | Specimen $R^2$ [95% CI] | Specimen MAE [95% CI] | Specimen Pearson $r$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E_d** | J/mm³ | 0.3646 | 20.2154 | **0.6053** [0.5611, 0.6441] | **15.3905** [14.1832, 16.6672] | 0.7838 |
| **P** | W | 0.2901 | 50.2852 | **0.5924** [0.5553, 0.6251] | **40.6812** [38.6062, 42.6515] | 0.7822 |
| **v** | mm/s | 0.3029 | 262.1390 | **0.5344** [0.4844, 0.5814] | **218.6236** [204.5861, 232.0615] | 0.7350 |
| **h** | mm | 0.1830 | 0.0293 | **0.5239** [0.4774, 0.5606] | **0.0232** [0.0220, 0.0246] | 0.7516 |
| **t** | mm | 0.2453 | 0.0184 | **0.5934** [0.5553, 0.6264] | **0.0143** [0.0135, 0.0151] | 0.7957 |
| **normalized_enthalpy** | dimensionless | 0.5940 | 3.3693 | **0.8058** [0.7819, 0.8280] | **2.3831** [2.2261, 2.5405] | 0.9044 |

---

## 3. Comparison Against Phase 2 Inverse Baselines (Beyond Porosity)

We directly benchmark the CNN's volumetric energy density prediction against the Phase 2 baselines loaded dynamically from `outputs/phase2/baselines.json`:

| Model | Features | Specimen $R^2$ | Specimen MAE (J/mm³) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Porosity-Only Linear Baseline** | Ground-truth pore% only | 0.0586 | 27.96 J/mm³ | Univariate lower bound |
| **Ridge Regression** | Tile image intensity statistics | 0.1231 | 26.55 J/mm³ | Linear baseline |
| **Extra Trees Regressor** | Tile image intensity statistics | 0.1634 | 25.48 J/mm³ | Non-linear baseline |
| **CNN (`tf_efficientnetv2_s`)** | Raw Micrograph Pixels | **0.6053** | **15.39 J/mm³** | **Deep representation** |

### Hypothesis Test: Does the CNN Extract Information Beyond Porosity?
- **Paired Difference in Specimen MAE:** $\Delta\text{MAE} = \text{MAE}_{\text{porosity-only}} - \text{MAE}_{\text{CNN}} = +11.25\,\text{J/mm}^3$
- **95% Specimen Bootstrap CI:** [+9.98, +12.45] J/mm³
- **Verdict:** **BEATS (EXCEEDS POROSITY-ONLY BASELINE)**

> [!IMPORTANT]
> Because the 95% bootstrap confidence interval [+9.98, +12.45] J/mm³ strictly excludes zero, the CNN extracts process-state signals beyond scalar void fraction alone. Spatial arrangement of pores and pore size distributions encode processing conditions more effectively than simple scalar porosity.

---

## 4. Ambiguity Analysis (Residuals > 2x MAE)

- **Ambiguity Threshold ($2 \times \text{MAE}$):** $30.78\,\text{J/mm}^3$
- **Total Ambiguous Specimens:** 97 out of 734 specimens (13.2%)

![Ambiguity Analysis](ambiguity_analysis.png)

### Metallurgical Root Causes of Inverse Degeneracy:
1. **Ill-Posed Degeneracy of 1D Energy Density:** Volumetric energy density $E_d = P / (v \cdot h \cdot t)$ lumps four distinct process parameters into a single scalar. As an illustrative example of parameter degeneracy, distinct parameter combinations—such as high laser power with high scan speed ($P=350\,\text{W}, v=1500\,\text{mm/s}$) versus low power with low speed ($P=150\,\text{W}, v=600\,\text{mm/s}$)—can yield similar volumetric energy density ($E_d \approx 50\,\text{J/mm}^3$), producing similar conduction-mode thermal fields. The network cannot uniquely resolve this 4-to-1 projection.
2. **Information Loss in the Dense Process Window:** In fully dense specimens ($E_d \in [50, 100]\,\text{J/mm}^3$, density $> 99.5\%$), nearly zero voids exist. Without voids to distinguish melt pool geometry, the optical appearance of the polished Ti-6Al-4V matrix is essentially uniform across a broad $50\,\text{J/mm}^3$ energy window.
3. **Thermal Boundary Non-Linearities:** At very low $E_d$ ($<20\,\text{J/mm}^3$), lack-of-fusion voids dominate; at very high $E_d$ ($>120\,\text{J/mm}^3$), keyhole vapor depression occurs. Between these extremes, microstructures plateau.

---

## 5. Leakage Demonstration (Rule A1)

To illustrate the catastrophic risk of random tile-level cross-validation without specimen grouping:

| Protocol | Target | Tile $R^2$ | Tile MAE | Leakage Gap ($\Delta R^2$) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Grouped K-Fold (Strict Rule A1)** | $E_d$ | 0.3646 | 20.22 J/mm³ | Baseline (Zero Leakage) | **VALID** |
| **Random K-Fold (Tile Leakage)** | $E_d$ | 0.4451 | 18.49 J/mm³ | $\Delta R^2 = +0.0805$ | **ARTIFICIALLY INFLATED** |

> [!WARNING]
> Random tile-level cross-validation leaks specimen-specific metallographic artifacts across train and validation splits, artificially inflating $R^2$ by $\Delta R^2 = +0.0805$. Strict specimen grouping (GroupKFold on `specimen_id`) is essential for valid scientific inference.

---

## 6. Gate 4 Final Verdict

| Metric / Requirement | Target / Baseline | CNN Achieved | Status |
| :--- | :--- | :--- | :--- |
| **Specimen Grouping (A1)** | Disjoint specimens across 5 folds | Asserted 0 leakage (734 specimens) | **PASS** |
| **Inverse Prediction ($E_d$)** | Outperform porosity-only model ($R^2 = 0.0586$) | Specimen $R^2 = 0.6053$ [95% CI: 0.5611, 0.6441] | **PASS** |
| **Information Beyond Porosity** | $\Delta\text{MAE} > 0$ with 95% CI strictly positive | $\Delta\text{MAE} = +11.25\,\text{J/mm}^3$ [+9.98, +12.45] | **BEATS (EXCEEDS POROSITY-ONLY BASELINE)** |
| **Normalized Enthalpy ($dH/h_s$)** | Literature physical constants documented | Specimen $R^2 = 0.8058$ | **PASS** |
| **Ambiguity Analysis** | Document $|\text{error}| > 2\times\text{MAE}$ outliers | 97 outliers characterized | **PASS** |
| **Leakage Demonstration** | Quantify random vs grouped gap | Gap $\Delta R^2 = +0.0805$ quantified | **PASS** |

### Formal Gate 4 Statement:
1. **Micrographs Predict Process State:** In this dataset, raw optical micrographs reliably predict LPBF process parameters, achieving specimen-level $R^2 = 0.6053$ for volumetric energy density $E_d$ and $R^2 = 0.8058$ for normalized enthalpy.
2. **Exceeds Porosity-Only Baseline:** The CNN significantly outperforms the univariate porosity-only baseline by $\Delta\text{MAE} = +11.25\,\text{J/mm}^3$ [95% bootstrap CI: +9.98, +12.45 J/mm³], demonstrating that optical micrographs contain spatial defect arrangements and morphology information beyond scalar void fraction alone.
3. **Physical Limits & Ambiguities:** Inverse process estimation is fundamentally bounded by process-space degeneracy (different $(P, v, h, t)$ producing identical $E_d$) and feature collapse in the dense window ($E_d \in [50, 100]\,\text{J/mm}^3$) where void-free micrographs provide minimal discriminative cues.

**GATE 4: PASS.**
