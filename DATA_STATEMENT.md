# Data Statement & Provenance

## 1. Dataset Identification & Citation
- **Dataset Title:** PBF-LB/M Ti6Al4V Micrographs and Labeled Manufacturing Defects
- **Version:** 1.0
- **Authors:** Lukas A. Meluhn, Selina Müller, Mika León Altmann, Thiemo Benthien, Anastasiya Toenjes
- **Host Institution:** Bundesanstalt für Materialforschung und -prüfung (BAM), Berlin, Germany
- **Publication Date:** September 1, 2023
- **Repository:** Zenodo
- **Digital Object Identifier (DOI):** [10.5281/zenodo.8303011](https://doi.org/10.5281/zenodo.8303011)

### Formal Citation
```bibtex
@dataset{meluhn_2023_8303011,
  author       = {Meluhn, Lukas A. and
                  Müller, Selina and
                  Altmann, Mika León and
                  Benthien, Thiemo and
                  Toenjes, Anastasiya},
  title        = {PBF-LB/M Ti6Al4V Micrographs and Labeled Manufacturing Defects},
  month        = sep,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.8303011},
  url          = {https://doi.org/10.5281/zenodo.8303011}
}
```

---

## 2. Licensing & Terms of Use
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Permissions:** You are free to share (copy and redistribute the material in any medium or format) and adapt (remix, transform, and build upon the material for any purpose, even commercially) under the terms that appropriate credit is given to the authors and changes are indicated.
- **Attribution Notice:** All derived tile datasets, trained deep learning checkpoints, baseline models, and statistical reports generated in this repository cite the original Zenodo record in accordance with CC BY 4.0.

---

## 3. Dataset Characteristics & Physical Provenance
- **Material Alloy:** Ti-6Al-4V (Grade 5 Titanium Alloy, $\alpha+\beta$).
- **Manufacturing Process:** Laser Powder Bed Fusion (PBF-LB/M, Selective Laser Melting).
- **Specimens:** 773 distinct cylindrical and cuboid specimens manufactured across wide process parameter ranges:
  - Laser Power ($P$): $100 - 400\,\text{W}$
  - Scan Speed ($v$): $400 - 2000\,\text{mm/s}$
  - Hatch Distance ($h$): $0.06 - 0.14\,\text{mm}$
  - Layer Thickness ($t$): $0.03 - 0.06\,\text{mm}$
  - Volumetric Energy Density ($E_d = \frac{P}{v \cdot h \cdot t}$): $19.8 - 250.0\,\text{J/mm}^3$
- **Microscopy Imaging:**
  - Optical microscopy panoramas (~181 MP stitched mosaics per specimen).
  - Unetched polished metallographic cross-sections.
  - Calibrated physical resolution: $0.55812\,\mu\text{m/pixel}$.
- **Verification Integrity:**
  - All 11 Zenodo archive files verified against published MD5 cryptographic checksums in Phase 0.
  - Zero image file corruption detected across all specimens.

