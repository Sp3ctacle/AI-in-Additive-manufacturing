"""
Script to update all run_info.json files with complete CC BY 4.0 license,
DOI, and formal scholarly citation metadata (Rule B10).
"""

import json
from pathlib import Path

DATASET_CITATION = (
    "Meluhn, L. A., Müller, S., Altmann, M. L., Benthien, T., & Toenjes, A. (2023). "
    "PBF-LB/M Ti6Al4V Micrographs and Labeled Manufacturing Defects (Version 1.0) [Data set]. "
    "Zenodo. https://doi.org/10.5281/zenodo.8303011"
)
DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
DATASET_DOI = "10.5281/zenodo.8303011"
DATASET_NAME = "PBF-LB/M Ti6Al4V Micrographs and Labeled Manufacturing Defects"

def main():
    root = Path(__file__).resolve().parent.parent
    run_info_files = list(root.glob("outputs/**/run_info.json"))
    
    print(f"Found {len(run_info_files)} run_info.json files:")
    for path in run_info_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            data["license"] = DATASET_LICENSE
            data["dataset_doi"] = DATASET_DOI
            data["dataset_name"] = DATASET_NAME
            data["dataset_citation"] = DATASET_CITATION
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"  [UPDATED] {path.relative_to(root)}")
        except Exception as e:
            print(f"  [ERROR] {path.relative_to(root)}: {e}")

if __name__ == "__main__":
    main()

