"""Central configuration for the Burotu pipeline."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root (src/burotu/config.py)
DATA_DIR = Path(os.environ.get("BUROTU_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Cancer signal origin (CSO) classes -> matching TCGA projects.
# Chosen for screening: common cancers and/or cancers with no screening today.
CANCER_CLASSES: dict[str, list[str]] = {
    "Lung": ["TCGA-LUAD", "TCGA-LUSC"],
    "Colorectal": ["TCGA-COAD", "TCGA-READ"],
    "Breast": ["TCGA-BRCA"],
    "Pancreas": ["TCGA-PAAD"],
    "Liver": ["TCGA-LIHC"],
    "Oesophagus-Stomach": ["TCGA-STAD", "TCGA-ESCA"],
    "Head and neck": ["TCGA-HNSC"],
    "Prostate": ["TCGA-PRAD"],
}
HEALTHY = "Healthy"

# Number of TCGA tumours downloaded per class.
TUMORS_PER_CLASS = int(os.environ.get("BUROTU_TUMORS_PER_CLASS", 48))

# Whole blood (the cfDNA background): Hannum et al. 2013, 450K.
BLOOD_GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/suppl/GSE40279_average_beta_GSM989827-GSM989990.txt.gz"
)
# Real cell-free DNA (Moss et al. 2018, Nat Commun), EPIC: used for validation only.
MOSS_CFDNA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122126/matrix/GSE122126-GPL21145_series_matrix.txt.gz"
)

SEED = 42
