"""Central configuration for the Burotu pipeline."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root (src/burotu/config.py)

# Data and model weights live outside Git, in one store (D:\Burotu\Data on the workstation):
#   <store>/raw/<origin>/<dataset>/      downloads, never modified          (pipeline step 1: ingest)
#   <store>/refined/<version>/          Burotu format, ready for training   (step 2: refine)
#   <store>/models/<model>/<version>/   weights + report of each trained run (step 3: train)
# See burotu.store for moving it between the workstation, the servers and Modal.
STORE = Path(os.environ.get("BUROTU_STORE", ROOT / "store"))
RAW_DIR = Path(os.environ.get("BUROTU_RAW_DIR", STORE / "raw"))
REFINED_VERSION = os.environ.get("BUROTU_REFINED", "v1")
DATA_DIR = Path(os.environ.get("BUROTU_DATA_DIR", STORE / "refined" / REFINED_VERSION))  # the refined data in use
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_STORE = Path(os.environ.get("BUROTU_MODELS_DIR", STORE / "models"))

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
# 12 more TCGA types: 20 classes in total since October 29, 2026 (the first prototype had 8).
EXTRA_CLASSES: dict[str, list[str]] = {
    "Kidney": ["TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"],
    "Bladder": ["TCGA-BLCA"],
    "Uterus": ["TCGA-UCEC", "TCGA-UCS"],
    "Cervix": ["TCGA-CESC"],
    "Thyroid": ["TCGA-THCA"],
    "Skin": ["TCGA-SKCM"],
    "Brain": ["TCGA-LGG", "TCGA-GBM"],
    "Soft tissue": ["TCGA-SARC"],
    "Mesothelium": ["TCGA-MESO"],
    "Adrenal": ["TCGA-ACC", "TCGA-PCPG"],
    "Testis": ["TCGA-TGCT"],
    "Thymus": ["TCGA-THYM"],
}
if os.environ.get("BUROTU_CLASSES", "20") == "20":  # BUROTU_CLASSES=8 rebuilds the first prototype
    CANCER_CLASSES = {**CANCER_CLASSES, **EXTRA_CLASSES}
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
