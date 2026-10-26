"""Configuration centrale de Burotu."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # racine du repo (src/burotu/config.py)
DATA_DIR = Path(os.environ.get("BUROTU_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Classes d'organe d'origine (CSO) -> projets TCGA correspondants.
# Choix orienté dépistage : cancers fréquents et/ou sans dépistage existant.
CANCER_CLASSES: dict[str, list[str]] = {
    "Poumon": ["TCGA-LUAD", "TCGA-LUSC"],
    "Colorectal": ["TCGA-COAD", "TCGA-READ"],
    "Sein": ["TCGA-BRCA"],
    "Pancréas": ["TCGA-PAAD"],
    "Foie": ["TCGA-LIHC"],
    "Œsophage-Estomac": ["TCGA-STAD", "TCGA-ESCA"],
    "Tête et cou": ["TCGA-HNSC"],
    "Prostate": ["TCGA-PRAD"],
}
HEALTHY = "Sain"

# Nombre de tumeurs TCGA téléchargées par classe.
TUMORS_PER_CLASS = int(os.environ.get("BUROTU_TUMORS_PER_CLASS", 48))

# Sang total (fond de l'ADN circulant) : Hannum et al. 2013, 450K.
BLOOD_GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/suppl/GSE40279_average_beta_GSM989827-GSM989990.txt.gz"
)
# ADN circulant réel (Moss et al. 2018, Nat Commun), EPIC : sert à la validation.
MOSS_CFDNA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122126/matrix/GSE122126-GPL21145_series_matrix.txt.gz"
)

SEED = 42
