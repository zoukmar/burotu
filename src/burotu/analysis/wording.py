"""Words Burotu uses with patients, in English, French and Spanish.

Burotu reports a methylation *signal*, never a disease. Patient-facing text never
names cancer or tumours: it says whether an unusual signal was found, which tissue it
most resembles, and that a doctor decides what happens next.
"""

from __future__ import annotations

import re

LANGS = ("en", "fr", "es")
LANG_NAMES = {"en": "English", "fr": "French", "es": "Spanish"}

# Tissue the signal most resembles, by model class.
TISSUES = {
    "en": {
        "Breast": "breast",
        "Colorectal": "colon and rectum",
        "Head and neck": "head and neck",
        "Liver": "liver",
        "Lung": "lung",
        "Oesophagus-Stomach": "oesophagus and stomach",
        "Pancreas": "pancreas",
        "Prostate": "prostate",
        "Kidney": "kidney",
        "Bladder": "bladder",
        "Uterus": "uterus",
        "Cervix": "cervix",
        "Thyroid": "thyroid",
        "Skin": "skin",
        "Brain": "brain",
        "Soft tissue": "soft tissue",
        "Mesothelium": "mesothelium (lining of the chest)",
        "Adrenal": "adrenal gland",
        "Testis": "testis",
        "Thymus": "thymus",
    },
    "fr": {
        "Breast": "sein",
        "Colorectal": "côlon et rectum",
        "Head and neck": "tête et cou",
        "Liver": "foie",
        "Lung": "poumon",
        "Oesophagus-Stomach": "œsophage et estomac",
        "Pancreas": "pancréas",
        "Prostate": "prostate",
        "Kidney": "rein",
        "Bladder": "vessie",
        "Uterus": "utérus",
        "Cervix": "col de l'utérus",
        "Thyroid": "thyroïde",
        "Skin": "peau",
        "Brain": "cerveau",
        "Soft tissue": "tissus mous",
        "Mesothelium": "mésothélium (enveloppe du thorax)",
        "Adrenal": "glande surrénale",
        "Testis": "testicule",
        "Thymus": "thymus",
    },
    "es": {
        "Breast": "mama",
        "Colorectal": "colon y recto",
        "Head and neck": "cabeza y cuello",
        "Liver": "hígado",
        "Lung": "pulmón",
        "Oesophagus-Stomach": "esófago y estómago",
        "Pancreas": "páncreas",
        "Prostate": "próstata",
        "Kidney": "riñón",
        "Bladder": "vejiga",
        "Uterus": "útero",
        "Cervix": "cuello uterino",
        "Thyroid": "tiroides",
        "Skin": "piel",
        "Brain": "cerebro",
        "Soft tissue": "tejidos blandos",
        "Mesothelium": "mesotelio (revestimiento del tórax)",
        "Adrenal": "glándula suprarrenal",
        "Testis": "testículo",
        "Thymus": "timo",
    },
}

# Quality-control findings, by code (see burotu.analysis.qc).
QC_TEXT = {
    "en": {
        "coverage": "only {coverage:.0%} of the measured sites could be read",
        "depth": "the sequencing was too shallow ({median:.0f} reads per site, {min_depth} needed)",
        "low_depth": "{low:.0%} of the sites were read too few times",
        "unknown": "{unknown} sites in the file are not part of the Burotu panel and were ignored",
    },
    "fr": {
        "coverage": "seuls {coverage:.0%} des sites mesurés ont pu être lus",
        "depth": "le séquençage était trop peu profond ({median:.0f} lectures par site, {min_depth} nécessaires)",
        "low_depth": "{low:.0%} des sites ont été lus trop peu de fois",
        "unknown": "{unknown} sites du fichier ne font pas partie du panel Burotu et ont été ignorés",
    },
    "es": {
        "coverage": "solo se pudo leer el {coverage:.0%} de los sitios medidos",
        "depth": "la secuenciación fue demasiado superficial "
        "({median:.0f} lecturas por sitio, se necesitan {min_depth})",
        "low_depth": "el {low:.0%} de los sitios se leyó demasiadas pocas veces",
        "unknown": "{unknown} sitios del archivo no forman parte del panel Burotu y se ignoraron",
    },
}

# Words that must never appear in a patient report (any language).
FORBIDDEN = re.compile(
    r"c[aá]nc[eé]r|cancer|canc[eé]reu|tumou?r|tumeur|maligno?|malin|carcinom|oncolog|m[ée]tasta|neoplas|n[ée]oplas",
    re.IGNORECASE,
)


def tissue(cls: str, lang: str) -> str:
    return TISSUES.get(lang, TISSUES["en"]).get(cls, cls)


def qc_findings(qc: dict, lang: str) -> list[str]:
    """Quality-control findings in plain words, from the report's codes and numbers."""
    from burotu.analysis.qc import MIN_MEDIAN_DEPTH

    texts = QC_TEXT.get(lang, QC_TEXT["en"])
    values = {
        "coverage": qc.get("panel_coverage", 0.0),
        "median": qc.get("median_depth", 0.0),
        "min_depth": MIN_MEDIAN_DEPTH,
        "low": qc.get("low_depth_share", 0.0),
        "unknown": qc.get("unknown_cpgs", 0),
    }
    return [texts[c].format(**values) for c in qc.get("codes", []) if c in texts]


def is_clean(text: str) -> bool:
    """True when the text names no disease (the rule for every patient-facing report)."""
    return not FORBIDDEN.search(text)
