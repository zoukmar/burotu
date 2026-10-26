"""Simulateur d'ADN circulant (cfDNA) pour le panel ciblé Burotu.

Modèle :
  1. Le plasma d'une personne saine contient surtout de l'ADN issu des cellules
     sanguines : on tire un individu sain réel (profil de sang total).
  2. Chez une personne atteinte de cancer, une fraction `tf` (fraction tumorale,
     typiquement 0,01 % à 10 %) provient de la tumeur : on tire une tumeur réelle
     TCGA de la classe voulue et on mélange les profils.
  3. Le séquençage bisulfite ciblé est simulé par un tirage binomial des lectures
     méthylées, avec une profondeur variable d'un CpG à l'autre.

Limite connue (à dire au jury) : on simule CpG par CpG, sans motifs de méthylation
au niveau de la lecture. Les tests commerciaux exploitent ces motifs et sont donc
plus sensibles à faible fraction tumorale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from burotu.config import HEALTHY
from burotu.data.reference import Reference


@dataclass
class SimConfig:
    depth: float = 500.0  # profondeur moyenne de lecture par CpG
    depth_sd_log: float = 0.5  # variabilité log-normale de la profondeur entre CpG
    tf_min: float = 1e-4
    tf_max: float = 0.1
    seed: int = 0


def mix_and_sequence(
    blood_beta: np.ndarray,
    tumor_beta: np.ndarray | None,
    tf: float,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (lectures méthylées, profondeur) pour un échantillon."""
    beta = blood_beta if tumor_beta is None or tf <= 0 else (1 - tf) * blood_beta + tf * tumor_beta
    beta = np.clip(beta, 0.0, 1.0)
    lam = cfg.depth * rng.lognormal(-(cfg.depth_sd_log**2) / 2, cfg.depth_sd_log, size=beta.shape)
    total = rng.poisson(lam)
    meth = rng.binomial(total, beta)
    return meth, total


def simulate_cohort(
    ref: Reference,
    split: str,
    n_healthy: int,
    n_cancer_per_class: int,
    cfg: SimConfig | None = None,
    tf_values: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Simule une cohorte à partir du split `train` ou `test` (aucune fuite entre les deux).

    Renvoie (meth, total, meta) avec meth/total de forme (échantillons, CpG du panel).
    Si `tf_values` est donné, chaque patient cancer reçoit successivement ces fractions
    (utile pour tracer la sensibilité en fonction de la fraction tumorale) ; sinon la
    fraction est tirée en log-uniforme entre tf_min et tf_max.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    blood = getattr(ref, f"blood_{split}").to_numpy(np.float32)  # CpG × individus
    tumors = getattr(ref, f"tumor_{split}")
    tmeta = ref.tumor_meta.set_index("file_id").loc[tumors.columns]
    tumors_np = tumors.to_numpy(np.float32)

    rows, meths, totals = [], [], []

    def add(tumor_idx: int | None, cls: str, tf: float) -> None:
        b = rng.integers(blood.shape[1])
        t = None if tumor_idx is None else tumors_np[:, tumor_idx]
        m, tot = mix_and_sequence(blood[:, b], t, tf, cfg, rng)
        meths.append(m)
        totals.append(tot)
        rows.append(
            {
                "label": cls,
                "is_cancer": tumor_idx is not None,
                "tumor_fraction": tf,
                "blood_donor": ref_col(getattr(ref, f"blood_{split}"), b),
                "tumor_id": None if tumor_idx is None else tumors.columns[tumor_idx],
            }
        )

    for _ in range(n_healthy):
        add(None, HEALTHY, 0.0)
    for cls in sorted(tmeta["cancer_class"].unique()):
        idx = np.flatnonzero(tmeta["cancer_class"].to_numpy() == cls)
        for i in range(n_cancer_per_class):
            tumor_idx = int(idx[i % len(idx)]) if i < len(idx) else int(rng.choice(idx))
            if tf_values:
                for tf in tf_values:
                    add(tumor_idx, cls, tf)
            else:
                tf = float(np.exp(rng.uniform(np.log(cfg.tf_min), np.log(cfg.tf_max))))
                add(tumor_idx, cls, tf)

    meta = pd.DataFrame(rows)
    meta.insert(0, "sample_id", [f"SIM-{split[:2].upper()}-{i:05d}" for i in range(len(meta))])
    return np.vstack(meths), np.vstack(totals), meta


def ref_col(df: pd.DataFrame, i: int) -> str:
    return str(df.columns[i])
