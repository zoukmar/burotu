"""Cell-free DNA (cfDNA) simulator for the Burotu targeted panel.

Model:
  1. Plasma from a healthy person is mostly DNA released by blood cells: we draw a
     real healthy individual (whole-blood profile).
  2. In a person with cancer, a fraction `tf` (tumour fraction, typically 0.01% to
     10%) comes from the tumour: we draw a real TCGA tumour of the chosen class and
     mix the two profiles.
  3. Targeted bisulfite sequencing is simulated by binomial sampling of methylated
     reads, with a read depth that varies from one CpG to the next.

Known limitation: the simulation works CpG by CpG, without read-level methylation
patterns. Commercial tests exploit those patterns and are therefore more sensitive
at very low tumour fractions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from burotu.config import HEALTHY
from burotu.data.reference import Reference


@dataclass
class SimConfig:
    depth: float = 500.0  # mean read depth per CpG
    depth_sd_log: float = 0.5  # log-normal spread of depth across CpGs
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
    """Return (methylated reads, read depth) for one sample."""
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
    """Simulate a cohort from the `train` or `test` split (no leakage between them).

    Returns (meth, total, meta) with meth/total shaped (samples, panel CpGs).
    If `tf_values` is given, each cancer patient is simulated at each of these
    fractions (useful to plot sensitivity against tumour fraction); otherwise the
    fraction is drawn log-uniformly between tf_min and tf_max.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    blood = getattr(ref, f"blood_{split}").to_numpy(np.float32)  # CpG × individuals
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
