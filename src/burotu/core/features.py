"""Turn read counts into model features.

Healthy plasma is almost unmethylated on the panel, so what matters is how many
methylated reads exceed what healthy blood would produce at the same depth. For each
CpG we compute a binomial z-score against the healthy background:

    z = (meth - total * mu) / sqrt(total * mu * (1 - mu))

where `mu` is the mean methylation of that CpG in the *training* blood donors only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-3  # floor for the background rate: avoids dividing by zero on fully clean CpGs


@dataclass
class Featurizer:
    mu: np.ndarray  # healthy background rate per panel CpG

    @classmethod
    def from_blood(cls, blood_train_beta: np.ndarray) -> Featurizer:
        """`blood_train_beta`: CpG × donors (training split only)."""
        return cls(mu=np.clip(blood_train_beta.mean(axis=1), EPS, 1 - EPS).astype(np.float32))

    def transform(self, meth: np.ndarray, total: np.ndarray) -> np.ndarray:
        """`meth`, `total`: samples × CpG. Returns samples × (CpG + 3) features."""
        meth = meth.astype(np.float32)
        total = total.astype(np.float32)
        expected = total * self.mu
        z = (meth - expected) / np.sqrt(total * self.mu * (1 - self.mu) + 1e-6)
        z = np.clip(z, -10, 50)
        excess = meth.sum(axis=1) - expected.sum(axis=1)
        summary = np.column_stack(
            [
                excess / np.maximum(total.sum(axis=1), 1),  # global excess methylation
                (z > 3).mean(axis=1),  # share of CpGs clearly above background
                np.sort(z, axis=1)[:, -20:].mean(axis=1),  # strength of the top 20 CpGs
            ]
        )
        return np.hstack([z, summary]).astype(np.float32)
