"""Make a result readable: where the score sits among healthy people, and what each tissue shows.

A score and a threshold are hard to read. This module turns one sample into numbers a
person can check:

* **Rank among healthy people**: the classic model's score against 3,000 simulated healthy
  plasma samples from calibration donors (never used to fit the model).
* **Lit sites per tissue**: for each tissue, how many of its marker sites carry more
  methylated reads than 99.5% of healthy samples do at that site, next to the most a
  healthy sample shows. Marker sites are panel sites methylated in that tissue's tumours
  and quiet in blood and in the other tissues, picked on training data only.
* **What pushed the score**: the classic model's own contributions (SHAP values from
  LightGBM), summed by group of sites.

At low signal only a few sites light up: the model adds up weak evidence spread over
thousands of sites. The page says so rather than overstating what the counts show.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_MARKERS = 150  # per tissue
MIN_GAIN = 0.30  # tissue tumour median minus blood mean
MIN_SPECIFICITY = 0.10  # tissue tumour median minus the median of the other tissues
SITE_QUANTILE = 0.995  # a site is lit above this quantile of healthy z-scores at that site
MIN_Z = 3.0


def build_markers(
    blood_train: pd.DataFrame, tumor_train: pd.DataFrame, classes: pd.Series, panel: list[str]
) -> dict[str, list[int]]:
    """Marker sites per tissue, as indices into `panel` (training data only)."""
    pos = {cpg: i for i, cpg in enumerate(panel)}
    blood = blood_train.loc[panel].mean(axis=1)
    t = tumor_train.loc[panel]
    out: dict[str, list[int]] = {}
    for cls in sorted(classes.unique()):
        cols = classes.index[classes == cls]
        own = t[cols].median(axis=1)
        other = t.drop(columns=cols).median(axis=1)
        gain, spec = own - blood, own - other
        ok = (gain >= MIN_GAIN) & (spec >= MIN_SPECIFICITY)
        best = (gain + spec)[ok].sort_values(ascending=False).index[:MAX_MARKERS]
        out[cls] = sorted(pos[c] for c in best)
    return out


def build_explainer(z_healthy: np.ndarray, healthy_scores: np.ndarray, markers: dict[str, list[int]]) -> dict:
    """Everything the explainer needs, from healthy calibration samples only."""
    cut = np.maximum(np.quantile(z_healthy, SITE_QUANTILE, axis=0), MIN_Z)
    lit = z_healthy > cut
    healthy_max = {k: int(np.quantile(lit[:, v].sum(axis=1), 0.99)) for k, v in markers.items()}
    return {
        "markers": markers,
        "site_cut": [round(float(c), 3) for c in cut],
        "healthy_lit_p99": healthy_max,
        "healthy_lit_total": {q: int(np.quantile(lit.sum(axis=1), v)) for q, v in (("p50", 0.5), ("p99", 0.99))},
        "healthy_scores": [round(float(s), 7) for s in np.quantile(healthy_scores, np.linspace(0, 1, 1001))],
        "n_healthy": len(healthy_scores),
    }


RANK_SCALE = 3.5  # the rank scale runs from 0% to 99.97% on a log axis: -log10(1 - rank)


def rank_position(rank: float, n_healthy: int) -> float:
    """Where a healthy-rank falls on a 0-1 log scale (50% -> 0.09, 99.5% -> 0.66, top of 3,000 -> ~1)."""
    import math

    tail = max(1.0 - rank, 1.0 / max(n_healthy, 1))
    return round(min(-math.log10(tail), RANK_SCALE) / RANK_SCALE, 4)


class Explainer:
    def __init__(self, path: Path) -> None:
        d = json.loads(path.read_text())
        self.markers: dict[str, list[int]] = d["markers"]
        self.cut = np.asarray(d["site_cut"], dtype=np.float32)
        self.healthy_max: dict[str, int] = d["healthy_lit_p99"]
        self.healthy_scores = np.asarray(d["healthy_scores"])
        self.n_healthy: int = d["n_healthy"]
        self.healthy_total: dict = d.get("healthy_lit_total", {})
        marked = sorted({i for v in self.markers.values() for i in v})
        self.unmarked = np.setdiff1d(np.arange(len(self.cut)), marked)

    def healthy_rank(self, score: float) -> float:
        """Share of healthy samples scoring below this one (0 to 1)."""
        return float(np.searchsorted(self.healthy_scores, score, side="right") - 1) / (len(self.healthy_scores) - 1)

    def explain(
        self, z: np.ndarray, total: np.ndarray, contrib: np.ndarray, score: float, classes: list[str], proba: np.ndarray
    ) -> dict:
        """One sample in panel order: z-scores, depths, detector SHAP values (features + bias), tissue probabilities."""
        lit = z > self.cut
        read = total > 0
        n_sites = len(self.cut)
        tissues = []
        for k, cls in enumerate(classes):
            idx = np.asarray(self.markers.get(cls, []), dtype=int)
            if idx.size == 0:
                continue
            tissues.append(
                {
                    "class": cls,
                    "markers": int(idx.size),
                    "read": int(read[idx].sum()),
                    "lit": int(lit[idx].sum()),
                    "healthy_max": int(self.healthy_max.get(cls, 0)),
                    "push": round(float(contrib[idx].sum()), 3),
                    "probability": round(float(proba[k]), 4),
                }
            )
        tissues.sort(key=lambda t: (t["lit"] - t["healthy_max"], t["lit"]), reverse=True)
        site_push = float(contrib[:n_sites].sum())
        return {
            "sites": n_sites,
            "sites_read": int(read.sum()),
            "lit": int(lit.sum()),
            "healthy_lit": self.healthy_total,
            "healthy_rank": round(self.healthy_rank(score), 4),
            "rank_pos": rank_position(self.healthy_rank(score), self.n_healthy),
            "n_healthy": self.n_healthy,
            "push": {
                "tissue_markers": round(sum(t["push"] for t in tissues), 3),
                "other_sites": round(float(contrib[self.unmarked].sum()), 3),
                "summary": round(float(contrib[n_sites:-1].sum()), 3),
                "all_sites": round(site_push, 3),
            },
            "tissues": tissues,
        }
