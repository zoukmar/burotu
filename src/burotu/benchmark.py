"""Frozen benchmark shared by every model (baseline classifier, Muse Glimmer).

Cohorts are simulated once with fixed seeds and cached in data/benchmark/, so every
model is trained and scored on exactly the same samples:

  * fit:          training-split donors and tumours, random tumour fractions
  * calibration:  healthy samples from training donors held out of model fitting;
                  the detection threshold is set here, on people no model has seen
  * test:         test-split donors and tumours at fixed tumour fractions

`evaluate` takes plain arrays (a cancer score and origin probabilities), so any model
that produces them is scored the same way.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from burotu.config import DATA_DIR, HEALTHY
from burotu.data.reference import Reference
from burotu.simulate.cfdna import SimConfig, simulate_cohort

BENCHMARK_DIR = DATA_DIR / "benchmark"
TARGET_SPECIFICITY = 0.995
TEST_TFS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
CALIB_DONORS = 35

# name -> (split, n_healthy, n_cancer_per_class, SimConfig, fixed tumour fractions)
COHORTS = {
    "fit": ("train", 4000, 500, SimConfig(seed=1, tf_min=3e-4), None),
    # Larger training set for Muse (same fitting donors and tumours, different draws).
    "fit_large": ("train", 12000, 2000, SimConfig(seed=11, tf_min=3e-4), None),
    "calibration": ("train", 3000, 0, SimConfig(seed=2), None),
    "test": ("test", 4000, 42, SimConfig(seed=3), TEST_TFS),
}


@dataclass
class Cohort:
    meth: np.ndarray  # samples × CpG
    total: np.ndarray  # samples × CpG
    meta: pd.DataFrame  # label, is_cancer, tumor_fraction, ...

    @property
    def labels(self) -> np.ndarray:
        return self.meta["label"].to_numpy()

    def __len__(self) -> int:
        return len(self.meta)


def split_donors(ref: Reference, n_calib: int = CALIB_DONORS) -> tuple[Reference, Reference]:
    """Split training donors into model-fitting donors and calibration-only donors."""
    donors = np.array(sorted(ref.blood_train.columns))
    np.random.default_rng(0).shuffle(donors)
    calib, fit = donors[:n_calib], donors[n_calib:]
    return (
        dataclasses.replace(ref, blood_train=ref.blood_train[fit]),
        dataclasses.replace(ref, blood_train=ref.blood_train[calib]),
    )


def load_cohort(name: str, ref: Reference | None = None, cache_dir: Path = BENCHMARK_DIR) -> Cohort:
    """Load a frozen cohort, simulating and caching it on first use."""
    npz, csv = cache_dir / f"{name}.npz", cache_dir / f"{name}.csv"
    if npz.exists() and csv.exists():
        arrays = np.load(npz)
        return Cohort(arrays["meth"].astype(np.int64), arrays["total"].astype(np.int64), pd.read_csv(csv))

    ref = ref or Reference.load()
    fit_ref, calib_ref = split_donors(ref)
    split, n_healthy, n_per_class, cfg, tfs = COHORTS[name]
    source = {"fit": fit_ref, "fit_large": fit_ref, "calibration": calib_ref, "test": ref}[name]
    meth, total, meta = simulate_cohort(source, split, n_healthy, n_per_class, cfg, tf_values=tfs)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, meth=meth.astype(np.uint16), total=total.astype(np.uint16))
        meta.to_csv(csv, index=False)
    except OSError:  # read-only data dir (e.g. a Modal mount): same seed, same cohort next time
        pass
    return Cohort(meth, total, meta)


def threshold_at_specificity(healthy_scores: np.ndarray, specificity: float = TARGET_SPECIFICITY) -> float:
    return float(np.quantile(healthy_scores, specificity))


def evaluate(
    score: np.ndarray,
    origin_proba: np.ndarray,
    classes: list[str],
    threshold: float,
    meta: pd.DataFrame,
    tfs: list[float] = TEST_TFS,
) -> dict:
    """Score a model's outputs on the test cohort (same metrics for every model)."""
    labels = meta["label"].to_numpy()
    is_cancer = labels != HEALTHY
    tf = meta["tumor_fraction"].to_numpy()
    detected = score > threshold
    rank = np.argsort(origin_proba, axis=1)[:, ::-1]
    truth = np.array([classes.index(c) if c in classes else -1 for c in labels])
    top1 = rank[:, 0] == truth
    top2 = (rank[:, :2] == truth[:, None]).any(axis=1)

    out = {
        "threshold": float(threshold),
        "specificity_observed": float(1 - detected[~is_cancer].mean()),
        "n_healthy": int((~is_cancer).sum()),
        "by_tumor_fraction": [],
        "by_class_at_1pct": {},
    }
    for f in tfs:
        m = is_cancer & np.isclose(tf, f)
        y = np.r_[np.zeros((~is_cancer).sum()), np.ones(m.sum())]
        out["by_tumor_fraction"].append(
            {
                "tumor_fraction": f,
                "n": int(m.sum()),
                "sensitivity": float(detected[m].mean()),
                "auc": float(roc_auc_score(y, np.r_[score[~is_cancer], score[m]])),
                "origin_top1": float(top1[m].mean()),
                "origin_top2": float(top2[m].mean()),
            }
        )
    for c in classes:
        m = (labels == c) & np.isclose(tf, 1e-2)
        out["by_class_at_1pct"][c] = {"sensitivity": float(detected[m].mean()), "origin_top1": float(top1[m].mean())}
    return out
