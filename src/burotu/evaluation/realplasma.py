"""Real plasma check: Moss et al. 2018 cell-free DNA (GSE122126, EPIC array).

Models are trained on simulated plasma built from whole-blood arrays. Real plasma differs
in two ways: it is measured on another platform, and healthy cell-free DNA is not pure
blood (endothelial, erythroid and liver DNA add background methylation; on the panel the
healthy real plasma average is about twice the whole-blood one). Scored against the
whole-blood background, every healthy sample then looks like cancer.

A lab deals with this the way it calibrates any new assay: by running healthy controls
on it. `recalibrated` re-estimates the healthy background from the healthy plasma
samples, leaving out the donor group being scored (both replicates of a pool, or one
individual), so no sample is ever scored against a background that contains it. The
model, its threshold and the pseudo read depth are left untouched.

With 12 healthy and 15 cancer samples this is a sanity check, not a performance figure.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from burotu.config import HEALTHY, PROCESSED_DIR
from burotu.core.features import EPS

PSEUDO_DEPTH = 500  # array beta values become read counts at this depth
SUBSET = PROCESSED_DIR / "real_plasma_panel.parquet"  # panel CpGs only, committed (small)


def _panel_subset(panel: list[str]):
    """Moss beta values on the panel (CpG × samples) and sample metadata, cached in SUBSET."""
    import pandas as pd

    if not SUBSET.exists():
        from burotu.data.geo import load_moss_cfdna

        beta, meta = load_moss_cfdna()
        sub = beta.reindex(panel).astype(np.float32)
        sub.columns = pd.MultiIndex.from_arrays([meta["title"], meta["label"]], names=["title", "label"])
        sub.to_parquet(SUBSET)
    sub = pd.read_parquet(SUBSET)
    return sub.reindex(panel), sub.columns.get_level_values("title"), sub.columns.get_level_values("label")


def load(panel: list[str]):
    beta, titles, labels = _panel_subset(panel)
    b = beta.to_numpy(np.float32).T  # samples × panel, a few NaN
    labels, titles = np.asarray(labels, dtype=object), np.asarray(titles, dtype=object)
    healthy = labels == HEALTHY
    groups = np.array([t.rsplit("_rep", 1)[0] if h else t for t, h in zip(titles, healthy, strict=True)])
    return b, labels, titles, groups


def _score(model, b: np.ndarray, mu: np.ndarray):
    b = np.where(np.isnan(b), mu[None, :], b)
    total = np.full(b.shape, PSEUDO_DEPTH, dtype=np.int64)
    meth = np.rint(b * PSEUDO_DEPTH).astype(np.int64)
    original = model.featurizer
    model.featurizer = dataclasses.replace(original, mu=mu.astype(np.float32))
    try:
        return model.cancer_score(meth, total), model.origin_proba(meth, total)
    finally:
        model.featurizer = original


def evaluate(model, panel: list[str], recalibrate: bool) -> dict:
    """`model` has featurizer, cancer_score, origin_proba, threshold and classes."""
    b, labels, titles, groups = load(panel)
    healthy = labels == HEALTHY
    score = np.zeros(len(b))
    proba = np.zeros((len(b), len(model.classes)))
    if not recalibrate:
        score[:], proba[:] = _score(model, b, model.featurizer.mu)
    else:
        for g in np.unique(groups):
            rows = groups == g
            controls = healthy & ~rows
            with warnings.catch_warnings():  # a CpG unread in every control keeps the model's background
                warnings.simplefilter("ignore", RuntimeWarning)
                mu = np.nanmean(b[controls], axis=0)
            mu = np.clip(np.where(np.isnan(mu), model.featurizer.mu, mu), EPS, 1 - EPS)
            score[rows], proba[rows] = _score(model, b[rows], mu)

    detected = score > model.threshold
    top1 = np.array([model.classes[i] for i in proba.argmax(axis=1)])
    known = ~healthy & np.isin(labels, model.classes)
    return {
        "background": "healthy plasma controls, leave-one-donor-group-out" if recalibrate else "whole blood",
        "auc_cancer_vs_healthy": round(float(roc_auc_score(~healthy, score)), 4),
        "detected_healthy": int(detected[healthy].sum()),
        "detected_cancer": int(detected[~healthy].sum()),
        "n_healthy": int(healthy.sum()),
        "n_cancer": int((~healthy).sum()),
        "origin_top1_correct": int((top1[known] == labels[known]).sum()),
        "n_known_origin": int(known.sum()),
        "samples": [
            {
                "title": str(t),
                "label": str(lab),
                "cancer_score": round(float(s), 4),
                "detected": bool(d),
                "origin_top1": str(o),
            }
            for t, lab, s, d, o in zip(titles, labels, score, detected, top1, strict=True)
        ],
    }


def report(model, panel: list[str]) -> dict:
    return {
        "note": "Moss et al. 2018 cfDNA on EPIC arrays, 12 healthy and 15 cancer samples: a sanity check, "
        "not a performance figure. Model and threshold unchanged; only the healthy background differs.",
        "whole_blood_background": evaluate(model, panel, recalibrate=False),
        "recalibrated": evaluate(model, panel, recalibrate=True),
    }
