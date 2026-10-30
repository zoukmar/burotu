"""Score every model on the open real-plasma datasets from GEO (burotu.data.public).

    python -m burotu.evaluation.public_plasma

Real patient plasma, measured on 450K/EPIC arrays by other labs: a check of how the models
trained on simulated plasma behave on real samples, not a validation. Two healthy
backgrounds are tried, as for Moss 2018 (burotu.evaluation.realplasma): whole blood (what the models
were trained with) and healthy plasma controls (the 12 healthy Moss samples, another lab,
never used for training).

Writes reports/public_plasma.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from burotu.config import DATA_DIR, HEALTHY, ROOT
from burotu.core.features import EPS
from burotu.evaluation import realplasma
from burotu.models.base import asset_dir
from burotu.models.base import panel as base_panel
from burotu.models.classic.model import BaselineModel
from burotu.models.logreg.model import LogRegEnsemble

PUBLIC = DATA_DIR / "public"
NO_CANCER = "No cancer"


def labels(gse: str) -> tuple[np.ndarray, str]:
    """Per-sample label (a Burotu tissue or NO_CANCER), and a description of the set."""
    m = pd.read_csv(PUBLIC / f"{gse}_samples.csv")
    if gse == "GSE277078":
        return np.array(["Lung"] * len(m)), "Non-small cell lung cancer, plasma, EPIC"
    if gse == "GSE108462":
        return np.array(["Prostate"] * len(m)), "Metastatic prostate cancer, plasma, 450K"
    if gse == "GSE298812":
        g = m.filter(like="characteristics").apply(lambda r: " ".join(map(str, r)), axis=1)
        y = np.where(g.str.contains("HCC-Pos"), "Liver", NO_CANCER)
        return y, "Liver cancer vs HIV+, fibrosis, cirrhosis, EPIC"
    if gse == "GSE129374":
        y = np.where(m["characteristics_ch1"].str.contains("with_HCC"), "Liver", NO_CANCER)
        return y, "Liver cancer vs cirrhosis, 450K"
    raise KeyError(gse)


def score(model, b: np.ndarray, mu: np.ndarray):
    return realplasma._score(model, b, mu)


def summary(model, s: np.ndarray, p: np.ndarray, y: np.ndarray) -> dict:
    cancer = y != NO_CANCER
    detected = s > model.threshold
    top1 = np.array([model.classes[i] for i in p.argmax(axis=1)])
    out = {
        "n_cancer": int(cancer.sum()),
        "n_no_cancer": int((~cancer).sum()),
        "detected": int(detected[cancer].sum()),
        "flagged_no_cancer": int(detected[~cancer].sum()),
        "tissue_right": int((top1[cancer] == y[cancer]).sum()),
        "top_tissues": pd.Series(top1[cancer]).value_counts().head(3).to_dict(),
    }
    if cancer.any() and (~cancer).any():
        out["auc"] = round(float(roc_auc_score(cancer, s)), 3)
    return out


def main() -> None:
    classic = BaselineModel.load(ROOT / "models" / "baseline.pkl")
    logreg = LogRegEnsemble.load(asset_dir("logreg"))
    panel = base_panel()
    moss, moss_labels, _, _ = realplasma.load(panel)
    with np.errstate(invalid="ignore"):
        plasma_mu = np.nanmean(moss[moss_labels == HEALTHY], axis=0)
    report = {
        "note": "Real plasma from other labs on 450K/EPIC arrays: a behaviour check, not a validation. "
        "'plasma controls' = healthy background from the 12 healthy Moss 2018 samples.",
        "datasets": {},
    }
    for gse in ("GSE277078", "GSE108462", "GSE298812", "GSE129374"):
        b = pd.read_parquet(PUBLIC / f"{gse}.parquet").reindex(panel).to_numpy(np.float32).T
        y, desc = labels(gse)
        entry = {"description": desc, "models": {}}
        for name, model in (("classic", classic), ("logreg", logreg)):
            entry["models"][name] = {}
            for bg, mu in (("whole blood", model.featurizer.mu), ("plasma controls", plasma_mu)):
                mu = np.clip(np.where(np.isnan(mu), model.featurizer.mu, mu), EPS, 1 - EPS)
                s, p = score(model, b, mu)
                entry["models"][name][bg] = summary(model, s, p, y)
        report["datasets"][gse] = entry
        print(gse, desc)
        for name, bgs in entry["models"].items():
            for bg, r in bgs.items():
                print(f"  {name:8s} {bg:16s} {r}")
    (ROOT / "reports" / "public_plasma.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
