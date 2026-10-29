"""Train Burotu's two heads on frozen CpGPT embeddings, and score them like every other model.

    python -m burotu.models.cpgpt.train_heads cpgpt_out/   (embeddings.npz + embeddings_meta.json)

The embeddings come from `modal run scripts/modal_muse.py::cpgpt_benchmark` (Train Muse
workflow, task "cpgpt"): CpGPT `large`, frozen, on the fit / calibration / test cohorts,
Moss 2018 real plasma and the open GEO plasma sets. The heads are gradient-boosted trees,
like the classic model, so differences come from what CpGPT sees, not from the head.

Writes src/burotu/analysis/assets/cpgpt/ and reports/cpgpt.json.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from burotu.config import HEALTHY, ROOT
from burotu.evaluation import benchmark
from burotu.evaluation.benchmark import TARGET_SPECIFICITY
from burotu.models.classic.model import LGB_PARAMS

OUT = Path(__file__).resolve().parent / "assets"


def main(src: Path) -> None:
    e = np.load(src / "embeddings.npz")
    meta = json.loads((src / "embeddings_meta.json").read_text())
    fit, cal, test = (e[k].astype(np.float32) for k in ("fit", "calibration", "test"))
    fit_labels = np.asarray(meta["fit"]["label"])
    is_cancer = fit_labels != HEALTHY
    classes = sorted(set(fit_labels[is_cancer]))
    detector = lgb.LGBMClassifier(**LGB_PARAMS, random_state=0).fit(fit, is_cancer.astype(int))
    y = np.array([classes.index(c) for c in fit_labels[is_cancer]])
    origin = lgb.LGBMClassifier(**LGB_PARAMS, random_state=0).fit(fit[is_cancer], y)
    threshold = float(np.quantile(detector.predict_proba(cal)[:, 1], TARGET_SPECIFICITY))
    result = benchmark.evaluate(
        detector.predict_proba(test)[:, 1], origin.predict_proba(test), classes, threshold, pd.DataFrame(meta["test"])
    )
    moss_labels = np.asarray(meta["moss"]["label"])
    xm = e["moss"].astype(np.float32)
    ms, mp = detector.predict_proba(xm)[:, 1], origin.predict_proba(xm)
    healthy = moss_labels == HEALTHY
    top1 = np.array([classes[i] for i in mp.argmax(axis=1)])
    known = ~healthy & np.isin(moss_labels, classes)
    real = {
        "background": "none needed (CpGPT reads beta values directly)",
        "detected_healthy": int((ms[healthy] > threshold).sum()),
        "detected_cancer": int((ms[~healthy] > threshold).sum()),
        "n_healthy": int(healthy.sum()),
        "n_cancer": int((~healthy).sum()),
        "origin_top1_correct": int((top1[known] == moss_labels[known]).sum()),
        "n_known_origin": int(known.sum()),
    }
    public = {}
    for gse in ("GSE277078", "GSE108462", "GSE298812", "GSE129374"):
        if gse in e:
            x = e[gse].astype(np.float32)
            s, p = detector.predict_proba(x)[:, 1], origin.predict_proba(x)
            public[gse] = {
                "detected": int((s > threshold).sum()),
                "n": len(s),
                "top_tissues": pd.Series([classes[i] for i in p.argmax(axis=1)]).value_counts().head(3).to_dict(),
            }
    report = {
        "model": "CpGPT large (frozen, ~100M parameters) + gradient-boosted heads",
        "target_specificity": TARGET_SPECIFICITY,
        "classes": classes,
        "embedding_dim": int(fit.shape[1]),
        "train_samples": len(fit),
        "threshold": threshold,
        "test": result,
        "real_plasma": {"note": "Moss et al. 2018, 12 healthy and 15 cancer samples: a sanity check.", "raw": real},
        "public_plasma": public,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, est in (("detector", detector), ("origin", origin)):
        (OUT / f"{name}.txt.gz").write_bytes(gzip.compress(est.booster_.model_to_string().encode(), 9, mtime=0))
    (OUT / "meta.json").write_text(json.dumps({"classes": classes, "threshold": threshold, "dim": int(fit.shape[1])}))
    (ROOT / "reports" / "cpgpt.json").write_text(json.dumps(report, indent=2))
    print(f"observed specificity: {result['specificity_observed']:.4f}")
    for r in result["by_tumor_fraction"]:
        print(
            f"TF {r['tumor_fraction']:>7.2%}  sens {r['sensitivity']:.2f}  AUC {r['auc']:.3f}"
            f"  origin top1 {r['origin_top1']:.2f} top2 {r['origin_top2']:.2f}"
        )
    print("real plasma (Moss):", real)
    print("public plasma:", json.dumps(public))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
