"""Score models trained on a newer refined version on the frozen v1 benchmark (same samples as before).

    BUROTU_REFINED=v2 python -m burotu.evaluation.cross_version     # -> reports/cross_v1_benchmark.json

The v1 test cohort (4,000 healthy samples + 42 TCGA tumours x 6 tumour fractions x 20 tissues)
was frozen on October 28. v2 keeps every v1 test donor and tumour out of training, so a v2 model
scored on it is directly comparable with the v1 models' published numbers. Two thresholds:

  * own:   the one each v2 model calibrated on its own (multi-cohort) calibration donors;
  * local: set on the v1 calibration cohort (Hannum donors never used for v1 testing), as a lab
           calibrates on healthy controls of its own batch; this is how the v1 models were set.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from burotu.config import ROOT, STORE
from burotu.evaluation import benchmark
from burotu.models.base import asset_dir

V1 = STORE / "refined" / "v1" / "benchmark"


def _summary(r: dict) -> dict:
    at = {round(x["tumor_fraction"], 6): x for x in r["by_tumor_fraction"]}
    return {"specificity": r["specificity_observed"], "signal_at_1pct": at[0.01]["sensitivity"],
            "tissue_at_1pct": at[0.01]["origin_top1"], "signal_at_0.1pct": at[0.001]["sensitivity"],
            "signal_at_10pct": at[0.1]["sensitivity"], "tissue_at_10pct": at[0.1]["origin_top1"]}  # fmt: skip


def main() -> None:
    from burotu.models.classic.model import BaselineModel
    from burotu.models.logreg.model import LogRegEnsemble

    arrays = np.load(V1 / "test.npz")
    meth, total = arrays["meth"].astype(np.int64), arrays["total"].astype(np.int64)
    meta = pd.read_csv(V1 / "test.csv")
    models = {
        "classic": BaselineModel.load(ROOT / "models" / "baseline.pkl"),
        "logreg": LogRegEnsemble.load(asset_dir("logreg")),
    }
    calib = np.load(V1 / "calibration.npz")
    c_meth, c_total = calib["meth"].astype(np.int64), calib["total"].astype(np.int64)
    out = {}
    for name, m in models.items():
        score, proba = m.cancer_score(meth, total), m.origin_proba(meth, total)
        r = benchmark.evaluate(score, proba, m.classes, m.threshold, meta)
        local = benchmark.threshold_at_specificity(m.cancer_score(c_meth, c_total))
        rl = benchmark.evaluate(score, proba, m.classes, local, meta)
        out[name] = {"v2_on_v1_test": _summary(r), "v2_on_v1_test_local": _summary(rl), "full": r, "full_local": rl}
    for name, old in (("classic", "baseline-20t.json"), ("logreg", "logreg-20t.json")):
        p = ROOT / "reports" / old
        if p.exists():
            out[name]["v1_on_v1_test"] = _summary(json.loads(p.read_text())["test"])
    (ROOT / "reports" / "cross_v1_benchmark.json").write_text(json.dumps(out, indent=2))
    for name, r in out.items():
        print(
            name,
            "v1:",
            r.get("v1_on_v1_test"),
            "\n  v2 own:",
            r["v2_on_v1_test"],
            "\n  v2 local:",
            r["v2_on_v1_test_local"],
        )


if __name__ == "__main__":
    main()
