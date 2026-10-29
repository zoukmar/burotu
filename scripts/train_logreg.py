"""Train and evaluate the logistic-regression ensembles (GRAIL-style) on the frozen benchmark.

    python scripts/train_logreg.py

Same cohorts, same threshold rule and same metrics as scripts/train_baseline.py. Writes the
portable model to src/burotu/analysis/assets/logreg/ and the report to reports/logreg.json.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from burotu import benchmark, realplasma
from burotu.benchmark import TARGET_SPECIFICITY
from burotu.config import HEALTHY, ROOT
from burotu.data.reference import Reference
from burotu.model.features import Featurizer
from burotu.model.logreg import LogRegEnsemble

OUT = ROOT / "src" / "burotu" / "analysis" / "assets" / "logreg"


def main() -> None:
    t0 = time.time()
    ref = Reference.load()
    fit_name = os.environ.get("BUROTU_FIT_COHORT", "fit")
    fit, calib, test = (benchmark.load_cohort(n, ref) for n in (fit_name, "calibration", "test"))
    featurizer = Featurizer.from_blood(ref.blood_train.to_numpy(np.float32))
    model = LogRegEnsemble.fit(featurizer, fit.meth, fit.total, fit.labels, HEALTHY)
    model.calibrate(calib.meth, calib.total, TARGET_SPECIFICITY)
    print(f"trained ({time.time() - t0:.0f}s)")
    result = benchmark.evaluate(
        model.cancer_score(test.meth, test.total),
        model.origin_proba(test.meth, test.total),
        model.classes,
        model.threshold,
        test.meta,
    )
    report = {
        "model": "Ensembles of logistic regressions (GRAIL-style), 8 bootstrap models each",
        "target_specificity": TARGET_SPECIFICITY,
        "panel_size": len(ref.panel),
        "classes": model.classes,
        "train_samples": len(fit),
        "threshold": model.threshold,
        "test": result,
        "real_plasma": realplasma.report(model, ref.panel),
    }
    tag = os.environ.get("BUROTU_RUN_TAG", "")
    model.save(OUT if not tag else ROOT / "models" / f"logreg{tag}")
    (ROOT / "reports" / f"logreg{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"observed specificity: {result['specificity_observed']:.4f}")
    for r in result["by_tumor_fraction"]:
        print(
            f"TF {r['tumor_fraction']:>7.2%}  sens {r['sensitivity']:.2f}  AUC {r['auc']:.3f}"
            f"  origin top1 {r['origin_top1']:.2f} top2 {r['origin_top2']:.2f}"
        )
    for key in ("whole_blood_background", "recalibrated"):
        r = report["real_plasma"][key]
        print(
            f"real plasma, {r['background']}: AUC {r['auc_cancer_vs_healthy']:.3f}, cancers "
            f"{r['detected_cancer']}/{r['n_cancer']}, healthy flagged {r['detected_healthy']}/{r['n_healthy']}, "
            f"origin {r['origin_top1_correct']}/{r['n_known_origin']}"
        )
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
