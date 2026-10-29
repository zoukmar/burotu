"""Export the trained baseline as portable assets for the tracking site.

    python -m burotu.models.classic.export   (after python -m burotu.models.classic.train)

Writes src/burotu/analysis/assets/baseline/: LightGBM text models (gzipped), the healthy
background per CpG, the panel order, classes and the calibrated threshold. Text models
avoid pickles, so the site can load them with any LightGBM 4.x.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from burotu.config import ROOT
from burotu.data.reference import Reference
from burotu.evaluation.benchmark import load_cohort
from burotu.models.classic.explain import build_explainer, build_markers
from burotu.models.classic.model import BaselineModel

OUT = Path(__file__).resolve().parent / "assets"


def main() -> None:
    model = BaselineModel.load(ROOT / "models" / "baseline.pkl")
    ref = Reference.load()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, est in [("detector", model.detector), ("origin", model.origin)]:
        text = est.booster_.model_to_string().encode()
        (OUT / f"{name}.txt.gz").write_bytes(gzip.compress(text, compresslevel=9, mtime=0))
    np.save(OUT / "mu.npy", model.featurizer.mu.astype(np.float32))
    meta = {"panel": ref.panel, "classes": model.classes, "threshold": model.threshold}
    (OUT / "meta.json").write_text(json.dumps(meta))

    # Explanation assets: marker sites per tissue, and how many sites light up in healthy people.
    classes = ref.tumor_meta.set_index("file_id").loc[ref.tumor_train.columns, "cancer_class"]
    markers = build_markers(ref.blood_train, ref.tumor_train, classes, ref.panel)
    calib = load_cohort("calibration")
    healthy = ~calib.meta["is_cancer"].to_numpy(bool)
    x = model.featurizer.transform(calib.meth[healthy], calib.total[healthy])
    scores = model.detector.predict_proba(x)[:, 1]
    explainer = build_explainer(x[:, : len(ref.panel)], scores, markers)
    (OUT / "explain.json").write_text(json.dumps(explainer))
    print("markers per tissue:", {k: len(v) for k, v in markers.items()})
    print("most lit markers in a healthy sample (p99):", explainer["healthy_lit_p99"])
    print(f"exported to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
