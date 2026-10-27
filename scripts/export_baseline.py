"""Export the trained baseline as portable assets for the tracking site.

    python scripts/export_baseline.py   (after scripts/train_baseline.py)

Writes src/burotu/analysis/assets/baseline/: LightGBM text models (gzipped), the healthy
background per CpG, the panel order, classes and the calibrated threshold. Text models
avoid pickles, so the site can load them with any LightGBM 4.x.
"""

from __future__ import annotations

import gzip
import json

import numpy as np

from burotu.config import ROOT
from burotu.data.reference import Reference
from burotu.model.baseline import BaselineModel

OUT = ROOT / "src" / "burotu" / "analysis" / "assets" / "baseline"


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
    print(f"exported to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
