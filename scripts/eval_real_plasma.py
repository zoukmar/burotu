"""Re-run the real plasma check on the saved baseline and update reports/baseline.json.

python scripts/eval_real_plasma.py   (after scripts/train_baseline.py)
"""

from __future__ import annotations

import json

from burotu import realplasma
from burotu.config import ROOT
from burotu.data.reference import Reference
from burotu.model.baseline import BaselineModel


def main() -> None:
    from train_baseline import print_real_plasma

    model = BaselineModel.load(ROOT / "models" / "baseline.pkl")
    path = ROOT / "reports" / "baseline.json"
    report = json.loads(path.read_text())
    report["real_plasma"] = realplasma.report(model, Reference.load().panel)
    path.write_text(json.dumps(report, indent=2))
    print_real_plasma(report["real_plasma"])


if __name__ == "__main__":
    main()
