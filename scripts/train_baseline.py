"""Train and evaluate the baseline classifier on the frozen benchmark.

    python scripts/train_baseline.py

Cohorts come from `burotu.benchmark` (simulated once, cached in data/benchmark/), so
the baseline and Muse Glimmer are scored on exactly the same samples. Real patient
plasma (Moss et al. 2018, EPIC array) is used as a sanity check only.

Writes models/baseline.pkl, reports/baseline.json and docs/assets/baseline_sensitivity.png.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from burotu import benchmark, realplasma
from burotu.benchmark import TARGET_SPECIFICITY
from burotu.config import HEALTHY, ROOT
from burotu.data.reference import Reference
from burotu.model.baseline import BaselineModel
from burotu.model.features import Featurizer


def naive_score(meth: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Day-1 reference: mean methylation over the whole panel."""
    return meth.sum(axis=1) / np.maximum(total.sum(axis=1), 1)


def print_real_plasma(rp: dict) -> None:
    for key in ("whole_blood_background", "recalibrated"):
        r = rp[key]
        print(
            f"real plasma, {r['background']}: AUC {r['auc_cancer_vs_healthy']:.3f}, "
            f"cancers {r['detected_cancer']}/{r['n_cancer']}, "
            f"healthy flagged {r['detected_healthy']}/{r['n_healthy']}, "
            f"origin {r['origin_top1_correct']}/{r['n_known_origin']}"
        )


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["test"]["by_tumor_fraction"]
    x = [r["tumor_fraction"] * 100 for r in rows]
    ink, muted, grid = "#1b2130", "#5c6370", "#e3e6ea"
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for key, label, color in [
        ("sensitivity", "Burotu baseline classifier", "#2a78d6"),
        ("sensitivity_naive", "Naive panel average", "#eb6834"),
    ]:
        y = [r[key] * 100 for r in rows]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, mec="white", mew=1.5, label=label)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{v:g}%" for v in x])
    ax.set_ylim(0, 102)
    ax.set_xlim(x[0] * 0.8, x[-1] * 1.25)  # room for the end markers
    ax.set_xlabel("Tumour fraction in plasma", color=muted)
    ax.set_ylabel(f"Sensitivity at {TARGET_SPECIFICITY:.1%} specificity (%)", color=muted)
    ax.set_title("Cancer signal detection on held-out simulated plasma", color=ink, loc="left", fontsize=11)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.tick_params(colors=muted, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main() -> None:
    t0 = time.time()
    ref = Reference.load()
    fit_ref, calib_ref = benchmark.split_donors(ref)
    fit, calib, test = (benchmark.load_cohort(n, ref) for n in ("fit", "calibration", "test"))
    print(f"cohorts ready: fit {len(fit)}, calibration {len(calib)}, test {len(test)} ({time.time() - t0:.0f}s)")

    featurizer = Featurizer.from_blood(ref.blood_train.to_numpy(np.float32))
    model = BaselineModel.fit(featurizer, fit.meth, fit.total, fit.labels, HEALTHY)
    model.calibrate(calib.meth, calib.total, TARGET_SPECIFICITY)
    print(f"trained ({time.time() - t0:.0f}s)")

    result = benchmark.evaluate(
        model.cancer_score(test.meth, test.total),
        model.origin_proba(test.meth, test.total),
        model.classes,
        model.threshold,
        test.meta,
    )
    # Day-1 naive score on the same samples, for the chart.
    naive_thr = benchmark.threshold_at_specificity(naive_score(calib.meth, calib.total))
    naive = naive_score(test.meth, test.total) > naive_thr
    is_cancer = test.labels != HEALTHY
    tf = test.meta["tumor_fraction"].to_numpy()
    for row in result["by_tumor_fraction"]:
        m = is_cancer & np.isclose(tf, row["tumor_fraction"])
        row["sensitivity_naive"] = float(naive[m].mean())

    report = {
        "target_specificity": TARGET_SPECIFICITY,
        "panel_size": len(ref.panel),
        "classes": model.classes,
        "train_samples": len(fit),
        "donors": {
            "fit": fit_ref.blood_train.shape[1],
            "calibration": calib_ref.blood_train.shape[1],
            "test": ref.blood_test.shape[1],
        },
        "threshold": model.threshold,
        "test": result,
        "real_plasma": realplasma.report(model, ref.panel),
    }
    model.save(ROOT / "models" / "baseline.pkl")
    out = ROOT / "reports" / "baseline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    plot(report, ROOT / "docs" / "assets" / "baseline_sensitivity.png")

    print(f"observed specificity: {result['specificity_observed']:.4f}")
    for r in result["by_tumor_fraction"]:
        print(
            f"TF {r['tumor_fraction']:>7.2%}  sens {r['sensitivity']:.2f} (naive {r['sensitivity_naive']:.2f})"
            f"  AUC {r['auc']:.3f}  origin top1 {r['origin_top1']:.2f} top2 {r['origin_top2']:.2f}"
        )
    print_real_plasma(report["real_plasma"])
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
