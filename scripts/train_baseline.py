"""Train and evaluate the baseline classifier.

    python scripts/train_baseline.py

Cohorts (people and tumours never cross between them):
  * train:       simulated from the training split, random tumour fractions
  * calibration: healthy draws from training-split donors held out of model fitting,
                 so the threshold is set on people the model has never seen
  * test:        simulated from the test split at fixed tumour fractions
  * real plasma: Moss et al. 2018 cfDNA (EPIC array), validation only

Writes models/baseline.pkl, reports/baseline.json and docs/assets/baseline_sensitivity.png.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from burotu.config import HEALTHY, ROOT
from burotu.data.reference import Reference
from burotu.model.baseline import TARGET_SPECIFICITY, BaselineModel
from burotu.model.features import Featurizer
from burotu.simulate.cfdna import SimConfig, simulate_cohort

TEST_TFS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def naive_score(meth: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Day-1 reference: mean methylation over the whole panel."""
    return meth.sum(axis=1) / np.maximum(total.sum(axis=1), 1)


def evaluate(model: BaselineModel, meth, total, meta, calib_naive: np.ndarray) -> dict:
    labels = meta["label"].to_numpy()
    is_cancer = labels != HEALTHY
    tf = meta["tumor_fraction"].to_numpy()
    score = model.cancer_score(meth, total)
    detected = score > model.threshold
    proba = model.origin_proba(meth, total)
    rank = np.argsort(proba, axis=1)[:, ::-1]
    truth_idx = np.array([model.classes.index(c) if c in model.classes else -1 for c in labels])
    top1 = rank[:, 0] == truth_idx
    top2 = (rank[:, :2] == truth_idx[:, None]).any(axis=1)

    naive = naive_score(meth, total)
    naive_thr = float(np.quantile(calib_naive, TARGET_SPECIFICITY))

    out = {
        "specificity_observed": float(1 - detected[~is_cancer].mean()),
        "n_healthy": int((~is_cancer).sum()),
        "by_tumor_fraction": [],
        "by_class_at_1pct": {},
    }
    for f in TEST_TFS:
        m = is_cancer & np.isclose(tf, f)
        auc = roc_auc_score(np.r_[np.zeros((~is_cancer).sum()), np.ones(m.sum())], np.r_[score[~is_cancer], score[m]])
        out["by_tumor_fraction"].append(
            {
                "tumor_fraction": f,
                "n": int(m.sum()),
                "sensitivity": float(detected[m].mean()),
                "sensitivity_naive": float((naive[m] > naive_thr).mean()),
                "auc": float(auc),
                "origin_top1": float(top1[m].mean()),
                "origin_top2": float(top2[m].mean()),
                "origin_top1_when_detected": float(top1[m & detected].mean()) if (m & detected).any() else None,
            }
        )
    for c in model.classes:
        m = (labels == c) & np.isclose(tf, 1e-2)
        out["by_class_at_1pct"][c] = {"sensitivity": float(detected[m].mean()), "origin_top1": float(top1[m].mean())}
    return out


def evaluate_real_plasma(model: BaselineModel, panel: list[str]) -> dict:
    """Moss et al. cfDNA: array beta values turned into pseudo read counts (depth 500)."""
    from burotu.data.geo import load_moss_cfdna

    beta, meta = load_moss_cfdna()
    b = beta.reindex(panel).to_numpy(np.float32).T  # samples × panel
    b = np.where(np.isnan(b), model.featurizer.mu[None, :], b)
    total = np.full(b.shape, 500, dtype=np.int64)
    meth = np.rint(b * 500).astype(np.int64)
    score = model.cancer_score(meth, total)
    proba = model.origin_proba(meth, total)
    labels = meta["label"].to_numpy()
    is_cancer = labels != HEALTHY
    samples = []
    for i, row in meta.iterrows():
        samples.append(
            {
                "title": row["title"],
                "label": labels[i],
                "cancer_score": round(float(score[i]), 4),
                "detected": bool(score[i] > model.threshold),
                "origin_top1": model.classes[int(np.argmax(proba[i]))],
            }
        )
    return {
        "auc_cancer_vs_healthy": float(roc_auc_score(is_cancer, score)),
        "detected_healthy": int(sum(s["detected"] for s in samples if s["label"] == HEALTHY)),
        "detected_cancer": int(sum(s["detected"] for s in samples if s["label"] != HEALTHY)),
        "n_healthy": int((~is_cancer).sum()),
        "n_cancer": int(is_cancer.sum()),
        "note": "Different platform (EPIC array, not sequencing): treat as a sanity check, not a performance figure.",
        "samples": samples,
    }


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-healthy", type=int, default=4000)
    ap.add_argument("--train-per-class", type=int, default=500)
    ap.add_argument("--test-healthy", type=int, default=4000)
    ap.add_argument("--test-per-class", type=int, default=42)
    ap.add_argument("--calib-donors", type=int, default=35)
    args = ap.parse_args()

    t0 = time.time()
    ref = Reference.load()
    featurizer = Featurizer.from_blood(ref.blood_train.to_numpy(np.float32))

    # Hold out some training donors: the detection threshold must be set on unseen people.
    donors = np.array(sorted(ref.blood_train.columns))
    np.random.default_rng(0).shuffle(donors)
    calib_donors, fit_donors = donors[: args.calib_donors], donors[args.calib_donors :]
    fit_ref = dataclasses.replace(ref, blood_train=ref.blood_train[fit_donors])
    calib_ref = dataclasses.replace(ref, blood_train=ref.blood_train[calib_donors])

    meth, total, meta = simulate_cohort(
        fit_ref, "train", args.train_healthy, args.train_per_class, SimConfig(seed=1, tf_min=3e-4)
    )
    model = BaselineModel.fit(featurizer, meth, total, meta["label"].to_numpy(), HEALTHY)
    print(f"trained on {len(meta)} samples ({time.time() - t0:.0f}s)")

    c_meth, c_total, _ = simulate_cohort(calib_ref, "train", 3000, 0, SimConfig(seed=2))
    model.calibrate(c_meth, c_total)
    calib_naive = naive_score(c_meth, c_total)

    t_meth, t_total, t_meta = simulate_cohort(
        ref, "test", args.test_healthy, args.test_per_class, SimConfig(seed=3), tf_values=TEST_TFS
    )
    report = {
        "target_specificity": TARGET_SPECIFICITY,
        "panel_size": len(ref.panel),
        "classes": model.classes,
        "train_samples": len(meta),
        "donors": {"fit": len(fit_donors), "calibration": len(calib_donors), "test": ref.blood_test.shape[1]},
        "threshold": model.threshold,
        "test": evaluate(model, t_meth, t_total, t_meta, calib_naive),
        "real_plasma": evaluate_real_plasma(model, ref.panel),
    }
    model.save(ROOT / "models" / "baseline.pkl")
    out = ROOT / "reports" / "baseline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    plot(report, ROOT / "docs" / "assets" / "baseline_sensitivity.png")

    print(f"observed specificity: {report['test']['specificity_observed']:.4f}")
    for r in report["test"]["by_tumor_fraction"]:
        print(
            f"TF {r['tumor_fraction']:>7.2%}  sens {r['sensitivity']:.2f} (naive {r['sensitivity_naive']:.2f})"
            f"  AUC {r['auc']:.3f}  origin top1 {r['origin_top1']:.2f} top2 {r['origin_top2']:.2f}"
        )
    rp = report["real_plasma"]
    print(f"real plasma AUC {rp['auc_cancer_vs_healthy']:.3f}, detected {rp['detected_cancer']}/{rp['n_cancer']} "
          f"cancers, {rp['detected_healthy']}/{rp['n_healthy']} healthy")  # fmt: skip
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
