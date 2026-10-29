"""A second, independent classic model: ensembles of logistic regressions.

This follows the approach GRAIL published for its multi-cancer test (Liu et al., Annals of
Oncology 2020): one ensemble of logistic regressions decides cancer signal vs no signal,
another ranks the tissue of origin. It reads the same binomial z-scores as the gradient
boosted model (features.py) but is a different family of model, so the two make different
mistakes: that is what makes it useful as a cross-check.

Each ensemble is a set of L2-regularised logistic regressions fitted on bootstrap samples
and averaged. The fitted model is plain arrays (scaler + weights), saved as .npz: no
pickles, no scikit-learn needed to run it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from burotu.model.features import Featurizer

TARGET_SPECIFICITY = 0.995
N_MODELS = 8
C_SIGNAL = 0.05
C_ORIGIN = 0.05
Z_CLIP = 10.0  # features are clipped before scaling: one extreme site should not dominate a linear model


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class LogRegEnsemble:
    featurizer: Featurizer
    classes: list[str]
    mean: np.ndarray = field(repr=False)  # feature scaler
    scale: np.ndarray = field(repr=False)
    w_signal: np.ndarray = field(repr=False)  # models × features
    b_signal: np.ndarray = field(repr=False)  # models
    w_origin: np.ndarray = field(repr=False)  # models × classes × features
    b_origin: np.ndarray = field(repr=False)  # models × classes
    threshold: float = 0.5

    # --- Training -------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        featurizer: Featurizer,
        meth: np.ndarray,
        total: np.ndarray,
        labels: np.ndarray,
        healthy_label: str,
        n_models: int = N_MODELS,
        seed: int = 0,
    ) -> LogRegEnsemble:
        from sklearn.linear_model import LogisticRegression

        x = np.clip(featurizer.transform(meth, total), -Z_CLIP, Z_CLIP)
        mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-6
        x = (x - mean) / scale
        is_cancer = labels != healthy_label
        classes = sorted(set(labels[is_cancer]))
        y_origin = np.array([classes.index(c) for c in labels[is_cancer]])
        xc = x[is_cancer]
        rng = np.random.default_rng(seed)
        ws, bs, wo, bo = [], [], [], []
        for _ in range(n_models):
            i = rng.integers(0, len(x), len(x))
            m = LogisticRegression(C=C_SIGNAL, max_iter=300).fit(x[i], is_cancer[i].astype(int))
            ws.append(m.coef_[0])
            bs.append(m.intercept_[0])
            j = rng.integers(0, len(xc), len(xc))
            o = LogisticRegression(C=C_ORIGIN, max_iter=300).fit(xc[j], y_origin[j])
            wo.append(o.coef_)
            bo.append(o.intercept_)
        return cls(
            featurizer=featurizer,
            classes=classes,
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            w_signal=np.array(ws, np.float32),
            b_signal=np.array(bs, np.float32),
            w_origin=np.array(wo, np.float32),
            b_origin=np.array(bo, np.float32),
        )

    def calibrate(self, meth: np.ndarray, total: np.ndarray, specificity: float = TARGET_SPECIFICITY) -> float:
        self.threshold = float(np.quantile(self.cancer_score(meth, total), specificity))
        return self.threshold

    # --- Inference --------------------------------------------------------------------

    def _x(self, meth: np.ndarray, total: np.ndarray) -> np.ndarray:
        x = np.clip(self.featurizer.transform(meth, total), -Z_CLIP, Z_CLIP)
        return (x - self.mean) / self.scale

    def cancer_score(self, meth: np.ndarray, total: np.ndarray) -> np.ndarray:
        """Mean log-odds of a cancer signal over the ensemble.

        Log-odds rather than probabilities: probabilities saturate at 1.0 for many samples,
        which would tie them at the threshold.
        """
        return (self._x(meth, total) @ self.w_signal.T + self.b_signal).mean(axis=1)

    def origin_proba(self, meth: np.ndarray, total: np.ndarray) -> np.ndarray:
        x = self._x(meth, total)
        probs = [_softmax(x @ w.T + b) for w, b in zip(self.w_origin, self.b_origin, strict=True)]
        return np.mean(probs, axis=0)

    # --- Portable assets ----------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "logreg.npz",
            mu=self.featurizer.mu,
            mean=self.mean,
            scale=self.scale,
            w_signal=self.w_signal,
            b_signal=self.b_signal,
            w_origin=self.w_origin.astype(np.float16),
            b_origin=self.b_origin,
        )
        (directory / "logreg.json").write_text(json.dumps({"classes": self.classes, "threshold": self.threshold}))

    @classmethod
    def load(cls, directory: Path) -> LogRegEnsemble:
        a = np.load(directory / "logreg.npz")
        meta = json.loads((directory / "logreg.json").read_text())
        return cls(
            featurizer=Featurizer(mu=a["mu"]),
            classes=meta["classes"],
            mean=a["mean"],
            scale=a["scale"],
            w_signal=a["w_signal"],
            b_signal=a["b_signal"],
            w_origin=a["w_origin"].astype(np.float32),
            b_origin=a["b_origin"],
            threshold=meta["threshold"],
        )
