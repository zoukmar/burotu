"""Baseline classifier: cancer signal detection and tissue of origin (CSO).

Two gradient-boosted models on the features from `features.py`:
  * detection: cancer vs healthy, thresholded at a fixed specificity;
  * tissue of origin: one of the cancer classes, trained on cancer samples only.

This is the yardstick the fine-tuned Muse model is compared against.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np

from burotu.model.features import Featurizer

TARGET_SPECIFICITY = 0.995

LGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.5,
    reg_lambda=1.0,
    n_jobs=2,
    verbose=-1,
)


@dataclass
class BaselineModel:
    featurizer: Featurizer
    classes: list[str]
    detector: lgb.LGBMClassifier = field(repr=False)
    origin: lgb.LGBMClassifier = field(repr=False)
    threshold: float = 0.5

    # --- Training -----------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        featurizer: Featurizer,
        meth: np.ndarray,
        total: np.ndarray,
        labels: np.ndarray,
        healthy_label: str,
        seed: int = 0,
    ) -> BaselineModel:
        x = featurizer.transform(meth, total)
        is_cancer = labels != healthy_label
        detector = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed).fit(x, is_cancer.astype(int))
        classes = sorted(set(labels[is_cancer]))
        y_origin = np.array([classes.index(c) for c in labels[is_cancer]])
        # LightGBM picks binary or multiclass from the number of labels.
        origin = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed).fit(x[is_cancer], y_origin)
        return cls(featurizer=featurizer, classes=classes, detector=detector, origin=origin)

    def calibrate(self, meth: np.ndarray, total: np.ndarray, specificity: float = TARGET_SPECIFICITY) -> float:
        """Set the detection threshold on healthy samples never used for training."""
        scores = self.cancer_score(meth, total)
        self.threshold = float(np.quantile(scores, specificity))
        return self.threshold

    # --- Inference -----------------------------------------------------------------

    def cancer_score(self, meth: np.ndarray, total: np.ndarray) -> np.ndarray:
        return self.detector.predict_proba(self.featurizer.transform(meth, total))[:, 1]

    def origin_proba(self, meth: np.ndarray, total: np.ndarray) -> np.ndarray:
        return self.origin.predict_proba(self.featurizer.transform(meth, total))

    def predict(self, meth: np.ndarray, total: np.ndarray) -> list[dict]:
        scores = self.cancer_score(meth, total)
        proba = self.origin_proba(meth, total)
        out = []
        for s, p in zip(scores, proba, strict=True):
            order = np.argsort(p)[::-1]
            out.append(
                {
                    "cancer_signal": bool(s > self.threshold),
                    "cancer_score": float(s),
                    "origin": [{"class": self.classes[i], "probability": float(p[i])} for i in order[:3]],
                }
            )
        return out

    # --- Persistence ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> BaselineModel:
        with open(path, "rb") as f:
            return pickle.load(f)
