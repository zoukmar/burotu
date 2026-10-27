"""Real plasma check: healthy controls recalibrate the background, never the sample scored."""

import numpy as np
import pytest

from burotu import realplasma
from burotu.analysis.engines import BaselineEngine
from burotu.model.features import Featurizer

pytestmark = pytest.mark.skipif(not realplasma.SUBSET.exists(), reason="real plasma subset not available")


class MeanExcess:
    """Toy model: average z-score above the healthy background."""

    classes = ("Colorectal", "Lung")
    threshold = 1.0

    def __init__(self, mu):
        self.featurizer = Featurizer(mu=mu)

    def cancer_score(self, meth, total):
        return self.featurizer.transform(meth, total)[:, :-3].mean(axis=1)

    def origin_proba(self, meth, total):
        return np.full((len(meth), 2), 0.5)


@pytest.fixture(scope="module")
def panel():
    return BaselineEngine().panel


def test_whole_blood_background_flags_healthy_plasma(panel):
    r = realplasma.evaluate(MeanExcess(BaselineEngine().featurizer.mu), panel, recalibrate=False)
    assert r["n_healthy"] == 12 and r["n_cancer"] == 15
    assert r["detected_healthy"] == 12  # the platform and biology shift this module corrects


def test_recalibrated_background_clears_healthy_plasma(panel):
    r = realplasma.evaluate(MeanExcess(BaselineEngine().featurizer.mu), panel, recalibrate=True)
    assert r["detected_healthy"] <= 1
    assert all(np.isfinite(s["cancer_score"]) for s in r["samples"])


def test_scored_group_is_left_out_of_its_background(panel):
    _, labels, _titles, groups = realplasma.load(panel)
    healthy = labels == "Healthy"
    assert len(set(groups[healthy])) == 8  # 4 pools (2 replicates each) + 4 individuals
    assert len(set(groups)) == 8 + 15
