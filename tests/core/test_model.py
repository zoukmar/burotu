import numpy as np

from burotu.core.features import Featurizer
from burotu.models.classic.model import BaselineModel


def _cohort(rng, n_per_label=60, n_cpg=50):
    """Healthy samples near background; each cancer class lights up its own CpG block."""
    mu = np.full(n_cpg, 0.01)
    labels, betas = [], []
    for label, block in [("Healthy", None), ("A", slice(0, 10)), ("B", slice(10, 20))]:
        for _ in range(n_per_label):
            b = mu.copy()
            if block is not None:
                b[block] = 0.2
            betas.append(b)
            labels.append(label)
    beta = np.array(betas)
    total = rng.poisson(400, beta.shape)
    meth = rng.binomial(total, beta)
    return meth, total, np.array(labels), mu


def test_featurizer_scores_excess_methylation():
    f = Featurizer(mu=np.full(3, 0.01, dtype=np.float32))
    total = np.array([[1000, 1000, 1000]])
    x_clean = f.transform(np.array([[10, 10, 10]]), total)
    x_hot = f.transform(np.array([[10, 10, 200]]), total)
    assert abs(x_clean[0, :3]).max() < 1
    assert x_hot[0, 2] > 20
    assert x_hot.shape == (1, 3 + 3)


def test_baseline_separates_signal_and_origin(tmp_path):
    rng = np.random.default_rng(0)
    meth, total, labels, mu = _cohort(rng)
    f = Featurizer(mu=mu.astype(np.float32))
    model = BaselineModel.fit(f, meth, total, labels, "Healthy")
    h_meth, h_total, h_labels, _ = _cohort(np.random.default_rng(1))
    model.calibrate(h_meth[h_labels == "Healthy"], h_total[h_labels == "Healthy"], specificity=0.99)

    preds = model.predict(h_meth, h_total)
    cancer = h_labels != "Healthy"
    detected = np.array([p["cancer_signal"] for p in preds])
    assert detected[cancer].mean() > 0.95 and detected[~cancer].mean() < 0.05
    origin = np.array([p["origin"][0]["class"] for p in preds])
    assert (origin[cancer] == h_labels[cancer]).mean() > 0.95

    model.save(tmp_path / "m.pkl")
    assert BaselineModel.load(tmp_path / "m.pkl").threshold == model.threshold


def test_registry_lists_every_model_package_with_a_card():
    from burotu.models.base import MODELS_DIR
    from burotu.models.registry import models

    ids = {m.id for m in models()}
    assert {"classic", "logreg", "glimmer", "cpgpt", "consensus"} <= ids
    for m in models():
        assert m.status in {"live", "training", "ready", "experiment"}
        if m.id != "spark":
            assert (MODELS_DIR / m.id / "README.md").exists(), m.id
        for w in m.weights:
            kind, _, where = w.partition(":")
            assert kind in {"git", "hf", "modal"}
            if kind == "git" and m.status == "live":
                assert (MODELS_DIR.parents[2] / where).exists(), where
