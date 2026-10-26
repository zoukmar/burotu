import numpy as np
import pandas as pd
import pytest

from burotu.data.reference import build_reference
from burotu.labformat import LabFormatError, read_lab_file, write_lab_file
from burotu.llm.client import MetaClient
from burotu.simulate.cfdna import SimConfig, mix_and_sequence, simulate_cohort


def _toy_data(n_cpg=400, n_blood=20, per_class=10, seed=0):
    rng = np.random.default_rng(seed)
    idx = [f"cg{i:08d}" for i in range(n_cpg)]
    blood = pd.DataFrame(
        rng.beta(1, 30, (n_cpg, n_blood)), index=idx, columns=[f"B{i}" for i in range(n_blood)]
    ).astype("float32")
    cols, meta = {}, []
    for c, cls in enumerate(["A", "B"]):
        for j in range(per_class):
            v = rng.beta(1, 30, n_cpg)
            v[c * 50 : (c + 1) * 50] = rng.beta(20, 3, 50)  # marqueurs spécifiques
            fid = f"T{cls}{j}"
            cols[fid] = v
            meta.append({"file_id": fid, "case_id": fid, "project": cls, "cancer_class": cls})
    return blood, pd.DataFrame(cols, index=idx).astype("float32"), pd.DataFrame(meta)


def test_panel_and_split_no_leakage():
    blood, tumors, meta = _toy_data()
    ref = build_reference(blood, tumors, meta, per_class=30)
    assert set(ref.blood_train.columns).isdisjoint(ref.blood_test.columns)
    assert set(ref.tumor_train.columns).isdisjoint(ref.tumor_test.columns)
    # Les marqueurs plantés doivent dominer le panel.
    planted = {f"cg{i:08d}" for i in range(100)}
    assert len(planted & set(ref.panel)) / len(ref.panel) > 0.5


def test_mixing_increases_methylation_with_tf():
    rng = np.random.default_rng(1)
    b, t = np.full(5000, 0.01), np.full(5000, 0.9)
    cfg = SimConfig(depth=1000)
    lo = [mix_and_sequence(b, t, 0.001, cfg, rng) for _ in range(3)]
    hi = [mix_and_sequence(b, t, 0.05, cfg, rng) for _ in range(3)]

    def frac(xs):
        return np.mean([m.sum() / tot.sum() for m, tot in xs])

    assert frac(hi) > frac(lo) > 0.009


def test_cohort_shapes():
    blood, tumors, meta = _toy_data()
    ref = build_reference(blood, tumors, meta, per_class=30)
    m, tot, md = simulate_cohort(ref, "train", n_healthy=5, n_cancer_per_class=3)
    assert m.shape == tot.shape == (5 + 2 * 3, len(ref.panel))
    assert (m <= tot).all()
    assert md["is_cancer"].sum() == 6


def test_labformat_roundtrip(tmp_path):
    p = write_lab_file(tmp_path / "s.tsv", "BRT-1", ["cg1", "cg2"], [3, 0], [100, 50], lab="X")
    s = read_lab_file(p)
    assert s.sample_id == "BRT-1" and s.header["lab"] == "X"
    assert s.beta.loc["cg1"] == pytest.approx(0.03)
    bad = tmp_path / "bad.tsv"
    bad.write_text("cpg_id\tmeth\ttotal\ncg1\t1\t2\n")
    with pytest.raises(LabFormatError):
        read_lab_file(bad)


def test_meta_client_mock():
    c = MetaClient(api_key=None)
    assert c.mock
    r = c.chat([{"role": "user", "content": "bonjour"}], json_mode=True)
    assert r.mock and r.json()["mock"] is True
