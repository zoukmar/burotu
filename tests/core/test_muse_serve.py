"""The Muse web API, served from a run directory built with the tiny random Glimmer."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
from fastapi.testclient import TestClient  # noqa: E402

from burotu.analysis.engines import MuseEngine  # noqa: E402
from burotu.labformat import parse_lab_text  # noqa: E402
from burotu.muse.reader import MuseReader  # noqa: E402
from burotu.muse.serve import MuseService, make_app  # noqa: E402
from burotu.muse.train import TrainConfig, load_lm  # noqa: E402

PANEL = [f"cg{i:08d}" for i in range(9)]
CLASSES = ["Colorectal", "Lung"]
TOKEN = "t" * 32


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    run = tmp_path_factory.mktemp("run")
    torch.manual_seed(0)
    lm, tok = load_lm(TrainConfig(model="tiny", lora_r=4))
    MuseReader(lm, tok, len(PANEL) + 3, CLASSES, n_tokens=4).save_trainable(run / "trainable.pt")
    np.save(run / "mu.npy", np.full(len(PANEL), 0.2, dtype=np.float32))
    meta = {"model": "tiny", "revision": "", "load_4bit": False, "lora_r": 4, "n_tokens": 4}
    (run / "serving.json").write_text(json.dumps({**meta, "panel": PANEL, "classes": CLASSES, "threshold": 0.0}))
    return TestClient(make_app(MuseService(run), TOKEN))


def body(meth=50, total=100):
    return {"sample_id": "S1", "meth": [meth] * len(PANEL), "total": [total] * len(PANEL)}


def test_health_is_public(client):
    assert client.get("/health").json()["classes"] == CLASSES


def test_predict_needs_the_token(client):
    assert client.post("/predict", json=body()).status_code == 401
    assert client.post("/predict", json=body(), headers={"Authorization": "Bearer nope"}).status_code == 401


def test_predict_answers_like_the_other_engines(client):
    d = client.post("/predict", json=body(), headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert d["sample_id"] == "S1" and isinstance(d["cancer_signal"], bool)
    assert [o["class"] for o in d["origin"]] and abs(sum(o["probability"] for o in d["origin"]) - 1) < 1e-3


def test_impossible_counts_are_refused(client):
    r = client.post("/predict", json=body(meth=200), headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 422


def test_tracking_site_engine_talks_to_the_api(client, monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: client.post("/predict", json=kw["json"], headers=kw["headers"])
    )
    rows = "\n".join(f"{c}\t30\t100" for c in PANEL)
    sample = parse_lab_text(f"# burotu-bmc v1\n# sample_id=S2\ncpg_id\tmeth\ttotal\n{rows}\n")
    d = MuseEngine("http://muse/predict", TOKEN, PANEL).analyse(sample)
    assert d.engine == "muse-glimmer" and d.origin[0]["class"] in CLASSES
