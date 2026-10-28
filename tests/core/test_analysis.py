import json
from pathlib import Path

import pytest

from burotu.analysis import qc
from burotu.analysis.engines import BaselineEngine
from burotu.analysis.report import template_report
from burotu.analysis.service import analyse
from burotu.labformat import LabFormatError, parse_lab_text, read_lab_file

DEMO = Path(__file__).resolve().parents[2] / "src" / "burotu_web" / "demo_samples"
MANIFEST = {m["sample_id"]: m for m in json.loads((DEMO / "manifest.json").read_text())}


@pytest.fixture(scope="module")
def engine():
    return BaselineEngine()


def test_exported_baseline_matches_the_panel(engine):
    assert len(engine.panel) == len(engine.featurizer.mu) == 1681
    assert len(engine.classes) == 8 and 0 < engine.threshold < 1


@pytest.mark.parametrize("sid", ["BRT-DEMO-01", "BRT-DEMO-02", "BRT-DEMO-05"])
def test_demo_samples_are_called_correctly(engine, sid):
    sample = read_lab_file(DEMO / f"{sid}.tsv")
    truth = MANIFEST[sid]
    d = engine.analyse(sample)
    assert d.cancer_signal == (truth["label"] != "Healthy")
    if d.cancer_signal:
        assert d.origin[0]["class"] == truth["label"]


def test_quality_control_stops_an_under_sequenced_sample(engine):
    sample = read_lab_file(DEMO / "BRT-DEMO-08.tsv")
    report = qc.check(sample, engine.panel)
    assert report.status == "fail" and "depth" in report.issues[0]
    result = analyse(sample, "en")
    assert result["detection"] is None
    assert "new blood draw" in result["report"]["text"]


def test_service_falls_back_to_a_template_report_without_a_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    result = analyse(read_lab_file(DEMO / "BRT-DEMO-02.tsv"), "fr")
    assert result["report"]["author"] == "template"
    assert "prototype de démonstration" in result["report"]["text"]


def test_template_never_states_a_diagnosis():
    detection = {"cancer_signal": True, "origin": [{"class": "Lung", "probability": 0.8}]}
    text = template_report({"status": "pass", "issues": []}, detection, "en")
    assert "not a diagnosis" in text and "not a medical result" in text


@pytest.mark.parametrize(
    "text",
    [
        "cpg_id\tmeth\ttotal\ncg1\t1\t2\n",  # no magic header
        "# burotu-bmc v1\ncpg_id\tmeth\ttotal\ncg1\t1\t2\n",  # no sample id
        "# burotu-bmc v1\n# sample_id=X\ncpg_id\tmeth\ttotal\ncg1\t5\t2\n",  # meth > total
        "# burotu-bmc v1\n# sample_id=X\ncpg_id\tmeth\ttotal\ncg1\t0.5\t2\n",  # not counts
    ],
)
def test_bad_lab_files_are_rejected(text):
    with pytest.raises(LabFormatError):
        parse_lab_text(text)


def test_hybrid_role_takes_origin_from_muse_and_signal_from_baseline(monkeypatch):
    from burotu.analysis import service
    from burotu.analysis.engines import Detection

    class FakeMuse:
        def analyse(self, sample):
            origin = [{"class": "Lung", "probability": 0.9}]
            return Detection("muse-glimmer", "Muse", False, -5.0, 0.0, origin)

    monkeypatch.setattr(service, "muse", lambda: FakeMuse())
    monkeypatch.setenv("BUROTU_MUSE_ROLE", "origin")
    d = service.analyse(read_lab_file(DEMO / "BRT-DEMO-02.tsv"), "en")["detection"]
    assert d["cancer_signal"] is True  # from the benchmark classifier
    assert d["origin"][0]["class"] == "Lung" and d["engine"] == "baseline+muse-glimmer"
