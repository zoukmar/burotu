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
    assert len(engine.panel) == len(engine.featurizer.mu) == 3148
    assert len(engine.classes) == 20 and 0 < engine.threshold < 1


@pytest.mark.parametrize("sid", ["BRT-DEMO-01", "BRT-DEMO-05", "BRT-DEMO-07", "BRT-DEMO-09"])
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
    result = analyse(read_lab_file(DEMO / "BRT-DEMO-05.tsv"), "fr")
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
        def analyse_all(self, sample):
            origin = [{"class": "Lung", "probability": 0.9}]
            return {
                "meta": Detection("muse-glimmer", "Muse", False, -5.0, 0.0, origin),
                "cpgpt": Detection("cpgpt", "CpGPT", True, 2.0, 1.0, [{"class": "Kidney", "probability": 0.7}]),
            }

    monkeypatch.setattr(service, "muse", lambda: FakeMuse())
    monkeypatch.setenv("BUROTU_MUSE_ROLE", "origin")
    a = service.analyse(read_lab_file(DEMO / "BRT-DEMO-05.tsv"), "en")
    d = a["detection"]
    assert d["cancer_signal"] is True  # from the benchmark classifier
    assert d["origin"][0]["class"] == "Lung" and d["engine"] == "baseline+muse-glimmer"
    assert set(a["models"]) == {"classic", "logreg", "meta", "cpgpt"}
    c = a["consensus"]
    assert c["n"] == 4 and c["signal_votes"] == 3 and not c["unanimous"]  # Muse said no
    assert c["tissue"] == "Kidney" and c["tissue_votes"] == 3


def test_answer_text_handles_parts_and_empty_answers():
    from burotu.llm.client import _answer_text

    assert _answer_text({"choices": [{"message": {"content": "Bonjour"}}]}) == "Bonjour"
    parts = {"choices": [{"message": {"content": [{"type": "text", "text": "Bon"}, {"type": "text", "text": "jour"}]}}]}
    assert _answer_text(parts) == "Bonjour"
    with pytest.raises(ValueError, match="finish_reason=length"):
        _answer_text({"choices": [{"message": {"content": None}, "finish_reason": "length"}]})


def _analysis():
    return analyse(read_lab_file(DEMO / "BRT-DEMO-05.tsv"), "en")


def test_question_is_answered_from_the_result_only():
    from burotu.analysis.qa import answer
    from burotu.llm.client import MetaClient

    seen = {}

    def handler(messages):
        seen["payload"] = messages[-1]["content"]
        return "The signal most resembles tissue from the kidney. Your doctor decides what comes next."

    out = answer(_analysis(), "What does this mean?", "en", client=MetaClient(mock=True, mock_handler=handler))
    assert out["text"].startswith("The signal") and "kidney" in seen["payload"]
    assert "cancer" not in seen["payload"].lower()


def test_an_answer_naming_a_disease_is_rewritten_then_replaced():
    from burotu.analysis.qa import FALLBACK, answer
    from burotu.llm.client import MetaClient

    calls = []

    def always_bad(messages):
        calls.append(1)
        return "You may have kidney cancer."

    out = answer(_analysis(), "Is it cancer?", "fr", client=MetaClient(mock=True, mock_handler=always_bad))
    assert len(calls) == 2 and out["text"] == FALLBACK["fr"]


def test_offline_answers_and_lab_notes_fall_back(monkeypatch):
    from burotu.analysis.qa import FALLBACK, answer, qc_notes

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert answer(_analysis(), "Hello?", "es")["text"] == FALLBACK["es"]
    failed = analyse(read_lab_file(DEMO / "BRT-DEMO-08.tsv"), "en")
    notes = failed["qc_notes"]
    assert set(notes) == {"en", "fr", "es"} and "re-sequence" in notes["en"]["text"]
    assert qc_notes({"status": "pass"}) == {}


def test_explanation_counts_lit_marker_sites():
    from burotu.analysis.engines import BaselineEngine
    from burotu.analysis.explain import rank_position
    from burotu.labformat import read_lab_file
    from burotu_web.admin import DEMO_DIR

    engine = BaselineEngine()
    kidney = engine.explain(read_lab_file(DEMO_DIR / "BRT-DEMO-05.tsv"))  # kidney, 10% tumour DNA
    healthy = engine.explain(read_lab_file(DEMO_DIR / "BRT-DEMO-01.tsv"))
    assert kidney["healthy_rank"] > 0.995 > healthy["healthy_rank"]
    assert kidney["lit"] > kidney["healthy_lit"]["p99"] >= healthy["lit"]
    best = max(kidney["tissues"], key=lambda t: t["lit"] - t["healthy_max"])
    assert best["class"] == "Kidney" and best["lit"] > best["healthy_max"]
    assert 0 < rank_position(0.5, 3000) < rank_position(0.995, 3000) < rank_position(1.0, 3000) <= 1.0
