"""One call from a lab file to a full result: quality control, detection, report."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache

from burotu.analysis import qc as qc_mod
from burotu.analysis.engines import BaselineEngine, MuseEngine
from burotu.analysis.report import write_reports
from burotu.labformat import LabSample


@lru_cache(maxsize=1)
def baseline() -> BaselineEngine:
    return BaselineEngine()


def muse() -> MuseEngine | None:
    url, token = os.environ.get("BUROTU_MUSE_URL"), os.environ.get("BUROTU_MUSE_TOKEN")
    return MuseEngine(url, token, baseline().panel) if url and token else None


def analyse(sample: LabSample, lang: str = "en") -> dict:
    """QC, then the signal and its likely tissue, then the letter in every language.

    `models` keeps both answers side by side: the classic classifier, and Muse Glimmer
    when it is served. `detection` is what Burotu reports: with BUROTU_MUSE_ROLE=origin,
    the classifier's signal (99.8% specificity) and Muse Glimmer's tissue.
    """
    panel = baseline().panel
    quality = qc_mod.check(sample, panel).to_dict()
    detection, notes, models, explain = None, [], {}, None
    if quality["status"] != "fail":
        models["classic"] = baseline().analyse(sample).to_dict()
        explain = baseline().explain(sample)
        engine = muse()
        if engine is not None:
            try:
                models["meta"] = engine.analyse(sample).to_dict()
            except Exception as e:  # Muse unreachable: fall back, and say so
                notes.append(f"Muse Glimmer was unavailable ({type(e).__name__}); the classic model answered alone.")
        meta = models.get("meta")
        if meta is not None and os.environ.get("BUROTU_MUSE_ROLE", "full") == "origin":
            detection = dict(models["classic"])
            detection["origin"] = meta["origin"]
            detection["engine"] = "baseline+muse-glimmer"
            detection["engine_label"] = "Classic model (signal) + Muse Glimmer 30B (tissue)"
        elif meta is not None:
            detection = meta
        else:
            detection = models["classic"]
    reports = write_reports(quality, detection)
    from burotu.analysis.qa import qc_notes

    notes_for_lab = qc_notes(quality)
    return {
        "sample_id": sample.sample_id,
        "lab": sample.header.get("lab", ""),
        "analysed_at": datetime.now().isoformat(timespec="seconds"),
        "quality": quality,
        "detection": detection,
        "models": models,
        "explain": explain,
        "reports": reports,
        "report": reports.get(lang, reports["en"]),
        "notes": notes,
        "qc_notes": notes_for_lab,
    }
