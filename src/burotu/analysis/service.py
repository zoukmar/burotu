"""One call from a lab file to a full result: quality control, detection, report."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache

from burotu.analysis import qc as qc_mod
from burotu.analysis.engines import BaselineEngine, MuseEngine
from burotu.analysis.report import write_report
from burotu.labformat import LabSample


@lru_cache(maxsize=1)
def baseline() -> BaselineEngine:
    return BaselineEngine()


def muse() -> MuseEngine | None:
    url, token = os.environ.get("BUROTU_MUSE_URL"), os.environ.get("BUROTU_MUSE_TOKEN")
    return MuseEngine(url, token, baseline().panel) if url and token else None


def analyse(sample: LabSample, lang: str = "en") -> dict:
    panel = baseline().panel
    quality = qc_mod.check(sample, panel).to_dict()
    detection, notes = None, []
    if quality["status"] != "fail":
        engine = muse()
        muse_result = None
        if engine is not None:
            try:
                muse_result = engine.analyse(sample).to_dict()
            except Exception as e:  # Muse unreachable: fall back, and say so
                notes.append(f"Muse Glimmer was unavailable ({type(e).__name__}); the benchmark classifier answered.")
        if muse_result is not None and os.environ.get("BUROTU_MUSE_ROLE", "full") == "origin":
            # Hybrid: the benchmark classifier raises the alarm (its threshold holds 99.8% specificity),
            # Muse Glimmer names the tissue of origin (where it beats the classifier at low tumour DNA).
            detection = baseline().analyse(sample).to_dict()
            detection["origin"] = muse_result["origin"]
            detection["engine"] = "baseline+muse-glimmer"
            detection["engine_label"] = "Benchmark classifier (cancer signal) + Muse Glimmer 30B (tissue of origin)"
        elif muse_result is not None:
            detection = muse_result
        else:
            detection = baseline().analyse(sample).to_dict()
    report = write_report(quality, detection, lang)
    return {
        "sample_id": sample.sample_id,
        "lab": sample.header.get("lab", ""),
        "analysed_at": datetime.now().isoformat(timespec="seconds"),
        "quality": quality,
        "detection": detection,
        "report": report,
        "notes": notes,
    }
