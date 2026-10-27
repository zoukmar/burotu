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
        if engine is not None:
            try:
                detection = engine.analyse(sample).to_dict()
            except Exception as e:  # Muse unreachable: fall back, and say so
                notes.append(f"Muse Glimmer was unavailable ({type(e).__name__}); the benchmark classifier answered.")
        if detection is None:
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
