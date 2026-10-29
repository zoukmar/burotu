"""Muse Glimmer at analysis time: an authenticated call to the GPU server on Modal.

The same call also returns the CpGPT sample embedding when the server runs CpGPT.
"""

from __future__ import annotations

import httpx

from burotu.core.labformat import LabSample
from burotu.models.base import Detection, counts_on_panel


class MuseEngine:
    """Fine-tuned Muse Glimmer behind an authenticated Modal web endpoint."""

    def __init__(self, url: str, token: str, panel: list[str], timeout: float = 180.0) -> None:
        self.url, self.token, self.panel, self.timeout = url, token, panel, timeout

    def analyse(self, sample: LabSample) -> Detection:
        return self.analyse_all(sample)["meta"]

    def analyse_all(self, sample: LabSample) -> dict[str, Detection]:
        """Muse Glimmer's answer, plus CpGPT's when the same GPU server also runs it."""
        meth, total = counts_on_panel(sample, self.panel)
        r = httpx.post(
            self.url,
            json={"sample_id": sample.sample_id, "meth": meth[0].tolist(), "total": total[0].tolist()},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        d = r.json()
        out = {
            "meta": Detection(
                engine="muse-glimmer",
                engine_label=d.get("engine_label", "Muse Glimmer 30B, fine-tuned (Meta open weights)"),
                cancer_signal=bool(d["cancer_signal"]),
                score=float(d["score"]),
                threshold=float(d["threshold"]),
                origin=d["origin"],
            )
        }
        from burotu.models.cpgpt.engine import cpgpt_heads

        heads = cpgpt_heads()
        if d.get("cpgpt_embedding") and heads is not None:
            out["cpgpt"] = heads.analyse(d["cpgpt_embedding"])
        return out
