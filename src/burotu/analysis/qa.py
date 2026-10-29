"""Muse Spark answers questions about one result, and explains quality-control problems.

Two jobs, both grounded only in the analysis JSON:

* `answer`: a question asked by the clinic for the patient or the doctor ("What does
  'colon and rectum' mean?", "Should I repeat the test?"). Same rules as the letter:
  a signal, never a disease; no diagnosis, no treatment advice; the doctor decides.
* `qc_note`: for the lab, when quality control did not pass, what went wrong in plain
  words and what to rerun.

Every answer goes through the same guard as the letter (burotu.analysis.wording), and
falls back to a fixed sentence when Muse Spark is unavailable or keeps breaking a rule.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from burotu.analysis.report import _facts
from burotu.analysis.wording import LANG_NAMES, LANGS, is_clean, qc_findings
from burotu.llm.client import MetaClient

AUDIENCES = ("patient", "doctor")
MAX_QUESTION = 600

QA_SYSTEM = """You answer questions about one Burotu result. Burotu is a demonstration prototype that \\
looks for an unusual DNA methylation signal in the cell-free DNA of a blood sample; it is not a medical \\
device. You receive the result as JSON, then a question from the clinic on behalf of the {audience}.

Rules:
- Answer in {language}, in 40 to 120 words, plainly{doctor_tone}. No headings.
- Use only facts from the JSON. If the answer is not in it, say so and suggest asking the doctor.
- Talk only about a "signal". Never write the words cancer, tumour, malignant, carcinoma, oncology or \\
metastasis, in any language, and never name a disease.
- Never give a diagnosis, a prognosis or treatment advice. The doctor decides what happens next.
- Name tissues exactly as written in the JSON.
- If the question is unrelated to this result, say you can only answer questions about it."""

QC_SYSTEM = """You help a lab technician understand why a sample failed or only partly passed quality \\
control in Burotu, a demonstration prototype that reads DNA methylation from a targeted sequencing panel \\
(1,681 sites). You receive the quality-control numbers as JSON.

Write in {language}, 50 to 110 words, plain sentences, no headings: what the numbers mean, the most likely \\
causes in the lab (for example too little cell-free DNA, a library that sequenced poorly, a wrong panel \\
file), and what to do next (re-sequence the same library, prepare a new library, or ask for a new blood \\
draw). Use only the numbers in the JSON. Never name a disease."""

FALLBACK = {
    "en": "This question cannot be answered automatically right now. Please ask the doctor.",
    "fr": "Cette question ne peut pas recevoir de réponse automatique pour le moment. Posez-la au médecin.",
    "es": "Ahora mismo no se puede responder a esta pregunta automáticamente. Pregúntale al médico.",
}

QC_FALLBACK = {
    "en": "Quality control did not pass: {findings}. Check the library and re-sequence it, or ask for a new "
    "blood draw if too little DNA was available.",
    "fr": "Le contrôle qualité n'est pas passé : {findings}. Vérifiez la librairie et reséquencez-la, ou "
    "demandez une nouvelle prise de sang si l'ADN était en trop faible quantité.",
    "es": "El control de calidad no se superó: {findings}. Revisa la librería y vuelve a secuenciarla, o pide "
    "una nueva extracción si había demasiado poco ADN.",
}


def result_facts(analysis: dict, lang: str) -> dict:
    """Everything Muse Spark may use about one analysis, in the target language."""
    facts = _facts(analysis["quality"], analysis.get("detection"), lang)
    models = analysis.get("models") or {}
    if models.get("classic") and models.get("meta"):
        facts["models"] = {
            "signal_decided_by": "classic model (tuned so that 99.5% of healthy people are not flagged)",
            "tissue_named_by": "Muse Glimmer, Meta's open-weight model fine-tuned by Burotu",
            "models_agree_on_tissue": models["classic"]["origin"][0]["class"] == models["meta"]["origin"][0]["class"],
        }
    reports = analysis.get("reports") or {}
    if lang in reports:
        facts["letter_already_sent"] = reports[lang]["text"]
    facts["about_burotu"] = (
        "Demonstration prototype, not a medical device, not validated. Looks at 8 tissue types today. "
        "Does not replace the screenings the doctor recommends."
    )
    return facts


def _ask(client: MetaClient, system: str, payload: str, max_tokens: int = 4000) -> str | None:
    """One Muse Spark answer that passes the guard (one rewrite allowed), or None."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": payload}]
    for _ in range(2):
        text = client.chat(messages, temperature=0.2, max_tokens=max_tokens).text.strip()
        if text and is_clean(text):
            return text
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "That answer broke a rule. Rewrite it following every rule."},
        ]
    return None


def answer(analysis: dict, question: str, lang: str, audience: str = "patient", client=None) -> dict:
    """Return {"text", "author"} for a question about this result."""
    lang = lang if lang in LANGS else "en"
    audience = audience if audience in AUDIENCES else "patient"
    question = question.strip()[:MAX_QUESTION]
    client = client or MetaClient()
    if client.mock and client.mock_handler is None:
        return {"text": FALLBACK[lang], "author": "fallback (Muse Spark not connected)"}
    system = QA_SYSTEM.format(
        audience=audience,
        language=LANG_NAMES[lang],
        doctor_tone=", with the technical details a doctor expects" if audience == "doctor" else "",
    )
    payload = json.dumps({"result": result_facts(analysis, lang), "question": question}, ensure_ascii=False)
    try:
        text = _ask(client, system, payload)
    except Exception as e:  # never break the page on a question
        return {"text": FALLBACK[lang], "author": f"fallback (Muse Spark unavailable: {type(e).__name__})"}
    if text is None:
        return {"text": FALLBACK[lang], "author": "fallback (answer broke a rule)"}
    return {"text": text, "author": client.model}


def qc_note(qc: dict, lang: str, client=None) -> dict:
    """Lab-facing explanation of a quality-control problem, or {} when QC passed cleanly."""
    if qc.get("status") == "pass":
        return {}
    lang = lang if lang in LANGS else "en"
    findings = "; ".join(qc_findings(qc, lang)) or "-"
    client = client or MetaClient()
    fallback = {"text": QC_FALLBACK[lang].format(findings=findings), "author": "template"}
    if client.mock and client.mock_handler is None:
        return fallback
    numbers = {k: qc.get(k) for k in ("status", "panel_coverage", "median_depth", "p10_depth", "low_depth_share")}
    numbers["findings"] = qc_findings(qc, lang)
    try:
        text = _ask(client, QC_SYSTEM.format(language=LANG_NAMES[lang]), json.dumps(numbers, ensure_ascii=False))
    except Exception as e:
        return {**fallback, "author": f"template (Muse Spark unavailable: {type(e).__name__})"}
    return {"text": text, "author": client.model} if text else fallback


def qc_notes(qc: dict, client=None) -> dict:
    """The lab note in every language, written in parallel ({} when QC passed)."""
    if qc.get("status") == "pass":
        return {}
    with ThreadPoolExecutor(max_workers=len(LANGS)) as pool:
        futures = {lang: pool.submit(qc_note, qc, lang, client) for lang in LANGS}
        return {lang: f.result() for lang, f in futures.items()}
