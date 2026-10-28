"""Plain-language report written by Muse Spark, with a template fallback.

The report never states a diagnosis: Burotu is a demonstration prototype, and even a
real screening test only says whether further examinations are warranted.
"""

from __future__ import annotations

import json

from burotu.llm.client import MetaClient

LANG_NAMES = {"en": "English", "fr": "French", "es": "Spanish"}

SYSTEM = """You write short result letters for Burotu, a demonstration prototype of a multi-cancer \
blood test based on cell-free DNA methylation. You receive the lab quality checks and the model's \
output as JSON.

Rules:
- Write for a patient, in plain words, in {language}. 120 to 180 words. No headings, no lists.
- Never state or suggest a diagnosis. A detected signal means further examinations with a doctor are \
warranted; no detected signal does not rule out cancer.
- If quality control failed, say the sample could not be analysed reliably and that a new blood draw \
is needed; give no result.
- Mention the most likely tissue of origin only if a signal was detected, as a lead for the doctor, \
not a certainty.
- End with exactly this sentence, translated: "Burotu is a demonstration prototype and this is not a \
medical result."
- Do not invent numbers that are not in the JSON."""

TEMPLATES = {
    "en": {
        "fail": "Your sample could not be analysed reliably: {issues} A new blood draw is needed.",
        "signal": "A cancer signal was detected in your sample. The pattern most resembles tissue from "
        "{origin}. This is not a diagnosis: it means further examinations with your doctor are warranted.",
        "none": "No cancer signal was detected in your sample. This does not rule out cancer: very early "
        "tumours can shed too little DNA to be seen. Keep up the screenings your doctor recommends.",
        "end": "Burotu is a demonstration prototype and this is not a medical result.",
    },
    "fr": {
        "fail": "Votre échantillon n'a pas pu être analysé de façon fiable : {issues} Une nouvelle prise de "
        "sang est nécessaire.",
        "signal": "Un signal de cancer a été détecté dans votre échantillon. Le profil ressemble le plus à "
        "celui du tissu suivant : {origin}. Ce n'est pas un diagnostic : des examens complémentaires avec "
        "votre médecin sont justifiés.",
        "none": "Aucun signal de cancer n'a été détecté dans votre échantillon. Cela n'exclut pas un cancer : "
        "une tumeur débutante peut libérer trop peu d'ADN pour être vue. Poursuivez les dépistages "
        "recommandés par votre médecin.",
        "end": "Burotu est un prototype de démonstration et ceci n'est pas un résultat médical.",
    },
    "es": {
        "fail": "Tu muestra no pudo analizarse de forma fiable: {issues} Es necesaria una nueva extracción.",
        "signal": "Se ha detectado una señal de cáncer en tu muestra. El patrón se parece sobre todo al "
        "tejido: {origin}. No es un diagnóstico: significa que conviene hacer pruebas adicionales con tu "
        "médico.",
        "none": "No se ha detectado ninguna señal de cáncer en tu muestra. Esto no descarta un cáncer: un "
        "tumor incipiente puede liberar demasiado poco ADN para verse. Sigue con los cribados que te "
        "recomiende tu médico.",
        "end": "Burotu es un prototipo de demostración y esto no es un resultado médico.",
    },
}


def template_report(qc: dict, detection: dict | None, lang: str) -> str:
    t = TEMPLATES.get(lang, TEMPLATES["en"])
    if qc["status"] == "fail" or detection is None:
        body = t["fail"].format(issues=" ".join(qc["issues"]))
    elif detection["cancer_signal"]:
        body = t["signal"].format(origin=detection["origin"][0]["class"])
    else:
        body = t["none"]
    return f"{body} {t['end']}"


def write_report(qc: dict, detection: dict | None, lang: str, client: MetaClient | None = None) -> dict:
    """Return {"text", "author"}; Muse Spark when a key is configured, a template otherwise."""
    client = client or MetaClient()
    if client.mock:
        return {"text": template_report(qc, detection, lang), "author": "template"}
    payload = {"quality_control": qc, "result": detection}
    try:
        resp = client.chat(
            [
                {"role": "system", "content": SYSTEM.format(language=LANG_NAMES.get(lang, "English"))},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.3,
            max_tokens=6000,  # Muse Spark reasons first; reasoning tokens count against this budget
        )
        return {"text": resp.text.strip(), "author": resp.model}
    except Exception as e:  # never block a result on the report
        return {
            "text": template_report(qc, detection, lang),
            "author": f"template (Muse Spark unavailable: {type(e).__name__})",
        }
