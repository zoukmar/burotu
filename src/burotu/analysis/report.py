"""Plain-language result letter written by Muse Spark, with a template fallback.

Burotu reports a methylation *signal*, never a disease: the letter says whether an
unusual signal was found and which tissue it most resembles, and that a doctor decides
what happens next. It never names cancer or tumours (burotu.analysis.wording), and a
Muse Spark answer that does is rewritten once, then replaced by the template.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from burotu.analysis.wording import LANG_NAMES, LANGS, is_clean, qc_findings, tissue
from burotu.llm.client import MetaClient

SYSTEM = """You write short result letters for Burotu, a demonstration prototype that looks for an \
unusual DNA methylation signal in the cell-free DNA of a blood sample. You receive the lab quality \
checks and the analysis result as JSON.

Rules:
- Write for the person who gave blood, in plain, calm words, in {language}. 110 to 170 words. No \
headings, no lists.
- Talk only about a "signal" (an unusual methylation signal). Never write the words cancer, tumour, \
malignant, carcinoma, oncology or metastasis, in any language, and never name a disease.
- Never state or suggest a diagnosis. An unusual signal is a lead for the doctor to check with further \
examinations; no unusual signal does not rule out a health problem.
- If quality control failed, say the sample could not be analysed reliably and that a new blood draw is \
needed; give no result.
- If a signal was found, name the tissue it most resembles exactly as given in the JSON, as a lead for \
the doctor, not a certainty. Invite the person to share the letter with their doctor.
- End with exactly this sentence, translated: "Burotu is a demonstration prototype, not a medical \
device, and this is not a medical result."
- Do not invent numbers that are not in the JSON."""

RETRY = (
    "Your letter named a disease or used a forbidden word. Rewrite it following every rule: talk only about "
    "an unusual methylation signal and the tissue it resembles."
)

TEMPLATES = {
    "en": {
        "fail": "Your blood sample could not be analysed reliably: {issues}. A new blood draw is needed; "
        "no result can be given from this sample.",
        "signal": "The analysis found an unusual methylation signal in the DNA circulating in your blood. "
        "The signal most resembles tissue from the {tissue}. This is not a diagnosis: it is a lead that "
        "your doctor can check with further examinations. Please share this letter with your doctor.",
        "none": "The analysis found no unusual methylation signal in the DNA circulating in your blood. This "
        "does not rule out a health problem: a very early change can release too little DNA to be seen. "
        "Keep up the check-ups your doctor recommends.",
        "end": "Burotu is a demonstration prototype, not a medical device, and this is not a medical result.",
    },
    "fr": {
        "fail": "Votre échantillon de sang n'a pas pu être analysé de façon fiable : {issues}. Une nouvelle "
        "prise de sang est nécessaire ; aucun résultat ne peut être donné à partir de cet échantillon.",
        "signal": "L'analyse a trouvé un signal de méthylation inhabituel dans l'ADN qui circule dans votre "
        "sang. Ce signal ressemble le plus à celui du tissu suivant : {tissue}. Ce n'est pas un "
        "diagnostic : c'est une piste que votre médecin peut vérifier par des examens complémentaires. "
        "Transmettez ce courrier à votre médecin.",
        "none": "L'analyse n'a trouvé aucun signal de méthylation inhabituel dans l'ADN qui circule dans votre "
        "sang. Cela n'exclut pas un problème de santé : une modification très précoce peut libérer trop "
        "peu d'ADN pour être vue. Poursuivez les suivis recommandés par votre médecin.",
        "end": "Burotu est un prototype de démonstration, pas un dispositif médical, et ceci n'est pas un "
        "résultat médical.",
    },
    "es": {
        "fail": "Tu muestra de sangre no pudo analizarse de forma fiable: {issues}. Es necesaria una nueva "
        "extracción; no se puede dar ningún resultado con esta muestra.",
        "signal": "El análisis encontró una señal de metilación inusual en el ADN que circula en tu sangre. "
        "La señal se parece sobre todo a la del tejido de {tissue}. No es un diagnóstico: es una pista que "
        "tu médico puede comprobar con pruebas adicionales. Comparte esta carta con tu médico.",
        "none": "El análisis no encontró ninguna señal de metilación inusual en el ADN que circula en tu "
        "sangre. Esto no descarta un problema de salud: un cambio muy temprano puede liberar demasiado poco "
        "ADN para verse. Sigue con los controles que te recomiende tu médico.",
        "end": "Burotu es un prototipo de demostración, no un dispositivo médico, y esto no es un resultado médico.",
    },
}


def _facts(qc: dict, detection: dict | None, lang: str) -> dict:
    """What Muse Spark may say, already in the target language and without disease names."""
    facts: dict = {
        "quality_control": {"status": qc["status"], "findings": qc_findings(qc, lang)},
    }
    if detection is not None and qc["status"] != "fail":
        facts["unusual_signal_found"] = bool(detection["cancer_signal"])
        if detection["cancer_signal"]:
            facts["tissues_the_signal_resembles"] = [
                {"tissue": tissue(o["class"], lang), "probability": o["probability"]} for o in detection["origin"]
            ]
    return facts


def template_report(qc: dict, detection: dict | None, lang: str) -> str:
    t = TEMPLATES.get(lang, TEMPLATES["en"])
    if qc["status"] == "fail" or detection is None:
        body = t["fail"].format(issues="; ".join(qc_findings(qc, lang)) or "-")
    elif detection["cancer_signal"]:
        body = t["signal"].format(tissue=tissue(detection["origin"][0]["class"], lang))
    else:
        body = t["none"]
    return f"{body} {t['end']}"


def write_report(qc: dict, detection: dict | None, lang: str, client: MetaClient | None = None) -> dict:
    """Return {"text", "author"}; Muse Spark when a key is configured, a template otherwise."""
    client = client or MetaClient()
    if client.mock:
        return {"text": template_report(qc, detection, lang), "author": "template"}
    messages = [
        {"role": "system", "content": SYSTEM.format(language=LANG_NAMES.get(lang, "English"))},
        {"role": "user", "content": json.dumps(_facts(qc, detection, lang), ensure_ascii=False)},
    ]
    try:
        for _ in range(2):
            # Muse Spark reasons first; reasoning tokens count against max_tokens.
            text = client.chat(messages, temperature=0.3, max_tokens=6000).text.strip()
            if is_clean(text):
                return {"text": text, "author": client.model}
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": RETRY}]
        reason = "answer named a disease"
    except Exception as e:  # never block a result on the letter
        reason = f"Muse Spark unavailable: {type(e).__name__}"
    return {"text": template_report(qc, detection, lang), "author": f"template ({reason})"}


def write_reports(qc: dict, detection: dict | None, client: MetaClient | None = None) -> dict:
    """The letter in every language Burotu speaks, written in parallel."""
    with ThreadPoolExecutor(max_workers=len(LANGS)) as pool:
        futures = {lang: pool.submit(write_report, qc, detection, lang, client) for lang in LANGS}
        return {lang: f.result() for lang, f in futures.items()}


DOCTOR_SYSTEM = """You write the research report Burotu sends to the doctor or lab who requested the \
analysis of one blood sample. Burotu is a research prototype that looks for an unusual DNA methylation \
signal in cell-free DNA. You receive the quality checks and every model's answer as JSON.

Rules:
- Write for a physician, in {language}, in a precise, factual tone. 90 to 140 words. No headings, no lists.
- Report the quality control, whether the models found an unusual methylation signal (with the score and \
threshold of the model that decides the signal), and the tissues the signal most resembles with their \
probabilities, saying which models agree.
- Talk only about a "signal". Never write the words cancer, tumour, malignant, carcinoma, oncology or \
metastasis, in any language, and never name a disease or give a diagnosis or a treatment advice.
- Say that the result is a research signal to interpret in its clinical context, and that further \
examinations are the physician's decision.
- End with exactly this sentence, translated: "Research use only: Burotu is not a medical device."
- Do not invent numbers that are not in the JSON."""

DOCTOR_TEMPLATES = {
    "en": (
        "Quality control: {qc}. {signal} Research use only: Burotu is not a medical device.",
        "Unusual methylation signal (score {score} above threshold {threshold}). Closest tissues: {tissues}. "
        "Models agreeing on the first tissue: {agree}. A research signal to interpret in its clinical "
        "context; further examinations are the physician's decision.",
        "No unusual methylation signal (score {score}, threshold {threshold}).",
        "Sample not analysable: a new blood draw is needed.",
    ),
    "fr": (
        "Contrôle qualité : {qc}. {signal} Usage recherche uniquement : Burotu n'est pas un dispositif médical.",
        "Signal de méthylation inhabituel (score {score} au-dessus du seuil {threshold}). Tissus les plus "
        "proches : {tissues}. Modèles d'accord sur le premier tissu : {agree}. Signal de recherche à "
        "interpréter dans son contexte clinique ; les examens complémentaires relèvent du médecin.",
        "Pas de signal de méthylation inhabituel (score {score}, seuil {threshold}).",
        "Échantillon non analysable : une nouvelle prise de sang est nécessaire.",
    ),
    "es": (
        "Control de calidad: {qc}. {signal} Solo para investigación: Burotu no es un producto sanitario.",
        "Señal de metilación inusual (puntuación {score} por encima del umbral {threshold}). Tejidos más "
        "parecidos: {tissues}. Modelos de acuerdo en el primer tejido: {agree}. Señal de investigación que "
        "debe interpretarse en su contexto clínico; las pruebas adicionales las decide el médico.",
        "Sin señal de metilación inusual (puntuación {score}, umbral {threshold}).",
        "Muestra no analizable: hace falta una nueva extracción.",
    ),
}
QC_WORD = {
    "en": {"pass": "passed", "warn": "passed with warnings", "fail": "failed"},
    "fr": {"pass": "réussi", "warn": "réussi avec réserves", "fail": "échoué"},
    "es": {"pass": "superado", "warn": "superado con avisos", "fail": "no superado"},
}


def _doctor_facts(qc: dict, detection: dict | None, models: dict, lang: str) -> dict:
    facts = {
        "quality_control": {
            "status": qc["status"],
            "panel_coverage": round(qc.get("panel_coverage", 0), 3),
            "median_depth": qc.get("median_depth"),
            "findings": qc_findings(qc, lang),
        }
    }
    if detection is not None and qc["status"] != "fail":
        facts["deciding_model"] = {
            "unusual_signal_found": bool(detection["cancer_signal"]),
            "score": round(detection["score"], 4),
            "threshold": round(detection["threshold"], 4),
        }
        facts["models"] = {
            k: {
                "unusual_signal_found": bool(m["cancer_signal"]),
                "tissues": [
                    {"tissue": tissue(o["class"], lang), "probability": round(o["probability"], 3)}
                    for o in m["origin"][:3]
                ],
            }
            for k, m in models.items()
        }
    return facts


def doctor_template(qc: dict, detection: dict | None, models: dict, lang: str) -> str:
    head, sig, none, fail = DOCTOR_TEMPLATES.get(lang, DOCTOR_TEMPLATES["en"])
    q = QC_WORD.get(lang, QC_WORD["en"])[qc["status"]]
    if qc["status"] == "fail" or detection is None:
        return head.format(qc=q, signal=fail)
    if not detection["cancer_signal"]:
        return head.format(
            qc=q, signal=none.format(score=f"{detection['score']:.3g}", threshold=f"{detection['threshold']:.3g}")
        )
    first = detection["origin"][0]["class"]
    agree = [k for k, m in models.items() if m["origin"] and m["origin"][0]["class"] == first]
    tissues = ", ".join(f"{tissue(o['class'], lang)} {o['probability']:.0%}" for o in detection["origin"][:3])
    return head.format(
        qc=q,
        signal=sig.format(
            score=f"{detection['score']:.3g}",
            threshold=f"{detection['threshold']:.3g}",
            tissues=tissues,
            agree=f"{len(agree)}/{len(models)}",
        ),
    )


def write_doctor_report(
    qc: dict, detection: dict | None, models: dict, lang: str, client: MetaClient | None = None
) -> dict:
    """Research report for the requesting physician: {"text", "author"} (Muse Spark, template fallback)."""
    client = client or MetaClient()
    if client.mock:
        return {"text": doctor_template(qc, detection, models, lang), "author": "template"}
    messages = [
        {"role": "system", "content": DOCTOR_SYSTEM.format(language=LANG_NAMES.get(lang, "English"))},
        {"role": "user", "content": json.dumps(_doctor_facts(qc, detection, models, lang), ensure_ascii=False)},
    ]
    try:
        for _ in range(2):
            text = client.chat(messages, temperature=0.2, max_tokens=6000).text.strip()
            if is_clean(text):
                return {"text": text, "author": client.model}
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": RETRY}]
        reason = "answer named a disease"
    except Exception as e:
        reason = f"Muse Spark unavailable: {type(e).__name__}"
    return {"text": doctor_template(qc, detection, models, lang), "author": f"template ({reason})"}
