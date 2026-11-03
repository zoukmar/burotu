"""Public live demo on burotu.com/{lang}/demo: anyone can watch the models read a lab file.

Ten demo lab files ship with the site (simulated plasma built from held-out tumours and blood
donors). Picking one runs the full analysis live: quality control, the classic models, Muse
Glimmer 30B on its Modal GPU, the consensus, "why this result" and the letter Muse Spark writes.
Results are cached a few minutes per sample, and each visitor is rate-limited, so the GPU and
the Meta Model API are only called when needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from burotu_web.i18n import LANGS, T
from burotu_web.templating import templates

router = APIRouter()
DEMO_DIR = Path(__file__).resolve().parent / "demo_samples"
# Shown in this order. 02 and 06 are left out: the models misname their tissue (kept in the admin).
SHOWN = ["BRT-DEMO-05", "BRT-DEMO-03", "BRT-DEMO-07", "BRT-DEMO-09", "BRT-DEMO-10", "BRT-DEMO-01", "BRT-DEMO-08"]
CACHE_SECONDS = 600
RUNS_PER_HOUR = 30
_cache: dict[str, tuple[float, dict]] = {}
_hits: dict[str, list[float]] = {}

TEXT = {
    "en": {
        "nav": "Live demo",
        "title": "Watch Burotu read a blood test",
        "lead": "Pick a lab file. Five models read it live, including Muse Glimmer 30B, Meta's open-weight "
        "model we fine-tuned to read DNA methylation, running on a GPU right now. The true answer is revealed "
        "at the end.",
        "pick": "Pick a lab file",
        "sample": "Sample",
        "hidden": "true answer hidden",
        "run": "Read it",
        "running": "Reading…",
        "qc": "Quality control",
        "qc_pass": "Passed",
        "qc_warn": "Passed with warnings",
        "qc_fail": "Refused: a new blood draw is needed",
        "coverage": "panel sites read",
        "depth": "reads per site (median)",
        "models": "Every model on this sample",
        "signal": "Signal",
        "no_signal": "No signal",
        "tissue": "Tissue it resembles",
        "classic": "Classic model",
        "classic_d": "gradient boosting · raises the alarm",
        "logreg": "Logistic regressions",
        "logreg_d": "GRAIL-style ensembles · cross-check",
        "meta": "Muse Glimmer 30B (Meta)",
        "meta_d": "open weights, fine-tuned by Burotu · names the tissue · Modal GPU",
        "meta_off": "Waking up the GPU or not served right now: the classic model answers alone.",
        "consensus": "Consensus",
        "consensus_d": "what the models say together",
        "of": "of",
        "why": "Why this result",
        "rank": "Rank among healthy people",
        "rank_d": "of simulated healthy samples score lower. A signal is raised above 99.5%.",
        "lit": "Lit sites",
        "lit_d": "sites carry more methylated DNA than 99.5% of healthy samples. A healthy sample shows up to",
        "letter": "Letter for the patient",
        "by_spark": "Written by Muse Spark (Meta Model API). Never a diagnosis: it goes through the doctor.",
        "by_template": "Written from a fixed template (Muse Spark not reachable).",
        "truth": "The true answer",
        "truth_healthy": "Healthy donor: no tumour DNA in this sample.",
        "truth_tumour": "Built from {tissue} tumour DNA at {tf} of the cell-free DNA, mixed into the blood of a donor "
        "no model saw in training.",
        "truth_shallow": "Same, but sequenced far too shallowly ({depth} reads per site): Burotu refuses to guess.",
        "time": "Read in {s} s",
        "cached": "result from {m} min ago (cached to spare the GPU)",
        "limit": "Too many runs from this connection. Try again in a few minutes.",
        "error": "Something went wrong. Try again.",
        "footnote": "Research prototype on simulated plasma, not a medical device. Lab files are simulated: no "
        "real patient data. The numbers behind each model are on the Technology page.",
    },
    "fr": {
        "nav": "Démo en direct",
        "title": "Regardez Burotu lire une prise de sang",
        "lead": "Choisissez un fichier de labo. Cinq modèles le lisent en direct, dont Muse Glimmer 30B, le modèle "
        "à poids ouverts de Meta que nous avons entraîné à lire la méthylation de l'ADN, qui tourne en ce moment "
        "sur une carte graphique. La vraie réponse est révélée à la fin.",
        "pick": "Choisissez un fichier de labo",
        "sample": "Échantillon",
        "hidden": "vraie réponse cachée",
        "run": "Le lire",
        "running": "Lecture…",
        "qc": "Contrôle qualité",
        "qc_pass": "Réussi",
        "qc_warn": "Réussi avec réserves",
        "qc_fail": "Refusé : une nouvelle prise de sang est nécessaire",
        "coverage": "des sites du panel lus",
        "depth": "lectures par site (médiane)",
        "models": "Chaque modèle sur cet échantillon",
        "signal": "Signal",
        "no_signal": "Pas de signal",
        "tissue": "Tissu le plus proche",
        "classic": "Modèle classique",
        "classic_d": "arbres boostés · donne l'alarme",
        "logreg": "Régressions logistiques",
        "logreg_d": "façon GRAIL · contrôle croisé",
        "meta": "Muse Glimmer 30B (Meta)",
        "meta_d": "poids ouverts, entraîné par Burotu · nomme le tissu · GPU Modal",
        "meta_off": "Carte graphique en cours de réveil ou non servie : le modèle classique répond seul.",
        "consensus": "Consensus",
        "consensus_d": "ce que disent les modèles ensemble",
        "of": "sur",
        "why": "Pourquoi ce résultat",
        "rank": "Rang parmi les personnes saines",
        "rank_d": "des échantillons sains simulés ont un score plus bas. Un signal est levé au-delà de 99,5 %.",
        "lit": "Sites allumés",
        "lit_d": "sites portent plus d'ADN méthylé que 99,5 % des échantillons sains. Un échantillon sain en montre "
        "jusqu'à",
        "letter": "Lettre pour le patient",
        "by_spark": "Écrite par Muse Spark (Meta Model API). Jamais un diagnostic : elle passe par le médecin.",
        "by_template": "Écrite à partir d'un modèle fixe (Muse Spark injoignable).",
        "truth": "La vraie réponse",
        "truth_healthy": "Donneur sain : aucun ADN tumoral dans cet échantillon.",
        "truth_tumour": "Construit à partir d'ADN tumoral de {tissue}, à {tf} de l'ADN circulant, mélangé au sang "
        "d'un donneur qu'aucun modèle n'a vu à l'entraînement.",
        "truth_shallow": "Idem, mais séquencé bien trop peu ({depth} lectures par site) : Burotu refuse de deviner.",
        "time": "Lu en {s} s",
        "cached": "résultat d'il y a {m} min (mis en cache pour ménager le GPU)",
        "limit": "Trop de lectures depuis cette connexion. Réessayez dans quelques minutes.",
        "error": "Un problème est survenu. Réessayez.",
        "footnote": "Prototype de recherche sur plasma simulé, pas un dispositif médical. Les fichiers de labo sont "
        "simulés : aucune donnée de patient réel. Les chiffres de chaque modèle sont sur la page Technologie.",
    },
    "es": {
        "nav": "Demo en vivo",
        "title": "Mire a Burotu leer un análisis de sangre",
        "lead": "Elija un archivo de laboratorio. Cinco modelos lo leen en vivo, entre ellos Muse Glimmer 30B, el "
        "modelo de pesos abiertos de Meta que entrenamos para leer la metilación del ADN, funcionando ahora mismo "
        "en una GPU. La respuesta real se revela al final.",
        "pick": "Elija un archivo de laboratorio",
        "sample": "Muestra",
        "hidden": "respuesta real oculta",
        "run": "Leerlo",
        "running": "Leyendo…",
        "qc": "Control de calidad",
        "qc_pass": "Superado",
        "qc_warn": "Superado con avisos",
        "qc_fail": "Rechazado: hace falta una nueva extracción",
        "coverage": "de los sitios del panel leídos",
        "depth": "lecturas por sitio (mediana)",
        "models": "Cada modelo con esta muestra",
        "signal": "Señal",
        "no_signal": "Sin señal",
        "tissue": "Tejido más parecido",
        "classic": "Modelo clásico",
        "classic_d": "árboles potenciados · da la alarma",
        "logreg": "Regresiones logísticas",
        "logreg_d": "al estilo GRAIL · control cruzado",
        "meta": "Muse Glimmer 30B (Meta)",
        "meta_d": "pesos abiertos, entrenado por Burotu · indica el tejido · GPU de Modal",
        "meta_off": "La GPU se está despertando o no está disponible: el modelo clásico responde solo.",
        "consensus": "Consenso",
        "consensus_d": "lo que dicen los modelos juntos",
        "of": "de",
        "why": "Por qué este resultado",
        "rank": "Posición entre personas sanas",
        "rank_d": "de las muestras sanas simuladas puntúan menos. Se da una señal por encima del 99,5 %.",
        "lit": "Sitios encendidos",
        "lit_d": "sitios llevan más ADN metilado que el 99,5 % de las muestras sanas. Una muestra sana muestra hasta",
        "letter": "Carta para el paciente",
        "by_spark": "Escrita por Muse Spark (Meta Model API). Nunca un diagnóstico: pasa por el médico.",
        "by_template": "Escrita a partir de una plantilla fija (Muse Spark no disponible).",
        "truth": "La respuesta real",
        "truth_healthy": "Donante sano: no hay ADN tumoral en esta muestra.",
        "truth_tumour": "Construida con ADN tumoral de {tissue}, al {tf} del ADN circulante, mezclado con la sangre "
        "de un donante que ningún modelo vio al entrenar.",
        "truth_shallow": "Igual, pero secuenciada con muy poca profundidad ({depth} lecturas por sitio): Burotu se "
        "niega a adivinar.",
        "time": "Leído en {s} s",
        "cached": "resultado de hace {m} min (en caché para ahorrar la GPU)",
        "limit": "Demasiadas lecturas desde esta conexión. Inténtelo de nuevo en unos minutos.",
        "error": "Algo ha fallado. Inténtelo de nuevo.",
        "footnote": "Prototipo de investigación con plasma simulado, no un producto sanitario. Los archivos de "
        "laboratorio son simulados: ningún dato de pacientes reales. Las cifras de cada modelo están en la página "
        "Tecnología.",
    },
}


def manifest() -> dict[str, dict]:
    rows = json.loads((DEMO_DIR / "manifest.json").read_text())
    return {r["sample_id"]: r for r in rows}


def _tissue(cls: str, lang: str) -> str:
    from burotu.analysis.wording import tissue

    return tissue(cls, lang)


@router.get("/{lang}/demo", response_class=HTMLResponse)
def demo_page(request: Request, lang: str):
    if lang not in LANGS:
        raise HTTPException(404)
    samples = [{"id": s, "n": s[-2:]} for s in SHOWN if s in manifest()]
    return templates.TemplateResponse(
        request,
        "public/demo.html",
        {
            "lang": lang,
            "t": T[lang],
            "d": TEXT[lang],
            "samples": samples,
            "page": "demo",
            "tech_nav": {"en": "Technology", "fr": "Technologie", "es": "Tecnología"}[lang],
            "text_json": json.dumps(TEXT[lang], ensure_ascii=False),
        },
    )


def _allowed(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _hits.get(ip, []) if now - t < 3600]
    _hits[ip] = hits
    if len(hits) >= RUNS_PER_HOUR:
        return False
    hits.append(now)
    return True


def _payload(sample_id: str, result: dict, elapsed: float) -> dict:
    """What the page shows: only results, never anything about the server."""
    models = {
        k: {
            "signal": bool(m["cancer_signal"]),
            "score": m["score"],
            "threshold": m["threshold"],
            "origin": m["origin"][:3],
        }
        for k, m in (result.get("models") or {}).items()
        if k in ("classic", "logreg", "meta")
    }
    x = result.get("explain")
    explain = None
    if x:
        explain = {
            "rank": x["healthy_rank"],
            "lit": x["lit"],
            "sites": x["sites"],
            "healthy_p99": x["healthy_lit"]["p99"],
            "tissues": sorted(x["tissues"], key=lambda t: -t["probability"])[:5],
        }
    q = result["quality"]
    return {
        "sample_id": sample_id,
        "quality": {"status": q["status"], "coverage": q["panel_coverage"], "depth": q["median_depth"]},
        "models": models,
        "consensus": result.get("consensus"),
        "explain": explain,
        "letters": {
            lang: {"text": r["text"], "spark": r["author"].lower().startswith("muse")}
            for lang, r in result["reports"].items()
        },
        "elapsed": round(elapsed, 1),
        "at": time.time(),
    }


@router.post("/api/demo/{sample_id}")
async def demo_run(request: Request, sample_id: str, lang: str = "en"):
    if sample_id not in SHOWN or lang not in LANGS:
        raise HTTPException(404)
    cached = _cache.get(sample_id)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        data = {**cached[1], "cached_min": int((time.time() - cached[0]) // 60)}
    else:
        ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
        if not _allowed(ip):
            return JSONResponse({"error": "limit"}, status_code=429)
        from burotu.analysis.service import analyse
        from burotu.core.labformat import read_lab_file

        sample = read_lab_file(DEMO_DIR / f"{sample_id}.tsv")
        t0 = time.time()
        result = await run_in_threadpool(analyse, sample, lang)
        data = _payload(sample_id, result, time.time() - t0)
        _cache[sample_id] = (time.time(), data)
    truth = manifest()[sample_id]
    classes = {o["class"] for m in data["models"].values() for o in m["origin"]}
    if data.get("consensus") and data["consensus"].get("tissue"):
        classes.add(data["consensus"]["tissue"])
    if data.get("explain"):
        classes |= {t["class"] for t in data["explain"]["tissues"]}
    if truth["label"] != "Healthy":
        classes.add(truth["label"])
    return {
        **data,
        "tissues": {c: _tissue(c, lang) for c in classes},
        "truth": {"label": truth["label"], "tf": truth["tumor_fraction"], "depth": truth["depth"]},
    }
