"""What Burotu runs, what it learned from, and where everything lives: the admin map page.

One source of truth for the tracking site's "Map" page (admin.burotu.com/map). Every text is
given in English, French and Spanish; numbers come from the run reports where they exist.
"""

from __future__ import annotations

# Status of a model or dataset in Burotu today.
LIVE, READY, RUNNING, EXPERIMENT, LATER = "live", "ready", "running", "experiment", "later"

STATUS = {
    LIVE: {"en": "Live", "fr": "En ligne", "es": "En línea"},
    READY: {"en": "Ready, going live", "fr": "Prêt, mise en ligne", "es": "Listo, en despliegue"},
    RUNNING: {"en": "Training", "fr": "En cours", "es": "En curso"},
    EXPERIMENT: {"en": "Experiment", "fr": "Expérience", "es": "Experimento"},
    LATER: {"en": "To do", "fr": "À faire", "es": "Por hacer"},
}

# The order a sample goes through, for the flow diagram.
FLOW = [
    ("qc", {"en": "Quality control", "fr": "Contrôle qualité", "es": "Control de calidad"}),
    (
        "models",
        {"en": "3 models read the sample", "fr": "3 modèles lisent l'échantillon", "es": "3 modelos leen la muestra"},
    ),
    ("consensus", {"en": "Consensus", "fr": "Consensus", "es": "Consenso"}),
    (
        "letter",
        {"en": "Letter + PDF (Muse Spark)", "fr": "Lettre + PDF (Muse Spark)", "es": "Carta + PDF (Muse Spark)"},
    ),
]

MODELS = [
    {
        "key": "classic",
        "name": "Classic model",
        "status": LIVE,
        "role": {"en": "Decides the signal", "fr": "Décide du signal", "es": "Decide la señal"},
        "type": {
            "en": "Gradient-boosted trees (LightGBM): one for the signal, one for the tissue",
            "fr": "Arbres de décision boostés (LightGBM) : un pour le signal, un pour le tissu",
            "es": "Árboles de decisión potenciados (LightGBM): uno para la señal, otro para el tejido",
        },
        "data": {
            "en": "Refined v2: 15,500 simulated plasma samples, 2,700 tumours from 22 public series (23 tissues) mixed into the blood of 3,683 people from 8 cohorts",
            "fr": "Données v2 : 15 500 plasmas simulés, 2 700 tumeurs de 22 séries publiques (23 tissus) mélangées au sang de 3 683 personnes de 8 cohortes",
            "es": "Datos v2: 15.500 plasmas simulados, 2.700 tumores de 22 series públicas (23 tejidos) mezclados con la sangre de 3.683 personas de 8 cohortes",
        },
        "where": {
            "en": "Weights in GitHub · runs on the tracking server (CPU)",
            "fr": "Poids sur GitHub · tourne sur le serveur du suivi (CPU)",
            "es": "Pesos en GitHub · se ejecuta en el servidor (CPU)",
        },
        "result": {
            "en": "Same v1 benchmark: 87% of signals at 1% tumour DNA (v1: 71%) · on the larger v2 benchmark: 80%, tissue 46% of 23, 99.0% of 1,745 healthy people not flagged · real plasma: 12/15 found, 0/12 healthy flagged",
            "fr": "Même banc v1 : 87 % des signaux à 1 % d'ADN tumoral (v1 : 71 %) · sur le banc v2, plus large : 80 %, tissu 46 % sur 23, 99,0 % de 1 745 sains non signalés · vrai plasma : 12/15 trouvés, 0/12 sain signalé",
            "es": "Mismo banco v1: 87 % de señales al 1 % de ADN tumoral (v1: 71 %) · en el banco v2, más amplio: 80 %, tejido 46 % de 23, 99,0 % de 1.745 sanos no señalados · plasma real: 12/15 hallados, 0/12 sanos señalados",
        },
        "link": "https://github.com/zoukmar/burotu/tree/main/src/burotu/analysis/assets/baseline",
    },
    {
        "key": "logreg",
        "name": "Logistic regressions",
        "status": LIVE,
        "role": {"en": "Cross-check", "fr": "Contrôle croisé", "es": "Control cruzado"},
        "type": {
            "en": "8 + 8 logistic regressions averaged, the method GRAIL published for Galleri (2020)",
            "fr": "8 + 8 régressions logistiques moyennées, la méthode publiée par GRAIL pour Galleri (2020)",
            "es": "8 + 8 regresiones logísticas promediadas, el método que GRAIL publicó para Galleri (2020)",
        },
        "data": {
            "en": "Same 15,500 simulated samples as the classic model (refined v2)",
            "fr": "Les mêmes 15 500 plasmas simulés que le modèle classique (données v2)",
            "es": "Las mismas 15.500 muestras simuladas que el modelo clásico (datos v2)",
        },
        "where": {
            "en": "Weights in GitHub · runs on the tracking server (CPU)",
            "fr": "Poids sur GitHub · tourne sur le serveur du suivi (CPU)",
            "es": "Pesos en GitHub · se ejecuta en el servidor (CPU)",
        },
        "result": {
            "en": "Same v1 benchmark, calibrated like v1: 70% of signals at 1% (v1: 57%) with 99.8% specificity (v1: 98.3%) · real plasma: 12/15 found, tissue 8 of 11",
            "fr": "Même banc v1, calibré comme en v1 : 70 % des signaux à 1 % (v1 : 57 %) avec 99,8 % de spécificité (v1 : 98,3 %) · vrai plasma : 12/15 trouvés, tissu 8 sur 11",
            "es": "Mismo banco v1, calibrado como en v1: 70 % de señales al 1 % (v1: 57 %) con 99,8 % de especificidad (v1: 98,3 %) · plasma real: 12/15 hallados, tejido 8 de 11",
        },
        "link": "https://github.com/zoukmar/burotu/blob/main/src/burotu/model/logreg.py",
    },
    {
        "key": "meta",
        "name": "Muse Glimmer v6 (Meta)",
        "status": LIVE,
        "role": {"en": "Names the tissue", "fr": "Nomme le tissu", "es": "Indica el tejido"},
        "type": {
            "en": "Meta's open-weight 30B language model, 4-bit, + our methylation adapter (3,148 sites → 16 tokens) + QLoRA (217M trained parameters)",
            "fr": "Le grand modèle de langage 30B à poids ouverts de Meta, en 4 bits, + notre adaptateur de méthylation (3 148 sites → 16 jetons) + QLoRA (217 M de paramètres entraînés)",
            "es": "El modelo de lenguaje de 30B con pesos abiertos de Meta, en 4 bits, + nuestro adaptador de metilación (3.148 sitios → 16 tokens) + QLoRA (217 M de parámetros entrenados)",
        },
        "data": {
            "en": "Refined v2: 35,000 simulated samples, 23 tissues, blood of 3,683 people from 8 cohorts, half the backgrounds mixed from 2-3 people; 66 min on one L40S GPU",
            "fr": "Données v2 : 35 000 plasmas simulés, 23 tissus, sang de 3 683 personnes de 8 cohortes, la moitié des fonds mélangés à partir de 2 ou 3 personnes ; 66 min sur une carte L40S",
            "es": "Datos v2: 35.000 muestras simuladas, 23 tejidos, sangre de 3.683 personas de 8 cohortes, la mitad de los fondos mezclados de 2 o 3 personas; 66 min en una GPU L40S",
        },
        "where": {
            "en": "Base model on Hugging Face (Meta) · our weights on Hugging Face (zoukmar75/burotu-glimmer) and Modal · runs on a Modal GPU",
            "fr": "Modèle de base sur Hugging Face (Meta) · nos poids sur Hugging Face (zoukmar75/burotu-glimmer) et Modal · tourne sur une carte graphique Modal",
            "es": "Modelo base en Hugging Face (Meta) · nuestros pesos en Hugging Face (zoukmar75/burotu-glimmer) y Modal · se ejecuta en una GPU de Modal",
        },
        "result": {
            "en": "v2 benchmark, 23 tissues: tissue right first time 58% at 1% (classic: 46%) · signal 75% at 1% with 99.5% specificity (v5: 38%) · 7 of 11 on real plasma",
            "fr": "Banc v2, 23 tissus : tissu juste du premier coup 58 % à 1 % (classique : 46 %) · signal 75 % à 1 % avec 99,5 % de spécificité (v5 : 38 %) · 7 sur 11 sur vrai plasma",
            "es": "Banco v2, 23 tejidos: tejido acertado a la primera 58 % al 1 % (clásico: 46 %) · señal 75 % al 1 % con 99,5 % de especificidad (v5: 38 %) · 7 de 11 en plasma real",
        },
        "link": "https://huggingface.co/zoukmar75/burotu-glimmer",
    },
    {
        "key": "cpgpt",
        "name": "CpGPT",
        "status": RUNNING,
        "role": {"en": "Cross-check", "fr": "Contrôle croisé", "es": "Control cruzado"},
        "type": {
            "en": "Methylation foundation model (transformer, ~100M parameters, Camillo et al. 2024), frozen, + our two heads",
            "fr": "Modèle de fondation de la méthylation (transformer, ~100 M de paramètres, Camillo et al. 2024), gelé, + nos deux têtes",
            "es": "Modelo fundacional de metilación (transformer, ~100 M de parámetros, Camillo et al. 2024), congelado, + nuestras dos cabezas",
        },
        "data": {
            "en": "Pre-trained by its authors on 150,000+ methylation arrays; our heads on the same 14,000 simulated samples",
            "fr": "Pré-entraîné par ses auteurs sur plus de 150 000 puces de méthylation ; nos têtes sur les mêmes 14 000 plasmas simulés",
            "es": "Preentrenado por sus autores con más de 150.000 arrays de metilación; nuestras cabezas con las mismas 14.000 muestras simuladas",
        },
        "where": {
            "en": "Weights on Hugging Face (lucascamillomd/cpgpt-models, MIT) · runs on a Modal GPU",
            "fr": "Poids sur Hugging Face (lucascamillomd/cpgpt-models, MIT) · tourne sur une carte graphique Modal",
            "es": "Pesos en Hugging Face (lucascamillomd/cpgpt-models, MIT) · se ejecuta en una GPU de Modal",
        },
        "result": {
            "en": "Tested, not kept: frozen, it finds 4% of signals at 1% (22% at 3%) and tissue near chance",
            "fr": "Testé, non retenu : gelé, il trouve 4 % des signaux à 1 % (22 % à 3 %) et le tissu presque au hasard",
            "es": "Probado, no adoptado: congelado, encuentra el 4 % de las señales al 1 % (22 % al 3 %) y el tejido casi al azar",
        },
        "link": "https://huggingface.co/lucascamillomd/cpgpt-models",
    },
    {
        "key": "consensus",
        "name": "Consensus",
        "status": LIVE,
        "role": {"en": "Fifth opinion", "fr": "Cinquième avis", "es": "Quinta opinión"},
        "type": {
            "en": "Vote: how many models see a signal, which tissue most of them name",
            "fr": "Vote : combien de modèles voient un signal, quel tissu la majorité nomme",
            "es": "Voto: cuántos modelos ven una señal, qué tejido nombra la mayoría",
        },
        "data": {
            "en": "None: it reads the other models",
            "fr": "Aucune : il lit les autres modèles",
            "es": "Ninguno: lee los demás modelos",
        },
        "where": {"en": "Code in GitHub", "fr": "Code sur GitHub", "es": "Código en GitHub"},
        "result": {
            "en": "Disagreement = the lab takes a second look",
            "fr": "Désaccord = le labo regarde à nouveau",
            "es": "Desacuerdo = el laboratorio lo revisa",
        },
        "link": "https://github.com/zoukmar/burotu/blob/main/src/burotu/analysis/consensus.py",
    },
    {
        "key": "spark",
        "name": "Muse Spark 1.3 (Meta)",
        "status": LIVE,
        "role": {"en": "Writes, never decides", "fr": "Écrit, ne décide jamais", "es": "Escribe, nunca decide"},
        "type": {
            "en": "Meta's language model through the Meta Model API: patient letters (EN/FR/ES), lab notes, answers to questions",
            "fr": "Le modèle de langage de Meta via la Meta Model API : lettres patient (EN/FR/ES), notes labo, réponses aux questions",
            "es": "El modelo de lenguaje de Meta vía la Meta Model API: cartas al paciente (EN/FR/ES), notas de laboratorio, respuestas",
        },
        "data": {
            "en": "Not trained by Burotu: only the facts of each result, with a guard against forbidden words",
            "fr": "Non entraîné par Burotu : seulement les faits de chaque résultat, avec un garde-fou sur les mots interdits",
            "es": "No entrenado por Burotu: solo los hechos de cada resultado, con un filtro de palabras prohibidas",
        },
        "where": {"en": "Meta Model API", "fr": "Meta Model API", "es": "Meta Model API"},
        "result": {
            "en": "Speaks of a signal, never of a disease",
            "fr": "Parle d'un signal, jamais d'une maladie",
            "es": "Habla de una señal, nunca de una enfermedad",
        },
        "link": "",
    },
]

DATASETS = [
    {
        "name": "Tumours: TCGA, CPTAC, TARGET, CCDI, HCMI, CGCI (GDC) + brain, sarcoma, blood cancer series (GEO)",
        "n": "3,832 tumours kept of 15,200 refined · 22 series · 23 tissues",
        "use": {
            "en": "Tumour DNA in the simulation",
            "fr": "ADN tumoral de la simulation",
            "es": "ADN tumoral de la simulación",
        },
        "where": {
            "en": "GDC + GEO (open) → workstation → refined v2 (GitHub Release)",
            "fr": "GDC + GEO (ouverts) → poste de travail → données v2 (Release GitHub)",
            "es": "GDC + GEO (abiertos) → estación de trabajo → datos v2 (Release de GitHub)",
        },
        "status": LIVE,
    },
    {
        "name": "Healthy blood: 8 cohorts (Hannum, GSE55763, GSE87571, GSE56105, GSE72680, GSE72775, GSE42861, GSE234190)",
        "n": "5,728 people: 3,683 training, 300 calibration, 1,745 test · each cohort aligned on the Hannum batch",
        "use": {
            "en": "Healthy background of the simulation",
            "fr": "Fond sain de la simulation",
            "es": "Fondo sano de la simulación",
        },
        "where": {"en": "GEO → GitHub", "fr": "GEO → GitHub", "es": "GEO → GitHub"},
        "status": LIVE,
    },
    {
        "name": "Simulated plasma (benchmark)",
        "n": "v2: 15,500 / 35,000 training · 3,000 calibration · 9,796 test (v1 benchmark kept for comparison)",
        "use": {
            "en": "Training and scoring every model on the same samples",
            "fr": "Entraîner et noter tous les modèles sur les mêmes échantillons",
            "es": "Entrenar y puntuar todos los modelos con las mismas muestras",
        },
        "where": {
            "en": "Rebuilt from a fixed seed (test and calibration lists in GitHub)",
            "fr": "Recréé à partir d'une graine fixe (listes test et calibration sur GitHub)",
            "es": "Regenerado con una semilla fija (listas de test y calibración en GitHub)",
        },
        "status": LIVE,
    },
    {
        "name": "Real plasma, Moss 2018 (GSE122126)",
        "n": "27 samples: 12 healthy, 15 with a signal",
        "use": {
            "en": "Check on real plasma, healthy controls",
            "fr": "Contrôle sur vrai plasma, témoins sains",
            "es": "Control en plasma real, controles sanos",
        },
        "where": {"en": "GEO → GitHub", "fr": "GEO → GitHub", "es": "GEO → GitHub"},
        "status": LIVE,
    },
    {
        "name": "Real plasma: lung (GSE277078), liver (GSE298812, GSE129374), prostate (GSE108462)",
        "n": "655 samples",
        "use": {
            "en": "Check every model on other labs' plasma",
            "fr": "Tester chaque modèle sur le plasma d'autres labos",
            "es": "Probar cada modelo con plasma de otros laboratorios",
        },
        "where": {
            "en": "GEO → GitHub (data/public)",
            "fr": "GEO → GitHub (data/public)",
            "es": "GEO → GitHub (data/public)",
        },
        "status": LIVE,
    },
    {
        "name": "Normal tissues, sorted cells, organoids (GEO, GDC)",
        "n": "2,100 normal tissues · 279 sorted cells · 584 organoids",
        "use": {
            "en": "Refined, not used for training yet",
            "fr": "Raffinés, pas encore utilisés pour l'entraînement",
            "es": "Refinados, aún no usados para entrenar",
        },
        "where": {
            "en": "GEO → GitHub (data/public)",
            "fr": "GEO → GitHub (data/public)",
            "es": "GEO → GitHub (data/public)",
        },
        "status": EXPERIMENT,
    },
    {
        "name": "Cell-type atlas, Loyfer 2023 (GSE186458)",
        "n": "253 samples · 39 cell types",
        "use": {
            "en": "Non-blood cells in healthy plasma (tested: kept as an option, off)",
            "fr": "Cellules non sanguines du plasma sain (testé : option, désactivée)",
            "es": "Células no sanguíneas del plasma sano (probado: opción, desactivada)",
        },
        "where": {
            "en": "GEO (352 GB) → refined v2 (253 samples)",
            "fr": "GEO (352 Go) → données v2 (253 échantillons)",
            "es": "GEO (352 GB) → datos v2 (253 muestras)",
        },
        "status": EXPERIMENT,
    },
]

PROGRESS = [
    (
        LIVE,
        {
            "en": "23 tissues on the site, admin and technology page (refined data v2)",
            "fr": "23 tissus sur le site, l'admin et la page Technologie (données v2)",
            "es": "23 tejidos en el sitio, la administración y la página de tecnología (datos v2)",
        },
    ),
    (
        LIVE,
        {
            "en": "Muse Glimmer v6 (refined v2, 23 tissues) served on Modal and published on Hugging Face",
            "fr": "Muse Glimmer v6 (données v2, 23 tissus) servi sur Modal et publié sur Hugging Face",
            "es": "Muse Glimmer v6 (datos v2, 23 tejidos) servido en Modal y publicado en Hugging Face",
        },
    ),
    (
        LIVE,
        {
            "en": '"Why this result" on every analysis',
            "fr": "« Pourquoi ce résultat » sur chaque analyse",
            "es": '"Por qué este resultado" en cada análisis',
        },
    ),
    (
        LIVE,
        {
            "en": "Logistic regressions + consensus on every analysis",
            "fr": "Régressions logistiques + consensus sur chaque analyse",
            "es": "Regresiones logísticas + consenso en cada análisis",
        },
    ),
    (
        LIVE,
        {
            "en": "Every model scored on 655 real plasma samples from GEO",
            "fr": "Chaque modèle testé sur 655 vrais plasmas de GEO",
            "es": "Cada modelo probado con 655 plasmas reales de GEO",
        },
    ),
    (
        RUNNING,
        {
            "en": "CpGPT on Modal, then live",
            "fr": "CpGPT sur Modal, puis mise en ligne",
            "es": "CpGPT en Modal, luego en línea",
        },
    ),
    (
        LATER,
        {
            "en": "Sequenced plasma sets (SPOT-MAS, Nassiri) converted to the panel",
            "fr": "Plasmas séquencés (SPOT-MAS, Nassiri) convertis vers le panel",
            "es": "Plasmas secuenciados (SPOT-MAS, Nassiri) convertidos al panel",
        },
    ),
    (
        LATER,
        {
            "en": "Real plasma from a partner lab (emails ready)",
            "fr": "Vrai plasma d'un labo partenaire (mails prêts)",
            "es": "Plasma real de un laboratorio socio (correos listos)",
        },
    ),
    (
        LATER,
        {
            "en": "3-minute video and X post (deadline 4 November)",
            "fr": "Vidéo de 3 minutes et post X (date limite 4 novembre)",
            "es": "Vídeo de 3 minutos y post en X (fecha límite 4 de noviembre)",
        },
    ),
]


REGISTRY_ID = {"meta": "glimmer"}  # inventory key -> registry id, where they differ
REGISTRY_STATUS = {"live": LIVE, "training": RUNNING, "ready": READY, "experiment": EXPERIMENT}


def models() -> list[dict]:
    """MODELS with status, version and card link taken from the model registry (single source)."""
    from burotu.models.registry import get

    out = []
    for m in MODELS:
        try:
            info = get(REGISTRY_ID.get(m["key"], m["key"]))
        except KeyError:
            out.append(m)
            continue
        out.append(
            {**m, "status": REGISTRY_STATUS.get(info.status, m["status"]), "version": info.version, "card": info.card}
        )
    return out
