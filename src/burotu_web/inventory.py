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
        {"en": "4 models read the sample", "fr": "4 modèles lisent l'échantillon", "es": "4 modelos leen la muestra"},
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
            "en": "14,000 simulated plasma samples: 680 TCGA tumours (20 tissues) mixed into the blood of 80 people",
            "fr": "14 000 plasmas simulés : 680 tumeurs TCGA (20 tissus) mélangées au sang de 80 personnes",
            "es": "14.000 plasmas simulados: 680 tumores TCGA (20 tejidos) mezclados con la sangre de 80 personas",
        },
        "where": {
            "en": "Weights in GitHub · runs on the tracking server (CPU)",
            "fr": "Poids sur GitHub · tourne sur le serveur du suivi (CPU)",
            "es": "Pesos en GitHub · se ejecuta en el servidor (CPU)",
        },
        "result": {
            "en": "71% of signals at 1% tumour DNA · tissue 55% · 99.9% of healthy people not flagged",
            "fr": "71 % des signaux à 1 % d'ADN tumoral · tissu 55 % · 99,9 % des sains non signalés",
            "es": "71 % de señales al 1 % de ADN tumoral · tejido 55 % · 99,9 % de sanos no señalados",
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
            "en": "Same 14,000 simulated samples as the classic model",
            "fr": "Les mêmes 14 000 plasmas simulés que le modèle classique",
            "es": "Las mismas 14.000 muestras simuladas que el modelo clásico",
        },
        "where": {
            "en": "Weights in GitHub · runs on the tracking server (CPU)",
            "fr": "Poids sur GitHub · tourne sur le serveur du suivi (CPU)",
            "es": "Pesos en GitHub · se ejecuta en el servidor (CPU)",
        },
        "result": {
            "en": "57% of signals at 1%, 97% at 3% · on real plasma: prostate recognised 163 times of 181",
            "fr": "57 % des signaux à 1 %, 97 % à 3 % · sur vrai plasma : prostate reconnue 163 fois sur 181",
            "es": "57 % de señales al 1 %, 97 % al 3 % · en plasma real: próstata reconocida 163 de 181",
        },
        "link": "https://github.com/zoukmar/burotu/blob/main/src/burotu/model/logreg.py",
    },
    {
        "key": "meta",
        "name": "Muse Glimmer v5 (Meta)",
        "status": LIVE,
        "role": {"en": "Names the tissue", "fr": "Nomme le tissu", "es": "Indica el tejido"},
        "type": {
            "en": "Meta's open-weight 30B language model, 4-bit, + our methylation adapter (3,148 sites → 16 tokens) + QLoRA (217M trained parameters)",
            "fr": "Le grand modèle de langage 30B à poids ouverts de Meta, en 4 bits, + notre adaptateur de méthylation (3 148 sites → 16 jetons) + QLoRA (217 M de paramètres entraînés)",
            "es": "El modelo de lenguaje de 30B con pesos abiertos de Meta, en 4 bits, + nuestro adaptador de metilación (3.148 sitios → 16 tokens) + QLoRA (217 M de parámetros entrenados)",
        },
        "data": {
            "en": "32,000 simulated samples, half the healthy backgrounds mixed from 2-3 people (pseudo-donors); 65 min on one L40S GPU",
            "fr": "32 000 plasmas simulés, la moitié des fonds sains mélangés à partir de 2 ou 3 personnes (pseudo-donneurs) ; 65 min sur une carte L40S",
            "es": "32.000 muestras simuladas, la mitad de los fondos sanos mezclados de 2 o 3 personas (pseudodonantes); 65 min en una GPU L40S",
        },
        "where": {
            "en": "Base model on Hugging Face (Meta) · our weights on Hugging Face (zoukmar75/burotu-glimmer) and Modal · runs on a Modal GPU",
            "fr": "Modèle de base sur Hugging Face (Meta) · nos poids sur Hugging Face (zoukmar75/burotu-glimmer) et Modal · tourne sur une carte graphique Modal",
            "es": "Modelo base en Hugging Face (Meta) · nuestros pesos en Hugging Face (zoukmar75/burotu-glimmer) y Modal · se ejecuta en una GPU de Modal",
        },
        "result": {
            "en": "Tissue right first time 60% at 1% (classic: 55%) · 7 of 11 on real plasma",
            "fr": "Tissu juste du premier coup 60 % à 1 % (classique : 55 %) · 7 sur 11 sur vrai plasma",
            "es": "Tejido acertado a la primera 60 % al 1 % (clásico: 55 %) · 7 de 11 en plasma real",
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
            "en": "Reading the 26,000 benchmark samples on Modal",
            "fr": "Lecture des 26 000 échantillons du banc d'essai sur Modal",
            "es": "Leyendo las 26.000 muestras del banco de pruebas en Modal",
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
        "name": "TCGA tumours (450K arrays)",
        "n": "960 tumours · 28 projects · 20 tissues",
        "use": {
            "en": "Tumour DNA in the simulation",
            "fr": "ADN tumoral de la simulation",
            "es": "ADN tumoral de la simulación",
        },
        "where": {
            "en": "GDC (open) → panel cut in GitHub",
            "fr": "GDC (ouvert) → extrait du panel sur GitHub",
            "es": "GDC (abierto) → extracto del panel en GitHub",
        },
        "status": LIVE,
    },
    {
        "name": "Healthy blood, Hannum (GSE40279)",
        "n": "164 people: 80 training, 35 calibration, 49 test",
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
        "n": "14,000 / 32,000 training · 3,000 calibration · 9,040 test",
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
        "name": "Healthy blood: GSE55763, GSE87571",
        "n": "3,443 people",
        "use": {
            "en": "More healthy donors for training (tested, not kept yet)",
            "fr": "Plus de donneurs sains pour l'entraînement (testé, pas encore retenu)",
            "es": "Más donantes sanos para entrenar (probado, aún no adoptado)",
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
            "en": "Non-blood cells in the simulation",
            "fr": "Cellules non sanguines dans la simulation",
            "es": "Células no sanguíneas en la simulación",
        },
        "where": {
            "en": "GEO (14 GB, to convert)",
            "fr": "GEO (14 Go, à convertir)",
            "es": "GEO (14 GB, por convertir)",
        },
        "status": LATER,
    },
]

PROGRESS = [
    (
        LIVE,
        {
            "en": "20 tissues on the site, admin and technology page",
            "fr": "20 tissus sur le site, l'admin et la page Technologie",
            "es": "20 tejidos en el sitio, la administración y la página de tecnología",
        },
    ),
    (
        LIVE,
        {
            "en": "Muse Glimmer v5 served on Modal and published on Hugging Face",
            "fr": "Muse Glimmer v5 servi sur Modal et publié sur Hugging Face",
            "es": "Muse Glimmer v5 servido en Modal y publicado en Hugging Face",
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
            "en": "Loyfer cell-type atlas, then the other open datasets",
            "fr": "Atlas Loyfer, puis les autres données ouvertes",
            "es": "Atlas Loyfer y luego los demás datos abiertos",
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
