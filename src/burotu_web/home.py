# ruff: noqa: E501  (long text strings)
"""Home page of burotu.com: the R&D story of a language model that learned to read DNA.

One page, in en/fr/es: what we built, the six fine-tuning runs of the hackathon week, the live
demo (the same widget as before, now on the home page), the results, the engine it became, and
where everything lives (video, code, weights). Numbers come from the run reports in `reports/`.
"""

from __future__ import annotations

VIDEO_URL = "https://youtu.be/LQVoz8USAHo"
GITHUB_URL = "https://github.com/zoukmar/burotu"
HF_URL = "https://huggingface.co/zoukmar75/burotu-glimmer"
PAPER_URL = "/paper"

# (day of the hackathon week, ISO date, run). Dates are the day each run was trained and reported.
RUNS = [
    (2, "2026-10-27", 1),
    (3, "2026-10-28", 2),
    (3, "2026-10-28", 3),
    (4, "2026-10-29", 4),
    (4, "2026-10-29", 5),
    (8, "2026-11-02", 6),
]

HOME = {
    "en": {
        "description": "R&D: in one week, we taught Meta's open-weight Muse Glimmer 30B to read DNA methylation from a blood sample. Live demo, video, code and weights.",
        "nav_how": "How it works",
        "nav_runs": "Six runs",
        "nav_demo": "Live demo",
        "nav_results": "Results",
        "nav_engine": "The engine",
        "nav_video": "Watch the video",
        "kicker": "R&D · Meta Global AI Developer Hackathon · first commit Oct 26 → v6 on Nov 2",
        "title": "We taught a language model to read DNA.",
        "lead": "Meta's Muse Glimmer 30B learned from text and images. It had never seen DNA. In one week, we gave it a new sense: it now reads DNA methylation from a blood sample, says whether there is an unusual signal, and names which of 23 tissues it comes from.",
        "btn_demo": "Try the live demo",
        "btn_video": "Watch the 3-min video",
        "links": "Open source",
        "paper": "Paper (PDF)",
        "video_alt": "Video: We taught Meta's LLM to read DNA in one week",
        "video_caption": "3 minutes: the idea, the six runs, the live demo and the engine.",
        "how_title": "A new sense for a language model",
        "how_lead": "Vision models read images through a small projector that turns pixels into tokens. We built the same thing for DNA.",
        "how": [
            (
                "3,148 methylation sites",
                "From cell-free DNA in a blood sample: at each site, the share of DNA molecules that carry a methyl mark. Tumours leave a different pattern from healthy blood cells.",
            ),
            (
                "A methylation adapter",
                "A small network turns those 3,148 values into 16 soft tokens in Glimmer's embedding space, trained first on its own, then with the language layers.",
            ),
            (
                "Muse Glimmer answers",
                "Fine-tuned with QLoRA (Meta's recipe, 4-bit), Glimmer reads the tokens and answers: is there a signal, and which tissue does it look like. The answer is read from its own logits.",
            ),
        ],
        "facts": [
            ("217M", "trained parameters"),
            ("66 min", "on one L40S GPU, on Modal"),
            ("35,000", "simulated plasma samples"),
            ("23", "tissues it can name"),
        ],
        "runs_title": "Six fine-tuning runs. One week.",
        "runs_lead": "It did not work the first time. Each run has its report in the repository, failures included. Scores below: on people never seen in training, at 1% tumour DNA.",
        "day": "Day",
        "runs": [
            (
                "Adapter and LoRA together",
                "LoRA learned the answer format faster than the adapter learned the sample.",
                "—",
                "Collapse: the same answer for everyone",
            ),
            (
                "Two stages",
                "The adapter alone first, then LoRA. It starts to read. 8 tissues.",
                "tissue 70% · classic ML 57%",
                "It reads, but memorises donors",
            ),
            (
                "A smaller adapter",
                "Regularised so it stops memorising each blood donor.",
                "signal 26% · specificity 99.9%",
                "Safe, but misses signals",
            ),
            (
                "Pseudo-donors",
                "Healthy backgrounds mixed from 2 to 3 real people, so no donor can be learned by heart.",
                "signal 42% · specificity 99.7%",
                "A better alarm",
            ),
            (
                "20 tissues",
                "Beats classic machine learning on the tissue of origin. Then more data made it worse.",
                "tissue 60% · classic ML 55%",
                "Fails on new people",
            ),
            (
                "5,728 people, 8 labs aligned",
                "Every lab batch moved onto one reference. 23 tissues, 35,000 training samples.",
                "tissue 58% of 23 · signal 75% at 99.5% specificity",
                "It reads, and it holds",
            ),
        ],
        "results_title": "What it does today",
        "results_lead": "Benchmark v2: 1,745 people and 1,132 tumours that no model saw in training, simulated plasma at 1% tumour DNA.",
        "results": [
            ("58%", "right tissue first time, among 23", "Muse Glimmer v6. Gradient boosting: 46%. Chance: 4%."),
            ("75%", "of signals found", "at 99.5% specificity: 5 healthy people in 1,000 flagged."),
            (
                "12 / 15",
                "real patient plasma samples found",
                "classic model, 0 of 12 healthy flagged (Moss 2018). Glimmer names 7 of 11 tissues.",
            ),
            ("26,155", "public samples refined", "3,832 tumours from 22 series, 5,728 blood donors from 8 cohorts."),
        ],
        "results_more": "Every run, every number: the Technology page",
        "engine_title": "Now it's a DNA-reading engine",
        "engine_lead": "Open weights, online on a Modal GPU, one API call. The language model stays the same: retrain the adapter on new labels and it answers a new question.",
        "uses": [
            ("Cancer signal and tissue", "What it does today, in this demo."),
            ("Treatment follow-up", "Tumour DNA going down, or up, between two draws."),
            ("Relapse watch", "A signal coming back after treatment."),
            ("Organ damage", "Which tissue is releasing DNA into the blood."),
            ("Biological age", "Epigenetic clocks from the same draw."),
            ("Your lab's question", "New labels, about an hour of GPU."),
        ],
        "uses_note": "Only the first is trained today. The others are research directions the same engine is built for.",
        "open_title": "Everything is open",
        "open": [
            ("Video", "3 minutes, the whole story", "youtu.be/LQVoz8USAHo"),
            ("Code", "Pipeline, training, serving, this site · AGPL-3.0", "github.com/zoukmar/burotu"),
            ("Weights", "Muse Glimmer + adapter, v1 to v6", "huggingface.co/zoukmar75/burotu-glimmer"),
            ("Paper", "Preprint, 5 pages: method, six runs, limits", "burotu.com/paper"),
        ],
        "built_with": "Built with Meta Muse Glimmer 30B (open weights) and Muse Spark (Meta Model API), trained and served on Modal.",
        "limits_title": "What it is not",
        "limits": [
            "A research prototype, trained on simulated plasma and checked on 27 real samples.",
            "Not a medical device and never a diagnosis: the report is written for a doctor, as a research signal.",
            "No test is sold or performed.",
        ],
    },
    "fr": {
        "description": "R&D : en une semaine, nous avons appris à Muse Glimmer 30B, le modèle à poids ouverts de Meta, à lire la méthylation de l'ADN dans une prise de sang. Démo en direct, vidéo, code et poids.",
        "nav_how": "Fonctionnement",
        "nav_runs": "Six versions",
        "nav_demo": "Démo en direct",
        "nav_results": "Résultats",
        "nav_engine": "Le moteur",
        "nav_video": "Voir la vidéo",
        "kicker": "R&D · Meta Global AI Developer Hackathon · premier commit le 26 oct. → v6 le 2 nov.",
        "title": "Nous avons appris à un modèle de langage à lire l'ADN.",
        "lead": "Muse Glimmer 30B, le modèle de Meta, a appris sur du texte et des images. Il n'avait jamais vu d'ADN. En une semaine, nous lui avons donné un nouveau sens : il lit la méthylation de l'ADN d'une prise de sang, dit s'il y a un signal inhabituel et nomme le tissu d'origine parmi 23.",
        "btn_demo": "Essayer la démo en direct",
        "btn_video": "Voir la vidéo (3 min)",
        "links": "Open source",
        "paper": "Article (PDF)",
        "video_alt": "Vidéo : nous avons appris à lire l'ADN au LLM de Meta en une semaine",
        "video_caption": "3 minutes : l'idée, les six versions, la démo en direct et le moteur.",
        "how_title": "Un nouveau sens pour un modèle de langage",
        "how_lead": "Les modèles de vision lisent les images grâce à un petit projecteur qui transforme les pixels en jetons. Nous avons construit la même chose pour l'ADN.",
        "how": [
            (
                "3 148 sites de méthylation",
                "Mesurés sur l'ADN circulant d'une prise de sang : pour chaque site, la part des molécules d'ADN qui portent une marque de méthylation. Les tumeurs laissent un motif différent de celui des cellules sanguines saines.",
            ),
            (
                "Un adaptateur de méthylation",
                "Un petit réseau transforme ces 3 148 valeurs en 16 jetons dans l'espace de Glimmer, entraîné d'abord seul, puis avec les couches de langage.",
            ),
            (
                "Muse Glimmer répond",
                "Affiné en QLoRA (la recette de Meta, en 4 bits), Glimmer lit ces jetons et répond : y a-t-il un signal, et à quel tissu ressemble-t-il. La réponse est lue dans ses propres logits.",
            ),
        ],
        "facts": [
            ("217 M", "paramètres entraînés"),
            ("66 min", "sur un GPU L40S, chez Modal"),
            ("35 000", "échantillons de plasma simulés"),
            ("23", "tissus qu'il sait nommer"),
        ],
        "runs_title": "Six entraînements. Une semaine.",
        "runs_lead": "Ça n'a pas marché du premier coup. Chaque version a son rapport dans le dépôt, échecs compris. Scores ci-dessous : sur des personnes jamais vues à l'entraînement, à 1 % d'ADN tumoral.",
        "day": "Jour",
        "runs": [
            (
                "Adaptateur et LoRA ensemble",
                "LoRA a appris le format de réponse plus vite que l'adaptateur n'a appris l'échantillon.",
                "—",
                "Effondrement : la même réponse pour tous",
            ),
            (
                "Deux étapes",
                "L'adaptateur seul d'abord, puis LoRA. Il commence à lire. 8 tissus.",
                "tissu 70 % · ML classique 57 %",
                "Il lit, mais mémorise les donneurs",
            ),
            (
                "Un adaptateur plus petit",
                "Régularisé pour ne plus mémoriser chaque donneur de sang.",
                "signal 26 % · spécificité 99,9 %",
                "Prudent, mais rate des signaux",
            ),
            (
                "Pseudo-donneurs",
                "Des fonds sains mélangés à partir de 2 ou 3 vraies personnes, pour qu'aucun donneur ne soit appris par cœur.",
                "signal 42 % · spécificité 99,7 %",
                "Une meilleure alarme",
            ),
            (
                "20 tissus",
                "Bat le machine learning classique sur le tissu d'origine. Puis plus de données l'ont dégradé.",
                "tissu 60 % · ML classique 55 %",
                "Échoue sur de nouvelles personnes",
            ),
            (
                "5 728 personnes, 8 labos alignés",
                "Chaque lot de labo ramené sur une même référence. 23 tissus, 35 000 échantillons d'entraînement.",
                "tissu 58 % sur 23 · signal 75 % à 99,5 % de spécificité",
                "Il lit, et ça tient",
            ),
        ],
        "results_title": "Ce qu'il fait aujourd'hui",
        "results_lead": "Banc v2 : 1 745 personnes et 1 132 tumeurs qu'aucun modèle n'a vues à l'entraînement, plasma simulé à 1 % d'ADN tumoral.",
        "results": [
            ("58 %", "bon tissu du premier coup, parmi 23", "Muse Glimmer v6. Gradient boosting : 46 %. Hasard : 4 %."),
            ("75 %", "des signaux trouvés", "à 99,5 % de spécificité : 5 personnes saines sur 1 000 signalées."),
            (
                "12 / 15",
                "vrais plasmas de patients trouvés",
                "modèle classique, 0 sain sur 12 signalé (Moss 2018). Glimmer nomme 7 tissus sur 11.",
            ),
            (
                "26 155",
                "échantillons publics raffinés",
                "3 832 tumeurs de 22 séries, 5 728 donneurs de sang de 8 cohortes.",
            ),
        ],
        "results_more": "Chaque version, chaque chiffre : la page Technologie",
        "engine_title": "C'est maintenant un moteur de lecture de l'ADN",
        "engine_lead": "Poids ouverts, en ligne sur un GPU Modal, un appel d'API. Le modèle de langage reste le même : on réentraîne l'adaptateur sur de nouvelles étiquettes et il répond à une nouvelle question.",
        "uses": [
            ("Signal de cancer et tissu", "Ce qu'il fait aujourd'hui, dans cette démo."),
            ("Suivi de traitement", "L'ADN tumoral qui baisse, ou remonte, entre deux prises de sang."),
            ("Surveillance de rechute", "Un signal qui revient après un traitement."),
            ("Lésion d'organe", "Quel tissu libère de l'ADN dans le sang."),
            ("Âge biologique", "Des horloges épigénétiques à partir de la même prise de sang."),
            ("La question de votre labo", "De nouvelles étiquettes, environ une heure de GPU."),
        ],
        "uses_note": "Seul le premier est entraîné aujourd'hui. Les autres sont des pistes de recherche pour lesquelles le même moteur est conçu.",
        "open_title": "Tout est ouvert",
        "open": [
            ("Vidéo", "3 minutes, toute l'histoire", "youtu.be/LQVoz8USAHo"),
            ("Code", "Pipeline, entraînement, service, ce site · AGPL-3.0", "github.com/zoukmar/burotu"),
            ("Poids", "Muse Glimmer + adaptateur, v1 à v6", "huggingface.co/zoukmar75/burotu-glimmer"),
            ("Article", "Prépublication, 5 pages : méthode, six versions, limites", "burotu.com/paper"),
        ],
        "built_with": "Construit avec Meta Muse Glimmer 30B (poids ouverts) et Muse Spark (Meta Model API), entraîné et servi sur Modal.",
        "limits_title": "Ce que ce n'est pas",
        "limits": [
            "Un prototype de recherche, entraîné sur du plasma simulé et vérifié sur 27 vrais échantillons.",
            "Pas un dispositif médical, jamais un diagnostic : le compte rendu est écrit pour un médecin, comme un signal de recherche.",
            "Aucun test n'est vendu ni réalisé.",
        ],
    },
    "es": {
        "description": "I+D: en una semana enseñamos a Muse Glimmer 30B, el modelo de pesos abiertos de Meta, a leer la metilación del ADN en una muestra de sangre. Demo en vivo, vídeo, código y pesos.",
        "nav_how": "Cómo funciona",
        "nav_runs": "Seis versiones",
        "nav_demo": "Demo en vivo",
        "nav_results": "Resultados",
        "nav_engine": "El motor",
        "nav_video": "Ver el vídeo",
        "kicker": "I+D · Meta Global AI Developer Hackathon · primer commit el 26 oct. → v6 el 2 nov.",
        "title": "Enseñamos a un modelo de lenguaje a leer el ADN.",
        "lead": "Muse Glimmer 30B, el modelo de Meta, aprendió con texto e imágenes. Nunca había visto ADN. En una semana le dimos un nuevo sentido: lee la metilación del ADN de una muestra de sangre, dice si hay una señal inusual y nombra el tejido de origen entre 23.",
        "btn_demo": "Probar la demo en vivo",
        "btn_video": "Ver el vídeo (3 min)",
        "links": "Código abierto",
        "paper": "Artículo (PDF)",
        "video_alt": "Vídeo: enseñamos al LLM de Meta a leer el ADN en una semana",
        "video_caption": "3 minutos: la idea, las seis versiones, la demo en vivo y el motor.",
        "how_title": "Un nuevo sentido para un modelo de lenguaje",
        "how_lead": "Los modelos de visión leen imágenes con un pequeño proyector que convierte píxeles en tokens. Construimos lo mismo para el ADN.",
        "how": [
            (
                "3.148 sitios de metilación",
                "Medidos en el ADN libre de una muestra de sangre: en cada sitio, la proporción de moléculas de ADN con una marca de metilación. Los tumores dejan un patrón distinto al de las células sanguíneas sanas.",
            ),
            (
                "Un adaptador de metilación",
                "Una pequeña red convierte esos 3.148 valores en 16 tokens en el espacio de Glimmer, entrenada primero sola y después con las capas de lenguaje.",
            ),
            (
                "Muse Glimmer responde",
                "Ajustado con QLoRA (la receta de Meta, en 4 bits), Glimmer lee los tokens y responde: si hay una señal y a qué tejido se parece. La respuesta se lee en sus propios logits.",
            ),
        ],
        "facts": [
            ("217 M", "parámetros entrenados"),
            ("66 min", "en una GPU L40S, en Modal"),
            ("35.000", "muestras de plasma simuladas"),
            ("23", "tejidos que sabe nombrar"),
        ],
        "runs_title": "Seis entrenamientos. Una semana.",
        "runs_lead": "No funcionó a la primera. Cada versión tiene su informe en el repositorio, fallos incluidos. Resultados: en personas nunca vistas en el entrenamiento, con un 1 % de ADN tumoral.",
        "day": "Día",
        "runs": [
            (
                "Adaptador y LoRA juntos",
                "LoRA aprendió el formato de respuesta más rápido que el adaptador la muestra.",
                "—",
                "Colapso: la misma respuesta para todos",
            ),
            (
                "Dos etapas",
                "Primero el adaptador solo, luego LoRA. Empieza a leer. 8 tejidos.",
                "tejido 70 % · ML clásico 57 %",
                "Lee, pero memoriza a los donantes",
            ),
            (
                "Un adaptador más pequeño",
                "Regularizado para dejar de memorizar a cada donante.",
                "señal 26 % · especificidad 99,9 %",
                "Prudente, pero pierde señales",
            ),
            (
                "Pseudodonantes",
                "Fondos sanos mezclados de 2 o 3 personas reales, para que ningún donante se aprenda de memoria.",
                "señal 42 % · especificidad 99,7 %",
                "Una alarma mejor",
            ),
            (
                "20 tejidos",
                "Supera al machine learning clásico en el tejido de origen. Luego más datos lo empeoraron.",
                "tejido 60 % · ML clásico 55 %",
                "Falla con personas nuevas",
            ),
            (
                "5.728 personas, 8 laboratorios alineados",
                "Cada lote de laboratorio llevado a una misma referencia. 23 tejidos, 35.000 muestras de entrenamiento.",
                "tejido 58 % de 23 · señal 75 % con 99,5 % de especificidad",
                "Lee, y se sostiene",
            ),
        ],
        "results_title": "Lo que hace hoy",
        "results_lead": "Banco v2: 1.745 personas y 1.132 tumores que ningún modelo vio en el entrenamiento, plasma simulado con un 1 % de ADN tumoral.",
        "results": [
            ("58 %", "tejido correcto a la primera, entre 23", "Muse Glimmer v6. Gradient boosting: 46 %. Azar: 4 %."),
            (
                "75 %",
                "de las señales encontradas",
                "con 99,5 % de especificidad: 5 personas sanas de cada 1.000 señaladas.",
            ),
            (
                "12 / 15",
                "plasmas reales de pacientes encontrados",
                "modelo clásico, 0 de 12 sanos señalados (Moss 2018). Glimmer nombra 7 de 11 tejidos.",
            ),
            (
                "26.155",
                "muestras públicas refinadas",
                "3.832 tumores de 22 series, 5.728 donantes de sangre de 8 cohortes.",
            ),
        ],
        "results_more": "Cada versión, cada cifra: la página Tecnología",
        "engine_title": "Ahora es un motor de lectura del ADN",
        "engine_lead": "Pesos abiertos, en línea en una GPU de Modal, una llamada a la API. El modelo de lenguaje no cambia: se reentrena el adaptador con nuevas etiquetas y responde a una nueva pregunta.",
        "uses": [
            ("Señal de cáncer y tejido", "Lo que hace hoy, en esta demo."),
            ("Seguimiento del tratamiento", "El ADN tumoral que baja, o sube, entre dos extracciones."),
            ("Vigilancia de recaída", "Una señal que vuelve tras el tratamiento."),
            ("Daño de órganos", "Qué tejido libera ADN en la sangre."),
            ("Edad biológica", "Relojes epigenéticos con la misma extracción."),
            ("La pregunta de su laboratorio", "Nuevas etiquetas, alrededor de una hora de GPU."),
        ],
        "uses_note": "Hoy solo el primero está entrenado. Los demás son líneas de investigación para las que está pensado el mismo motor.",
        "open_title": "Todo es abierto",
        "open": [
            ("Vídeo", "3 minutos, toda la historia", "youtu.be/LQVoz8USAHo"),
            ("Código", "Pipeline, entrenamiento, servicio, este sitio · AGPL-3.0", "github.com/zoukmar/burotu"),
            ("Pesos", "Muse Glimmer + adaptador, v1 a v6", "huggingface.co/zoukmar75/burotu-glimmer"),
            ("Artículo", "Preprint, 5 páginas: método, seis versiones, límites", "burotu.com/paper"),
        ],
        "built_with": "Construido con Meta Muse Glimmer 30B (pesos abiertos) y Muse Spark (Meta Model API), entrenado y servido en Modal.",
        "limits_title": "Lo que no es",
        "limits": [
            "Un prototipo de investigación, entrenado con plasma simulado y comprobado con 27 muestras reales.",
            "No es un producto sanitario ni un diagnóstico: el informe se escribe para un médico, como señal de investigación.",
            "No se vende ni se realiza ninguna prueba.",
        ],
    },
}

MONTHS = {
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "fr": ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."],
    "es": ["ene.", "feb.", "mar.", "abr.", "may.", "jun.", "jul.", "ago.", "sept.", "oct.", "nov.", "dic."],
}


def short_date(iso: str, lang: str) -> str:
    _, m, d = iso.split("-")
    month = MONTHS[lang][int(m) - 1]
    return f"{month} {int(d)}" if lang == "en" else f"{int(d)} {month}"


def runs(lang: str) -> list[dict]:
    """The six runs, with their day of the week, date and texts."""
    out = []
    for (day, iso, n), (title, text, metric, verdict) in zip(RUNS, HOME[lang]["runs"], strict=True):
        out.append(
            {
                "n": n,
                "day": day,
                "date": short_date(iso, lang),
                "iso": iso,
                "title": title,
                "text": text,
                "metric": metric,
                "verdict": verdict,
                "failed": n == 1,
                "best": n == 6,
            }
        )
    return out
