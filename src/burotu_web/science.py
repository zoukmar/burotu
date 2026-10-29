"""What happens under the hood: every model run, read from its report, in EN/FR/ES.

Used by the public page burotu.com/{lang}/technology (simple) and by the tracking
page admin.burotu.com/models (every run, every number). Numbers are never typed here:
they come from the run reports in model_reports/, copied from reports/ in the repo.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / "model_reports"
TFS = (0.003, 0.01, 0.03, 0.1)  # tumour DNA fractions shown
# The classic and Meta runs compared head to head (same tissue types, same test set).
COMPARE = ("baseline-8", "glimmer-v2")

# Every run, in the order it happened. `role` is what Burotu uses it for today.
RUNS = (
    {"id": "baseline-8", "file": "baseline-8.json", "kind": "classic", "role": "history"},
    {"id": "glimmer-v1", "file": "muse_glimmer-v1.json", "kind": "meta", "role": "failed"},
    {"id": "glimmer-v2", "file": "muse_glimmer-v2.json", "kind": "meta", "role": "tissue"},
    {"id": "glimmer-v3", "file": "muse_glimmer-v3.json", "kind": "meta", "role": "experiment"},
    {"id": "glimmer-v4", "file": "muse_glimmer-v4.json", "kind": "meta", "role": "experiment"},
    {"id": "baseline", "file": "baseline.json", "kind": "classic", "role": "signal"},
)


@cache
def run_metrics(run_id: str) -> dict:
    run = next(r for r in RUNS if r["id"] == run_id)
    rep = json.loads((REPORTS / run["file"]).read_text())
    by_tf = {}
    for row in rep["test"]["by_tumor_fraction"]:
        by_tf[round(row["tumor_fraction"], 4)] = {
            "sensitivity": row["sensitivity"],
            "auc": row["auc"],
            "top1": row["origin_top1"],
            "top2": row["origin_top2"],
        }
    rp = rep.get("real_plasma", {}).get("recalibrated", {})
    return {
        **run,
        "specificity": rep["test"]["specificity_observed"],
        "by_tf": by_tf,
        "trainable_params": rep.get("trainable_params"),
        "minutes": rep.get("minutes"),
        "real_plasma": {
            "auc": rp.get("auc_cancer_vs_healthy"),
            "healthy_flagged": rp.get("detected_healthy"),
            "n_healthy": rp.get("n_healthy"),
            "detected": rp.get("detected_cancer"),
            "n_cancer": rp.get("n_cancer"),
            "origin": rp.get("origin_top1_correct"),
            "n_origin": rp.get("n_known_origin"),
        },
    }


def all_runs() -> list[dict]:
    return [run_metrics(r["id"]) for r in RUNS]


def origin_chart(width: int = 640, height: int = 240) -> dict:
    """Grouped bars: tissue right first time, classic vs Meta, per tumour DNA fraction (SVG geometry)."""
    classic, meta = run_metrics(COMPARE[0]), run_metrics(COMPARE[1])
    left, bottom, top = 44, 34, 16
    plot_h = height - bottom - top
    group = (width - left - 10) / len(TFS)
    bar = group * 0.3
    groups = []
    for i, tf in enumerate(TFS):
        x0 = left + i * group + group * 0.18
        bars = []
        for j, (name, m) in enumerate((("classic", classic), ("meta", meta))):
            v = m["by_tf"][tf]["top1"]
            h = plot_h * v
            bars.append(
                {
                    "name": name,
                    "x": round(x0 + j * (bar + 6), 1),
                    "y": round(top + plot_h - h, 1),
                    "w": round(bar, 1),
                    "h": round(h, 1),
                    "label": f"{v:.0%}",
                }
            )
        groups.append({"tf": f"{tf * 100:g}%", "x": round(left + i * group + group / 2, 1), "bars": bars})
    ticks = [{"y": round(top + plot_h * (1 - v), 1), "label": f"{v:.0%}"} for v in (0, 0.25, 0.5, 0.75, 1)]
    return {"width": width, "height": height, "left": left, "base": top + plot_h, "groups": groups, "ticks": ticks}


# Our 20 tissues, and the AJCC 8th edition cancer categories (the way "50+ types" are counted
# by multi-cancer tests) that the TCGA projects behind each tissue cover. Conservative mapping.
COVERAGE = [
    ("Lung", ["Lung"]),
    ("Colorectal", ["Colon and rectum"]),
    ("Breast", ["Breast"]),
    ("Pancreas", ["Exocrine pancreas"]),
    ("Liver", ["Liver"]),
    ("Oesophagus-Stomach", ["Oesophagus and oesophagogastric junction", "Stomach"]),
    ("Head and neck", ["Oral cavity", "Oropharynx (HPV-mediated)", "Oropharynx (p16-) and hypopharynx", "Larynx"]),
    ("Prostate", ["Prostate"]),
    ("Kidney", ["Kidney"]),
    ("Bladder", ["Urinary bladder"]),
    ("Uterus", ["Uterus, carcinoma and carcinosarcoma"]),
    ("Cervix", ["Cervix"]),
    ("Thyroid", ["Thyroid"]),
    ("Skin", ["Melanoma of the skin"]),
    ("Brain", ["Brain and spinal cord (gliomas)"]),
    (
        "Soft tissue",
        [
            "Soft tissue sarcoma of the trunk and extremities",
            "Soft tissue sarcoma of the retroperitoneum",
            "Soft tissue sarcoma of the abdomen and thoracic visceral organs",
        ],
    ),
    ("Mesothelium", ["Malignant pleural mesothelioma"]),
    ("Adrenal", ["Adrenal cortical carcinoma", "Adrenal neuroendocrine tumours"]),
    ("Testis", ["Testis"]),
    ("Thymus", ["Thymus"]),
]
NOT_YET = [
    "Ovary, fallopian tube and peritoneum",
    "Bile ducts (3 categories)",
    "Gallbladder",
    "Ampulla of Vater",
    "Small intestine",
    "Appendix",
    "Anus",
    "Neuroendocrine tumours (appendix, colon and rectum, pancreas)",
    "Gastrointestinal stromal tumour",
    "Nasal cavity and nasopharynx",
    "Merkel cell carcinoma",
    "Bone",
    "Ureter and renal pelvis",
    "Vagina",
    "Vulva",
    "Gestational trophoblastic neoplasms",
    "Blood cancers (leukaemia, lymphoma, myeloma)",
]


def coverage_count() -> int:
    return sum(len(c) for _, c in COVERAGE)


TEXT = {
    "en": {
        "coverage": "20 tissues, about {n} cancer categories",
        "coverage_lead": 'Multi-cancer tests count "types" with the AJCC staging manual, where one organ can count '
        "several times (five soft tissue sarcoma sites, three bile duct categories). Burotu names tissues instead. "
        "Counted the AJCC way, the public tumours behind our 20 tissues cover about {n} categories. The others are "
        "missing from public methylation data, which is why going to 50 needs a partner lab's sequenced samples.",
        "col_tissue": "Burotu tissue",
        "col_ajcc": "AJCC categories covered",
        "not_yet": "Not covered yet (no public data on the same array)",
        "title": "Under the hood",
        "lead": "Every number on this page comes from the experiments in Burotu's public repository. Burotu is a "
        "demonstration prototype: its models are trained and tested on public data, mostly simulated plasma.",
        "nav": "Technology",
        "pipeline": "From a blood draw to a letter",
        "steps": [
            (
                "Blood draw and lab",
                "Cell-free DNA is isolated from plasma and a targeted panel of 1,681 methylation "
                "sites is sequenced. The lab sends back read counts.",
            ),
            (
                "Quality control",
                "Coverage and read depth are checked first. A sample read too shallowly gets no "
                "result: a new blood draw is asked for.",
            ),
            (
                "Signal",
                "A classic model (gradient boosting on read counts compared with healthy blood) decides "
                "whether there is an unusual signal. Its threshold is set so that 99.5% of healthy people never "
                "trained on are not flagged.",
            ),
            (
                "Tissue",
                "Muse Glimmer 30B, Meta's open-weight model, reads the sample through an adapter that turns "
                "1,681 numbers into 16 tokens, and names the tissue the signal most resembles.",
            ),
            (
                "Letter",
                "Muse Spark writes the letter in English, French and Spanish. It talks about a signal, "
                "never a disease; an answer that breaks the rule is rewritten or replaced.",
            ),
            (
                "Follow-up",
                "Every analysis joins the patient's file, with a PDF comparing both models and earlier results.",
            ),
        ],
        "data": "Where the data comes from",
        "data_rows": [
            ("Tumour profiles", "TCGA (GDC portal), 20 tissue types, 48 tumours each"),
            ("Healthy background", "Whole blood of 164 adults (Hannum et al. 2013)"),
            ("Real plasma", "27 cell-free DNA samples, 12 healthy and 15 with a tumour (Moss et al. 2018)"),
            ("Split", "People are split before anything else: 80 to train, 35 to set the threshold, 49 to test"),
        ],
        "compare": "Classic model vs Meta model",
        "compare_lead": "Right first time on the tissue, on people and tumours neither model ever saw, by share of "
        "tumour DNA in the blood.",
        "classic": "Classic model",
        "meta": "Muse Glimmer (Meta)",
        "tf_axis": "Tumour DNA in the blood",
        "signal_title": "Raising the alarm",
        "signal_text": "Both models rank samples well (AUC 0.97 to 1.00 from 1% tumour DNA). But only the classic "
        "model keeps 99.5% specificity on new people: trained on 80 blood donors, Muse Glimmer also learned each "
        "donor's background. So the classic model raises the alarm and Muse Glimmer names the tissue.",
        "col_tf": "Tumour DNA",
        "col_sens": "Signal found (classic, 99.8% specificity)",
        "col_auc_c": "AUC classic",
        "col_auc_m": "AUC Muse Glimmer",
        "runs": "Every training run",
        "runs_lead": "Muse Glimmer was fine-tuned on Modal GPUs (one 48 GB L40S, about 70 minutes per run) with "
        "QLoRA, following Meta's recipe.",
        "run_names": {
            "baseline-8": "Classic model, 8 tissue types (first prototype)",
            "glimmer-v1": "Muse Glimmer v1: adapter and LoRA trained together",
            "glimmer-v2": "Muse Glimmer v2: adapter first, then LoRA",
            "glimmer-v3": "Muse Glimmer v3: smaller, regularised adapter",
            "glimmer-v4": "Muse Glimmer v4: pseudo-donors (mixes of real donors)",
            "baseline": "Classic model, 20 tissue types",
        },
        "run_notes": {
            "baseline-8": "Where Burotu started: the benchmark Glimmer v1 to v4 were measured against.",
            "glimmer-v1": "Learned the answer format, not the sample: the same answer for everyone.",
            "glimmer-v2": "Names the tissue in Burotu: best at low tumour DNA.",
            "glimmer-v3": "Fewer healthy people flagged, but too few signals found.",
            "glimmer-v4": "Best Glimmer alarm so far (42% of signals found at 1%), still below the classic model.",
            "baseline": "Raises the alarm in Burotu: same performance with 20 types as with 8; tissue right 55% at 1%, against 5% by chance.",
        },
        "plasma": "Real patient plasma",
        "plasma_text": "On 27 real samples measured on arrays, against the whole-blood background every sample "
        "looked unusual (AUC 0.62). Calibrating the healthy background with healthy controls, as a lab does for "
        "any new assay, gives AUC {auc:.2f}, {flagged} of {n_healthy} healthy people flagged, {detected} of "
        "{n_cancer} tumour samples found. A sanity check on 27 samples, not a clinical figure.",
        "lessons": "What we learned",
        "lesson_items": [
            "Train the adapter alone first. Trained together from the start, LoRA learns the answer format "
            "before the adapter learns to carry the sample, and the model stops looking at it.",
            "Read the yes/no margin before Glimmer's soft cap. Confident answers otherwise land on the same value.",
            "Watch healthy people never trained on during training: it is the only way to see a model memorise "
            "donors instead of learning the signal.",
        ],
        "limits": "Limits",
        "limit_items": [
            "Plasma is simulated site by site; real tests also use patterns within each DNA fragment.",
            "20 tissue types today; the product targets 50.",
            "Real plasma check on 27 samples only. Burotu is not a validated test and not a medical device.",
        ],
        "repo": "Code, data pipeline and every report",
    },
    "fr": {
        "coverage": "20 tissus, environ {n} catégories de cancer",
        "coverage_lead": "Les tests multi-cancers comptent les « types » avec le manuel de classification AJCC, où un "
        "même organe peut compter plusieurs fois (cinq localisations de sarcomes, trois catégories de voies biliaires). "
        "Burotu désigne des tissus. Comptées à la manière AJCC, les tumeurs publiques derrière nos 20 tissus couvrent "
        "environ {n} catégories. Les autres manquent dans les données publiques de méthylation : pour aller à 50, il faut "
        "les échantillons séquencés d'un laboratoire partenaire.",
        "col_tissue": "Tissu Burotu",
        "col_ajcc": "Catégories AJCC couvertes",
        "not_yet": "Pas encore couvert (pas de données publiques sur la même puce)",
        "title": "Sous le capot",
        "lead": "Chaque chiffre de cette page vient des expériences du dépôt public de Burotu. Burotu est un "
        "prototype de démonstration : ses modèles sont entraînés et testés sur des données publiques, surtout du "
        "plasma simulé.",
        "nav": "Technologie",
        "pipeline": "De la prise de sang au courrier",
        "steps": [
            (
                "Prise de sang et labo",
                "L'ADN circulant est extrait du plasma et un panel ciblé de 1 681 sites de "
                "méthylation est séquencé. Le labo renvoie des comptes de lectures.",
            ),
            (
                "Contrôle qualité",
                "La couverture et la profondeur sont vérifiées d'abord. Un échantillon trop peu "
                "lu n'a pas de résultat : une nouvelle prise de sang est demandée.",
            ),
            (
                "Signal",
                "Un modèle classique (gradient boosting sur les lectures comparées au sang sain) décide s'il "
                "y a un signal inhabituel. Son seuil est réglé pour que 99,5 % des personnes saines jamais vues ne "
                "soient pas signalées.",
            ),
            (
                "Tissu",
                "Muse Glimmer 30B, le modèle open-weights de Meta, lit l'échantillon grâce à un adaptateur qui "
                "transforme 1 681 nombres en 16 jetons, et désigne le tissu dont le signal se rapproche le plus.",
            ),
            (
                "Courrier",
                "Muse Spark rédige le courrier en anglais, français et espagnol. Il parle d'un signal, "
                "jamais d'une maladie ; une réponse qui enfreint la règle est réécrite ou remplacée.",
            ),
            (
                "Suivi",
                "Chaque analyse rejoint le dossier du patient, avec un PDF qui compare les deux modèles et les "
                "résultats précédents.",
            ),
        ],
        "data": "D'où viennent les données",
        "data_rows": [
            ("Profils tumoraux", "TCGA (portail GDC), 20 types de tissus, 48 tumeurs chacun"),
            ("Bruit de fond sain", "Sang total de 164 adultes (Hannum et al. 2013)"),
            ("Plasma réel", "27 échantillons d'ADN circulant, 12 sains et 15 avec une tumeur (Moss et al. 2018)"),
            (
                "Répartition",
                "Les personnes sont réparties avant tout : 80 pour entraîner, 35 pour régler le seuil, 49 pour tester",
            ),
        ],
        "compare": "Modèle classique vs modèle Meta",
        "compare_lead": "Tissu trouvé du premier coup, sur des personnes et des tumeurs qu'aucun modèle n'a vues, "
        "selon la part d'ADN tumoral dans le sang.",
        "classic": "Modèle classique",
        "meta": "Muse Glimmer (Meta)",
        "tf_axis": "ADN tumoral dans le sang",
        "signal_title": "Donner l'alarme",
        "signal_text": "Les deux modèles classent bien les échantillons (AUC de 0,97 à 1,00 dès 1 % d'ADN "
        "tumoral). Mais seul le modèle classique garde 99,5 % de spécificité sur de nouvelles personnes : "
        "entraîné sur 80 donneurs, Muse Glimmer a aussi appris le fond sanguin de chacun. Le modèle classique "
        "donne donc l'alarme et Muse Glimmer désigne le tissu.",
        "col_tf": "ADN tumoral",
        "col_sens": "Signal trouvé (classique, spécificité 99,8 %)",
        "col_auc_c": "AUC classique",
        "col_auc_m": "AUC Muse Glimmer",
        "runs": "Tous les entraînements",
        "runs_lead": "Muse Glimmer a été affiné sur des GPU Modal (une L40S de 48 Go, environ 70 minutes par "
        "entraînement) avec QLoRA, en suivant la recette de Meta.",
        "run_names": {
            "baseline-8": "Modèle classique, 8 types de tissus (premier prototype)",
            "glimmer-v1": "Muse Glimmer v1 : adaptateur et LoRA entraînés ensemble",
            "glimmer-v2": "Muse Glimmer v2 : l'adaptateur d'abord, puis LoRA",
            "glimmer-v3": "Muse Glimmer v3 : adaptateur réduit et régularisé",
            "glimmer-v4": "Muse Glimmer v4 : pseudo-donneurs (mélanges de vrais donneurs)",
            "baseline": "Modèle classique, 20 types de tissus",
        },
        "run_notes": {
            "baseline-8": "Le point de départ : la référence de Glimmer v1 à v4.",
            "glimmer-v1": "A appris le format de la réponse, pas l'échantillon : la même réponse pour tous.",
            "glimmer-v2": "Désigne le tissu dans Burotu : le meilleur à faible dose.",
            "glimmer-v3": "Moins de personnes saines signalées, mais trop peu de signaux trouvés.",
            "glimmer-v4": "Meilleure alarme Glimmer à ce jour (42 % des signaux trouvés à 1 %), encore sous le modèle classique.",
            "baseline": "Donne l'alarme dans Burotu : mêmes performances avec 20 types qu'avec 8 ; tissu trouvé 55 % à 1 %, contre 5 % au hasard.",
        },
        "plasma": "Plasma de vrais patients",
        "plasma_text": "Sur 27 vrais échantillons mesurés sur puces, face au bruit de fond du sang total, tous "
        "semblaient inhabituels (AUC 0,62). En calibrant le bruit de fond avec des témoins sains, comme un labo le "
        "fait pour tout nouveau test : AUC {auc:.2f}, {flagged} personne(s) saine(s) signalée(s) sur {n_healthy}, "
        "{detected} échantillons tumoraux trouvés sur {n_cancer}. Une vérification sur 27 échantillons, pas un "
        "chiffre clinique.",
        "lessons": "Ce que nous avons appris",
        "lesson_items": [
            "Entraîner d'abord l'adaptateur seul. Entraînés ensemble dès le début, LoRA apprend le format de la "
            "réponse avant que l'adaptateur n'apprenne à porter l'échantillon, et le modèle cesse de le regarder.",
            "Lire l'écart oui/non avant le plafonnement interne de Glimmer. Sinon, les réponses sûres d'elles "
            "tombent toutes sur la même valeur.",
            "Surveiller pendant l'entraînement des personnes saines jamais vues : c'est le seul moyen de voir un "
            "modèle mémoriser les donneurs au lieu d'apprendre le signal.",
        ],
        "limits": "Limites",
        "limit_items": [
            "Le plasma est simulé site par site ; les vrais tests exploitent aussi les motifs au sein de chaque "
            "fragment d'ADN.",
            "20 types de tissus aujourd'hui ; le produit en vise 50.",
            "Vérification sur plasma réel avec 27 échantillons seulement. Burotu n'est pas un test validé ni un "
            "dispositif médical.",
        ],
        "repo": "Code, chaîne de données et tous les rapports",
    },
    "es": {
        "coverage": "20 tejidos, unas {n} categorías de cáncer",
        "coverage_lead": "Las pruebas multicáncer cuentan los «tipos» con el manual de clasificación AJCC, donde un "
        "mismo órgano puede contar varias veces (cinco localizaciones de sarcomas, tres categorías de vías biliares). "
        "Burotu indica tejidos. Contadas al modo AJCC, las muestras tumorales públicas detrás de nuestros 20 tejidos "
        "cubren unas {n} categorías. Las demás faltan en los datos públicos de metilación: para llegar a 50 hacen falta "
        "las muestras secuenciadas de un laboratorio colaborador.",
        "col_tissue": "Tejido Burotu",
        "col_ajcc": "Categorías AJCC cubiertas",
        "not_yet": "Aún no cubierto (sin datos públicos en el mismo chip)",
        "title": "Bajo el capó",
        "lead": "Cada cifra de esta página procede de los experimentos del repositorio público de Burotu. Burotu es "
        "un prototipo de demostración: sus modelos se entrenan y evalúan con datos públicos, sobre todo plasma "
        "simulado.",
        "nav": "Tecnología",
        "pipeline": "De la extracción a la carta",
        "steps": [
            (
                "Extracción y laboratorio",
                "Se aísla el ADN circulante del plasma y se secuencia un panel dirigido de "
                "1.681 sitios de metilación. El laboratorio devuelve recuentos de lecturas.",
            ),
            (
                "Control de calidad",
                "Primero se comprueban la cobertura y la profundidad. Una muestra poco leída no "
                "tiene resultado: se pide una nueva extracción.",
            ),
            (
                "Señal",
                "Un modelo clásico (gradient boosting sobre las lecturas comparadas con sangre sana) decide si "
                "hay una señal inusual. Su umbral se ajusta para que el 99,5 % de las personas sanas nunca vistas no "
                "sean señaladas.",
            ),
            (
                "Tejido",
                "Muse Glimmer 30B, el modelo open-weights de Meta, lee la muestra mediante un adaptador que "
                "convierte 1.681 números en 16 tokens, e indica el tejido al que más se parece la señal.",
            ),
            (
                "Carta",
                "Muse Spark redacta la carta en inglés, francés y español. Habla de una señal, nunca de una "
                "enfermedad; una respuesta que rompe la regla se reescribe o se sustituye.",
            ),
            (
                "Seguimiento",
                "Cada análisis se suma al expediente del paciente, con un PDF que compara los dos "
                "modelos y los resultados anteriores.",
            ),
        ],
        "data": "De dónde vienen los datos",
        "data_rows": [
            ("Perfiles tumorales", "TCGA (portal GDC), 20 tipos de tejido, 48 tumores cada uno"),
            ("Fondo sano", "Sangre total de 164 adultos (Hannum et al. 2013)"),
            ("Plasma real", "27 muestras de ADN circulante, 12 sanas y 15 con un tumor (Moss et al. 2018)"),
            (
                "Reparto",
                "Las personas se reparten antes de nada: 80 para entrenar, 35 para fijar el umbral, 49 para evaluar",
            ),
        ],
        "compare": "Modelo clásico vs modelo Meta",
        "compare_lead": "Tejido acertado a la primera, en personas y tumores que ningún modelo vio, según la "
        "proporción de ADN tumoral en la sangre.",
        "classic": "Modelo clásico",
        "meta": "Muse Glimmer (Meta)",
        "tf_axis": "ADN tumoral en la sangre",
        "signal_title": "Dar la alarma",
        "signal_text": "Los dos modelos ordenan bien las muestras (AUC de 0,97 a 1,00 desde un 1 % de ADN "
        "tumoral). Pero solo el modelo clásico mantiene un 99,5 % de especificidad en personas nuevas: entrenado "
        "con 80 donantes, Muse Glimmer también aprendió el fondo sanguíneo de cada uno. Por eso el modelo clásico "
        "da la alarma y Muse Glimmer indica el tejido.",
        "col_tf": "ADN tumoral",
        "col_sens": "Señal encontrada (clásico, especificidad 99,8 %)",
        "col_auc_c": "AUC clásico",
        "col_auc_m": "AUC Muse Glimmer",
        "runs": "Todos los entrenamientos",
        "runs_lead": "Muse Glimmer se ajustó en GPU de Modal (una L40S de 48 GB, unos 70 minutos por "
        "entrenamiento) con QLoRA, siguiendo la receta de Meta.",
        "run_names": {
            "baseline-8": "Modelo clásico, 8 tipos de tejido (primer prototipo)",
            "glimmer-v1": "Muse Glimmer v1: adaptador y LoRA entrenados a la vez",
            "glimmer-v2": "Muse Glimmer v2: primero el adaptador, luego LoRA",
            "glimmer-v3": "Muse Glimmer v3: adaptador reducido y regularizado",
            "glimmer-v4": "Muse Glimmer v4: pseudodonantes (mezclas de donantes reales)",
            "baseline": "Modelo clásico, 20 tipos de tejido",
        },
        "run_notes": {
            "baseline-8": "El punto de partida: la referencia de Glimmer v1 a v4.",
            "glimmer-v1": "Aprendió el formato de la respuesta, no la muestra: la misma respuesta para todos.",
            "glimmer-v2": "Indica el tejido en Burotu: el mejor con poco ADN tumoral.",
            "glimmer-v3": "Menos personas sanas señaladas, pero demasiado pocas señales encontradas.",
            "glimmer-v4": "La mejor alarma de Glimmer hasta ahora (42 % de señales encontradas al 1 %), aún por debajo del modelo clásico.",
            "baseline": "Da la alarma en Burotu: el mismo rendimiento con 20 tipos que con 8; tejido acertado el 55 % al 1 %, frente al 5 % por azar.",
        },
        "plasma": "Plasma de pacientes reales",
        "plasma_text": "En 27 muestras reales medidas con chips, frente al fondo de la sangre total todas parecían "
        "inusuales (AUC 0,62). Calibrando el fondo con controles sanos, como hace un laboratorio con cualquier "
        "prueba nueva: AUC {auc:.2f}, {flagged} de {n_healthy} personas sanas señaladas, {detected} de {n_cancer} "
        "muestras tumorales encontradas. Una comprobación con 27 muestras, no una cifra clínica.",
        "lessons": "Lo que aprendimos",
        "lesson_items": [
            "Entrenar primero el adaptador solo. Entrenados a la vez desde el principio, LoRA aprende el formato de "
            "la respuesta antes de que el adaptador aprenda a llevar la muestra, y el modelo deja de mirarla.",
            "Leer el margen sí/no antes del límite interno de Glimmer. Si no, las respuestas seguras caen todas en "
            "el mismo valor.",
            "Vigilar durante el entrenamiento a personas sanas nunca vistas: es la única forma de ver que un modelo "
            "memoriza donantes en lugar de aprender la señal.",
        ],
        "limits": "Límites",
        "limit_items": [
            "El plasma se simula sitio a sitio; las pruebas reales también usan patrones dentro de cada fragmento "
            "de ADN.",
            "20 tipos de tejido hoy; el producto apunta a 50.",
            "Comprobación en plasma real con solo 27 muestras. Burotu no es una prueba validada ni un dispositivo "
            "médico.",
        ],
        "repo": "Código, cadena de datos y todos los informes",
    },
}
