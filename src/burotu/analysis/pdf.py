"""The result letter as a two-page PDF, in English, French or Spanish.

Page 1: who, when, the signal and the tissue it resembles, the letter, quality control.
Page 2: the classic and Meta models side by side, the patient's follow-up, and where
Burotu stands among other tests. Like the letter, it talks about a methylation signal,
never a disease, and says on every page that Burotu is a prototype, not a medical device.
"""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from burotu.analysis.wording import qc_findings, tissue

INK = colors.HexColor("#1B2130")
SOFT = colors.HexColor("#4A5263")
VENOUS = colors.HexColor("#7A1F33")
PLASMA = colors.HexColor("#F6EFD2")
LINE = colors.HexColor("#D6DDD9")
GLASS = colors.HexColor("#F6F7F4")
GREEN = colors.HexColor("#1F5E35")
META = colors.HexColor("#2D5BD7")

T = {
    "en": {
        "title": "Methylation signal report",
        "banner": "Demonstration prototype. Not a medical device, not a validated test, not a medical result.",
        "name": "Name",
        "born": "Born in",
        "ref": "Reference",
        "sample": "Sample",
        "drawn": "Blood draw",
        "site": "Collection site",
        "analysed": "Analysed",
        "result": "Result",
        "signal": "Unusual signal detected",
        "clear": "No unusual signal detected",
        "noresult": "No result: a new blood draw is needed",
        "resembles": "Tissue the signal most resembles",
        "letter": "Your letter",
        "by_muse": "Written by Muse Spark (Meta) from the analysis results.",
        "by_template": "Written from Burotu's standard wording.",
        "qc": "Quality control",
        "qc_pass": "Passed",
        "qc_warn": "Passed with warnings",
        "qc_fail": "Failed",
        "coverage": "panel coverage",
        "depth": "median depth",
        "reads": "reads",
        "models": "Two models, one result",
        "models_lead": "The classic model, tuned to almost never flag a healthy person, decides whether there is a "
        "signal. Muse Glimmer, Meta's open-weight model fine-tuned by Burotu, names the tissue the signal most "
        "resembles, where it is more accurate at very low levels of circulating DNA.",
        "col_model": "Model",
        "col_signal": "Signal",
        "col_score": "Score / threshold",
        "col_tissue": "Top tissue",
        "classic": "Classic model (gradient boosting)",
        "meta": "Meta model (Muse Glimmer 30B)",
        "yes": "Detected",
        "no": "Not detected",
        "not_served": "Not available for this analysis",
        "agree": "The two models agree on the tissue.",
        "differ": "The two models differ on the tissue; the Meta model's answer is reported.",
        "followup": "Your follow-up",
        "followup_one": "This is your first analysis with Burotu. Later analyses will appear here, so your doctor can "
        "see how the signal evolves.",
        "followup_many": "Every analysis on your file, newest first.",
        "col_date": "Date",
        "col_ref": "Reference",
        "col_result": "Result",
        "qc_failed": "Quality control failed",
        "market": "Where Burotu stands",
        "market_lead": "Burotu does not replace the screenings your doctor recommends. Other tests, as of October 2026:",
        "col_test": "Test",
        "col_looks": "Looks for",
        "col_status": "Status",
        "m1": (
            "Organ screenings (mammography, colonoscopy or stool test, low-dose lung scan, HPV test)",
            "One organ each",
            "Established, recommended by health authorities",
        ),
        "m2": ("Shield, Guardant Health (blood)", "Colon and rectum", "Approved by the US FDA in July 2024"),
        "m3": (
            "Galleri, GRAIL (blood, methylation)",
            "Signals from many tissues",
            "Backed by an FDA advisory panel in September 2026, not yet approved",
        ),
        "m4": ("Burotu (blood, methylation)", "8 tissues today, 50 targeted", "Demonstration prototype, not validated"),
        "sources": "Sources",
        "footer": "Burotu is a demonstration prototype, not a medical device. This document is not a medical result.",
        "page": "Page",
    },
    "fr": {
        "title": "Rapport de signal de méthylation",
        "banner": "Prototype de démonstration. Pas un dispositif médical, pas un test validé, pas un résultat médical.",
        "name": "Nom",
        "born": "Né(e) en",
        "ref": "Référence",
        "sample": "Échantillon",
        "drawn": "Prise de sang",
        "site": "Centre de prélèvement",
        "analysed": "Analysé le",
        "result": "Résultat",
        "signal": "Signal inhabituel détecté",
        "clear": "Aucun signal inhabituel détecté",
        "noresult": "Pas de résultat : une nouvelle prise de sang est nécessaire",
        "resembles": "Tissu dont le signal se rapproche le plus",
        "letter": "Votre courrier",
        "by_muse": "Rédigé par Muse Spark (Meta) à partir des résultats de l'analyse.",
        "by_template": "Rédigé à partir de la formulation standard de Burotu.",
        "qc": "Contrôle qualité",
        "qc_pass": "Réussi",
        "qc_warn": "Réussi avec réserves",
        "qc_fail": "Échoué",
        "coverage": "couverture du panel",
        "depth": "profondeur médiane",
        "reads": "lectures",
        "models": "Deux modèles, un résultat",
        "models_lead": "Le modèle classique, réglé pour ne presque jamais signaler une personne saine, décide s'il y a "
        "un signal. Muse Glimmer, le modèle open-weights de Meta affiné par Burotu, désigne le tissu dont le "
        "signal se rapproche le plus : il est plus précis quand l'ADN circulant est en très faible quantité.",
        "col_model": "Modèle",
        "col_signal": "Signal",
        "col_score": "Score / seuil",
        "col_tissue": "Tissu principal",
        "classic": "Modèle classique (gradient boosting)",
        "meta": "Modèle Meta (Muse Glimmer 30B)",
        "yes": "Détecté",
        "no": "Non détecté",
        "not_served": "Non disponible pour cette analyse",
        "agree": "Les deux modèles s'accordent sur le tissu.",
        "differ": "Les deux modèles divergent sur le tissu ; la réponse du modèle Meta est retenue.",
        "followup": "Votre suivi",
        "followup_one": "C'est votre première analyse avec Burotu. Les suivantes s'ajouteront ici, pour que votre "
        "médecin voie l'évolution du signal.",
        "followup_many": "Toutes les analyses de votre dossier, de la plus récente à la plus ancienne.",
        "col_date": "Date",
        "col_ref": "Référence",
        "col_result": "Résultat",
        "qc_failed": "Contrôle qualité échoué",
        "market": "Où se situe Burotu",
        "market_lead": "Burotu ne remplace pas les dépistages recommandés par votre médecin. Les autres tests, en "
        "octobre 2026 :",
        "col_test": "Test",
        "col_looks": "Recherche",
        "col_status": "Statut",
        "m1": (
            "Dépistages par organe (mammographie, coloscopie ou test dans les selles, scanner faible dose, test HPV)",
            "Un organe chacun",
            "Établis, recommandés par les autorités de santé",
        ),
        "m2": ("Shield, Guardant Health (sang)", "Côlon et rectum", "Approuvé par la FDA américaine en juillet 2024"),
        "m3": (
            "Galleri, GRAIL (sang, méthylation)",
            "Signaux de nombreux tissus",
            "Soutenu par un comité consultatif de la FDA en septembre 2026, pas encore approuvé",
        ),
        "m4": (
            "Burotu (sang, méthylation)",
            "8 tissus aujourd'hui, 50 visés",
            "Prototype de démonstration, non validé",
        ),
        "sources": "Sources",
        "footer": "Burotu est un prototype de démonstration, pas un dispositif médical. Ce document n'est pas un "
        "résultat médical.",
        "page": "Page",
    },
    "es": {
        "title": "Informe de señal de metilación",
        "banner": "Prototipo de demostración. No es un dispositivo médico, ni una prueba validada, ni un resultado "
        "médico.",
        "name": "Nombre",
        "born": "Año de nacimiento",
        "ref": "Referencia",
        "sample": "Muestra",
        "drawn": "Extracción",
        "site": "Centro de extracción",
        "analysed": "Analizada el",
        "result": "Resultado",
        "signal": "Señal inusual detectada",
        "clear": "Ninguna señal inusual detectada",
        "noresult": "Sin resultado: es necesaria una nueva extracción",
        "resembles": "Tejido al que más se parece la señal",
        "letter": "Tu carta",
        "by_muse": "Redactada por Muse Spark (Meta) a partir de los resultados del análisis.",
        "by_template": "Redactada con la formulación estándar de Burotu.",
        "qc": "Control de calidad",
        "qc_pass": "Superado",
        "qc_warn": "Superado con avisos",
        "qc_fail": "No superado",
        "coverage": "cobertura del panel",
        "depth": "profundidad mediana",
        "reads": "lecturas",
        "models": "Dos modelos, un resultado",
        "models_lead": "El modelo clásico, ajustado para casi nunca señalar a una persona sana, decide si hay una "
        "señal. Muse Glimmer, el modelo open-weights de Meta ajustado por Burotu, indica el tejido al que más se "
        "parece la señal: es más preciso cuando hay muy poco ADN circulante.",
        "col_model": "Modelo",
        "col_signal": "Señal",
        "col_score": "Puntuación / umbral",
        "col_tissue": "Tejido principal",
        "classic": "Modelo clásico (gradient boosting)",
        "meta": "Modelo Meta (Muse Glimmer 30B)",
        "yes": "Detectada",
        "no": "No detectada",
        "not_served": "No disponible para este análisis",
        "agree": "Los dos modelos coinciden en el tejido.",
        "differ": "Los dos modelos difieren en el tejido; se informa la respuesta del modelo Meta.",
        "followup": "Tu seguimiento",
        "followup_one": "Es tu primer análisis con Burotu. Los siguientes aparecerán aquí, para que tu médico vea "
        "cómo evoluciona la señal.",
        "followup_many": "Todos los análisis de tu expediente, del más reciente al más antiguo.",
        "col_date": "Fecha",
        "col_ref": "Referencia",
        "col_result": "Resultado",
        "qc_failed": "Control de calidad no superado",
        "market": "Dónde se sitúa Burotu",
        "market_lead": "Burotu no sustituye los cribados que recomienda tu médico. Otras pruebas, en octubre de 2026:",
        "col_test": "Prueba",
        "col_looks": "Busca",
        "col_status": "Situación",
        "m1": (
            "Cribados por órgano (mamografía, colonoscopia o test en heces, TAC de baja dosis, test de VPH)",
            "Un órgano cada uno",
            "Establecidos, recomendados por las autoridades sanitarias",
        ),
        "m2": ("Shield, Guardant Health (sangre)", "Colon y recto", "Aprobado por la FDA de EE. UU. en julio de 2024"),
        "m3": (
            "Galleri, GRAIL (sangre, metilación)",
            "Señales de muchos tejidos",
            "Respaldado por un comité asesor de la FDA en septiembre de 2026, aún no aprobado",
        ),
        "m4": ("Burotu (sangre, metilación)", "8 tejidos hoy, 50 previstos", "Prototipo de demostración, no validado"),
        "sources": "Fuentes",
        "footer": "Burotu es un prototipo de demostración, no un dispositivo médico. Este documento no es un "
        "resultado médico.",
        "page": "Página",
    },
}


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK)
    return {
        "base": base,
        "small": ParagraphStyle("small", parent=base, fontSize=8, leading=11, textColor=SOFT),
        "label": ParagraphStyle("label", parent=base, fontName="Helvetica-Bold", fontSize=8, textColor=SOFT),
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=20, leading=24),
        "h2": ParagraphStyle(
            "h2", parent=base, fontName="Helvetica-Bold", fontSize=13, leading=17, spaceBefore=6, spaceAfter=4
        ),
        "verdict": ParagraphStyle("verdict", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=21),
        "letter": ParagraphStyle("letter", parent=base, fontSize=10.5, leading=16),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, leading=11.5),
        "cellb": ParagraphStyle("cellb", parent=base, fontName="Helvetica-Bold", fontSize=8.5, leading=11.5),
    }


def _bars(origins: list[dict], lang: str, width: float = 160 * mm) -> Drawing:
    """Horizontal bars, one per tissue, in the report's colours."""
    row, label_w, track_w = 7 * mm, 55 * mm, width - 75 * mm
    d = Drawing(width, row * len(origins))
    for i, o in enumerate(origins):
        y = row * (len(origins) - 1 - i) + 1.5 * mm
        d.add(String(0, y, tissue(o["class"], lang).capitalize(), fontName="Helvetica", fontSize=9.5, fillColor=INK))
        d.add(Rect(label_w, y - 0.5 * mm, track_w, 3.2 * mm, rx=1.6 * mm, fillColor=LINE, strokeColor=None))
        w = max(track_w * o["probability"], 0.8 * mm)
        r = min(1.6 * mm, w / 2)
        d.add(
            Rect(label_w, y - 0.5 * mm, w, 3.2 * mm, rx=r, ry=r, fillColor=VENOUS if i == 0 else SOFT, strokeColor=None)
        )
        d.add(
            String(
                label_w + track_w + 3 * mm,
                y,
                f"{o['probability']:.0%}",
                fontName="Helvetica-Bold",
                fontSize=9.5,
                fillColor=INK,
            )
        )
    return d


def _trend(points: list[tuple[str, float, float]], width: float = 160 * mm, height: float = 32 * mm) -> Drawing:
    """Signal score over time (oldest left) with the threshold as a dashed line."""
    d = Drawing(width, height)
    ys = [s for _, s, _ in points] + [t for _, _, t in points]
    lo, hi = min(ys), max(ys)
    lo, hi = (lo - 0.05, hi + 0.05) if hi - lo < 1e-6 else (lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.1)

    def y(v: float) -> float:
        return 4 * mm + (v - lo) / (hi - lo) * (height - 10 * mm)

    xs = [10 * mm + i * (width - 20 * mm) / max(1, len(points) - 1) for i in range(len(points))]
    thr = points[-1][2]
    d.add(Line(8 * mm, y(thr), width - 8 * mm, y(thr), strokeColor=SOFT, strokeDashArray=[3, 3], strokeWidth=0.6))
    if len(points) > 1:
        d.add(
            PolyLine(
                [c for x, (_, s, _) in zip(xs, points, strict=True) for c in (x, y(s))],
                strokeColor=VENOUS,
                strokeWidth=1.4,
            )
        )
    for x, (label, s, t) in zip(xs, points, strict=True):
        d.add(Circle(x, y(s), 1.6 * mm, fillColor=VENOUS if s > t else colors.white, strokeColor=VENOUS))
        d.add(String(x, 0, label, fontName="Helvetica", fontSize=7, fillColor=SOFT, textAnchor="middle"))
    return d


def build_pdf(booking: dict, analysis: dict, followup: list[dict], lang: str) -> bytes:
    """`booking`: the booking row as a dict; `analysis`: the stored result; `followup`: Database.patient_analyses."""
    lang = lang if lang in T else "en"
    t, st = T[lang], _styles()
    q, d = analysis["quality"], analysis.get("detection")
    reports = analysis.get("reports") or {booking["lang"]: analysis.get("report", {"text": "", "author": ""})}
    letter = reports.get(lang) or next(iter(reports.values()))
    buf = io.BytesIO()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def frame(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColor(INK)
        for i, filled in enumerate((False, True, False)):
            cx = 20 * mm + i * 5 * mm
            canvas.setFillColor(VENOUS if filled else colors.white)
            canvas.setStrokeColor(INK if not filled else VENOUS)
            canvas.circle(cx, A4[1] - 16 * mm, 1.6 * mm, stroke=1, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(36 * mm, A4[1] - 17.6 * mm, "burotu")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(SOFT)
        canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 17.6 * mm, f"{booking['ref']} · {generated}")
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
        canvas.drawString(18 * mm, 11 * mm, t["footer"])
        canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f"{t['page']} {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=26 * mm,
        bottomMargin=22 * mm,
        title=f"Burotu {booking['ref']}",
        author="Burotu",
    )
    full = A4[0] - 36 * mm
    story: list = [Paragraph(t["title"], st["h1"]), Spacer(1, 3 * mm)]

    banner = Table([[Paragraph(t["banner"], st["cell"])]], colWidths=[full])
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PLASMA),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story += [banner, Spacer(1, 5 * mm)]

    facts = [
        (t["name"], f"{booking['first_name']} {booking['last_name']}"),
        (t["born"], str(booking["birth_year"])),
        (t["ref"], booking["ref"]),
        (t["sample"], analysis.get("sample_id", "")),
        (t["drawn"], f"{booking['day']} {booking['slot']}"),
        (t["site"], f"{booking.get('site_name', '')}, {booking.get('city', '')}".strip(", ")),
        (t["analysed"], analysis.get("at", analysis.get("analysed_at", "")).replace("T", " ")[:16]),
    ]
    rows = [
        [Paragraph(k, st["label"]), Paragraph(v, st["cell"]), Paragraph(k2, st["label"]), Paragraph(v2, st["cell"])]
        for (k, v), (k2, v2) in zip(facts[::2], [*facts[1::2], ("", "")], strict=True)
    ]
    ft = Table(rows, colWidths=[30 * mm, full / 2 - 30 * mm, 30 * mm, full / 2 - 30 * mm])
    ft.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story += [ft, Spacer(1, 5 * mm)]

    # The result
    if d is None or q["status"] == "fail":
        verdict, colour = t["noresult"], SOFT
    elif d["cancer_signal"]:
        verdict, colour = t["signal"], VENOUS
    else:
        verdict, colour = t["clear"], GREEN
    box: list = [
        Paragraph(t["result"], st["label"]),
        Spacer(1, 1.5 * mm),
        Paragraph(f'<font color="#{colour.hexval()[2:]}">{verdict}</font>', st["verdict"]),
    ]
    if d is not None and d["cancer_signal"]:
        box += [
            Spacer(1, 3 * mm),
            Paragraph(t["resembles"], st["label"]),
            Spacer(1, 1.5 * mm),
            _bars(d["origin"], lang, full - 12 * mm),
        ]
    rb = Table([[box]], colWidths=[full])
    rb.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("LEFTPADDING", (0, 0), (-1, -1), 6 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
                ("LINEBEFORE", (0, 0), (0, -1), 3, colour),
            ]
        )
    )
    story += [KeepTogether(rb), Spacer(1, 6 * mm)]

    # The letter
    by = t["by_muse"] if str(letter.get("author", "")).lower().startswith("muse") else t["by_template"]
    story += [
        Paragraph(t["letter"], st["h2"]),
        Paragraph(letter.get("text", ""), st["letter"]),
        Spacer(1, 1.5 * mm),
        Paragraph(by, st["small"]),
        Spacer(1, 6 * mm),
    ]

    # Quality control
    qc_line = (
        f"<b>{t['qc_' + q['status']]}</b> · {t['coverage']} {q['panel_coverage']:.1%} · {t['depth']} "
        f"{q['median_depth']:.0f} {t['reads']}"
    )
    findings = qc_findings(q, lang) if "codes" in q else []
    story += [Paragraph(t["qc"], st["h2"]), Paragraph(qc_line, st["base"])]
    story += [Paragraph(f"{f[:1].upper()}{f[1:]}.", st["small"]) for f in findings]

    # Page 2: models, follow-up, market
    story += [PageBreak(), Paragraph(t["models"], st["h2"]), Paragraph(t["models_lead"], st["base"]), Spacer(1, 3 * mm)]
    models = analysis.get("models") or {}
    head = [Paragraph(x, st["label"]) for x in (t["col_model"], t["col_signal"], t["col_score"], t["col_tissue"])]
    mrows = [head]
    for key, name in (("classic", t["classic"]), ("meta", t["meta"])):
        m = models.get(key)
        if m is None:
            mrows.append([Paragraph(name, st["cellb"]), Paragraph(t["not_served"], st["cell"]), "", ""])
            continue
        fmt = "{:.4f} / {:.4f}" if key == "classic" else "{:.2f} / {:.2f}"
        top = m["origin"][0]
        mrows.append(
            [
                Paragraph(name, st["cellb"]),
                Paragraph(t["yes"] if m["cancer_signal"] else t["no"], st["cell"]),
                Paragraph(fmt.format(m["score"], m["threshold"]), st["cell"]),
                Paragraph(f"{tissue(top['class'], lang).capitalize()} ({top['probability']:.0%})", st["cell"]),
            ]
        )
    mt = Table(mrows, colWidths=[full * 0.36, full * 0.18, full * 0.2, full * 0.26])
    mt.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#F5F8FF")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story += [mt]
    if models.get("classic") and models.get("meta"):
        same = models["classic"]["origin"][0]["class"] == models["meta"]["origin"][0]["class"]
        story += [Spacer(1, 2 * mm), Paragraph(t["agree"] if same else t["differ"], st["small"])]

    story += [
        Spacer(1, 6 * mm),
        Paragraph(t["followup"], st["h2"]),
        Paragraph(t["followup_many"] if len(followup) > 1 else t["followup_one"], st["base"]),
        Spacer(1, 2 * mm),
    ]
    frows = [[Paragraph(x, st["label"]) for x in (t["col_date"], t["col_ref"], t["col_result"], t["col_tissue"])]]
    for f in followup:
        res = t["qc_failed"] if f["signal"] is None else (t["yes"] if f["signal"] else t["no"])
        frows.append(
            [
                Paragraph(f["at"].replace("T", " ")[:16], st["cell"]),
                Paragraph(f["ref"], st["cell"]),
                Paragraph(res, st["cellb" if f["signal"] else "cell"]),
                Paragraph(tissue(f["tissue"], lang).capitalize() if f["tissue"] else "", st["cell"]),
            ]
        )
    fl = Table(frows, colWidths=[full * 0.24, full * 0.2, full * 0.28, full * 0.28])
    fl.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story += [fl]
    points = [(f["at"][5:10], f["score"], f["threshold"]) for f in reversed(followup) if f["score"] is not None]
    if len(points) > 1:
        story += [Spacer(1, 3 * mm), _trend(points, full)]

    story += [
        Spacer(1, 6 * mm),
        Paragraph(t["market"], st["h2"]),
        Paragraph(t["market_lead"], st["base"]),
        Spacer(1, 2 * mm),
    ]
    krows = [[Paragraph(x, st["label"]) for x in (t["col_test"], t["col_looks"], t["col_status"])]]
    for k in ("m1", "m2", "m3", "m4"):
        name, looks, status = t[k]
        krows.append(
            [
                Paragraph(name, st["cellb" if k == "m4" else "cell"]),
                Paragraph(looks, st["cell"]),
                Paragraph(status, st["cell"]),
            ]
        )
    kt = Table(krows, colWidths=[full * 0.42, full * 0.22, full * 0.36])
    kt.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
                ("BACKGROUND", (0, 4), (-1, 4), PLASMA),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story += [kt]

    doc.build(story, onFirstPage=frame, onLaterPages=frame)
    return buf.getvalue()
