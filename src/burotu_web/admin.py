"""Tracking site admin.burotu.com: bookings, sample journey, collection sites and hours."""

from __future__ import annotations

import csv
import hmac
import io
import json
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, StreamingResponse

from burotu_web.admin_i18n import ADMIN_LANGS, COOKIE, admin_lang, translator
from burotu_web.db import STATUS_KEYS
from burotu_web.security import LoginThrottle, check_csrf, client_ip, csrf_token
from burotu_web.templating import templates

router = APIRouter()
TIME_RE = re.compile(r"^\d{2}:\d{2}$")


class LoginRequired(Exception):
    """Raised when a tracking page is requested without an admin session."""


def require_admin(request: Request) -> str:
    user = request.session.get("admin")
    if not user:
        raise LoginRequired
    return user


def page(request: Request, name: str, status_code: int = 200, **ctx):
    lang = admin_lang(request)
    return templates.TemplateResponse(
        request,
        f"admin/{name}",
        {
            "csrf": csrf_token(request),
            "user": request.session.get("admin"),
            "alang": lang,
            "ADMIN_LANGS": ADMIN_LANGS,
            "_": translator(lang),
            **ctx,
        },
        status_code=status_code,
    )


@router.get("/lang/{code}")
def set_admin_lang(code: str, next: str = "/"):
    """Remember the tracking language for this browser, then go back where we were."""
    target = next if next.startswith("/") and not next.startswith("//") else "/"
    resp = RedirectResponse(target, status_code=303)
    if code in ADMIN_LANGS:
        resp.set_cookie(COOKIE, code, max_age=365 * 24 * 3600, samesite="lax", httponly=True)
    return resp


@router.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return "User-agent: *\nDisallow: /\n"


@router.get("/healthz")
def healthz():
    return {"ok": True}


# --- Sign-in -------------------------------------------------------------------


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return page(request, "login.html", error=None)


@router.post("/login", response_class=HTMLResponse)
def login(request: Request, csrf: str = Form(""), username: str = Form(""), password: str = Form("")):
    check_csrf(request, csrf)
    settings = request.app.state.settings
    throttle: LoginThrottle = request.app.state.throttle
    ip = client_ip(request)
    if throttle.blocked(ip):
        msg = translator(admin_lang(request))("Too many attempts. Try again in 15 minutes.")
        return page(request, "login.html", 429, error=msg)
    if not settings.admin_password:
        return page(request, "login.html", 503, error="No admin password is configured (BUROTU_ADMIN_PASSWORD).")
    ok = hmac.compare_digest(username, settings.admin_user) & hmac.compare_digest(password, settings.admin_password)
    if not ok:
        throttle.fail(ip)
        return page(request, "login.html", 401, error=translator(admin_lang(request))("Wrong username or password."))
    throttle.reset(ip)
    request.session.clear()  # fresh session: prevents session fixation
    request.session["admin"] = username
    return RedirectResponse("/", status_code=303)


@router.post("/logout")
def logout(request: Request, csrf: str = Form("")):
    check_csrf(request, csrf)
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# --- Bookings and tracking -----------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def bookings(request: Request, status: str = "", q: str = ""):
    require_admin(request)
    db = request.app.state.db
    status = status if status in STATUS_KEYS else ""
    return page(
        request,
        "bookings.html",
        bookings=db.list_bookings(status or None, q.strip() or None),
        stats=db.stats(),
        status=status,
        q=q,
        today=date.today().isoformat(),
    )


@router.get("/bookings/{ref}", response_class=HTMLResponse)
def booking_detail(request: Request, ref: str):
    require_admin(request)
    db = request.app.state.db
    b = db.get_booking(ref)
    if b is None:
        raise HTTPException(404)
    letter = request.query_params.get("letter", b["lang"])
    return page(
        request,
        "booking.html",
        b=b,
        events=db.events(ref),
        analysis=db.latest_analysis(ref),
        followup=db.patient_analyses(b["email"]),
        questions=db.questions(ref),
        letter_lang=letter if letter in ADMIN_LANGS else "en",
        demo_samples=demo_samples(),
        error=request.query_params.get("error"),
        tissue=tissue_name,
        qc_findings=qc_text,
    )


@router.get("/bookings/{ref}/report.pdf")
def booking_report_pdf(request: Request, ref: str, lang: str = ""):
    require_admin(request)
    db = request.app.state.db
    b = db.get_booking(ref)
    analysis = db.latest_analysis(ref) if b is not None else None
    if analysis is None:
        raise HTTPException(404)
    from burotu.analysis.pdf import build_pdf

    lang = lang if lang in ADMIN_LANGS else b["lang"]
    pdf = build_pdf(dict(b), analysis, db.patient_analyses(b["email"]), lang)
    name = f"burotu-{ref}-{lang}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf), media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{name}"'}
    )


@router.get("/models", response_class=HTMLResponse)
def models_page(request: Request):
    require_admin(request)
    import os

    from burotu_web import science

    lang = admin_lang(request)
    try:
        from burotu.analysis.pdf import T as PDF_TEXT

        market = {k: PDF_TEXT[lang][k] for k in ("m1", "m2", "m3", "m4")}
    except ImportError:
        market = {}
    status = {
        "muse": bool(os.environ.get("BUROTU_MUSE_URL") and os.environ.get("BUROTU_MUSE_TOKEN")),
        "role": os.environ.get("BUROTU_MUSE_ROLE", "full"),
        "spark": bool(os.environ.get("MODEL_API_KEY")),
    }
    return page(
        request,
        "models.html",
        s=science.TEXT[lang],
        runs=science.all_runs(),
        chart=science.origin_chart(),
        tfs=science.TFS,
        coverage=science.COVERAGE,
        not_yet=science.NOT_YET,
        coverage_n=science.coverage_count(),
        tissue=tissue_name,
        status=status,
        market=market,
    )


@router.get("/map", response_class=HTMLResponse)
def map_page(request: Request):
    """What Burotu runs, what each model learned from, where it lives, and where the work stands."""
    require_admin(request)
    import json as _json

    from burotu_web import inventory

    db = request.app.state.db
    try:
        public = _json.loads((Path(__file__).resolve().parent / "model_reports" / "public_plasma.json").read_text())
    except OSError:
        public = None
    return page(
        request,
        "map.html",
        inv=inventory,
        models=inventory.models(),
        lang=admin_lang(request),
        public=public,
        counts=db.model_counts(),
        recent=db.recent_analyses(),
        tissue=tissue_name,
    )


def tissue_name(cls: str, lang: str) -> str:
    try:
        from burotu.analysis.wording import tissue
    except ImportError:
        return cls
    return tissue(cls, lang)


def qc_text(qc: dict, lang: str) -> list[str]:
    try:
        from burotu.analysis.wording import qc_findings
    except ImportError:
        return qc.get("issues", [])
    return qc_findings(qc, lang) if "codes" in qc else qc.get("issues", [])


# --- Lab results ----------------------------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parent / "demo_samples"
MAX_LAB_FILE = 2 * 1024 * 1024


@lru_cache(maxsize=1)
def demo_samples() -> list[dict]:
    path = DEMO_DIR / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else []


@router.post("/bookings/{ref}/analyse")
async def booking_analyse(
    request: Request,
    ref: str,
    csrf: str = Form(""),
    demo: str = Form(""),
    lab_file: UploadFile | None = File(None),  # noqa: B008
):
    user = require_admin(request)
    check_csrf(request, csrf)
    db = request.app.state.db
    b = db.get_booking(ref)
    if b is None:
        raise HTTPException(404)

    def fail(msg: str) -> RedirectResponse:
        return RedirectResponse(f"/bookings/{ref}?error={quote(msg)}#lab", status_code=303)

    truth = None
    if lab_file is not None and lab_file.filename:
        raw = await lab_file.read(MAX_LAB_FILE + 1)
        if len(raw) > MAX_LAB_FILE:
            return fail("The lab file is larger than 2 MB.")
        text, name = raw.decode("utf-8", errors="replace"), lab_file.filename
    elif demo:
        match = next((d for d in demo_samples() if d["sample_id"] == demo), None)
        if match is None:
            return fail("Unknown demo sample.")
        text, name, truth = (DEMO_DIR / f"{demo}.tsv").read_text(), f"{demo}.tsv", match
    else:
        return fail("Choose a lab file or a demo sample.")

    try:
        from burotu.analysis.service import analyse
        from burotu.core.labformat import LabFormatError, parse_lab_text
    except ImportError:
        return fail("Analysis is not installed on this server.")
    try:
        sample = parse_lab_text(text, name)
    except LabFormatError as e:
        return fail(f"This file cannot be read: {e}")

    result = await run_in_threadpool(analyse, sample, b["lang"])
    result["ground_truth"] = truth
    detection = result["detection"]
    engine = detection["engine"] if detection else "quality-control"
    db.add_analysis(ref, engine, result, user)

    if detection is None:
        db.set_status(ref, b["status"], "Quality control failed: a new blood draw is needed.", actor=user)
    elif b["status"] not in ("cancelled", "reported"):
        verdict = "Signal detected" if detection["cancer_signal"] else "No signal detected"
        db.set_status(ref, "analysed", f"{verdict} ({detection['engine_label']}).", actor=user)
    return RedirectResponse(f"/bookings/{ref}#lab", status_code=303)


@router.post("/bookings/{ref}/ask")
async def booking_ask(
    request: Request,
    ref: str,
    csrf: str = Form(""),
    question: str = Form(""),
    lang: str = Form("en"),
    audience: str = Form("patient"),
):
    """A question about the latest result, answered by Muse Spark from that result only."""
    user = require_admin(request)
    check_csrf(request, csrf)
    db = request.app.state.db
    analysis = db.latest_analysis(ref)
    question = question.strip()
    if analysis is None or not question:
        return RedirectResponse(f"/bookings/{ref}#ask", status_code=303)
    from burotu.analysis.qa import AUDIENCES, MAX_QUESTION, answer

    lang = lang if lang in ADMIN_LANGS else "en"
    audience = audience if audience in AUDIENCES else "patient"
    result = await run_in_threadpool(answer, analysis, question, lang, audience)
    db.add_question(ref, lang, audience, question[:MAX_QUESTION], result, user)
    return RedirectResponse(f"/bookings/{ref}#ask", status_code=303)


@router.post("/bookings/{ref}/status")
def booking_status(request: Request, ref: str, csrf: str = Form(""), status: str = Form(...), note: str = Form("")):
    user = require_admin(request)
    check_csrf(request, csrf)
    if status not in STATUS_KEYS:
        raise HTTPException(400, "Unknown status")
    try:
        request.app.state.db.set_status(ref, status, note.strip(), actor=user)
    except KeyError as e:
        raise HTTPException(404) from e
    return RedirectResponse(f"/bookings/{ref}", status_code=303)


@router.post("/bookings/{ref}/note")
def booking_note(request: Request, ref: str, csrf: str = Form(""), note: str = Form("")):
    require_admin(request)
    check_csrf(request, csrf)
    request.app.state.db.set_note(ref, note.strip())
    return RedirectResponse(f"/bookings/{ref}", status_code=303)


@router.get("/export.csv")
def export_csv(request: Request):
    require_admin(request)
    cols = [
        "ref", "created_at", "status", "day", "slot", "site_name", "city", "first_name", "last_name",
        "email", "phone", "birth_year", "doctor", "lang", "payment_status", "admin_note",
    ]  # fmt: skip
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in request.app.state.db.list_bookings():
        # Neutralise spreadsheet formulas (CSV injection).
        w.writerow([f"'{r[c]}" if str(r[c])[:1] in "=+-@" else r[c] for c in cols])
    name = f"burotu-reservations-{datetime.now():%Y%m%d}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# --- Collection sites and hours --------------------------------------------------


@router.get("/sites", response_class=HTMLResponse)
def sites(request: Request):
    require_admin(request)
    db = request.app.state.db
    return page(
        request,
        "sites.html",
        sites=db.list_sites(active_only=False),
        closures=db.list_closures(),
        today=date.today().isoformat(),
    )


@router.post("/sites")
def site_add(
    request: Request, csrf: str = Form(""), name: str = Form(...), city: str = Form(...), address: str = Form(...)
):
    require_admin(request)
    check_csrf(request, csrf)
    request.app.state.db.add_site(name.strip(), city.strip(), address.strip())
    return RedirectResponse("/sites", status_code=303)


@router.post("/sites/{site_id}")
def site_update(
    request: Request,
    site_id: int,
    csrf: str = Form(""),
    name: str = Form(...),
    city: str = Form(...),
    address: str = Form(...),
    open_from: str = Form(...),
    open_to: str = Form(...),
    slot_minutes: int = Form(...),
    capacity: int = Form(...),
    weekdays: list[str] = Form([]),  # noqa: B008
    active: str = Form(""),
):
    require_admin(request)
    check_csrf(request, csrf)
    if not (TIME_RE.match(open_from) and TIME_RE.match(open_to)) or open_from >= open_to:
        raise HTTPException(400, "Invalid opening hours")
    request.app.state.db.update_site(
        site_id,
        name=name.strip(),
        city=city.strip(),
        address=address.strip(),
        open_from=open_from,
        open_to=open_to,
        slot_minutes=max(5, min(slot_minutes, 120)),
        capacity=max(1, min(capacity, 20)),
        weekdays=",".join(d for d in weekdays if d in "0123456"),
        active=1 if active else 0,
    )
    return RedirectResponse("/sites", status_code=303)


@router.post("/closures")
def closure(
    request: Request, csrf: str = Form(""), site_id: int = Form(...), day: str = Form(...), action: str = Form("close")
):
    require_admin(request)
    check_csrf(request, csrf)
    try:
        date.fromisoformat(day)
    except ValueError as e:
        raise HTTPException(400, "Invalid date") from e
    request.app.state.db.set_closure(site_id, day, action == "close")
    return RedirectResponse("/sites", status_code=303)
