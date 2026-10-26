"""Public site burotu.com: one-pager in en/fr/es with built-in booking."""

from __future__ import annotations

import random
import re
from datetime import date

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

from burotu_web.i18n import CANCERS, DEFAULT_LANG, LANGS, TARGET_CANCER_TYPES, T
from burotu_web.security import check_csrf, csrf_token
from burotu_web.templating import templates

router = APIRouter()
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
OPTIONAL_FIELDS = {"doctor"}  # no prescription needed: the doctor is optional


def sequencing_reads(seed: int = 11) -> list[dict]:
    """Cell-free DNA reads drawn in the hero: rows of CpG sites.

    Most come from blood (unmethylated CpGs, hollow circles); a few tumour reads
    see their CpGs fill in during the animation.
    """
    rng = random.Random(seed)
    reads, gap, width_max = [], 26, 1160
    tumour_rows = {2, 5, 8}
    for row in range(10):
        x = 0
        while True:
            n = rng.randint(4, 9)
            width = (n - 1) * gap
            if x + width > width_max:
                break
            tumour = row in tumour_rows and rng.random() < 0.2
            sites = []
            for i in range(n):
                filled = (tumour and rng.random() < 0.8) or (not tumour and rng.random() < 0.04)
                sites.append({"cx": x + i * gap, "filled": filled, "tumour": tumour and filled})
            reads.append({"x1": x, "x2": x + width, "y": row * 34, "sites": sites, "tumour": tumour})
            x += width + rng.randint(40, 90)
    order = 0
    for r in reads:
        for s in r["sites"]:
            if s["tumour"]:
                s["delay"] = round(0.4 + order * 0.05, 2)
                order += 1
    return reads


READS = sequencing_reads()


def pick_lang(request: Request) -> str:
    """Pick the site language from the browser's Accept-Language header.

    Languages are ranked by their q-value (browser preference order breaks ties);
    the first one the site supports wins. English is the default.
    """
    ranked = []
    for pos, part in enumerate(request.headers.get("accept-language", "").lower().split(",")):
        code, _, params = part.strip().partition(";")
        try:
            q = float(params.strip()[2:]) if params.strip().startswith("q=") else 1.0
        except ValueError:
            q = 0.0
        if code and q > 0:
            ranked.append((-q, pos, code[:2]))
    for _, _, code in sorted(ranked):
        if code in LANGS:
            return code
    return DEFAULT_LANG


def need_lang(lang: str) -> str:
    if lang not in LANGS:
        raise HTTPException(404)
    return lang


def render_page(request: Request, lang: str, *, form=None, errors=None, booking=None, status_code=200):
    db = request.app.state.db
    return templates.TemplateResponse(
        request,
        "public/onepager.html",
        {
            "lang": lang,
            "t": T[lang],
            "csrf": csrf_token(request),
            "cancers": CANCERS[lang],
            "target_cancers": TARGET_CANCER_TYPES,
            "reads": READS,
            "sites": db.list_sites(),
            "today": date.today(),
            "form": form or {},
            "errors": errors or [],
            "booking": booking,
        },
        status_code=status_code,
    )


@router.get("/")
def root(request: Request):
    return RedirectResponse(f"/{pick_lang(request)}/", status_code=302)


@router.get("/healthz")
def healthz():
    return {"ok": True}


@router.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return "User-agent: *\nAllow: /\nDisallow: /api/\n"


@router.get("/api/slots")
def api_slots(request: Request, site: int, day: str):
    try:
        d = date.fromisoformat(day)
    except ValueError as e:
        raise HTTPException(400, "invalid date") from e
    return JSONResponse({"slots": request.app.state.db.available_slots(site, d)})


@router.get("/{lang}/", response_class=HTMLResponse)
def home(request: Request, lang: str, booked: str | None = None):
    lang = need_lang(lang)
    booking = None
    # The confirmation is only shown to the session that just booked.
    if booked and request.session.get("last_booking") == booked:
        booking = request.app.state.db.get_booking(booked)
    return render_page(request, lang, booking=booking)


@router.get("/{lang}/book")
def book_anchor(lang: str):
    return RedirectResponse(f"/{need_lang(lang)}/#book", status_code=301)


@router.post("/{lang}/book", response_class=HTMLResponse)
def book_submit(
    request: Request,
    lang: str,
    csrf: str = Form(""),
    site_id: int = Form(0),
    day: str = Form(""),
    slot: str = Form(""),
    first_name: str = Form(""),
    last_name: str = Form(""),
    email: str = Form(""),
    phone: str = Form(""),
    birth_year: str = Form(""),
    doctor: str = Form(""),
    consent: str = Form(""),
    website: str = Form(""),  # honeypot for bots, invisible to humans
):
    lang = need_lang(lang)
    check_csrf(request, csrf)
    t = T[lang]
    form = {
        "site_id": str(site_id) if site_id else "",
        "day": day.strip(),
        "slot": slot.strip(),
        "first_name": first_name.strip()[:120],
        "last_name": last_name.strip()[:120],
        "email": email.strip()[:120],
        "phone": phone.strip()[:40],
        "birth_year": birth_year.strip(),
        "doctor": doctor.strip()[:120],
    }
    errors = []
    if website or not all(v for k, v in form.items() if k not in OPTIONAL_FIELDS):
        errors.append(t["err_required"])
    if form["email"] and not EMAIL_RE.match(form["email"]):
        errors.append(t["err_email"])
    if form["birth_year"] and not (form["birth_year"].isdigit() and 1920 <= int(form["birth_year"]) <= 2008):
        errors.append(t["err_birth"])
    if not consent:
        errors.append(t["err_consent"])
    try:
        if form["day"]:
            date.fromisoformat(form["day"])
    except ValueError:
        errors.append(t["err_slot"])

    if not errors:
        ref = request.app.state.db.create_booking(
            {**form, "site_id": site_id, "birth_year": int(form["birth_year"]), "lang": lang}
        )
        if ref:
            request.session["last_booking"] = ref
            return RedirectResponse(f"/{lang}/?booked={ref}#book", status_code=303)
        errors.append(t["err_slot"])

    return render_page(request, lang, form=form, errors=list(dict.fromkeys(errors)), status_code=422)
