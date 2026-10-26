"""Tracking site admin.burotu.com: bookings, sample journey, collection sites and hours."""

from __future__ import annotations

import csv
import hmac
import io
import re
from datetime import date, datetime

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, StreamingResponse

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
    return templates.TemplateResponse(
        request,
        f"admin/{name}",
        {"csrf": csrf_token(request), "user": request.session.get("admin"), **ctx},
        status_code=status_code,
    )


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
        return page(request, "login.html", 429, error="Too many attempts. Try again in 15 minutes.")
    if not settings.admin_password:
        return page(request, "login.html", 503, error="No admin password is configured (BUROTU_ADMIN_PASSWORD).")
    ok = hmac.compare_digest(username, settings.admin_user) & hmac.compare_digest(password, settings.admin_password)
    if not ok:
        throttle.fail(ip)
        return page(request, "login.html", 401, error="Wrong username or password.")
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
    return page(request, "booking.html", b=b, events=db.events(ref))


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
