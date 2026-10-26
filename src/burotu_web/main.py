"""Entry point: one process serves two sites, routed by host name.

    burotu.com        → public site (one-pager, booking)
    admin.burotu.com  → tracking (bookings, samples, collection sites)

Locally: http://localhost:8000 and http://admin.localhost:8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.routing import Host, Mount

from burotu_web import admin, public
from burotu_web.db import Database
from burotu_web.security import LoginThrottle, SecurityHeadersMiddleware
from burotu_web.settings import Settings, get_settings
from burotu_web.templating import STATIC_DIR


def _subapp(settings: Settings, db: Database, router, cookie: str) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.db = db
    app.state.throttle = LoginThrottle()  # one failure counter per app instance
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie=cookie,
        same_site="lax",
        https_only=settings.secure_cookies,
        max_age=8 * 3600,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(router)
    return app


def create_app(settings: Settings | None = None) -> Starlette:
    settings = settings or get_settings()
    db = Database(settings.db_path)

    public_app = _subapp(settings, db, public.router, "burotu_session")
    admin_app = _subapp(settings, db, admin.router, "burotu_admin")

    @admin_app.exception_handler(admin.LoginRequired)
    async def _to_login(_request: Request, _exc: admin.LoginRequired):
        return RedirectResponse("/login", status_code=303)

    @asynccontextmanager
    async def lifespan(_app):
        db.init()
        yield

    routes = [Host(settings.admin_host, admin_app)]
    routes += [Host(h, public_app) for h in settings.public_hosts]
    routes.append(Mount("/", public_app))  # unknown host: the reverse proxy already filters domains
    return Starlette(routes=routes, middleware=[Middleware(SecurityHeadersMiddleware)], lifespan=lifespan)


app = create_app()
