"""Entry point for burotu.com: the R&D page of a language model that reads DNA, with its live demo.

The old tracking site (admin.burotu.com) is retired: its host now redirects to burotu.com.

Locally: http://localhost:8000
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Host, Mount, Route

from burotu_web import public
from burotu_web.security import SecurityHeadersMiddleware
from burotu_web.settings import Settings, get_settings
from burotu_web.templating import STATIC_DIR


def _public_app(settings: Settings) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(public.router)
    return app


async def _to_public(_request: Request):
    return RedirectResponse("https://burotu.com/", status_code=301)


def create_app(settings: Settings | None = None) -> Starlette:
    settings = settings or get_settings()
    public_app = _public_app(settings)
    retired = Starlette(routes=[Route("/{path:path}", _to_public)])
    routes = [Host(settings.admin_host, retired)]
    routes += [Host(h, public_app) for h in settings.public_hosts]
    routes.append(Mount("/", public_app))  # unknown host: the reverse proxy already filters domains
    return Starlette(routes=routes, middleware=[Middleware(SecurityHeadersMiddleware)])


app = create_app()
