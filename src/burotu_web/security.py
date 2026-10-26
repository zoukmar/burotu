"""Briques de sécurité partagées : jeton CSRF, en-têtes HTTP, limitation des connexions."""

from __future__ import annotations

import hmac
import secrets
import time
from collections import defaultdict

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

CSP = (
    "default-src 'self'; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "script-src 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CSP)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(24)
        request.session["csrf"] = token
    return token


def check_csrf(request: Request, token: str) -> None:
    if not hmac.compare_digest(token or "", request.session.get("csrf", "")):
        raise HTTPException(400, "Form expired. Reload the page and try again.")


class LoginThrottle:
    """Bloque une adresse IP après `limit` échecs sur `window` secondes."""

    def __init__(self, limit: int = 8, window: int = 900) -> None:
        self.limit, self.window = limit, window
        self._fails: dict[str, list[float]] = defaultdict(list)

    def blocked(self, ip: str) -> bool:
        now = time.time()
        self._fails[ip] = [t for t in self._fails[ip] if now - t < self.window]
        return len(self._fails[ip]) >= self.limit

    def fail(self, ip: str) -> None:
        self._fails[ip].append(time.time())

    def reset(self, ip: str) -> None:
        self._fails.pop(ip, None)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"
