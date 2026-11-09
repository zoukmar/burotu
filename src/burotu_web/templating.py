"""Template engine of the site."""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

from burotu_web.i18n import LANG_NAMES, LANGS

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

templates = Jinja2Templates(directory=HERE / "templates")
# Cookie banner (static/consent.js): Google Analytics only loads after "Accept".
CONSENT = {
    "en": {
        "title": "Cookies",
        "text": "We use Google Analytics to count visits, only if you accept. No advertising.",
        "accept": "Accept",
        "refuse": "Refuse",
        "manage": "Cookies",
    },
    "fr": {
        "title": "Cookies",
        "text": "Nous utilisons Google Analytics pour compter les visites, si vous acceptez. Aucune publicité.",
        "accept": "Accepter",
        "refuse": "Refuser",
        "manage": "Cookies",
    },
    "es": {
        "title": "Cookies",
        "text": "Usamos Google Analytics para contar las visitas, solo si usted acepta. Sin publicidad.",
        "accept": "Aceptar",
        "refuse": "Rechazar",
        "manage": "Cookies",
    },
}


def _consent(lang: str | None) -> dict:
    return CONSENT.get(lang or "en", CONSENT["en"])


templates.env.globals.update(LANGS=LANGS, LANG_NAMES=LANG_NAMES, consent=_consent)
