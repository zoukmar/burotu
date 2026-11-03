"""Template engine of the site."""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

from burotu_web.i18n import LANG_NAMES, LANGS

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

templates = Jinja2Templates(directory=HERE / "templates")
templates.env.globals.update(LANGS=LANGS, LANG_NAMES=LANG_NAMES)
