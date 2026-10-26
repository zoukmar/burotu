"""Template engine shared by the public site and tracking."""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

from burotu_web.db import PIPELINE, STATUS_LABELS, STATUSES
from burotu_web.i18n import LANG_NAMES, LANGS

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

templates = Jinja2Templates(directory=HERE / "templates")
templates.env.globals.update(
    LANGS=LANGS,
    LANG_NAMES=LANG_NAMES,
    STATUSES=STATUSES,
    STATUS_LABELS=STATUS_LABELS,
    PIPELINE=PIPELINE,
)


def euros(cents: int) -> str:
    return f"€{cents / 100:,.0f}"


templates.env.filters["euros"] = euros
