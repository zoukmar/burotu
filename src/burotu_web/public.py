"""Public site burotu.com: the R&D story of a language model that reads DNA, with the live demo."""

from __future__ import annotations

import json
import random

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse

from burotu_web import home as H
from burotu_web.i18n import DEFAULT_LANG, LANGS, T
from burotu_web.templating import templates

router = APIRouter()
TECH_NAV = {"en": "Technology", "fr": "Technologie", "es": "Tecnología"}


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
                s["delay"] = min(order, MAX_DELAY_STEP)  # CSS class d0..d39 (the CSP forbids inline styles)
                order += 1
    return reads


MAX_DELAY_STEP = 39
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


@router.get("/")
def root(request: Request):
    return RedirectResponse(f"/{pick_lang(request)}/", status_code=302)


@router.get("/healthz")
def healthz():
    return {"ok": True}


@router.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return "User-agent: *\nAllow: /\nDisallow: /api/\n"


@router.get("/demo")
def demo_short(request: Request):
    """burotu.com/demo (shown in the video): the demo lives on the home page."""
    return RedirectResponse(f"/{pick_lang(request)}/#demo", status_code=302)


@router.get("/paper")
def paper():
    """The preprint, as a PDF."""
    return RedirectResponse("/static/burotu-paper.pdf", status_code=302)


@router.get("/{lang}/demo")
def demo_old(lang: str):
    return RedirectResponse(f"/{need_lang(lang)}/#demo", status_code=301)


@router.get("/{lang}/book")
def book_gone(lang: str):
    return RedirectResponse(f"/{need_lang(lang)}/", status_code=301)


@router.get("/{lang}/", response_class=HTMLResponse)
def home(request: Request, lang: str):
    from burotu_web import demo

    lang = need_lang(lang)
    h = H.HOME[lang]
    samples = [{"id": s, "n": s[-2:]} for s in demo.SHOWN if s in demo.manifest()]
    return templates.TemplateResponse(
        request,
        "public/home.html",
        {
            "lang": lang,
            "t": T[lang],
            "h": h,
            "d": demo.TEXT[lang],
            "text_json": json.dumps(demo.TEXT[lang], ensure_ascii=False),
            "samples": samples,
            "runs": H.runs(lang),
            "reads": READS,
            "tech_nav": TECH_NAV[lang],
            "video_url": H.VIDEO_URL,
            "github_url": H.GITHUB_URL,
            "hf_url": H.HF_URL,
            "open_items": [
                (*o, u) for o, u in zip(h["open"], (H.VIDEO_URL, H.GITHUB_URL, H.HF_URL, H.PAPER_URL), strict=True)
            ],
        },
    )


def _tissue(cls: str, lang: str) -> str:
    try:
        from burotu.analysis.wording import tissue
    except ImportError:
        return cls
    return tissue(cls, lang)


@router.get("/{lang}/technology", response_class=HTMLResponse)
def technology(request: Request, lang: str):
    from burotu_web import science

    lang = need_lang(lang)
    return templates.TemplateResponse(
        request,
        "public/technology.html",
        {
            "lang": lang,
            "t": T[lang],
            "h": H.HOME[lang],
            "s": science.TEXT[lang],
            "tech_nav": science.TEXT[lang]["nav"],
            "page": "technology",
            "runs": science.all_runs(),
            "chart": science.origin_chart(),
            "classic": science.run_metrics(science.COMPARE[0]),
            "meta": science.run_metrics(science.COMPARE[1]),
            "current": science.run_metrics("baseline"),
            "tfs": science.TFS,
            "coverage": science.COVERAGE,
            "not_yet": science.NOT_YET,
            "coverage_n": science.coverage_count(),
            "tissue": _tissue,
        },
    )


# Live demo API (/api/demo/...). Its page is now the #demo section of the home page.
from burotu_web.demo import router as _demo_router  # noqa: E402

router.include_router(_demo_router)
