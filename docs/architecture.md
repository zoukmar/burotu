# Architecture

## Overview

```
 Patient ── blood draw ──► Partner collection site ──► Partner lab
                                                         │ targeted bisulfite sequencing
                                                         ▼
                                              burotu-bmc file (read counts per CpG)
                                                         │
                       ┌─────────────────────────────────┴──────────────────────┐
                       │ Burotu                                                  │
                       │  1. Data quality check ....... Muse Spark (API)         │
                       │  2. Cancer signal ............ Muse Glimmer + adapter   │
                       │  3. Tissue of origin ......... Muse Glimmer + adapter   │
                       │  4. Plain-language report .... Muse Spark (API)         │
                       └─────────────────────────────────┬──────────────────────┘
                                                         ▼
                                                      Patient
```

Muse Glimmer, Meta's open-weight model, receives the sample through a small methylation adapter
that projects the per-CpG signal into its token space, and is fine-tuned with LoRA. Numbers are
never fed to it as text. A gradient-boosted classifier runs on the same features as the benchmark
and safety net. Muse Spark, through the Meta Model API, handles the steps where language matters:
checking a lab file, explaining a result clearly, and answering follow-up questions.

## Repository components

| Path | Role |
| --- | --- |
| `src/burotu/core` | `burotu-bmc` lab file format, features (binomial z-scores against healthy blood) |
| `src/burotu/data` | Sources (GDC, GEO, open plasma sets), targeted panel, cell-free DNA simulator |
| `src/burotu/models` | One package per model (classic, logreg, glimmer, cpgpt, consensus) and `registry.toml` |
| `src/burotu/evaluation` | Frozen benchmark, real-plasma checks (Moss 2018, open GEO sets) |
| `src/burotu/analysis` | The pipeline for one sample: QC, every model, consensus, letters, Q&A, PDF |
| `src/burotu/llm` | Meta Model API client (OpenAI-compatible), mock mode without a key |
| `src/burotu_web` | burotu.com (site, booking) and admin.burotu.com (tracking, map of models and data) |
| `scripts/modal_muse.py` | Modal app: train and serve Muse Glimmer, run CpGPT, publish weights |
| `deploy/` | Production compose file, Caddy block, deploy script |

### How a model plugs in

```
lab file ─► core.labformat ─► analysis.service ─┬─► models.classic.engine   (CPU)  ── signal
                                                ├─► models.logreg.engine    (CPU)  ── cross-check
                                                ├─► models.glimmer.engine ─► Modal GPU ── tissue
                                                │        └─ same call ─► CpGPT embedding ─► models.cpgpt.engine
                                                └─► models.consensus ─► letters (Muse Spark), PDF, admin
```

Every engine returns the same `Detection` (signal, score, threshold, top tissues), so the
pipeline, the admin pages and the PDF never depend on how a model works or where it runs. Which
model plays which role, its version and where its weights live are declared once in
`src/burotu/models/registry.toml`.

## Website and tracking

One FastAPI process serves two sites, routed by host name (`starlette.routing.Host`):

- **burotu.com**: one-page site in English, French and Spanish, with blood-draw booking. The
  root URL redirects to the visitor's browser language, and to English when the browser asks for
  a language the site does not offer.
- **admin.burotu.com**: bookings and the journey of each sample (requested → appointment
  confirmed → blood drawn → in the lab → analysed → report sent), with a timestamped history,
  collection sites, opening hours and closures.

Both sites use separate session cookies, and tracking routes do not exist on the public domain.
Forms carry a CSRF token, sign-in is throttled to 8 failures per 15 minutes, HTTP headers are
strict (a CSP with no third-party scripts), and the CSV export is protected against formula
injection.

Storage is SQLite (WAL mode) in a Docker volume. That is enough for the prototype; moving to
PostgreSQL only touches `db.py`.
