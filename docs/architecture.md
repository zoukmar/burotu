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
| `src/burotu/data` | Download TCGA (GDC) and GEO data, build the targeted panel |
| `src/burotu/simulate` | Cell-free DNA simulator built from real profiles |
| `src/burotu/model` | Features and benchmark classifier (Muse Glimmer adapter next) |
| `src/burotu/llm` | Meta Model API client (OpenAI-compatible), mock mode without a key |
| `src/burotu/labformat.py` | `burotu-bmc` exchange format with the partner lab |
| `src/burotu_web` | burotu.com (one-pager, booking) and admin.burotu.com (tracking) |
| `deploy/` | Production compose file, Caddy block, deploy script |

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
