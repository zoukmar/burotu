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
                       │  1. Data quality check ....... Muse Spark (Meta)        │
                       │  2. Cancer signal ............ dedicated classifier     │
                       │  3. Tissue of origin ......... dedicated classifier     │
                       │  4. Plain-language report .... Muse Spark (Meta)        │
                       └─────────────────────────────────┬──────────────────────┘
                                                         ▼
                                                      Patient
```

Detection relies on a tabular classifier trained on methylation profiles: a language model is
not the right tool to classify thousands of numeric values. Meta's models are used where
language matters: checking that a lab file can be trusted, explaining a result clearly, and
answering follow-up questions.

## Repository components

| Path | Role |
| --- | --- |
| `src/burotu/data` | Download TCGA (GDC) and GEO data, build the targeted panel |
| `src/burotu/simulate` | Cell-free DNA simulator built from real profiles |
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
