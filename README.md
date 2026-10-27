<p align="center">
  <img src="docs/assets/onepager.png" alt="burotu.com" width="760">
</p>

<h1 align="center">Burotu</h1>

<p align="center">
  <strong>50 cancers. One blood draw.</strong><br>
  Teaching Meta's Muse models to read the DNA that tumours leave in our blood.
</p>

<p align="center">
  <a href="https://burotu.com">Live demo</a> ·
  <a href="#how-meta-models-power-burotu">Meta models</a> ·
  <a href="#muse-glimmer-learns-to-read-methylation">Muse Glimmer adapter</a> ·
  <a href="#benchmark">Benchmark</a> ·
  <a href="#quick-start">Quick start</a>
</p>

> [!IMPORTANT]
> **Demonstration prototype** built for the Meta Global AI Developer Hackathon 2026 (indie track,
> *Push the Frontier*). Burotu is not a medical device, is not clinically validated, has no CE
> marking and must not be used to make any medical decision. Nothing is sold: payments run in test
> mode only.

## The idea

Most cancers are found late, once symptoms appear. Screening exists for only a handful of them,
and each one needs its own exam. Yet tumours shed fragments of their DNA into the bloodstream from
early on, and that DNA carries a chemical signature, **methylation**, that reveals both *that* it
is cancerous and *which tissue* it comes from.

Language models already read text, code and images. Burotu asks a simple question: **can a Meta
model learn to read methylation too?** We give Meta's open-weight **Muse Glimmer** a new input, a
blood sample, and teach it to answer two questions: is there a cancer signal, and where does it
most likely come from? **Muse Spark** then turns that answer into something a patient can
understand.

## How Meta models power Burotu

```
Blood draw ─► Partner lab ─► burotu-bmc file (read counts on 1,681 CpG sites)
                                   │
                                   ▼
                   ┌───────────────────────────────────────┐
                   │ Muse Spark · quality control          │  Is this lab file trustworthy?
                   └───────────────────┬───────────────────┘
                                       ▼
                   ┌───────────────────────────────────────┐
                   │ Muse Glimmer + methylation adapter    │  Cancer signal? Tissue of origin?
                   │ (open weights, fine-tuned with LoRA)  │
                   └───────────────────┬───────────────────┘
                                       ▼
                   ┌───────────────────────────────────────┐
                   │ Muse Spark · report and Q&A           │  Plain language, EN / FR / ES
                   └───────────────────────────────────────┘
```

| Step | Meta model | What it does | Status |
| --- | --- | --- | --- |
| Read the sample | **Muse Glimmer** (open weights, 30B) + methylation adapter + LoRA | Detects a cancer signal and ranks the likely tissues of origin | Training code ready, GPU run pending |
| Check lab data | **Muse Spark** (Meta Model API) | Reads coverage and lab metadata, flags files that cannot be trusted and says what to rerun | Client ready |
| Explain the result | **Muse Spark** | Writes a careful report and answers follow-up questions, grounded in the model's output | Client ready |

A gradient-boosted classifier runs alongside as the **benchmark and safety net**: every claim about
Muse is measured against it on the same held-out data.

## Muse Glimmer learns to read methylation

Feeding 1,681 numbers to a language model as text does not work well: the model reads digits, not
biology. Burotu follows the recipe that taught language models to see images instead.

```
read counts ─► binomial z-scores ─► methylation adapter (MLP) ─► 16 soft tokens
  (1,681 CpG)    vs healthy blood                                      │
                                                                       ▼
                              "Cancer signal: yes. Most likely origin: colorectal (0.71) ..."
                                                     ▲
                              Muse Glimmer 30B, 4-bit, LoRA adapters ──┘
```

- **Adapter**: a small network projects the sample into a handful of embeddings that live in
  Glimmer's token space, the same idea as vision projectors in multimodal models.
- **Fine-tuning**: QLoRA on Glimmer (about 18 GB in 4-bit, one 80 GB GPU), adapter and LoRA
  weights trained together.
- **Training data**: tens of thousands of plasma samples simulated from the training split only.
- **Evaluation**: the exact held-out people, tumours and threshold protocol used for the
  benchmark below.
- **Compute**: Modal serverless GPUs, through the hackathon's partner credits.

If it works, the recipe is reusable: any open-weight Meta model could learn to read other omics
data the same way.

## Benchmark

The bar Muse Glimmer has to clear. Simulated plasma built only from **held-out people and
tumours**; the detection threshold is set on blood donors no model ever saw.

<p align="center"><img src="docs/assets/baseline_sensitivity.png" alt="Sensitivity by tumour fraction" width="640"></p>

| Tumour fraction in plasma | 0.3% | 1% | 3% | 10% |
| --- | --- | --- | --- | --- |
| Cancer signal detected, baseline | 7% | 71% | 99.7% | 100% |
| Tissue of origin top-1, baseline | 25% | 57% | 87% | 96% |
| Tissue of origin top-2, baseline | 41% | 74% | 95% | 99.7% |
| Muse Glimmer + adapter | *pending* | *pending* | *pending* | *pending* |

Specificity measured on unseen healthy donors: **99.8%** (target 99.5%).

What these numbers do **not** say:
- The plasma is simulated CpG by CpG; real tests also use read-level patterns.
- On real patient plasma (Moss et al., measured on arrays rather than sequencing), the platform
  shift is not corrected yet: every sample crosses the threshold and the healthy/cancer AUC is
  only 0.62. Correcting it is on the roadmap.

## Data

All public, no patient data collected.

| Role | Dataset |
| --- | --- |
| Tumour methylation, 8 types × 48 tumours | TCGA via the GDC portal (open access, Illumina 450K) |
| Healthy blood background, 164 adults | Hannum et al., *Mol Cell* 2013, GEO GSE40279 |
| Real plasma for validation | Moss et al., *Nat Commun* 2018, GEO GSE122126 |

Plasma is simulated by mixing real tumour profiles into real healthy blood at tumour fractions
from 0.01% to 10%, then sampling sequencing reads. People and tumours are split into training and
test sets **before** simulation and panel selection. Details in [docs/data.md](docs/data.md).

**Coverage.** The product targets 50 cancer types. The prototype starts with **8**: lung,
colorectal, breast, pancreas, liver, oesophagus and stomach, head and neck, prostate.

## The product

A model alone does not reach patients, so Burotu ships as a product from day one:

- **[burotu.com](https://burotu.com)**: one-page site in English, French and Spanish (picked from
  the browser language), with booking at partner collection sites.
- **admin.burotu.com**: tracking for every sample, from request to blood draw, lab, analysis and
  report, with a timestamped history. Drop the lab's read-count file on a sample and get, in
  seconds: quality control, cancer signal, likely tissue of origin, and a plain-language report
  for the patient written by Muse Spark in their language. Detection comes from the fine-tuned
  Muse Glimmer when it is served, from the benchmark classifier otherwise; the page says which.

<p align="center"><img src="docs/assets/tracking.png" alt="Sample tracking" width="640"></p>

## Quick start

```bash
make install        # Python 3.11+
make check          # lint + tests, same as CI
make dev            # http://localhost:8000 (site) · http://admin.localhost:8000 (tracking, password "dev")
```

Data and benchmark (about 5 GB downloaded from GDC and GEO):

```bash
make data           # TCGA tumours, whole blood, real plasma
make reference      # targeted panel and train/test split
make baseline       # benchmark classifier and metrics (about 5 min on CPU)
make demo-samples   # simulated lab files for the demo
```

Fine-tuning Muse Glimmer runs on Modal GPUs, launched from GitHub: **Actions → Train Muse → Run
workflow** (needs the `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` repository secrets). The same
training code runs on CPU with a tiny random Glimmer, for tests and smoke runs:

```bash
pip install -e ".[dev,muse]"
python -m burotu.muse.train --model tiny --steps 30 --eval-stride 20
```

Once trained, **Actions → Deploy Muse** serves the run on a Modal GPU behind an authenticated API
([serve.py](src/burotu/muse/serve.py), scales to zero when idle; needs a `BUROTU_MUSE_TOKEN`
secret). Setting `BUROTU_MUSE_URL` and `BUROTU_MUSE_TOKEN` on the server switches the tracking
site to Muse Glimmer.

Every model is scored on the same frozen benchmark (`data/benchmark/`): the calibration cohort
sets the threshold, the test cohort gives the numbers.

Set `MODEL_API_KEY` to call Muse Spark through the Meta Model API. Without it, the
[client](src/burotu/llm/client.py) runs in mock mode so everything works offline. `make help`
lists every command.

## Repository

```
src/burotu/        pipeline: data, panel, cfDNA simulator, benchmark, models, Muse reader, Meta client
src/burotu_web/    burotu.com (one-pager) and admin.burotu.com (tracking)
tests/             pipeline, models, site, booking, tracking, security
deploy/            production compose, Caddy block, deploy script
docs/              architecture, data and method, deployment
.github/workflows  CI (lint, tests, image) and continuous deployment
```

Every push to `main` runs the tests, publishes `ghcr.io/zoukmar/burotu-web:<sha>` and deploys it,
with automatic rollback if the site does not answer. See [docs/deployment.md](docs/deployment.md).

## Roadmap

- [x] Public data pipeline, targeted panel, cfDNA simulator
- [x] burotu.com and sample tracking, continuous deployment
- [x] Benchmark classifier on held-out simulated plasma
- [x] Methylation adapter and QLoRA training code, tested end to end on a tiny Glimmer
- [x] Lab-file analysis in the tracking site, with a GPU serving API for the fine-tuned model
- [ ] Full QLoRA fine-tuning of Muse Glimmer 30B on Modal
- [ ] Muse Glimmer vs benchmark on the same held-out data
- [ ] Muse Spark quality control, patient report and Q&A
- [ ] Platform correction and validation on real plasma
- [ ] From 8 to 50 cancer types
