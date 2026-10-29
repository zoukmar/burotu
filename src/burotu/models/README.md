# Models

Every model Burotu runs lives in its own package here, with the same layout:

| File | What it holds |
| --- | --- |
| `README.md` | Model card: what it is, what it learned from, results, limits |
| `model.py` | The model itself and how it is trained (when Burotu trains it) |
| `train.py` | Command that trains it on the benchmark and writes its report |
| `engine.py` | What the analysis pipeline calls: `analyse(sample) -> Detection` |
| `assets/` | Portable weights served by the tracking site (when small enough for git) |

[`registry.toml`](registry.toml) lists every model, its version, role, status and where its
weights live (`git:`, `hf:` Hugging Face, `modal:` Modal volume). [`base.py`](base.py) holds
what they share: the `Detection` answer and the panel of 3,148 CpG sites.

| Model | Role | Family | Runs on |
| --- | --- | --- | --- |
| [classic](classic/) | Decides the reported signal | Gradient-boosted trees | Server CPU |
| [logreg](logreg/) | Cross-check | Logistic-regression ensembles (GRAIL-style) | Server CPU |
| [glimmer](glimmer/) | Names the reported tissue | Muse Glimmer 30B + methylation adapter + QLoRA | Modal GPU |
| [cpgpt](cpgpt/) | Experiment (not used on samples) | CpGPT foundation model + Burotu heads | Modal GPU |
| [consensus](consensus/) | Fifth opinion | Vote over the models present | Server |

Muse Spark (letters, Q&A) is a writer, not a model of the result: it lives in
[`burotu/llm`](../llm) and [`burotu/analysis`](../analysis).

Adding a model: create `models/<id>/` with this layout, add an entry to `registry.toml`, and
call its engine from `burotu.analysis.service`. Every model is scored with the same code
([`burotu/evaluation`](../evaluation)) on the same frozen benchmark.
