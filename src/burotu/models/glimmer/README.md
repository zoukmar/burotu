# Muse Glimmer v6 (Meta open weights)

**Role in Burotu: reads the sample, signal and tissue of origin.**

- **Family:** Meta's open-weight [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Apache-2.0), loaded in 4-bit, with a methylation adapter that turns the features of 3,148 CpG sites into 16 soft tokens ([`reader.py`](reader.py)), and QLoRA (rank 16): 217M trained parameters. The model answers yes/no for the signal and names the tissue; scores are read from the logits.
- **Trained on:** refined data v2, `fit_mix` cohort: 35,000 simulated samples from 3,683 blood donors (8 cohorts aligned on one batch) and tumours of 23 tissue classes, half the healthy backgrounds mixed from 2-3 donors (pseudo-donors). Two stages: adapter alone (500 steps), then adapter + LoRA (1,500 steps in all). 66 minutes on one L40S.
- **Weights:** [huggingface.co/zoukmar75/burotu-glimmer](https://huggingface.co/zoukmar75/burotu-glimmer) (ours only, not the base model) and the `burotu-runs` Modal volume.
- **Serving:** [`serve.py`](serve.py) on a Modal GPU behind a bearer token ([`scripts/modal_muse.py`](../../../../scripts/modal_muse.py)); the site and its live demo call it through [`engine.py`](engine.py).
- **Paper:** [`paper/burotu-paper.pdf`](../../../../paper/burotu-paper.pdf).

| Tumour DNA in plasma (23 tissues, held-out people and tumours) | 0.3% | 1% | 3% | 10% |
| --- | --- | --- | --- | --- |
| Signal found (99.5% specificity) | 8% | **75%** | 92% | 95% |
| Tissue right first time (chance 4%) | 15% | **58%** (classic 46%) | 85% | 92% |

Real plasma (Moss 2018, healthy-control background): 12 of 15 cancers found but 5 of 12 healthy people flagged, so its alarm needs recalibration on real plasma; 7 of 11 tissues right. Report: [`reports/muse_glimmer-v6.json`](../../../../reports/muse_glimmer-v6.json); earlier runs: `muse_glimmer-v1.json` to `-v5.json`.

```bash
python -m burotu.models.glimmer.train --model tiny --steps 30   # CPU smoke test
# Full run: GitHub Actions → Train Muse (Modal GPU); serving: Actions → Deploy Muse
```
