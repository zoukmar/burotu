# Muse Glimmer v5 (Meta open weights)

**Role in Burotu: names the reported tissue.**

- **Family:** Meta's open-weight [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Apache-2.0), loaded in 4-bit, with a methylation adapter that turns the 3,154 features into 16 soft tokens ([`reader.py`](reader.py)), and QLoRA (rank 16): 217M trained parameters. The model answers yes/no for the signal and names the tissue; scores are read from the logits.
- **Trained on:** the `fit_mix` cohort, 32,000 simulated samples, half the healthy backgrounds mixed from 2-3 real donors (pseudo-donors). Two stages: adapter alone (500 steps), then adapter + LoRA. 65 minutes on one L40S.
- **Weights:** [huggingface.co/zoukmar75/burotu-glimmer](https://huggingface.co/zoukmar75/burotu-glimmer) (ours only, not the base model) and the `burotu-runs` Modal volume.
- **Serving:** [`serve.py`](serve.py) on a Modal GPU behind a bearer token ([`scripts/modal_muse.py`](../../../../scripts/modal_muse.py)); the tracking site calls it through [`engine.py`](engine.py).

| Tumour DNA in plasma (20 tissues) | 0.3% | 1% | 3% | 10% |
| --- | --- | --- | --- | --- |
| Tissue right first time | 15% | **60%** (classic 55%) | 88% | 92% |
| Signal found | 9% | 38% | 42% | 37% |

It names tissues well but does not hold 99.5% specificity on new people (it also learned each donor's blood background), so the classic model keeps the signal. Real plasma: 7 of 11 tissues right. Report: [`reports/muse_glimmer-v5.json`](../../../../reports/muse_glimmer-v5.json).

```bash
python -m burotu.models.glimmer.train --model tiny --steps 30   # CPU smoke test
# Full run: GitHub Actions → Train Muse (Modal GPU); serving: Actions → Deploy Muse
```
