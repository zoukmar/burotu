# Logistic regressions (GRAIL-style)

**Role in Burotu: independent cross-check** of the signal and the tissue.

- **Family:** two ensembles of 8 L2-regularised logistic regressions on bootstrap samples, averaged: the approach GRAIL published for its multi-cancer test (Liu et al., *Annals of Oncology* 2020). A different family from the classic model, so the two make different mistakes.
- **Inputs:** the same z-scores as the classic model, clipped and standardised. Score = mean log-odds.
- **Trained on:** the same `fit` cohort (14,000 simulated samples).
- **Weights:** [`assets/`](assets/) as plain arrays (`.npz` + `.json`): numpy only at analysis time.

| Tumour DNA in plasma | 0.3% | 1% | 3% | 10% |
| --- | --- | --- | --- | --- |
| Signal found (98.3% specificity observed) | 9% | 57% | 97% | 100% |
| Tissue right first time | 16% | 54% | 88% | 93% |

Real plasma from other labs ([`evaluation/public_plasma.py`](../../evaluation/public_plasma.py)): prostate recognised 163 times of 181, lung 60 of 185.

```bash
python -m burotu.models.logreg.train
```
