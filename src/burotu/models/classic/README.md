# Classic model (gradient-boosted trees)

**Role in Burotu: decides the reported signal.** Its threshold is the one calibrated so that
99.5% of healthy people are not flagged.

- **Family:** two LightGBM models: a detector (signal vs no signal) and a tissue classifier (20 tissues).
- **Inputs:** for each of the 3,148 panel sites, a binomial z-score of methylated reads against healthy blood ([`core/features.py`](../../core/features.py)), plus 3 summary features.
- **Trained on:** the `fit` cohort, 14,000 simulated plasma samples (680 TCGA tumours of 20 tissues mixed into the blood of 80 people, tumour DNA 0.03% to 30%). Threshold set on 3,000 samples from 35 other donors.
- **Weights:** [`assets/`](assets/) as LightGBM text files (no pickle), plus `explain.json` for the "why this result" view ([`explain.py`](explain.py)).

| Tumour DNA in plasma | 0.3% | 1% | 3% | 10% |
| --- | --- | --- | --- | --- |
| Signal found (99.9% specificity observed) | 9% | 71% | 97% | 99.6% |
| Tissue right first time | 15% | 55% | 88% | 93% |

Real plasma (Moss 2018, healthy plasma background): 12/15 signals, 0/12 healthy flagged.
Report: [`reports/baseline.json`](../../../../reports/baseline.json).

```bash
python -m burotu.models.classic.train    # trains on the benchmark, writes models/baseline.pkl + report
python -m burotu.models.classic.export   # writes assets/ for the tracking site
```

**Limits:** trained only on simulated plasma; on real plasma from other labs it often names the wrong tissue (prostate read as liver).
