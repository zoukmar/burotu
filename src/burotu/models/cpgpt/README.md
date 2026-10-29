# CpGPT + Burotu heads

**Role in Burotu: independent cross-check** from a model pre-trained on real methylation data.

- **Family:** [CpGPT](https://www.biorxiv.org/content/10.1101/2024.10.24.619766) (Camillo et al., 2024), a transformer foundation model for DNA methylation pre-trained on more than 150,000 arrays. Each CpG enters as its value plus the Nucleotide Transformer v2 embedding of the 2,001 bases around it. We use the `large` checkpoint (~100M parameters, MIT licence, [lucascamillomd/cpgpt-models](https://huggingface.co/lucascamillomd/cpgpt-models)) frozen.
- **Burotu's part:** [`reader.py`](reader.py) keeps only the panel's 3,148 sites (about 1 GB of memory instead of the package's 324-species tables) and returns one sample embedding; two LightGBM heads trained on those embeddings give the signal and the tissue ([`train_heads.py`](train_heads.py), [`engine.py`](engine.py)).
- **Trained on:** heads on the `fit` cohort embeddings; threshold on the calibration cohort.
- **Serving:** CpGPT runs on its own Modal GPU, started in parallel with Muse Glimmer; the heads run on the tracking server.

**Status: experiment, not used on samples.** Frozen, CpGPT's sample embedding does not keep
the faint signal of a little tumour DNA in blood: it was pre-trained to summarise whole methylation
profiles (age, cell mix), where 1% of foreign DNA barely moves the summary.

| Tumour DNA in plasma (20 tissues) | 0.3% | 1% | 3% | 10% |
| --- | --- | --- | --- | --- |
| Signal found (99.0% specificity observed) | 1% | 4% | 22% | 93% |
| Tissue right first time | 5% | 6% | 7% | 18% |

On real plasma it flags every healthy Moss sample (12/12) and names no tissue right. Report:
[`reports/cpgpt.json`](../../../../reports/cpgpt.json). Next step, if pursued: fine-tune CpGPT
(or read its per-site reconstruction errors) instead of using the frozen summary embedding.
