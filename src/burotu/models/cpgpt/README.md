# CpGPT + Burotu heads

**Role in Burotu: independent cross-check** from a model pre-trained on real methylation data.

- **Family:** [CpGPT](https://www.biorxiv.org/content/10.1101/2024.10.24.619766) (Camillo et al., 2024), a transformer foundation model for DNA methylation pre-trained on more than 150,000 arrays. Each CpG enters as its value plus the Nucleotide Transformer v2 embedding of the 2,001 bases around it. We use the `large` checkpoint (~100M parameters, MIT licence, [lucascamillomd/cpgpt-models](https://huggingface.co/lucascamillomd/cpgpt-models)) frozen.
- **Burotu's part:** [`reader.py`](reader.py) keeps only the panel's 3,148 sites (about 1 GB of memory instead of the package's 324-species tables) and returns one sample embedding; two LightGBM heads trained on those embeddings give the signal and the tissue ([`train_heads.py`](train_heads.py), [`engine.py`](engine.py)).
- **Trained on:** heads on the `fit` cohort embeddings; threshold on the calibration cohort.
- **Serving:** CpGPT runs on its own Modal GPU, started in parallel with Muse Glimmer; the heads run on the tracking server.

**Status:** embedding the benchmark on Modal; results will be added here.
