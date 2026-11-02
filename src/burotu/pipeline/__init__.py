"""Burotu data and training pipeline: ingest -> refine -> train -> evaluate -> publish.

    python -m burotu.pipeline status            # where each stage stands, read from the store
    python -m burotu.pipeline ingest            # step 1: download raw data (cloud: GDC/GEO modules)
    python -m burotu.pipeline refine            # step 2: raw -> refined/<version> (burotu.pipeline.refine)
    python -m burotu.pipeline train [model...]  # step 3: train the CPU models (Glimmer: Modal workflow)
    python -m burotu.pipeline evaluate          # step 4: frozen benchmark + open real plasma
    python -m burotu.pipeline publish           # step 5: bundle the weights for the site (GitHub Release)

Inference is not a stage: the site loads the published weights (burotu.analysis.service).
"""
