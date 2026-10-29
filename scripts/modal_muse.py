"""Fine-tune Muse Glimmer on Modal GPUs, then serve it to the tracking site.

    modal run scripts/modal_muse.py::main                        # full Muse Glimmer run
    modal run scripts/modal_muse.py::main --steps 30 --eval-stride 20  # quick smoke run
    BUROTU_GPU=H100 modal run scripts/modal_muse.py::main        # pick another GPU

    BUROTU_MUSE_RUN=glimmer modal deploy scripts/modal_muse.py   # serve a trained run

Usually launched from GitHub Actions ("Train Muse" and "Deploy Muse" workflows), which
hold the Modal token. The base model is cached in a Modal volume after the first
download; adapter and LoRA weights and every report are kept in the `burotu-runs` volume.
The served API (burotu.muse.serve) needs the `burotu-muse` Modal secret, holding
BUROTU_MUSE_TOKEN, the same bearer token admin.burotu.com sends.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get("BUROTU_GPU", "L40S")  # 48 GB: enough for Glimmer 30B in 4-bit
RUN = os.environ.get("BUROTU_MUSE_RUN", "glimmer")  # run served by `modal deploy`

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.21.0",
        "peft==0.21.2",
        "accelerate",
        "bitsandbytes",
        "numpy==2.5.3",
        "pandas==3.0.5",
        "pyarrow==26.0.0",
        "scikit-learn==1.9.1",
        "httpx",
        "fastapi==0.162.0",
    )
    .env({"BUROTU_DATA_DIR": "/root/data", "HF_HOME": "/cache/hf", "PYTHONPATH": "/root/src"})
    .add_local_dir(ROOT / "data" / "processed", "/root/data/processed")
    .add_local_dir(ROOT / "data" / "benchmark", "/root/data/benchmark")
    .add_local_dir(ROOT / "src", "/root/src")
)

# CpGPT needs a newer torch than the Glimmer stack, so it gets its own image.
cpgpt_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "numpy==2.5.3",
        "pandas==3.0.5",
        "pyarrow==26.0.0",
        "scikit-learn==1.9.1",
        "huggingface-hub",
        "transformers",
        "lightning",
        "hydra-core",
        "hydra-colorlog",
        "omegaconf",
        "pyfaidx",
        "sqlitedict",
        "rootutils",
        "loguru",
        "schedulefree<2",
        "boto3",
        "torchmetrics",
        "rich",
        "biopython",
        "lifelines",
        "litmodels",
        "litlogger",
        "pyaging",
    )
    .pip_install("cpgpt==0.0.13", extra_options="--no-deps")
    .env({"BUROTU_DATA_DIR": "/root/data", "HF_HOME": "/cache/hf", "PYTHONPATH": "/root/src"})
    .add_local_dir(ROOT / "data" / "processed", "/root/data/processed")
    .add_local_dir(ROOT / "data" / "benchmark", "/root/data/benchmark")
    .add_local_dir(ROOT / "data" / "public", "/root/data/public")
    .add_local_dir(ROOT / "src", "/root/src")
)
CPGPT_GPU = os.environ.get("BUROTU_CPGPT_GPU", "L4")
CPGPT_MODEL = "large"

app = modal.App("burotu-muse", image=image)
runs = modal.Volume.from_name("burotu-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("burotu-hf-cache", create_if_missing=True)


@app.function(volumes={"/cache": hf_cache}, cpu=4, memory=8192, timeout=2 * 3600)
def download(model: str, revision: str) -> str:
    """Fetch the base model into the cache volume on a CPU machine, so GPU time is not spent downloading."""
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]
    path = snapshot_download(model, revision=revision or None, allow_patterns=patterns)
    hf_cache.commit()
    return path


@app.function(gpu=GPU, volumes={"/runs": runs, "/cache": hf_cache}, timeout=6 * 3600, memory=32768)
def train(args: dict) -> dict:
    from burotu.muse.train import TrainConfig, run

    cfg = TrainConfig(**args)
    cfg.out = f"/runs/{cfg.out}"
    try:
        report = run(cfg)
    finally:
        runs.commit()
        hf_cache.commit()
    report["gpu"] = GPU
    return report


@app.function(gpu=GPU, volumes={"/runs": runs, "/cache": hf_cache}, timeout=3600)
def diagnose(name: str) -> dict:
    from pathlib import Path

    from burotu.muse.diagnose import run as diagnose_run_dir

    return diagnose_run_dir(Path(f"/runs/{name}"))


def _cpgpt_reader():
    """Download CpGPT (weights + the human DNA embeddings) once into the cache volume, keep the panel's slice."""
    from huggingface_hub import hf_hub_download, snapshot_download

    from burotu.analysis.engines import ASSETS
    from burotu.cpgpt_reader import DEPS_REPO, MODEL_REPO, CpGPTReader, prepare_panel

    panel = json.loads((ASSETS / "baseline" / "meta.json").read_text())["panel"]
    config = Path(hf_hub_download(MODEL_REPO, f"config/{CPGPT_MODEL}.yaml"))
    ckpt = Path(hf_hub_download(MODEL_REPO, f"weights/{CPGPT_MODEL}.ckpt"))
    assets = Path(f"/runs/cpgpt/panel_{len(panel)}.npz")
    if not assets.exists():
        deps = Path(snapshot_download(DEPS_REPO))
        print(prepare_panel(panel, deps, assets))
        runs.commit()
    hf_cache.commit()
    return CpGPTReader(config, ckpt, assets)


@app.function(
    image=cpgpt_image, gpu=CPGPT_GPU, volumes={"/runs": runs, "/cache": hf_cache}, timeout=4 * 3600, memory=32768
)
def cpgpt_embed_benchmark() -> dict:
    """CpGPT embeddings of every benchmark cohort, Moss real plasma and the public plasma sets."""
    import glob

    import numpy as np
    import pandas as pd

    from burotu import benchmark, realplasma
    from burotu.cpgpt_reader import beta_from_counts
    from burotu.data.reference import Reference

    reader = _cpgpt_reader()
    ref = Reference.load()
    out, meta = {}, {}
    for name in ("fit", "calibration", "test"):
        c = benchmark.load_cohort(name, ref)
        out[name] = reader.embed(beta_from_counts(c.meth, c.total), batch_size=32).astype(np.float16)
        meta[name] = c.meta.to_dict(orient="list")
        print(name, out[name].shape, flush=True)
    b, labels, titles, groups = realplasma.load(ref.panel)
    out["moss"] = reader.embed(b).astype(np.float16)
    meta["moss"] = {"label": list(map(str, labels)), "title": list(map(str, titles)), "group": list(map(str, groups))}
    for f in sorted(glob.glob("/root/data/public/GSE*.parquet")):
        gse = Path(f).stem
        beta = pd.read_parquet(f).reindex(ref.panel)
        out[gse] = reader.embed(beta.to_numpy(np.float32).T).astype(np.float16)
        meta[gse] = {"sample": list(map(str, beta.columns))}
        print(gse, out[gse].shape, flush=True)
    np.savez_compressed("/runs/cpgpt/embeddings.npz", **out)
    Path("/runs/cpgpt/embeddings_meta.json").write_text(json.dumps(meta))
    runs.commit()
    return {k: list(v.shape) for k, v in out.items()}


@app.local_entrypoint()
def cpgpt_benchmark() -> None:
    """modal run scripts/modal_muse.py::cpgpt_benchmark  (then modal volume get burotu-runs cpgpt/...)"""
    print(json.dumps(cpgpt_embed_benchmark.remote(), indent=2))


MODEL_CARD = """---
license: apache-2.0
base_model: meta-models/Muse-Glimmer-30B
tags: [dna-methylation, cfdna, liquid-biopsy, qlora, peft, burotu]
---

# Burotu methylation reader for Muse Glimmer 30B ({run})

Trainable weights that teach Meta's open-weight **Muse Glimmer 30B** to read a cell-free DNA
methylation profile: a methylation adapter (3,148 CpG sites -> 16 soft tokens) and QLoRA weights
(rank 16), trained in two stages on simulated plasma (TCGA tumours mixed into healthy blood).
The base model is not included: load it from `meta-models/Muse-Glimmer-30B` (revision
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, 4-bit nf4) and apply these weights with the code in
[github.com/zoukmar/burotu](https://github.com/zoukmar/burotu) (`burotu.muse`).

**Research prototype, not a medical device.** Trained and tested on simulated data; the tests on
real plasma are small sanity checks.

| Tumour DNA in plasma (20 tissue types, held-out people and tumours) | 0.3% | 1% | 3% | 10% |
| --- | --- | --- | --- | --- |
| Tissue of origin right first time | 15% | 60% | 88% | 92% |

Files: `trainable.pt` (adapter + LoRA), `serving.json` (panel, classes, threshold, adapter
settings), `mu.npy` (healthy background per CpG), `report.json` (full benchmark).
Built for the Meta Global AI Developer Hackathon (2026).
"""


@app.function(volumes={"/runs": runs}, timeout=3600)
def publish_hf(run: str, repo_id: str, token: str) -> str:
    """Upload a trained run's own weights (not the base model) to a Hugging Face model repo."""
    import shutil
    import tempfile

    from huggingface_hub import HfApi

    src = Path(f"/runs/{run}")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        for name in ("trainable.pt", "serving.json", "mu.npy"):
            shutil.copy(src / name, out / name)
        if (src / "report.json").exists():
            shutil.copy(src / "report.json", out / "report.json")
        (out / "README.md").write_text(MODEL_CARD.format(run=run))
        api = HfApi(token=token)
        if "/" not in repo_id:  # only a name, or nothing: publish under the token's own account
            repo_id = f"{api.whoami()['name']}/{repo_id or 'burotu-glimmer'}"
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=repo_id, commit_message=f"Burotu {run}")
    return f"https://huggingface.co/{repo_id}"


@app.local_entrypoint()
def publish(run: str = "glimmer-v5", repo_id: str = "") -> None:
    """HF_TOKEN=... modal run scripts/modal_muse.py::publish --run glimmer-v5 --repo-id user/name"""
    print(publish_hf.remote(run, repo_id, os.environ["HF_TOKEN"]))


@app.local_entrypoint()
def diagnose_run(name: str = "glimmer") -> None:
    """modal run scripts/modal_muse.py::diagnose_run --name glimmer"""
    report = diagnose.remote(name)
    out = ROOT / "reports" / f"muse_{name}_diagnosis.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# The serving class needs the `burotu-muse` secret, so it is only defined when deploying
# (the Deploy Muse workflow sets BUROTU_MUSE_SERVE=1); training runs never need it.
# Inside Modal's containers the module is imported again without that variable, so the class
# must also exist there (modal.is_local() is False in a container).
SERVE = os.environ.get("BUROTU_MUSE_SERVE") == "1" or not modal.is_local()


if SERVE:

    @app.cls(
        gpu=GPU,
        volumes={"/runs": runs, "/cache": hf_cache},
        # The run name travels as a secret: image.env() after add_local_dir would force a rebuild step.
        secrets=[modal.Secret.from_name("burotu-muse"), modal.Secret.from_dict({"BUROTU_MUSE_RUN": RUN})],
        scaledown_window=15 * 60,  # stay warm through a demo, then scale to zero
        timeout=600,
    )
    class Detector:
        @modal.enter()
        def load(self) -> None:
            from burotu.muse.serve import MuseService

            self.service = MuseService(f"/runs/{os.environ['BUROTU_MUSE_RUN']}")

        @modal.asgi_app(label="burotu-muse")
        def web(self):
            from burotu.muse.serve import make_app

            def cpgpt(meth, total):
                """Start CpGPT on its own GPU while Glimmer reads the sample; returns a waiter."""
                call = CpGPTEmbedder().embed.spawn(meth.tolist(), total.tolist())
                return lambda: {"cpgpt_embedding": call.get(timeout=300)}

            return make_app(self.service, os.environ["BUROTU_MUSE_TOKEN"], extra=cpgpt)

    @app.cls(
        image=cpgpt_image,
        gpu=CPGPT_GPU,
        volumes={"/runs": runs, "/cache": hf_cache},
        scaledown_window=15 * 60,
        timeout=600,
    )
    class CpGPTEmbedder:
        @modal.enter()
        def load(self) -> None:
            self.reader = _cpgpt_reader()

        @modal.method()
        def embed(self, meth: list[int], total: list[int]) -> list[float]:
            import numpy as np

            from burotu.cpgpt_reader import beta_from_counts

            beta = beta_from_counts(np.asarray([meth]), np.asarray([total]))
            return self.reader.embed(beta)[0].round(5).tolist()


@app.local_entrypoint()
def main(
    model: str = "meta-models/Muse-Glimmer-30B",
    steps: int = 1500,
    batch_size: int = 16,
    grad_accum: int = 1,
    lora_r: int = 16,
    n_tokens: int = 16,
    eval_stride: int = 1,
    name: str = "glimmer",
    stage1_steps: int = 0,
    min_tf: float = 0.0,
    probe_every: int = 0,
    eval_only: bool = False,
    train_cohort: str = "fit_large",
    weights_file: str = "trainable.pt",
    adapter_hidden: int = 1024,
    adapter_dropout: float = 0.1,
    adapter_input_norm: str = "layer",
    weight_decay: float = 0.01,
    lr_adapter: float = 1e-3,
    lr_lora: float = 2e-4,
    revision: str = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",  # revision pinned in Meta's QLoRA recipe
) -> None:
    args = {
        "model": model,
        "load_4bit": model != "tiny",
        "steps": steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lora_r": lora_r,
        "n_tokens": n_tokens,
        "eval_stride": eval_stride,
        "out": name,
        "stage1_steps": stage1_steps,
        "min_tf": min_tf,
        "probe_every": probe_every,
        "eval_only": eval_only,
        "train_cohort": train_cohort,
        "weights_file": weights_file,
        "adapter_hidden": adapter_hidden,
        "adapter_dropout": adapter_dropout,
        "adapter_input_norm": adapter_input_norm,
        "weight_decay": weight_decay,
        "lr_adapter": lr_adapter,
        "lr_lora": lr_lora,
        "revision": revision if model == "meta-models/Muse-Glimmer-30B" else "",
    }
    if model != "tiny":
        print(f"Caching {model} on a CPU machine (skipped when already cached)...")
        print(download.remote(model, args["revision"]))
    print(f"Training on Modal ({GPU}): {json.dumps(args)}")
    report = train.remote(args)
    out = ROOT / "reports" / f"muse_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out.relative_to(ROOT)}")
    for r in report["test"]["by_tumor_fraction"]:
        print(
            f"TF {r['tumor_fraction']:>7.2%}  sens {r['sensitivity']:.2f}"
            f"  AUC {r['auc']:.3f}  origin top1 {r['origin_top1']:.2f}"
        )
