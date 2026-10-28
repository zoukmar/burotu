"""Fine-tune Muse Glimmer on Modal GPUs, then serve it to the tracking site.

    modal run scripts/modal_muse.py                              # full Muse Glimmer run
    modal run scripts/modal_muse.py --steps 30 --eval-stride 20  # quick smoke run
    BUROTU_GPU=H100 modal run scripts/modal_muse.py              # pick another GPU

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


@app.function(gpu=GPU, volumes={"/runs": runs, "/cache": hf_cache}, timeout=6 * 3600)
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
SERVE = os.environ.get("BUROTU_MUSE_SERVE") == "1"


if SERVE:

    @app.cls(
        gpu=GPU,
        image=image.env({"BUROTU_MUSE_RUN": RUN}),
        volumes={"/runs": runs, "/cache": hf_cache},
        secrets=[modal.Secret.from_name("burotu-muse")],
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

            return make_app(self.service, os.environ["BUROTU_MUSE_TOKEN"])


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
