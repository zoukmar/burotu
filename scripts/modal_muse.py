"""Fine-tune Muse Glimmer on Modal GPUs and bring the benchmark report back.

    modal run scripts/modal_muse.py                              # full Muse Glimmer run
    modal run scripts/modal_muse.py --steps 30 --eval-stride 20  # quick smoke run
    BUROTU_GPU=H100 modal run scripts/modal_muse.py              # pick another GPU

Usually launched from GitHub Actions ("Train Muse" workflow), which holds the Modal token.
The base model is cached in a Modal volume after the first download; adapter and LoRA
weights and every report are kept in the `burotu-runs` volume.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get("BUROTU_GPU", "L40S")  # 48 GB: enough for Glimmer 30B in 4-bit

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
    )
    .env({"BUROTU_DATA_DIR": "/root/data", "HF_HOME": "/cache/hf", "PYTHONPATH": "/root/src"})
    .add_local_dir(ROOT / "data" / "processed", "/root/data/processed")
    .add_local_dir(ROOT / "data" / "benchmark", "/root/data/benchmark")
    .add_local_dir(ROOT / "src", "/root/src")
)

app = modal.App("burotu-muse", image=image)
runs = modal.Volume.from_name("burotu-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("burotu-hf-cache", create_if_missing=True)


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
    }
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
