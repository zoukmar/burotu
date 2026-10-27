"""Fine-tune a Meta model to read methylation, then score it on the frozen benchmark.

    # CPU smoke run with a tiny random Glimmer (seconds, no download)
    python -m burotu.muse.train --model tiny --steps 30 --out runs/tiny

    # The real run (on a GPU, see scripts/modal_muse.py)
    python -m burotu.muse.train --model meta-models/Muse-Glimmer-30B --4bit --steps 1500 --out runs/glimmer

Trains the methylation adapter and LoRA weights together on the `fit_large` cohort,
sets the detection threshold on the `calibration` cohort and reports metrics on the
`test` cohort: the exact samples and protocol used for the baseline classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from burotu import benchmark
from burotu.config import HEALTHY, ROOT
from burotu.data.reference import Reference
from burotu.model.features import Featurizer
from burotu.muse.reader import MuseReader

# Linear layers of the language model only, as in Meta's official Glimmer QLoRA recipe
# (meta-oss-cookbook/muse-glimmer/post-training): the vision tower stays frozen.
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


@dataclass
class TrainConfig:
    model: str = "tiny"
    load_4bit: bool = False
    steps: int = 1500
    batch_size: int = 16
    grad_accum: int = 1
    lr_adapter: float = 1e-3
    lr_lora: float = 2e-4  # Meta's Glimmer QLoRA recipe uses 2e-4
    lora_r: int = 16
    n_tokens: int = 16
    warmup: int = 50
    eval_batch: int = 32
    train_cohort: str = "fit_large"
    revision: str = ""  # pin a model revision for reproducible runs
    eval_stride: int = 1  # score every n-th calibration/test sample (smoke runs use > 1)
    seed: int = 0
    out: str = "runs/latest"


# --- Model loading ------------------------------------------------------------------


def load_lm(cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    if cfg.model == "tiny":
        from burotu.muse.tiny import tiny_glimmer

        lm, tokenizer = tiny_glimmer()
        target = LORA_TARGETS
    else:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision or None)
        kwargs: dict = {"dtype": torch.bfloat16, "device_map": {"": 0} if torch.cuda.is_available() else None}
        if cfg.revision:
            kwargs["revision"] = cfg.revision
        if cfg.load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,  # required to reload QLoRA adapters (Meta's guide)
            )
        multimodal = hasattr(AutoConfig.from_pretrained(cfg.model, revision=cfg.revision or None), "text_config")
        loader = AutoModelForMultimodalLM if multimodal else AutoModelForCausalLM
        lm = loader.from_pretrained(cfg.model, **kwargs)
        target = LORA_TARGETS if multimodal else r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        if cfg.load_4bit:
            from peft import prepare_model_for_kbit_training

            lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=True)

    for p in lm.parameters():
        p.requires_grad_(False)
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, lora_dropout=0.05, target_modules=target, bias="none")
    return get_peft_model(lm, lora), tokenizer


# --- Data -----------------------------------------------------------------------------


def features_of(cohort: benchmark.Cohort, featurizer: Featurizer) -> torch.Tensor:
    return torch.from_numpy(featurizer.transform(cohort.meth, cohort.total))


def subsample(cohort: benchmark.Cohort, stride: int) -> benchmark.Cohort:
    if stride <= 1:
        return cohort
    idx = np.arange(0, len(cohort), stride)
    return benchmark.Cohort(cohort.meth[idx], cohort.total[idx], cohort.meta.iloc[idx].reset_index(drop=True))


def batched(n: int, size: int):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


# --- Training ---------------------------------------------------------------------------


def train(reader: MuseReader, x: torch.Tensor, labels: np.ndarray, cfg: TrainConfig, log) -> None:
    lora_params = [p for p in reader.lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": reader.adapter.parameters(), "lr": cfg.lr_adapter},
            {"params": lora_params, "lr": cfg.lr_lora},
        ],
        weight_decay=0.01,
    )
    base_lrs = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(cfg.seed)
    use_cuda = torch.cuda.is_available()
    reader.train()
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        # Linear warm-up, then cosine decay.
        f = step / cfg.warmup if step <= cfg.warmup else 0.5 * (1 + math.cos(math.pi * step / cfg.steps))
        for g, lr in zip(opt.param_groups, base_lrs, strict=True):
            g["lr"] = lr * f
        total = 0.0
        for _ in range(cfg.grad_accum):
            idx = rng.integers(0, len(x), cfg.batch_size)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                loss = reader.loss(x[idx], labels[idx]) / cfg.grad_accum
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_([*reader.adapter.parameters(), *lora_params], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 1 or step % 25 == 0 or step == cfg.steps:
            log({"step": step, "loss": round(total, 4), "lr_scale": round(f, 3), "elapsed_s": round(time.time() - t0)})
    reader.eval()


# --- Evaluation -------------------------------------------------------------------------


@torch.no_grad()
def score(reader: MuseReader, x: torch.Tensor, labels: np.ndarray, batch: int, with_origin: bool):
    use_cuda = torch.cuda.is_available()
    scores = []
    for s in batched(len(x), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            scores.append(reader.cancer_score(x[s]))
    score_arr = torch.cat(scores).numpy()
    origin = np.full((len(x), len(reader.classes)), 1 / len(reader.classes), dtype=np.float32)
    if with_origin:
        cancer_idx = np.flatnonzero(labels != HEALTHY)  # origin only matters for cancer samples
        for s in batched(len(cancer_idx), batch):
            rows = cancer_idx[s]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                origin[rows] = reader.origin_proba(x[rows]).numpy()
    return score_arr, origin


def run(cfg: TrainConfig) -> dict:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"

    def log(entry: dict) -> None:
        print(json.dumps(entry), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    torch.manual_seed(cfg.seed)
    t0 = time.time()
    ref = Reference.load()
    featurizer = Featurizer.from_blood(ref.blood_train.to_numpy(np.float32))
    fit = benchmark.load_cohort(cfg.train_cohort, ref)
    calib = subsample(benchmark.load_cohort("calibration", ref), cfg.eval_stride)
    test = subsample(benchmark.load_cohort("test", ref), cfg.eval_stride)
    classes = sorted(set(fit.labels) - {HEALTHY})
    log({"event": "data", "fit": len(fit), "calibration": len(calib), "test": len(test), "classes": classes})

    lm, tokenizer = load_lm(cfg)
    in_dim = len(ref.panel) + 3
    reader = MuseReader(lm, tokenizer, in_dim, classes, n_tokens=cfg.n_tokens)
    trainable = sum(p.numel() for p in reader.parameters() if p.requires_grad)
    log({"event": "model", "model": cfg.model, "trainable_params": trainable, "load_s": round(time.time() - t0)})

    train(reader, features_of(fit, featurizer), fit.labels, cfg, log)
    reader.save_trainable(out / "trainable.pt")

    c_score, _ = score(reader, features_of(calib, featurizer), calib.labels, cfg.eval_batch, with_origin=False)
    threshold = benchmark.threshold_at_specificity(c_score)
    t_score, t_origin = score(reader, features_of(test, featurizer), test.labels, cfg.eval_batch, with_origin=True)
    report = {
        "model": cfg.model,
        "config": asdict(cfg),
        "trainable_params": trainable,
        "train_samples": len(fit),
        "test": benchmark.evaluate(t_score, t_origin, classes, threshold, test.meta),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    log({"event": "done", "minutes": report["minutes"], "specificity": report["test"]["specificity_observed"]})
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    for name, field in TrainConfig.__dataclass_fields__.items():
        flag = "--4bit" if name == "load_4bit" else f"--{name.replace('_', '-')}"
        if field.type in ("bool", bool):
            ap.add_argument(flag, dest=name, action="store_true")
        else:
            typ = {"int": int, "float": float, "str": str}[str(field.type)]
            ap.add_argument(flag, dest=name, type=typ, default=field.default)
    cfg = TrainConfig(**vars(ap.parse_args()))
    if cfg.out == TrainConfig.out:
        cfg.out = str(ROOT / "runs" / cfg.model.replace("/", "_"))
    report = run(cfg)
    for r in report["test"]["by_tumor_fraction"]:
        print(
            f"TF {r['tumor_fraction']:>7.2%}  sens {r['sensitivity']:.2f}"
            f"  AUC {r['auc']:.3f}  origin top1 {r['origin_top1']:.2f}"
        )


if __name__ == "__main__":
    main()
