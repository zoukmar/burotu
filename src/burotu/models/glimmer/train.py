"""Fine-tune a Meta model to read methylation, then score it on the frozen benchmark.

    # CPU smoke run with a tiny random Glimmer (seconds, no download)
    python -m burotu.models.glimmer.train --model tiny --steps 30 --out runs/tiny

    # The real run (on a GPU, see scripts/modal_muse.py)
    python -m burotu.models.glimmer.train --model meta-models/Muse-Glimmer-30B --4bit --steps 1500 --out runs/glimmer

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

from burotu.config import HEALTHY, ROOT
from burotu.core.features import Featurizer
from burotu.data.reference import Reference
from burotu.evaluation import benchmark
from burotu.models.glimmer.reader import MuseReader

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
    stage1_steps: int = 0  # first steps train the adapter only (LoRA frozen), as LLaVA-style models do
    min_tf: float = 0.0  # drop training cancers below this tumour fraction (0 = keep all)
    adapter_hidden: int = 1024
    adapter_dropout: float = 0.1
    adapter_input_norm: str = "layer"  # "layer" (per-sample LayerNorm) or "fixed" (keeps magnitudes)
    weight_decay: float = 0.01
    weights_file: str = "trainable.pt"  # what eval-only loads, e.g. trainable_stage1.pt
    eval_only: bool = False  # re-score an existing run (its trainable.pt) without training
    probe_every: int = 0  # every n steps, log the AUC on a fixed probe set (healthy vs cancers at 3-10%)
    out: str = "runs/latest"


# --- Model loading ------------------------------------------------------------------


def load_lm(cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    if cfg.model == "tiny":
        from burotu.models.glimmer.tiny import tiny_glimmer

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


Probe = dict[str, tuple[torch.Tensor, np.ndarray]]


def probe_auc(reader: MuseReader, probes: Probe | None, batch: int) -> dict | None:
    """AUC per probe set: `seen` uses training donors, `unseen` healthy people never trained on."""
    if not probes:
        return None
    from sklearn.metrics import roc_auc_score

    was_training = reader.training
    reader.eval()
    out = {}
    for name, (x, y) in probes.items():
        s, _ = score(reader, x, np.full(len(x), HEALTHY), batch, with_origin=False)
        out[name] = round(float(roc_auc_score(y, s)), 4)
    reader.train(was_training)
    return out


def train(
    reader: MuseReader,
    x: torch.Tensor,
    labels: np.ndarray,
    cfg: TrainConfig,
    log,
    probe: Probe | None = None,
    checkpoint: Path | None = None,
) -> None:
    lora_params = [p for p in reader.lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": reader.adapter.parameters(), "lr": cfg.lr_adapter},
            {"params": lora_params, "lr": cfg.lr_lora},
        ],
        weight_decay=cfg.weight_decay,
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
        if step <= cfg.stage1_steps:
            opt.param_groups[1]["lr"] = 0.0  # stage 1: adapter only
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
        if step == cfg.stage1_steps < cfg.steps and checkpoint is not None:
            reader.save_trainable(checkpoint)  # adapter-only model, evaluable later with weights_file
            log({"event": "checkpoint", "step": step, "path": str(checkpoint)})
        probe_now = bool(cfg.probe_every) and (step % cfg.probe_every == 0 or step == cfg.steps)
        if step == 1 or step % 25 == 0 or step == cfg.steps or probe_now:
            entry = {"step": step, "loss": round(total, 4), "lr_scale": round(f, 3)}
            entry["elapsed_s"] = round(time.time() - t0)
            if probe_now:
                entry["probe_auc"] = probe_auc(reader, probe, cfg.eval_batch)
            log(entry)
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


class ReaderAsModel:
    """The baseline model's interface over a MuseReader, for burotu.evaluation.realplasma."""

    def __init__(self, reader: MuseReader, featurizer: Featurizer, threshold: float, batch: int) -> None:
        self.reader, self.featurizer, self.threshold, self.batch = reader, featurizer, threshold, batch
        self.classes = reader.classes

    def _x(self, meth: np.ndarray, total: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self.featurizer.transform(meth, total))

    def cancer_score(self, meth: np.ndarray, total: np.ndarray) -> np.ndarray:
        x = self._x(meth, total)
        return score(self.reader, x, np.full(len(x), HEALTHY), self.batch, with_origin=False)[0]

    def origin_proba(self, meth: np.ndarray, total: np.ndarray) -> np.ndarray:
        x = self._x(meth, total)
        return score(self.reader, x, np.full(len(x), "cancer"), self.batch, with_origin=True)[1]


def adapter_kwargs(cfg: TrainConfig) -> dict:
    return {"hidden": cfg.adapter_hidden, "dropout": cfg.adapter_dropout, "input_norm": cfg.adapter_input_norm}


def _rows(cohort: benchmark.Cohort, rows: np.ndarray) -> benchmark.Cohort:
    return benchmark.Cohort(cohort.meth[rows], cohort.total[rows], cohort.meta.iloc[rows].reset_index(drop=True))


def by_donor(scores: np.ndarray, meta) -> list[dict]:
    """Healthy scores per blood donor, highest first: shows whether a few donors set the threshold."""
    import pandas as pd

    healthy = (meta["label"] == HEALTHY).to_numpy()
    df = pd.DataFrame({"donor": meta["blood_donor"].to_numpy()[healthy], "score": scores[healthy]})
    g = df.groupby("donor")["score"].agg(["count", "median", "max"]).sort_values("median", ascending=False)
    return [{"donor": str(d), "n": int(r["count"]), "median": round(float(r["median"]), 3),
             "max": round(float(r["max"]), 3)} for d, r in g.iterrows()]  # fmt: skip


def save_for_serving(out: Path, cfg: TrainConfig, featurizer: Featurizer, panel, classes, threshold: float) -> None:
    """Everything burotu.models.glimmer.serve needs next to trainable.pt to answer like the benchmark."""
    np.save(out / "mu.npy", featurizer.mu.astype(np.float32))
    meta = {
        "model": cfg.model,
        "revision": cfg.revision,
        "load_4bit": cfg.load_4bit,
        "lora_r": cfg.lora_r,
        "n_tokens": cfg.n_tokens,
        "adapter": adapter_kwargs(cfg),
        "panel": list(panel),
        "classes": list(classes),
        "threshold": float(threshold),
    }
    (out / "serving.json").write_text(json.dumps(meta))


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
    reader = MuseReader(lm, tokenizer, in_dim, classes, n_tokens=cfg.n_tokens, adapter_kwargs=adapter_kwargs(cfg))
    trainable = sum(p.numel() for p in reader.parameters() if p.requires_grad)
    log(
        {
            "event": "model",
            "model": cfg.model,
            "trainable_params": trainable,
            "output_head": reader.head_precision,
            "load_s": round(time.time() - t0),
        }
    )

    if cfg.min_tf > 0:
        keep = (fit.labels == HEALTHY) | (fit.meta["tumor_fraction"].to_numpy() >= cfg.min_tf)
        fit = benchmark.Cohort(fit.meth[keep], fit.total[keep], fit.meta[keep].reset_index(drop=True))
        log({"event": "min_tf", "kept": int(keep.sum())})
    probe = None
    if cfg.probe_every:
        # Cancers at 3-10% from the small `fit` cohort (training donors), against healthy samples
        # from training donors (`seen`) or from calibration donors never trained on (`unseen`):
        # the second one shows whether the model learns tumours or memorises donors.
        small = benchmark.load_cohort("fit", ref)
        tf = small.meta["tumor_fraction"].to_numpy()
        cancers = np.flatnonzero(tf >= 0.03)[:64]
        seen = np.flatnonzero(small.labels == HEALTHY)[:64]
        unseen = np.flatnonzero(calib.labels == HEALTHY)[:: max(1, len(calib) // 128)][:128]
        x_cancer = features_of(_rows(small, cancers), featurizer)

        def probe_set(cohort: benchmark.Cohort, healthy: np.ndarray):
            x = torch.cat([features_of(_rows(cohort, healthy), featurizer), x_cancer])
            return x, np.r_[np.zeros(len(healthy)), np.ones(len(cancers))]

        probe = {"seen": probe_set(small, seen), "unseen": probe_set(calib, unseen)}
        log({"event": "probe", "auc_before_training": probe_auc(reader, probe, cfg.eval_batch)})
    if cfg.eval_only:
        reader.load_trainable(out / cfg.weights_file)
        reader.eval()
        log({"event": "eval_only", "weights": str(out / cfg.weights_file)})
    else:
        train(reader, features_of(fit, featurizer), fit.labels, cfg, log, probe, out / "trainable_stage1.pt")
        reader.save_trainable(out / "trainable.pt")

    c_score, _ = score(reader, features_of(calib, featurizer), calib.labels, cfg.eval_batch, with_origin=False)
    threshold = benchmark.threshold_at_specificity(c_score)
    save_for_serving(out, cfg, featurizer, ref.panel, classes, threshold)
    t_score, t_origin = score(reader, features_of(test, featurizer), test.labels, cfg.eval_batch, with_origin=True)
    report = {
        "model": cfg.model,
        "config": asdict(cfg),
        "trainable_params": trainable,
        "train_samples": len(fit),
        "test": benchmark.evaluate(t_score, t_origin, classes, threshold, test.meta),
    }
    try:
        from burotu.evaluation import realplasma

        model = ReaderAsModel(reader, featurizer, threshold, cfg.eval_batch)
        report["real_plasma"] = realplasma.report(model, ref.panel)
    except FileNotFoundError as e:  # real plasma subset not available
        log({"event": "real_plasma_skipped", "reason": str(e)})
    report["minutes"] = round((time.time() - t0) / 60, 1)
    np.savez_compressed(out / "scores.npz", calibration=c_score, test=t_score, test_origin=t_origin)
    report["healthy_scores_by_donor"] = {
        name: by_donor(score_arr, cohort.meta)
        for name, score_arr, cohort in [("calibration", c_score, calib), ("test", t_score, test)]
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
