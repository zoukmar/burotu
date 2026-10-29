"""Why does a Muse run score every sample the same? A few minutes of GPU instead of a full run.

    python -m burotu.models.glimmer.diagnose --run runs/glimmer          (locally, tiny model)
    modal run scripts/modal_muse.py::diagnose_run --name glimmer

On a handful of test samples (healthy, and cancers at 10% tumour DNA) it reports:
  * collapse: how much the adapter's soft tokens differ between samples, trained vs fresh;
  * sensitivity: how much the answer logits move when only the sample changes, in float32;
  * scores: the cancer scores themselves, and how many distinct values they take;
  * gradients: adapter vs LoRA gradient norms on one fresh training batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from burotu.config import HEALTHY
from burotu.core.features import Featurizer
from burotu.data.reference import Reference
from burotu.evaluation import benchmark
from burotu.models.glimmer.reader import MethylationAdapter, MuseReader
from burotu.models.glimmer.train import TrainConfig, load_lm


def _spread(t: torch.Tensor) -> float:
    """Between-sample spread of soft tokens, relative to their overall size."""
    t = t.float()
    return float(t.std(dim=0).mean() / (t.std() + 1e-8))


@torch.no_grad()
def _answer_logits(reader: MuseReader, x: torch.Tensor) -> torch.Tensor:
    embeds, mask, head_len = reader._sequences(x, [[] for _ in range(len(x))])
    out = reader.lm(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True)
    h = out.hidden_states[-1][:, head_len - 1].float()
    head = reader.lm.get_output_embeddings()
    with torch.autocast("cuda", enabled=False):  # true float32 logits, not bf16
        if hasattr(head, "weight") and head.weight.dtype in (torch.float32, torch.bfloat16, torch.float16):
            return (h @ head.weight.float().T).float()
        return out.logits[:, head_len - 1].float()


def run(run_dir: Path, n: int = 16) -> dict:
    meta = json.loads((run_dir / "serving.json").read_text()) if (run_dir / "serving.json").exists() else None
    cfg = TrainConfig(**(json.loads((run_dir / "report.json").read_text())["config"]))
    torch.manual_seed(0)
    ref = Reference.load()
    featurizer = Featurizer.from_blood(ref.blood_train.to_numpy(np.float32))
    test = benchmark.load_cohort("test", ref)
    tf = test.meta["tumor_fraction"].to_numpy()
    rows = np.r_[np.flatnonzero(test.labels == HEALTHY)[:n], np.flatnonzero(np.isclose(tf, 0.1))[:n]]
    x = torch.from_numpy(featurizer.transform(test.meth[rows], test.total[rows]))
    is_cancer = test.labels[rows] != HEALTHY

    lm, tok = load_lm(cfg)
    classes = meta["classes"] if meta else sorted(set(test.labels) - {HEALTHY})
    from burotu.models.glimmer.train import adapter_kwargs

    reader = MuseReader(lm, tok, x.shape[1], classes, n_tokens=cfg.n_tokens, adapter_kwargs=adapter_kwargs(cfg))
    fresh = MethylationAdapter(x.shape[1], reader.adapter.d_model, cfg.n_tokens, **adapter_kwargs(cfg))
    fresh = fresh.to(reader.device)
    fresh.embed_scale.copy_(reader.adapter.embed_scale)
    fresh.eval()
    report: dict = {"run": str(run_dir), "embed_scale": float(reader.adapter.embed_scale)}

    # Gradients on one fresh batch, before loading the trained weights.
    reader.train()
    labels = test.labels[rows]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        loss = reader.loss(x, labels)
    loss.backward()
    norm = lambda ps: float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ps if p.grad is not None)))  # noqa: E731
    report["fresh_loss"] = float(loss)
    report["grad_norm_adapter"] = norm(list(reader.adapter.parameters()))
    report["grad_norm_lora"] = norm([p for p in reader.lm.parameters() if p.requires_grad])
    reader.zero_grad(set_to_none=True)
    reader.eval()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        xd = x.to(reader.device)
        report["spread_fresh_adapter"] = _spread(fresh(xd))
        fresh_logits = _answer_logits(reader, x)
        reader.load_trainable(run_dir / "trainable.pt")
        reader.eval()
        report["spread_trained_adapter"] = _spread(reader.adapter(xd))
        report["soft_token_std"] = float(reader.adapter(xd).float().std())
        report["text_token_std"] = float(reader._embed(reader._question).float().std())
        trained_logits = _answer_logits(reader, x)

    for name, lg in [("fresh", fresh_logits), ("trained", trained_logits)]:
        yes_no = (lg[:, reader._yes] - lg[:, reader._no]).cpu().numpy()
        report[f"{name}_logit_spread_between_samples"] = float(lg.std(dim=0).mean())
        report[f"{name}_score_float32"] = {
            "healthy_mean": float(yes_no[~is_cancer].mean()),
            "cancer_mean": float(yes_no[is_cancer].mean()),
            "std": float(yes_no.std()),
            "distinct": len(np.unique(np.round(yes_no, 6))),
        }
    scores = reader.cancer_score(x).numpy()
    report["trained_score_as_scored"] = {"distinct": len(np.unique(scores)), "values": scores[:6].round(4).tolist()}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    print(json.dumps(run(Path(a.run), a.n), indent=2))


if __name__ == "__main__":
    main()
