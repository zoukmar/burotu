"""MuseReader: a language model that reads a methylation profile through an adapter.

    features (1,681 CpG z-scores + summaries)
        └─► MethylationAdapter ─► N soft tokens in the model's embedding space
                                        │
    [PREFIX tokens] [soft tokens] [QUESTION tokens] [answer tokens] ─► causal LM (+ LoRA)

The same recipe multimodal models use for images: a small projector maps a new kind
of input into the token space, and the language model learns to reason over it.

Scoring is done from the model's own probabilities, so it can be compared with any
classifier on the same benchmark:
  * cancer score  = log P(" yes") - log P(" no") right after QUESTION;
  * tissue origin = softmax over the log-likelihood of each class name.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from burotu.muse import prompts


class MethylationAdapter(nn.Module):
    """Project a methylation feature vector into `n_tokens` embeddings of size `d_model`."""

    def __init__(self, in_dim: int, d_model: int, n_tokens: int = 16, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.n_tokens, self.d_model = n_tokens, d_model
        self.norm_in = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tokens * d_model),
        )
        self.norm_out = nn.LayerNorm(d_model)
        # Set by MuseReader to the typical size of the model's own input embeddings.
        self.register_buffer("embed_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sign(x) * torch.log1p(x.abs())  # z-scores span orders of magnitude
        h = self.net(self.norm_in(x)).view(x.shape[0], self.n_tokens, self.d_model)
        return self.norm_out(h) * self.embed_scale


class MuseReader(nn.Module):
    def __init__(self, lm: nn.Module, tokenizer, in_dim: int, classes: Sequence[str], n_tokens: int = 16):
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.classes = list(classes)
        embed = lm.get_input_embeddings()
        self.adapter = MethylationAdapter(in_dim, embed.embedding_dim, n_tokens)
        self.adapter.to(embed.weight.device)
        with torch.no_grad():
            # Measure what real token embeddings look like at the model's input (some models,
            # Glimmer included, normalise embeddings on the way in), and match that scale.
            sample = torch.arange(min(embed.num_embeddings, 4096), device=embed.weight.device)
            self.adapter.embed_scale.fill_(float(embed(sample).float().std()))

        self._prefix = self._ids(prompts.PREFIX)
        self._question = self._ids(prompts.QUESTION)
        self._yes = self._ids(prompts.YES)[0]  # first token of each answer
        self._no = self._ids(prompts.NO)[0]
        self._origin_lead = self._ids(prompts.ORIGIN_LEAD)
        self._origin_options = [self._ids(prompts.origin_option(c)) for c in self.classes]
        eos = getattr(tokenizer, "eos_token_id", None)
        self._eos = [eos] if eos is not None else []

    # --- Helpers ----------------------------------------------------------------

    def _ids(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    @property
    def device(self) -> torch.device:
        return self.lm.get_input_embeddings().weight.device

    def _embed(self, ids: list[int]) -> torch.Tensor:
        t = torch.tensor(ids, dtype=torch.long, device=self.device)
        return self.lm.get_input_embeddings()(t)

    def _sequences(self, features: torch.Tensor, tails: list[list[int]]):
        """Build [prefix][soft tokens][question][tail] for each sample, right-padded.

        Returns inputs_embeds, attention_mask and the index where each tail starts.
        """
        soft = self.adapter(features.to(self.device, torch.float32))
        dtype = self.lm.get_input_embeddings().weight.dtype
        soft = soft.to(dtype)
        prefix, question = self._embed(self._prefix), self._embed(self._question)
        head_len = len(self._prefix) + soft.shape[1] + len(self._question)
        max_len = head_len + max(len(t) for t in tails)
        embeds = torch.zeros(len(tails), max_len, soft.shape[2], dtype=dtype, device=self.device)
        mask = torch.zeros(len(tails), max_len, dtype=torch.long, device=self.device)
        for i, tail in enumerate(tails):
            parts = [prefix, soft[i], question]
            if tail:
                parts.append(self._embed(tail))
            seq = torch.cat(parts)
            embeds[i, : len(seq)] = seq
            mask[i, : len(seq)] = 1
        return embeds, mask, head_len

    def _logits(self, embeds: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Logits at the given sequence positions only.

        Glimmer's vocabulary has 202,048 tokens: computing logits everywhere would cost
        several GB per batch for positions we never read.
        """
        try:
            return self.lm(inputs_embeds=embeds, attention_mask=mask, logits_to_keep=positions).logits
        except TypeError:  # models without logits_to_keep
            return self.lm(inputs_embeds=embeds, attention_mask=mask).logits[:, positions]

    def _span(self, start: int, stop: int) -> torch.Tensor:
        return torch.arange(start, stop, device=self.device)

    # --- Training ----------------------------------------------------------------

    def loss(self, features: torch.Tensor, labels: Sequence[str]) -> torch.Tensor:
        """Cross-entropy on the answer tokens only."""
        tails = [self._ids(prompts.target_text(lab)) + self._eos for lab in labels]
        embeds, mask, head_len = self._sequences(features, tails)
        targets = torch.full(mask.shape, -100, dtype=torch.long, device=self.device)
        for i, tail in enumerate(tails):
            targets[i, head_len : head_len + len(tail)] = torch.tensor(tail, device=self.device)
        # Position t predicts token t + 1: keep the logits that predict the answer tokens.
        positions = self._span(head_len - 1, mask.shape[1] - 1)
        logits = self._logits(embeds, mask, positions).float()
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets[:, positions + 1].reshape(-1), ignore_index=-100
        )

    # --- Scoring ----------------------------------------------------------------------

    def _continuation_logprob(self, features: torch.Tensor, lead: list[int], option: list[int]) -> torch.Tensor:
        """Log-likelihood of `option` written after QUESTION + `lead`, for each sample."""
        embeds, mask, head_len = self._sequences(features, [lead + option] * len(features))
        start = head_len + len(lead)
        logp = torch.log_softmax(self._logits(embeds, mask, self._span(start - 1, start - 1 + len(option))).float(), -1)
        return sum(logp[:, j, tok] for j, tok in enumerate(option))

    def _next_token_margin(self, embeds: torch.Tensor, mask: torch.Tensor, pos: int, a: int, b: int) -> torch.Tensor:
        """logit(a) - logit(b) for the token after `pos`, computed in float32.

        Glimmer returns bf16 logits squeezed by a tanh soft cap, so confident answers land
        on the same few representable values and tie. Ties at the top break the threshold
        set at 99.5% specificity. We capture the final hidden state going into the output
        head and redo the last step (projection, output multiplier, soft cap) in float32.
        """
        head = self.lm.get_output_embeddings()
        seen: dict = {}
        hook = head.register_forward_hook(lambda _m, inp, _out: seen.__setitem__("h", inp[0]))
        try:
            self._logits(embeds, mask, self._span(pos, pos + 1))
        finally:
            hook.remove()
        with torch.autocast(self.device.type, enabled=False):
            h = seen["h"][:, -1].float()
            raw = h @ head.weight[[a, b]].float().T
            cfg = getattr(self.lm.config, "text_config", self.lm.config)
            logits = raw * float(getattr(cfg, "output_multiplier", 1.0) or 1.0)
            cap = getattr(cfg, "final_logit_softcapping", None)
            if cap:
                logits = cap * torch.tanh(logits / cap)
        return logits[:, 0] - logits[:, 1]

    @torch.no_grad()
    def cancer_score(self, features: torch.Tensor) -> torch.Tensor:
        """log P(" yes") - log P(" no"): higher means a stronger cancer signal."""
        head_w = self.lm.get_output_embeddings().weight
        if self._yes != self._no and head_w.is_floating_point():  # one pass, compare the two next tokens
            embeds, mask, head_len = self._sequences(features, [[] for _ in range(len(features))])
            return self._next_token_margin(embeds, mask, head_len - 1, self._yes, self._no).cpu()
        yes = self._continuation_logprob(features, [], self._ids(prompts.YES))
        no = self._continuation_logprob(features, [], self._ids(prompts.NO))
        return (yes - no).cpu()

    @torch.no_grad()
    def origin_proba(self, features: torch.Tensor) -> torch.Tensor:
        scores = []
        for option in self._origin_options:
            ll = self._continuation_logprob(features, self._origin_lead, option)
            scores.append(ll / math.sqrt(max(1, len(option))))  # mild length normalisation
        return torch.softmax(torch.stack(scores, dim=1), dim=1).cpu()

    # --- Persistence ------------------------------------------------------------------

    def trainable_state(self) -> dict:
        """Adapter weights plus LoRA weights: the only parts that change in training."""
        state = {f"adapter.{k}": v.cpu() for k, v in self.adapter.state_dict().items()}
        for name, p in self.lm.named_parameters():
            if p.requires_grad:
                state[f"lm.{name}"] = p.detach().cpu()
        return state

    def save_trainable(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.trainable_state(), "classes": self.classes}, path)

    def load_trainable(self, path: str | Path) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        state = blob["state"]
        self.adapter.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("adapter.")})
        lm_state = {k[3:]: v for k, v in state.items() if k.startswith("lm.")}
        missing = set(lm_state) - {n for n, _ in self.lm.named_parameters()}
        if missing:
            raise KeyError(f"unknown LoRA weights: {sorted(missing)[:3]}")
        self.lm.load_state_dict(lm_state, strict=False)
