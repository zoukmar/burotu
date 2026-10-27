"""A tiny randomly initialised Muse Glimmer and a byte tokenizer, for tests and CPU smoke runs.

Same architecture class as the real 30B model, a few hundred thousand parameters, no
download. It lets the whole training and scoring path run anywhere in seconds.
"""

from __future__ import annotations


class ByteTokenizer:
    """Minimal tokenizer: one token per UTF-8 byte, plus an end-of-sequence token."""

    eos_token_id = 256
    pad_token_id = 257

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))


def tiny_glimmer():
    from transformers.models.muse_glimmer import MuseGlimmerConfig, MuseGlimmerForConditionalGeneration

    text = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=300,
        layer_types=["sliding_attention", "full_attention"],
        max_position_embeddings=512,
        sliding_window=64,
        # Larger init than real models: a frozen random output head with the default 0.02 init can only
        # produce near-uniform logits, so LoRA alone could never fit the answers in a smoke test.
        initializer_range=0.5,
        bos_token_id=None,
        eos_token_id=ByteTokenizer.eos_token_id,
    )
    vision = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, layer_types=["full_attention"]
    )
    cfg = MuseGlimmerConfig(text_config=text, vision_config=vision, out_hidden_size=64, projector_hidden_size=32)
    return MuseGlimmerForConditionalGeneration(cfg), ByteTokenizer()
