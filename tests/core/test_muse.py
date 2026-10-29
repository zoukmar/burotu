"""Muse reader on a tiny random Glimmer: shapes, gradients, scoring and persistence."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from burotu.muse import prompts  # noqa: E402
from burotu.muse.reader import MuseReader  # noqa: E402
from burotu.muse.train import TrainConfig, load_lm  # noqa: E402

CLASSES = ["Colorectal", "Lung"]


@pytest.fixture(scope="module")
def reader():
    torch.manual_seed(0)
    lm, tok = load_lm(TrainConfig(model="tiny"))
    return MuseReader(lm, tok, in_dim=12, classes=CLASSES, n_tokens=4).eval()  # no dropout when scoring


def test_answers_are_what_we_score():
    assert prompts.target_text("Healthy") == " no."
    assert prompts.target_text("Lung") == " yes. Most likely origin: Lung."


def test_loss_backpropagates_to_adapter_and_lora(reader):
    x = torch.randn(3, 12)
    loss = reader.loss(x, np.array(["Healthy", "Lung", "Colorectal"]))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in reader.adapter.parameters())
    lora = [p for n, p in reader.lm.named_parameters() if "lora_" in n]
    assert lora and any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora)
    reader.zero_grad()


def test_scores_have_the_right_shape(reader):
    x = torch.randn(5, 12)
    assert reader.cancer_score(x).shape == (5,)
    proba = reader.origin_proba(x)
    assert proba.shape == (5, len(CLASSES))
    assert torch.allclose(proba.sum(dim=1), torch.ones(5), atol=1e-5)


def test_trainable_weights_roundtrip(reader, tmp_path):
    x = torch.randn(2, 12)
    before = reader.cancer_score(x)
    reader.save_trainable(tmp_path / "w.pt")
    with torch.no_grad():
        for p in reader.adapter.parameters():
            p.add_(1.0)
    assert not torch.allclose(before, reader.cancer_score(x))
    reader.load_trainable(tmp_path / "w.pt")
    assert torch.allclose(before, reader.cancer_score(x), atol=1e-5)


def test_float32_margin_matches_the_model_logits(reader):
    """The float32 yes/no margin must equal the model's own logit difference."""
    x = torch.randn(3, 12)
    embeds, mask, head_len = reader._sequences(x, [[], [], []])
    with torch.no_grad():
        margin = reader._next_token_margin(embeds, mask, head_len - 1, 10, 20, capped=True)
        logits = reader._logits(embeds, mask, reader._span(head_len - 1, head_len))[:, 0].float()
    assert torch.allclose(margin, logits[:, 10] - logits[:, 20], atol=1e-4)


def test_one_pass_origin_matches_the_model_next_token_distribution(reader):
    """With distinct first tokens, tissue probabilities come from one forward pass."""
    saved = reader._origin_options
    reader._origin_options = [[65, 46], [66, 46]]  # "A." and "B.": distinct first tokens
    try:
        x = torch.randn(3, 12)
        fast = reader.origin_proba(x)
        embeds, mask, head_len = reader._sequences(x, [reader._origin_lead] * 3)
        pos = head_len + len(reader._origin_lead) - 1
        with torch.no_grad():
            logits = reader._logits(embeds, mask, reader._span(pos, pos + 1))[:, 0].float()
        expected = torch.softmax(logits[:, [65, 66]], dim=1)
        assert torch.allclose(fast, expected, atol=1e-4)
    finally:
        reader._origin_options = saved
