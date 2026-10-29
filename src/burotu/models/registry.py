"""The model registry: one TOML file listing every model, read the same way everywhere.

    from burotu.models.registry import models, get
    get("glimmer").weights   # ["hf:zoukmar75/burotu-glimmer", "modal:burotu-runs/glimmer-v5"]

Adding a model = a package in burotu/models/<id>/ (engine.py, README.md model card) and an
entry in registry.toml. Weights are referenced as `git:<path>`, `hf:<repo>` or
`modal:<volume>/<run>`; `links()` turns them into URLs for people.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / "registry.toml"
REPO = "https://github.com/zoukmar/burotu"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    version: str
    role: str
    status: str
    family: str
    inputs: str
    trained_on: str
    runs_on: str
    weights: list[str] = field(default_factory=list)
    report: str = ""
    base_model: str = ""

    @property
    def card(self) -> str:
        """The model card on GitHub."""
        return f"{REPO}/tree/main/src/burotu/models/{self.id}"

    def links(self) -> list[tuple[str, str]]:
        """(label, url) for each place the weights live."""
        out = []
        for w in self.weights:
            kind, _, where = w.partition(":")
            if kind == "git":
                out.append(("GitHub", f"{REPO}/tree/main/{where}"))
            elif kind == "hf":
                out.append(("Hugging Face", f"https://huggingface.co/{where}"))
            elif kind == "modal":
                out.append(("Modal", ""))
        return out


@cache
def models() -> tuple[ModelInfo, ...]:
    data = tomllib.loads(REGISTRY.read_text())
    return tuple(ModelInfo(**m) for m in data["model"])


def get(model_id: str) -> ModelInfo:
    for m in models():
        if m.id == model_id:
            return m
    raise KeyError(model_id)
