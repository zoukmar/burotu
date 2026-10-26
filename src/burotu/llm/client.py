"""Client pour les modèles Meta (Muse Spark) via la Meta Model API.

L'API est compatible OpenAI (chat completions) :
    POST {base_url}/chat/completions   Authorization: Bearer $MODEL_API_KEY

Variables d'environnement :
    MODEL_API_KEY       clé de la Meta Model API (absente → mode mock)
    META_API_BASE_URL   défaut https://api.meta.ai/v1
    META_MODEL          défaut muse-spark-1.3
    BUROTU_LLM_MOCK=1   force le mode mock (tests, démo hors ligne)

Le mode mock permet de développer et de tester tout le pipeline sans clé :
il renvoie une réponse déterministe et le signale explicitement.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"


@dataclass
class LLMResponse:
    text: str
    model: str
    mock: bool
    usage: dict[str, Any] | None = None

    def json(self) -> Any:
        """Extrait le premier objet JSON de la réponse (tolère un bloc ```json)."""
        t = self.text.strip()
        if "```" in t:
            t = t.split("```")[1]
            t = t[4:] if t.startswith("json") else t
        start, end = t.find("{"), t.rfind("}")
        return json.loads(t[start : end + 1])


class MetaClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        mock: bool | None = None,
        mock_handler: Callable[[list[dict]], str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("MODEL_API_KEY")
        self.base_url = (base_url or os.environ.get("META_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("META_MODEL") or DEFAULT_MODEL
        env_mock = os.environ.get("BUROTU_LLM_MOCK") == "1"
        self.mock = mock if mock is not None else (env_mock or not self.api_key)
        self.mock_handler = mock_handler
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
        retries: int = 3,
    ) -> LLMResponse:
        if self.mock:
            text = self.mock_handler(messages) if self.mock_handler else _default_mock(messages, json_mode)
            return LLMResponse(text=text, model=f"mock:{self.model}", mock=True)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last: Exception | None = None
        for attempt in range(retries):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    model=data.get("model", self.model),
                    mock=False,
                    usage=data.get("usage"),
                )
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                if (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code < 500
                    and e.response.status_code != 429
                ):
                    raise
                time.sleep(2**attempt)
        raise RuntimeError(f"Meta Model API indisponible après {retries} essais : {last}")


def _default_mock(messages: list[dict[str, str]], json_mode: bool) -> str:
    if json_mode:
        return json.dumps({"mock": True, "note": "Réponse simulée : définir MODEL_API_KEY pour utiliser Muse Spark."})
    return "[MODE MOCK] Réponse simulée. Définir MODEL_API_KEY pour utiliser Muse Spark."
