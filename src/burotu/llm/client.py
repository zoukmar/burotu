"""Client for Meta's Muse models through the Meta Model API.

The API is OpenAI-compatible (chat completions):
    POST {base_url}/chat/completions   Authorization: Bearer $MODEL_API_KEY

Environment variables:
    MODEL_API_KEY       Meta Model API key (missing → mock mode)
    META_API_BASE_URL   default https://api.meta.ai/v1
    META_MODEL          default muse-spark-1.3
    BUROTU_LLM_MOCK=1   force mock mode (tests, offline demo)

Mock mode lets the whole pipeline run and be tested without a key: it returns a
deterministic answer and says so explicitly.
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
        """Extract the first JSON object from the answer (tolerates a ```json block)."""
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
                    text=_answer_text(data),
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
        raise RuntimeError(f"Meta Model API unavailable after {retries} attempts: {last}")


def _answer_text(data: dict) -> str:
    """The answer text, whether `content` is a string or a list of parts.

    Muse Spark reasons before answering, and reasoning tokens count against
    max_completion_tokens: a budget that is too small leaves an empty answer.
    """
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not content or not content.strip():
        raise ValueError(f"empty answer (finish_reason={choice.get('finish_reason')}); raise max_tokens")
    return content


def _default_mock(messages: list[dict[str, str]], json_mode: bool) -> str:
    if json_mode:
        return json.dumps({"mock": True, "note": "Mock answer: set MODEL_API_KEY to use Muse Spark."})
    return "[MOCK MODE] Simulated answer. Set MODEL_API_KEY to use Muse Spark."
