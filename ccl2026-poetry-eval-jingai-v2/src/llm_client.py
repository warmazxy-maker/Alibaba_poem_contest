"""Small OpenAI-compatible chat completions client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import Settings, settings


class LLMError(RuntimeError):
    """Raised when a model request fails."""


class LLMClient:
    def __init__(self, cfg: Settings = settings, dry_run: bool | None = None) -> None:
        self.cfg = cfg
        self.dry_run = cfg.dry_run if dry_run is None else dry_run

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self.dry_run or not self.cfg.api_key:
            return self._dry_response(messages)

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.cfg.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code < 500 and exc.code != 429:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise LLMError(f"LLM HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < self.cfg.max_retries:
                time.sleep(2**attempt)

        raise LLMError(f"LLM request failed after retries: {last_error}") from last_error

    @staticmethod
    def _dry_response(messages: list[dict[str, str]]) -> str:
        user_text = next(
            (msg["content"] for msg in reversed(messages) if msg.get("role") == "user"),
            "",
        )
        excerpt = user_text.replace("\n", " ")[:120]
        return json.dumps(
            {"prediction": "DRY_RUN", "reason": excerpt},
            ensure_ascii=False,
        )
