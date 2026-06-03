"""Small OpenAI-compatible chat completions client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import Settings, settings


class LLMError(RuntimeError):
    """功能：模型请求失败或配置不合法时抛出的异常类型。"""


class LLMClient:
    """功能：封装 OpenAI 兼容 Chat Completions API 请求。"""

    def __init__(self, cfg: Settings = settings, dry_run: bool | None = None) -> None:
        """功能：初始化模型客户端。

        参数：
            cfg：全局配置对象，提供 API 地址、模型名、超时、重试和 dry-run 默认值。
            dry_run：是否强制使用模拟响应；为 None 时使用 cfg.dry_run。
        """
        self.cfg = cfg
        self.dry_run = cfg.dry_run if dry_run is None else dry_run

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """功能：发送一轮聊天补全请求，返回模型文本结果。

        参数：
            messages：OpenAI chat 格式消息列表，每项包含 role 和 content。
            temperature：本次请求的采样温度；为 None 时使用配置中的默认值。
            max_tokens：本次请求允许生成的最大 token 数；为 None 时使用配置中的默认值。

        返回：
            str：模型返回的 message.content 文本；dry-run 时返回模拟 JSON 字符串。
        """
        if self.dry_run:
            return self._dry_response(messages)
        if not self.cfg.api_key:
            raise LLMError("OPENAI_API_KEY is required unless dry_run=True.")

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
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
        """功能：生成 dry-run 模式下的模拟模型响应。

        参数：
            messages：原始 chat 消息列表，用于截取最后一条用户消息作为 reason 摘要。

        返回：
            str：JSON 字符串，包含 prediction=DRY_RUN 和 reason。
        """
        user_text = next(
            (msg["content"] for msg in reversed(messages) if msg.get("role") == "user"),
            "",
        )
        excerpt = user_text.replace("\n", " ")[:120]
        return json.dumps(
            {"prediction": "DRY_RUN", "reason": excerpt},
            ensure_ascii=False,
        )
