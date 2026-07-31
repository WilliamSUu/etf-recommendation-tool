from __future__ import annotations

import json
import re
from typing import Any

import requests


class DeepSeekError(RuntimeError):
    """Raised when DeepSeek API is unavailable or returns invalid content."""


class DeepSeekClient:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        if not api_key:
            raise DeepSeekError("未配置 DEEPSEEK_API_KEY，请在 .env 中填写 DeepSeek API Key。")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        timeout: int = 60,
    ) -> Any:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - network/API boundary
            raise DeepSeekError(f"DeepSeek API 调用失败：{exc}") from exc

        try:
            return json.loads(_strip_code_fence(content))
        except json.JSONDecodeError as exc:
            raise DeepSeekError(f"DeepSeek 返回内容不是合法 JSON：{content[:300]}") from exc


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", content, flags=re.DOTALL)
    return match.group(1).strip() if match else content

