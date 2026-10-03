"""OpenAI-compatible chat backends.

The paper drives navigation with a proprietary frontier model; this module is the
single swap point that lets any chat.completions endpoint stand in for it.
"""
from __future__ import annotations

import base64
import io
import os
import random
import time
from typing import Any, Dict, List, Optional

from PIL import Image


def encode_image(rgb, quality: int = 85) -> str:
    img = Image.fromarray(rgb.astype("uint8"), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def image_block(rgb) -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(rgb)}"},
    }


def text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


class BackendError(RuntimeError):
    pass


def _lookup_key(api_key_env: str) -> Optional[str]:
    """Prefer the environment, then a local .env so secrets need not be exported into a shell."""
    value = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    if value:
        return value
    dotenv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(dotenv):
        with open(dotenv) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                if key.strip() in (api_key_env, "OPENAI_API_KEY"):
                    return val.strip().strip('"').strip("'")
    return None


class ChatBackend:
    """Bounded-retry wrapper over an OpenAI-compatible chat.completions endpoint."""

    name = "chat"

    def __init__(
        self,
        model: str = "glm-5.3-flash",
        base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: Optional[str] = "minimal",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 6,
        timeout: float = 180.0,
        retry_sleep: float = 5.0,
    ) -> None:
        from openai import OpenAI

        key = api_key or _lookup_key(api_key_env)
        if not key:
            raise BackendError(
                f"No API key found. Export ${api_key_env}, or add a line "
                f"`{api_key_env}=...` to gpt6astra-r2rce/.env"
            )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.base_url = base_url
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {
            "calls": 0,
            "failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "latency_s": 0.0,
        }

    def infer(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            params["extra_body"] = {"reasoning_effort": self.reasoning_effort}

        last_error = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(**params)
                choice = resp.choices[0]
                text = choice.message.content
                reasoning = getattr(choice.message, "reasoning_content", None) or ""
                usage = getattr(resp, "usage", None)
                if not text:
                    # Thinking models can spend the entire token budget on
                    # reasoning_content and still answer with finish=length.
                    raise BackendError(
                        f"empty content (finish={choice.finish_reason}, "
                        f"reasoning_chars={len(reasoning)}); max_tokens={self.max_tokens} too small"
                    )
                self.stats["calls"] += 1
                self.stats["latency_s"] += time.time() - t0
                if usage is not None:
                    self.stats["prompt_tokens"] += usage.prompt_tokens or 0
                    self.stats["completion_tokens"] += usage.completion_tokens or 0
                    details = getattr(usage, "completion_tokens_details", None)
                    self.stats["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0
                return {
                    "text": text,
                    "reasoning_chars": len(reasoning),
                    "attempts": attempt + 1,
                    "latency_s": round(time.time() - t0, 2),
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                }
            except Exception as error:  # noqa: BLE001 - endpoint errors are untyped
                status = getattr(error, "status_code", None)
                detail = str(error)
                if status == 400 and "reasoning_effort" in detail and self.reasoning_effort:
                    # Provider rejects the knob; drop it once and keep going.
                    self.reasoning_effort = None
                    params.pop("extra_body", None)
                    continue
                last_error = error
                self.stats["failures"] += 1
                if attempt == self.max_retries - 1:
                    break
                is_rate_limited = status == 429 or "rate" in detail.lower()
                cap = self.retry_sleep * (8 if is_rate_limited else 2)
                time.sleep(min(cap, 60) * (0.5 + random.random()))
        raise BackendError(f"backend failed after {self.max_retries} attempts: {last_error}")
