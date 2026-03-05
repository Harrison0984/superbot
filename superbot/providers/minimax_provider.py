"""MiniMax native API provider - bypasses LiteLLM for MiniMax."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import string
from typing import Any

import requests

from superbot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_ALNUM = string.ascii_letters + string.digits


def _sanitize_error_message(message: str) -> str:
    """Remove sensitive information from error messages."""
    # Remove API keys (various patterns)
    patterns = [
        r'sk-[a-zA-Z0-9-]{20,}',  # OpenAI-style keys
        r'Bearer\s+[a-zA-Z0-9-]+',  # Bearer tokens
        r'api[_-]?key["\s:=]+["\']?[^"\'\s]+',  # api_key=xxx
        r'Authorization["\s:=]+["\'][^"\'\s]+',  # Authorization: xxx
    ]
    sanitized = message
    for pattern in patterns:
        sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
    return sanitized


def _short_tool_id() -> str:
    """Generate a 9-char alphanumeric ID for tool calls."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


class MiniMaxProvider(LLMProvider):
    """MiniMax native API provider using the text/chatcompletion_v2 endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "MiniMax-M2.5",
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.api_base = api_base or "https://api.minimaxi.com/v1"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request via MiniMax native API."""
        model = model or self.default_model
        messages = self._sanitize_empty_content(messages)

        # Build request payload for MiniMax native API
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }

        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        # Add tools if provided (MiniMax supports function calling)
        if tools:
            payload["tools"] = tools

        # Ensure API base has /v1 suffix for native endpoint
        base = self.api_base.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        url = f"{base}/text/chatcompletion_v2"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            # Run sync requests in a thread to avoid blocking the event loop
            response = await asyncio.to_thread(
                requests.post,
                url,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            logger.debug("MiniMax response: {}", json.dumps(data, ensure_ascii=False)[:500])
            return self._parse_response(data)
        except requests.exceptions.Timeout:
            return LLMResponse(
                content="Error calling MiniMax: Request timed out. Please try again.",
                finish_reason="error",
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            # Friendly error messages for common HTTP status codes
            if status_code == 401:
                error_msg = "Authentication failed. Please check your API key."
            elif status_code == 403:
                error_msg = "Access forbidden. Please check your API permissions."
            elif status_code == 429:
                error_msg = "Rate limit exceeded. Please try again later."
            elif status_code >= 500:
                error_msg = "MiniMax server error. Please try again later."
            else:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get("message", str(e))
                except Exception:
                    error_msg = str(e)
            # Sanitize error message to avoid leaking sensitive info
            error_msg = _sanitize_error_message(error_msg)
            return LLMResponse(content=f"Error calling MiniMax: {error_msg}", finish_reason="error")
        except Exception as e:
            # Sanitize error message to avoid leaking sensitive info
            error_msg = _sanitize_error_message(str(e))
            return LLMResponse(content=f"Error calling MiniMax: {error_msg}", finish_reason="error")

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse MiniMax native API response."""
        if not data:
            logger.error("MiniMax response data is empty or None")
            return LLMResponse(content="Error: Empty response from MiniMax", finish_reason="error")

        choices = data.get("choices")
        if not choices:
            logger.error("MiniMax response has no choices: {}", data)
            return LLMResponse(content="Error: No choices in MiniMax response", finish_reason="error")

        choice = choices[0]
        if not choice:
            logger.error("MiniMax response first choice is empty: {}", data)
            return LLMResponse(content="Error: Empty choice in MiniMax response", finish_reason="error")

        message = choice.get("message", {}) if choice else {}

        content = message.get("content", "")
        reasoning = message.get("reasoning_content")

        # Parse tool calls if present
        tool_calls = []
        raw_tool_calls = message.get("tool_calls", []) or []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", {})
            # Arguments can be a string or dict
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCallRequest(
                id=tc.get("id", _short_tool_id()),
                name=func.get("name", ""),
                arguments=args,
            ))

        usage = data.get("usage", {})

        finish_reason = choice.get("finish_reason") if choice else "stop"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            reasoning_content=reasoning,
        )

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
