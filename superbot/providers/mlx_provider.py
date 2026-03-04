"""MLX VLM provider for Apple Silicon local models."""

from __future__ import annotations

import json
import re
from typing import Any

from superbot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


# Match <tool_call>...</tool_call> block (with or without closing tag)
# The model sometimes doesn't output the closing </tool_call> tag
_TOOL_CALL_PATTERN = re.compile(r'<tool_call>.*?(</tool_call>|$)', re.DOTALL)


class MLXProvider(LLMProvider):
    """MLX provider for Apple Silicon local models.

    Uses mlx_lm library directly for inference.
    Example models: Qwen-MLX, Llama-3-MLX, etc.

    Usage:
        1. Download MLX model (e.g., from HuggingFace)
        2. Set model path in config
        3. Run superbot agent

    Config example:
        {
            "local_model": {"enabled": true, "provider": "mlx", "path": "/path/to/Qwen-MLX"}
        }
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "qwen2.5-0.5b-instruct-mlx",
    ):
        super().__init__(api_key, api_base)
        # api_base is used as model path (full path required)
        self.model_path = api_base or default_model
        self.default_model = default_model
        self._model = None
        self._tokenizer = None
        self._tools: list[dict[str, Any]] | None = None

    def _load_model(self):
        """Load MLX model and tokenizer."""
        import os

        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError(
                "mlx_lm not installed. Run: pip install mlx-lm"
            )

        # Check if model path exists
        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"MLX model path does not exist: {self.model_path}\n"
                "Please download a model or update the path in config.json"
            )

        try:
            if self._model is None:
                self._model, self._tokenizer = load(self.model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MLX model from {self.model_path}: {e}"
            )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Generate response using MLX model."""
        # Store tools for tool call detection
        self._tools = tools

        # Load model if not loaded
        self._load_model()

        # Convert messages to prompt with tools description
        prompt = self._build_prompt(messages, tools)

        try:
            from mlx_lm import generate

            # Generate response
            response = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )

            # Check for tool calls in response
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                return LLMResponse(
                    content=response,
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    usage={},
                )

            return LLMResponse(
                content=response,
                tool_calls=[],
                finish_reason="stop",
                usage={},
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error generating with MLX: {str(e)}",
                finish_reason="error",
            )

    def _parse_tool_calls(self, response: str) -> list[ToolCallRequest]:
        """Parse tool calls from model response."""
        tool_calls = []

        # Find all <tool_call> tags and extract JSON
        for match in _TOOL_CALL_PATTERN.finditer(response):
            text = match.group(0)
            try:
                data = self._extract_json(text)
                if data:
                    name = data.get("name", "")
                    params = data.get("params", data.get("arguments", {}))
                    if name:
                        tool_calls.append(ToolCallRequest(
                            id=f"mlx_{len(tool_calls)}",
                            name=name,
                            arguments=params,
                        ))
            except Exception:
                continue

        return tool_calls

    def _extract_json(self, text: str) -> dict | None:
        """Extract valid JSON from text, handling nested objects."""
        # Find the first { and track brackets
        start = text.find('{')
        if start == -1:
            return None

        # Count brackets to find matching closing
        depth = 0
        end = start
        # Use substring starting from start for correct indexing
        substr = text[start:]
        for i, c in enumerate(substr):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = start + i + 1
                    break

        json_str = text[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _build_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        """Build prompt from messages in chat format."""
        parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Skip tool messages - we'll add their results manually
            if role == "tool":
                tool_name = msg.get("name", "unknown")
                tool_result = msg.get("content", "")
                parts.append(f"Tool Result ({tool_name}): {tool_result}")
                continue

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        # Add tools description if available
        if tools:
            tools_desc = self._format_tools(tools)
            parts.append(f"\n{tools_desc}\n")

        # Add final prompt for assistant
        parts.append("Assistant:")

        return "\n\n".join(parts)

    def _format_tools(self, tools: list[dict[str, Any]]) -> str:
        """Format tools for the prompt."""
        if not tools:
            return ""

        lines = ["You can call these tools when needed:"]
        lines.append("<tools>")

        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})

            lines.append(f'<tool_call>')
            lines.append(f'{{"name": "{name}", "params": ')
            # Build params schema
            param_lines = []
            required = params.get("required", [])
            properties = params.get("properties", {})
            for pname, pinfo in properties.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                required_str = ", required" if pname in required else ""
                param_lines.append(f'  "{pname}": {ptype}  # {pdesc}{required_str}')

            if param_lines:
                lines.append("  {")
                lines.append(",\n".join(param_lines))
                lines.append("  }")
            else:
                lines.append("  {}")

            lines.append(f'}}')
            lines.append(f'</tool_call>')
            lines.append(f"# {desc}")

        lines.append("</tools>")
        lines.append("When you need to call a tool, use the format: <tool_call>{\"name\": \"tool_name\", \"params\": {...}}</tool_call>")

        return "\n".join(lines)

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
