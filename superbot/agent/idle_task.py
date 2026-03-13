"""Idle task system for AgentLoop."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superbot.agent.loop import AgentLoop

import requests
from loguru import logger

DEFAULT_IDLE_THRESHOLD = 180  # 3 minutes


def get_provider_quota(agent: "AgentLoop" = None) -> dict[str, Any]:
    """Get quota info from provider. Call manually when needed.

    Args:
        agent: Optional AgentLoop instance. If not provided, reads from config.

    Returns: {"total": float, "used": float, "provider": str}
    """
    from superbot.providers.minimax_provider import MiniMaxProvider

    # Try to get from agent first
    if agent:
        provider = agent.memory_provider or agent.provider
        if provider:
            if isinstance(provider, MiniMaxProvider):
                api_key = getattr(provider, "api_key", None)
                if api_key:
                    return _fetch_minimax_quota(api_key)

    # Fallback: read from config
    try:
        from superbot.config import load_config
        config = load_config()
        api_key = config.providers.minimax.api_key
        if api_key:
            return _fetch_minimax_quota(api_key)
        return {"error": "No API key in config"}
    except Exception as e:
        return {"error": str(e)}


def _fetch_minimax_quota(api_key: str) -> dict[str, Any]:
    """Fetch quota from MiniMax API."""
    try:
        url = "https://www.minimaxi.com/v1/api/openplatform/coding_plan/remains"
        resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return _parse_minimax_response(data)
        return {"error": f"HTTP {resp.status_code}", "detail": resp.text[:100]}
    except Exception as e:
        return {"error": str(e)}


def _parse_minimax_response(data: dict) -> dict[str, Any]:
    """Parse MiniMax quota response."""
    model_remains = data.get("model_remains", [])
    if not model_remains:
        return {"total": 0, "used": 0, "provider": "MiniMax", "raw": data}

    # Models share the same quota pool, use the first one
    m = model_remains[0]
    total = m.get("current_interval_total_count", 0)
    used = m.get("current_interval_usage_count", 0)

    # Note: field name is misleading, usage_count actually seems to be remaining
    remaining = used
    used = total - remaining

    models = []
    for m in model_remains:
        models.append({
            "model": m.get("model_name", "unknown"),
            "total": m.get("current_interval_total_count", 0),
            "used": m.get("current_interval_total_count", 0) - m.get("current_interval_usage_count", 0),
            "remaining": m.get("current_interval_usage_count", 0)
        })

    return {
        "total": total,
        "used": used,
        "remaining": remaining,
        "provider": "MiniMax",
        "models": models
    }


class IdleTask(ABC):
    """Base class for tasks executed during idle time."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Task name for logging."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Task type (unique identifier for same-type filtering)."""
        pass

    @property
    def idle_threshold_seconds(self) -> int:
        """Minimum idle time required for task in seconds, default 3 minutes."""
        return DEFAULT_IDLE_THRESHOLD

    @property
    def enabled(self) -> bool:
        """Whether task is enabled."""
        return True

    @abstractmethod
    async def should_run(self, agent: "AgentLoop", idle_seconds: float) -> bool:
        """Whether to execute this task.

        Args:
            agent: AgentLoop instance
            idle_seconds: Current idle duration in seconds

        Returns:
            True if should execute, False to skip
        """
        pass

    @abstractmethod
    async def execute(self, agent: "AgentLoop") -> None:
        """Execute task.

        Note: This method is called during idle time, task can implement its own loop logic internally.
        """
        pass
