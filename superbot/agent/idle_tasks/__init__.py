"""Idle tasks for AgentLoop."""

from superbot.agent.idle_task import IdleTask, DEFAULT_IDLE_THRESHOLD
from superbot.agent.idle_tasks.cleanup import CleanupIdleTask

__all__ = [
    "IdleTask",
    "DEFAULT_IDLE_THRESHOLD",
    "CleanupIdleTask",
]
