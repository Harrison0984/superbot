"""Idle tasks for AgentLoop."""

from superbot.agent.idle_task import IdleTask, DEFAULT_IDLE_THRESHOLD
from superbot.agent.idle_tasks.cleanup import CleanupIdleTask
from superbot.agent.idle_tasks.entity_normalization import EntityNormalizationIdleTask

__all__ = [
    "IdleTask",
    "DEFAULT_IDLE_THRESHOLD",
    "CleanupIdleTask",
    "EntityNormalizationIdleTask",
]
