"""Reflection idle task for MemorySystem."""

from typing import TYPE_CHECKING

from loguru import logger

from superbot.agent.idle_task import IdleTask

if TYPE_CHECKING:
    from superbot.agent.loop import AgentLoop
    from superbot.memory.facade.memory_system import MemorySystem


class ReflectionIdleTask(IdleTask):
    """Reflection task executed during idle time."""

    def __init__(self, memory_system: "MemorySystem"):
        self._memory_system = memory_system

    @property
    def name(self) -> str:
        return "reflection"

    @property
    def task_type(self) -> str:
        return "reflection"

    @property
    def idle_threshold_seconds(self) -> int:
        return self._memory_system.config.reflection_timeout

    async def should_run(self, agent: "AgentLoop", idle_seconds: float) -> bool:
        """Check if reflection should run."""
        # Check if idle time meets threshold
        if idle_seconds < self.idle_threshold_seconds:
            return False

        # Check quota (pass agent for provider access)
        return self._memory_system._check_quota_sufficient(agent)

    async def execute(self, agent: "AgentLoop") -> None:
        """Execute reflection task."""
        logger.info("[Memory] ReflectionIdleTask executing")
        try:
            await self._memory_system._do_reflection()
        except Exception as e:
            logger.error("[Memory] ReflectionIdleTask error: {}", e)
