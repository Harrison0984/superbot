"""Memory consolidation idle task."""

import time
from typing import TYPE_CHECKING

from loguru import logger

from superbot.agent.idle_task import IdleTask, DEFAULT_IDLE_THRESHOLD

if TYPE_CHECKING:
    from superbot.agent.loop import AgentLoop


class MemoryConsolidationIdleTask(IdleTask):
    """空闲时执行记忆 consolidation"""

    def __init__(self, min_idle_seconds: int = DEFAULT_IDLE_THRESHOLD):
        self.min_idle_seconds = min_idle_seconds
        self._last_run: float | None = None

    @property
    def name(self) -> str:
        return "memory_consolidation"

    @property
    def task_type(self) -> str:
        return "memory_consolidation"

    @property
    def idle_threshold_seconds(self) -> int:
        return self.min_idle_seconds

    async def should_run(self, agent: "AgentLoop", idle_seconds: float) -> bool:
        # 检查空闲时间
        if idle_seconds < self.min_idle_seconds:
            return False

        # 检查是否有需要 consolidation 的 session
        for session in agent.sessions.all():
            unconsolidated = len(session.messages) - session.last_consolidated
            if unconsolidated >= agent.memory_window:
                return True
        return False

    async def execute(self, agent: "AgentLoop") -> None:
        for session in agent.sessions.all():
            unconsolidated = len(session.messages) - session.last_consolidated
            if unconsolidated >= agent.memory_window:
                logger.info("Running consolidation for session {}", session.key)
                await agent._consolidate_memory(session)

        self._last_run = time.time()
