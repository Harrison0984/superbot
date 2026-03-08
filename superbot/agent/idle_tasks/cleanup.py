"""Cleanup idle task."""

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from superbot.agent.idle_task import IdleTask

if TYPE_CHECKING:
    from superbot.agent.loop import AgentLoop


class CleanupIdleTask(IdleTask):
    """空闲时执行清理任务"""

    def __init__(self, min_idle_seconds: int = 30):
        self.min_idle_seconds = min_idle_seconds

    @property
    def name(self) -> str:
        return "cleanup"

    @property
    def task_type(self) -> str:
        return "cleanup"

    @property
    def idle_threshold_seconds(self) -> int:
        return self.min_idle_seconds

    async def should_run(self, agent: "AgentLoop", idle_seconds: float) -> bool:
        return idle_seconds >= self.min_idle_seconds

    async def execute(self, agent: "AgentLoop") -> None:
        # 清理临时文件
        temp_dir = agent.workspace / "temp"
        if temp_dir.exists():
            for f in temp_dir.glob("*.tmp"):
                try:
                    f.unlink()
                    logger.debug("Deleted temp file: {}", f)
                except Exception as e:
                    logger.warning("Failed to delete temp file {}: {}", f, e)
