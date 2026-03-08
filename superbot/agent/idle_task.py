"""Idle task system for AgentLoop."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superbot.agent.loop import AgentLoop

DEFAULT_IDLE_THRESHOLD = 180  # 3 minutes


class IdleTask(ABC):
    """空闲时执行的任务基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """任务名称，用于日志"""
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        """任务类型（唯一标识，用于同类型过滤）"""
        pass

    @property
    def idle_threshold_seconds(self) -> int:
        """任务需要的最小空闲时间（秒），默认 3 分钟"""
        return DEFAULT_IDLE_THRESHOLD

    @property
    def enabled(self) -> bool:
        """任务是否启用"""
        return True

    @abstractmethod
    async def should_run(self, agent: "AgentLoop", idle_seconds: float) -> bool:
        """是否执行此任务

        Args:
            agent: AgentLoop 实例
            idle_seconds: 当前空闲时长（秒）

        Returns:
            True 表示应该执行，False 表示跳过
        """
        pass

    @abstractmethod
    async def execute(self, agent: "AgentLoop") -> None:
        """执行任务

        注意：此方法在空闲时调用，任务内部可自行实现循环逻辑
        """
        pass
