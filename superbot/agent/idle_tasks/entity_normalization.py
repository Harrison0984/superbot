"""Entity normalization idle task."""

from superbot.agent.idle_task import IdleTask


class EntityNormalizationIdleTask(IdleTask):
    """Idle task for normalizing entities in memory."""

    def __init__(self, idle_threshold_seconds: int = 3600):
        """Initialize with 1 hour threshold."""
        self._idle_threshold = idle_threshold_seconds

    @property
    def name(self) -> str:
        return "Entity Normalization"

    @property
    def task_type(self) -> str:
        return "entity_normalization"

    @property
    def idle_threshold_seconds(self) -> int:
        return self._idle_threshold

    async def should_run(self, agent, idle_seconds: float) -> bool:
        # Only run if memory system is available
        return agent.memory_system is not None

    async def execute(self, agent) -> None:
        """Execute entity normalization."""
        if agent.memory_system is None:
            return

        try:
            # Call normalize on memory system
            normalized = agent.memory_system.normalize_entities()
            if normalized > 0:
                agent.bus.logger.info("Normalized {} entity pairs", normalized)
        except Exception as e:
            agent.bus.logger.error("Entity normalization failed: {}", e)
