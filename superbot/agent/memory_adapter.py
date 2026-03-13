"""Memory adapter for vector-based memory system.

This module provides an adapter to make the MemorySystem compatible with
the existing MemoryStore interface used by ContextBuilder.
"""

from pathlib import Path
from typing import Any

from loguru import logger


class MemoryAdapter:
    """Adapter that wraps MemorySystem to provide MemoryStore-compatible interface.

    This adapter provides:
    - get_memory_context(): Returns formatted memory for system prompt
    - remember(): Stores new memories in real-time
    - recall(): Retrieves relevant memories for a query
    """

    def __init__(
        self,
        memory_system,
        embedding_provider=None,
        llm_provider=None,
    ):
        """Initialize the memory adapter.

        Args:
            memory_system: MemorySystem instance from superbot.memory.
            embedding_provider: EmbeddingProvider for vector encoding.
            llm_provider: LLMProvider for triple extraction and context understanding.
        """
        self._memory_system = memory_system

        # Set providers if provided
        if embedding_provider:
            self._memory_system.set_embedding(embedding_provider)
        if llm_provider:
            self._memory_system.set_llm(llm_provider)

        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider

    def get_memory_context(self, query: str = "") -> str:
        """Get formatted memory context for system prompt.

        Args:
            query: Current query to recall relevant memories.

        Returns:
            Formatted memory context string for system prompt.
        """
        if self._memory_system is None:
            return ""

        try:
            return self._memory_system.get_memory_context(query)
        except Exception as e:
            logger.error("Error getting memory context: {}", e)
            return ""

    async def remember(self, text: str) -> bool:
        """Store new memory in real-time (异步版本).

        Args:
            text: Text content to remember.

        Returns:
            True if successfully stored, False otherwise.
        """
        if self._memory_system is None:
            logger.debug("Memory system not available, skipping remember")
            return False

        try:
            await self._memory_system.remember(text)
            return True
        except Exception as e:
            logger.warning("Error storing memory: {}", e)
            return False

    def recall(self, query: str, top_n: int = 5) -> dict[str, Any]:
        """Retrieve relevant memories for a query.

        Args:
            query: Query text to search memories.
            top_n: Number of results to return.

        Returns:
            Dictionary with facts, raw_logs, relations, etc.
        """
        if self._memory_system is None:
            return {}

        try:
            return self._memory_system.recall(query_text=query, top_n=top_n)
        except Exception as e:
            logger.error("Error recalling memories: {}", e)
            return {}

    def shutdown(self):
        """Shutdown the memory system."""
        if self._memory_system is not None:
            try:
                self._memory_system.shutdown()
            except Exception as e:
                logger.error("Error shutting down memory system: {}", e)


def create_memory_adapter(
    workspace: Path,
    embedding_provider=None,
    llm_provider=None,
    memory_config=None,
) -> tuple[MemoryAdapter | None, Any]:
    """Create MemoryAdapter from workspace and providers.

    Args:
        workspace: Workspace path for data storage.
        embedding_provider: EmbeddingProvider instance.
        llm_provider: LLMProvider instance.
        memory_config: MemoryConfig instance (optional, will create if not provided).

    Returns:
        Tuple of (MemoryAdapter instance or None, memory_config or None).
    """
    try:
        from superbot.memory.facade.memory_system import MemorySystem
        from superbot.memory.config import Config as MemoryConfig

        # Create memory system with data directory in workspace
        data_dir = str(workspace / "memory" / "data")

        # Create memory config with reasonable defaults if not provided
        if memory_config is None:
            memory_config = MemoryConfig()

        memory_system = MemorySystem(
            data_dir=data_dir,
            config=memory_config,
        )

        # Set providers
        if embedding_provider:
            memory_system.set_embedding(embedding_provider)
        if llm_provider:
            memory_system.set_llm(llm_provider)

        adapter = MemoryAdapter(
            memory_system=memory_system,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

        return adapter, memory_config

    except ImportError as e:
        logger.error("Failed to import memory system: {}", e)
        return None, None
    except Exception as e:
        logger.error("Failed to create memory adapter: {}", e)
        return None, None
