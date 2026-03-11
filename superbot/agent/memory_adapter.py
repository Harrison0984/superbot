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

    def get_memory_context(self, query: str = "", top_n: int = 5) -> str:
        """Get formatted memory context for system prompt.

        Args:
            query: Current query to recall relevant memories.
            top_n: Number of memories to retrieve.

        Returns:
            Formatted memory context string for system prompt.
        """
        if self._memory_system is None:
            return ""

        try:
            results = self._memory_system.recall(query_text=query, top_n=top_n)

            # Format memory for system prompt
            memory_parts = []

            # Add facts if available
            facts = results.get("facts", [])
            if facts:
                memory_parts.append("## Relevant Facts")
                for fact in facts:
                    content = fact.get("content", "")
                    if content:
                        memory_parts.append(f"- {content}")

            # Add raw logs if available
            raw_logs = results.get("raw_logs", [])
            if raw_logs:
                memory_parts.append("\n## Conversation Context")
                for log in raw_logs[:3]:  # Limit to 3 most relevant
                    content = log.get("content", "")
                    if content:
                        memory_parts.append(f"- {content[:200]}...")

            # Add LLM understanding if available
            understanding = results.get("understanding")
            if understanding:
                memory_parts.append(f"\n## Context Understanding\n{understanding}")

            # Add relations if available
            relations = results.get("relations", [])
            if relations:
                memory_parts.append("\n## Related Entities")
                for rel in relations[:5]:  # Limit to 5 most relevant
                    head = rel.get("head", "")
                    relation = rel.get("relation", "")
                    tail = rel.get("tail", "")
                    if head and relation and tail:
                        memory_parts.append(f"- {head} → {relation} → {tail}")

            if not memory_parts:
                return ""

            return "\n".join(memory_parts)

        except Exception as e:
            logger.error("Error getting memory context: {}", e)
            return ""

    def remember(self, text: str, analyze: bool = True) -> bool:
        """Store new memory in real-time.

        Args:
            text: Text content to remember.
            analyze: If True, analyze and extract triples. If False, only save without analysis.

        Returns:
            True if successfully stored, False otherwise.
        """
        if self._memory_system is None:
            logger.debug("Memory system not available, skipping remember")
            return False

        try:
            result = self._memory_system.remember(text, analyze=analyze)
            if not result:
                logger.debug("Memory filtered by entropy gatekeeper: {}", text[:50])
            return result
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
