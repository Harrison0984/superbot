"""Provider adapters for vector-based memory system.

This module provides adapters to make superbot's LLM and embedding providers
compatible with the MemorySystem's protocol requirements.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from superbot.config.schema import EmbeddingConfig

# Import from superbot.memory if available, otherwise use local Protocol definitions
try:
    from superbot.memory.models.protocols import EmbeddingProvider as _EmbeddingProvider
    from superbot.memory.models.protocols import LLMProvider as _LLMProvider

    class EmbeddingProvider(_EmbeddingProvider):
        """Protocol for embedding providers."""
        pass

    class LLMProvider(_LLMProvider):
        """Protocol for LLM providers."""
        pass
except ImportError:
    from typing import Protocol as _Protocol

    class EmbeddingProvider(_Protocol):
        """Protocol for embedding providers."""

        def encode(self, text: str) -> np.ndarray:
            """Encode a single text to vector."""
            ...

        def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
            """Encode multiple texts to vectors."""
            ...

        def dimension(self) -> int:
            """Return embedding dimension."""
            ...

    class LLMProvider(_Protocol):
        """Protocol for LLM providers."""

        def generate(self, prompt: str, **kwargs) -> str:
            """Generate text from prompt."""
            ...

        def extract_triples(self, text: str) -> list[dict[str, Any]]:
            """Extract knowledge triples from text."""
            ...

        def compress(self, text: str) -> str:
            """Compress text to summary."""
            ...

        def understand_context(
            self,
            query: str,
            memory: list[str],
            history: list[str],
        ) -> str:
            """Understand context from query, memory, and history."""
            ...


# Reusable thread pool for async LLM calls (module-level singleton)
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Get or create shared ThreadPoolExecutor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4)
    return _executor


def shutdown_executor():
    """Shutdown the shared ThreadPoolExecutor."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


# Cleanup executor on application exit
import atexit
atexit.register(shutdown_executor)


class SuperbotEmbeddingAdapter:
    """Adapter for superbot's embedding functionality using sentence-transformers.

    This adapter loads a local sentence-transformers model and provides
    the encode methods required by the MemorySystem.
    """

    def __init__(self, model_path: str):
        """Initialize the embedding adapter.

        Args:
            model_path: Path to the sentence-transformers model directory.
        """
        self._model = None
        self._model_path = model_path
        self._dimension: int | None = None

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_path)
            self._dimension = self._model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text to vector.

        Args:
            text: Text to encode.

        Returns:
            Embedding vector as numpy array.
        """
        self._load_model()
        return self._model.encode(text)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts to vectors.

        Args:
            texts: List of texts to encode.

        Returns:
            List of embedding vectors.
        """
        self._load_model()
        return self._model.encode(texts).tolist()

    def dimension(self) -> int:
        """Return embedding dimension.

        Returns:
            Dimension of the embedding vectors.
        """
        self._load_model()
        return self._dimension


class SuperbotLLMAdapter:
    """Adapter for superbot's LLM provider.

    This adapter wraps a superbot provider (e.g., MiniMaxProvider, OpenAIProvider)
    and provides the methods required by the MemorySystem:
    - generate: Generate text from prompt
    - extract_triples: Extract knowledge triples
    - compress: Compress text to summary
    - understand_context: Understand context from query/memory/history
    """

    def __init__(
        self,
        provider: Any,
        default_model: str = "default",
        config: "EmbeddingConfig | None" = None,
    ):
        """Initialize the LLM adapter.

        Args:
            provider: Superbot LLM provider instance.
            default_model: Default model to use for generation.
            config: EmbeddingConfig instance for LLM parameters.
        """
        self._provider = provider
        self._default_model = default_model
        self._config = config

    def _calculate_max_input_chars(self, output_tokens: int) -> int:
        """Calculate max input characters based on output token limit.

        Args:
            output_tokens: Number of tokens reserved for output.

        Returns:
            Maximum input characters.
        """
        max_tokens = self._config.llm_max_tokens if self._config else 262144
        chars_per_token = self._config.chars_per_token if self._config else 4
        return (max_tokens - output_tokens - 200) * chars_per_token

    def _get_temperature(self, default: float = 0.7, method: str = "") -> float:
        """Get temperature from config or use default.

        Args:
            default: Default temperature if not configured.
            method: Method name for method-specific temperature (e.g., "triple", "compress", "context").

        Returns:
            Temperature value.
        """
        if self._config:
            # Check method-specific temperature first
            if method:
                temp = getattr(self._config, f"{method}_temperature", None)
                if temp is not None:
                    return temp
            # Fall back to general llm_temperature
            temp = getattr(self._config, "llm_temperature", None)
            if temp is not None:
                return temp
        return default

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt.
            **kwargs: Additional generation parameters (max_tokens, temperature, etc.).

        Returns:
            Generated text.
        """
        import asyncio

        # Use config values as defaults
        default_max_tokens = self._config.llm_max_tokens if self._config else 1024
        default_temperature = self._config.llm_temperature if self._config else 0.7

        max_tokens = kwargs.get("max_tokens", default_max_tokens)
        temperature = kwargs.get("temperature", default_temperature)

        # Check if provider has async chat
        chat_method = getattr(self._provider, "chat", None)
        if chat_method and asyncio.iscoroutinefunction(chat_method):
            # Provider is async - use shared ThreadPoolExecutor to avoid event loop issues
            def _run_async():
                return asyncio.run(chat_method(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._default_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ))
            executor = _get_executor()
            response = executor.submit(_run_async).result()
        else:
            # Provider is sync
            response = self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self._default_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return response.content or ""

    def extract_triples(self, text: str) -> list[dict[str, Any]]:
        """Extract knowledge triples from text using LLM.

        Args:
            text: Input text to extract triples from.

        Returns:
            List of triple dictionaries with subject, relation, object keys.
        """
        default_max_tokens = self._config.triple_max_tokens if self._config else 512
        max_input_chars = self._calculate_max_input_chars(default_max_tokens)

        # If text fits in one chunk, process normally
        if len(text) <= max_input_chars:
            return self._extract_triples_single(text, default_max_tokens)

        # Split into chunks and process each, then merge results
        all_triples = []
        chunk_size = int(max_input_chars * 0.9)  # 90% to leave some buffer

        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            triples = self._extract_triples_single(chunk, default_max_tokens)
            all_triples.extend(triples)

        # Deduplicate by subject-relation-object
        return self._deduplicate_triples(all_triples)

    def _extract_triples_single(self, text: str, max_tokens: int) -> list[dict[str, Any]]:
        """Extract triples from a single chunk of text."""
        prompt = f"""Extract knowledge triples from the following text.
Return a JSON array of triples in the format: [{{"subject": "...", "relation": "...", "object": "..."}}]

Text: {text}

Triples:"""

        temperature = self._get_temperature(0.3, method="triple")
        response = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        return self._parse_json(response)

    def _deduplicate_triples(self, triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate triples based on subject-relation-object."""
        seen = set()
        unique = []
        for t in triples:
            key = (t.get("subject", ""), t.get("relation", ""), t.get("object", ""))
            if key not in seen and all(key):
                seen.add(key)
                unique.append(t)
        return unique

    def compress(self, text: str) -> str:
        """Compress text to a concise summary.

        Args:
            text: Input text to compress.

        Returns:
            Compressed summary.
        """
        default_max_tokens = self._config.compress_max_tokens if self._config else 128
        max_input_chars = self._calculate_max_input_chars(default_max_tokens)

        # If text fits in one chunk, process normally
        if len(text) <= max_input_chars:
            return self._compress_single(text, default_max_tokens)

        # Split into chunks and compress each, then merge
        summaries = []
        chunk_size = int(max_input_chars * 0.9)

        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            summary = self._compress_single(chunk, default_max_tokens)
            summaries.append(summary)

        # Merge all summaries into one
        merged = " ".join(summaries)
        # If merged is still too long, compress again
        if len(merged) > max_input_chars:
            return self._compress_single(merged[:max_input_chars], default_max_tokens)
        return merged

    def _compress_single(self, text: str, max_tokens: int) -> str:
        """Compress a single chunk of text."""
        prompt = f"""Compress the following text into a concise summary (max 100 characters).
Keep only the most important information.

Text: {text}

Summary:"""

        temperature = self._get_temperature(0.3, method="compress")
        response = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        return response.strip()

    def understand_context(
        self,
        query: str,
        memory: list[str],
        history: list[str],
    ) -> str:
        """Understand context from query, memory, and history.

        Args:
            query: Current user query.
            memory: Relevant memories retrieved.
            history: Recent conversation history.

        Returns:
            Understanding of the context.
        """
        default_max_tokens = self._config.context_max_tokens if self._config else 128

        # Reserve extra 100 tokens for the larger prompt
        max_input_chars = self._calculate_max_input_chars(default_max_tokens + 100)

        # Build prompt and truncate if needed
        memory_str = "\n".join(f"- {m}" for m in memory) if memory else "No relevant memories"
        history_str = "\n".join(f"- {h}" for h in history) if history else "No conversation history"

        prompt = f"""Given the user's current message and context from memory and conversation history,
infer what the user means (especially pronouns like "it", "this", "that").

Current message: {query}

Relevant memory:
{memory_str}

Conversation history:
{history_str}

Understanding (one sentence):"""

        # Truncate prompt if too long
        if len(prompt) > max_input_chars:
            # Truncate memory and history first (keep query)
            memory_chars = max_input_chars // 3
            history_chars = max_input_chars // 3
            query_chars = max_input_chars // 3

            if len(memory_str) > memory_chars:
                memory_str = memory_str[:memory_chars] + "..."
            if len(history_str) > history_chars:
                history_str = history_str[:history_chars] + "..."
            if len(query) > query_chars:
                query = query[:query_chars] + "..."

            prompt = f"""Given the user's current message and context from memory and conversation history,
infer what the user means (especially pronouns like "it", "this", "that").

Current message: {query}

Relevant memory:
{memory_str}

Conversation history:
{history_str}

Understanding (one sentence):"""

        temperature = self._get_temperature(0.3, method="context")
        response = self.generate(prompt, max_tokens=default_max_tokens, temperature=temperature)
        return response.strip()

    @staticmethod
    def _parse_json(response: str) -> list[dict[str, Any]]:
        """Parse JSON from LLM response.

        Args:
            response: LLM response text.

        Returns:
            Parsed JSON as list of dicts, or empty list on failure.
        """
        import re

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON array/object in response
        match = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", response)
        if match:
            try:
                result = json.loads(match.group())
                return result if isinstance(result, list) else [result]
            except json.JSONDecodeError:
                pass

        return []


def create_embedding_provider(config) -> EmbeddingProvider | None:
    """Create embedding provider from config.

    Args:
        config: EmbeddingConfig instance.

    Returns:
        SuperbotEmbeddingAdapter instance if enabled, None otherwise.
    """
    if not config.enabled:
        return None

    return SuperbotEmbeddingAdapter(model_path=config.model_path)


def create_llm_adapter(provider, default_model: str, config=None) -> LLMProvider:
    """Create LLM adapter from superbot provider.

    Args:
        provider: Superbot LLM provider instance.
        default_model: Default model name.
        config: EmbeddingConfig instance for LLM parameters.

    Returns:
        SuperbotLLMAdapter instance.
    """
    return SuperbotLLMAdapter(provider=provider, default_model=default_model, config=config)
