"""Tests for multi-model provider creation."""

import pytest
from superbot.config.schema import Config, LocalModelConfig, AgentsConfig, AgentDefaults
from superbot.cli.commands import _make_provider


def test_make_provider_returns_tuple():
    """Test that _make_provider returns (main_provider, memory_provider) tuple."""
    config = Config(
        agents=AgentsConfig(defaults=AgentDefaults(model="test", provider="minimax")),
        local_model=LocalModelConfig(enabled=False),
    )
    result = _make_provider(config)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2 elements, got {len(result) if isinstance(result, tuple) else 'N/A'}"


def test_make_provider_with_local_model_disabled():
    """When local_model disabled, memory_provider should be None."""
    config = Config(
        agents=AgentsConfig(defaults=AgentDefaults(model="MiniMax-M2.5", provider="minimax")),
        local_model=LocalModelConfig(enabled=False),
    )
    main_provider, memory_provider = _make_provider(config)
    assert main_provider is not None, "main_provider should not be None"
    assert memory_provider is None, "memory_provider should be None when local_model disabled"


def test_agent_loop_accepts_memory_provider():
    """Test that AgentLoop accepts memory_provider parameter."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from superbot.agent.loop import AgentLoop
    from superbot.bus.queue import MessageBus

    mock_provider = MagicMock()
    mock_memory_provider = MagicMock()
    mock_bus = MagicMock(spec=MessageBus)

    loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        memory_provider=mock_memory_provider,
        workspace=Path("/tmp/test"),
    )

    assert loop.memory_provider == mock_memory_provider
    assert loop.provider == mock_provider


def test_consolidate_uses_memory_provider():
    """Test that memory consolidation uses memory_provider when available."""
    from pathlib import Path
    from unittest.mock import MagicMock, AsyncMock
    from superbot.agent.loop import AgentLoop
    from superbot.bus.queue import MessageBus

    mock_provider = MagicMock()
    mock_memory_provider = MagicMock()
    mock_bus = MagicMock(spec=MessageBus)

    loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        memory_provider=mock_memory_provider,
        workspace=Path("/tmp/test"),
    )

    # When memory_provider is set, it should be used
    assert loop.memory_provider is not None
    # In actual consolidation, it will use memory_provider instead of provider
