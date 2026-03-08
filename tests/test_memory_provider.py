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
