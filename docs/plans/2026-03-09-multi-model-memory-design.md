# Multi-Model Architecture Design

## Overview

Support multiple LLM models simultaneously in superbot, where:
- **LocalModelConfig** (MLX) = Memory system (dedicated local model for memory consolidation)
- **Other providers** (minimax, openai, etc.) = Main agent for user conversations

## Problem

Currently superbot uses a single model for both conversation and memory consolidation. Memory consolidation is a background task that can use a smaller, local model, freeing up the main model for user interactions.

## Solution

### 1. Configuration Layer

Use existing `LocalModelConfig` in `config/schema.py`:

```python
local_model:
  enabled: true
  provider: mlx
  path: /path/to/local/model
```

When `LocalModelConfig` is enabled:
- MLX provider is used for memory consolidation
- Main agent uses `agents.defaults.provider` configuration

### 2. Provider Initialization

In `cli/commands.py`, modify `_make_provider()` to return both providers:

```python
def _make_provider(config: Config):
    # Existing: main agent provider
    main_provider = _create_main_provider(config)

    # New: memory-specific MLX provider
    memory_provider = None
    if config.local_model and config.local_model.enabled:
        memory_provider = _create_mlx_provider(config.local_model)

    return main_provider, memory_provider
```

### 3. AgentLoop Changes

Modify `AgentLoop.__init__` to accept memory provider:

```python
def __init__(
    self,
    bus: MessageBus,
    provider: LLMProvider,  # Main agent provider
    memory_provider: LLMProvider | None = None,  # MLX for memory
    ...
):
    self.provider = provider
    self.memory_provider = memory_provider
    ...
```

### 4. Memory Consolidation

In `memory.py`, use `memory_provider` when available, fallback to main provider:

```python
async def consolidate(self, session, main_provider, memory_provider, ...):
    provider = memory_provider or main_provider
    ...
```

### 5. Data Flow

```
User Message
    │
    ▼
Main Agent (minimax/openai/etc.) ─────► Conversation
    │
    ▼
Memory Consolidation (idle time)
    │
    ▼
Memory Provider (MLX local) ─────► MEMORY.md + HISTORY.md
```

## Configuration Example

```yaml
agents:
  defaults:
    model: MiniMax-M2.5
    provider: minimax

local_model:
  enabled: true
  provider: mlx
  path: /Users/heyunpeng/models/llama-3-8b
```

## Error Handling

- If `LocalModelConfig` is disabled or not configured: fallback to main provider for memory
- If MLX provider initialization fails: log warning, fallback to main provider
- Memory consolidation failure should not block main agent operation

## Testing

1. Unit test provider initialization with both configs
2. Integration test memory consolidation with MLX provider
3. Fallback test when local model is disabled
