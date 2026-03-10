"""Tool registry for dynamic tool management."""

from typing import TYPE_CHECKING, Any

from superbot.agent.tools.base import Tool, tool_error

if TYPE_CHECKING:
    from superbot.bus.queue import MessageBus


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._bus: "MessageBus | None" = None

    def set_bus(self, bus: "MessageBus") -> None:
        """Set the message bus for all tools."""
        self._bus = bus

    def register(self, tool: Tool) -> None:
        """Register a tool and auto-initialize it."""
        self._tools[tool.name] = tool
        # Auto-initialize if bus is set
        if self._bus:
            tool.initialize(self._bus)

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        params: dict[str, Any],
    ) -> str:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return tool_error("not_found", f"Tool '{name}' not found", available=list(self.tool_names))

        try:
            errors = tool.validate_params(params)
            if errors:
                return tool_error("invalid_params", f"Invalid parameters for tool '{name}': " + "; ".join(errors))
            result = await tool.execute(channel, sender_id, chat_id, content, **params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
