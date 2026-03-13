"""
ClaudeTool - Uses robust MCP client to avoid asyncio issues while maintaining context
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

from superbot.agent.tools.base import Tool

logger = logging.getLogger(__name__)


class ClaudeToolDirect(Tool):
    """Claude Code integration using robust MCP client."""

    def __init__(self, config: Any = None):
        self.config = config
        self.workdir = ""
        self.timeout = 120

        # Parse config
        if config:
            self.workdir = getattr(config, 'workdir', '') or ""
            self.timeout = getattr(config, 'timeout', 120)

        # MCP client (persistent connection)
        self._mcp_client = None

    def _get_cwd(self) -> str:
        """Get the working directory."""
        return self.workdir or os.getcwd()

    def _get_mcp_client(self):
        """Get or create MCP client with persistent connection."""
        if self._mcp_client is None:
            from superbot.agent.tools.robust_mcp import RobustMCPClient
            script_path = "scripts/fastmcp_server.py"
            self._mcp_client = RobustMCPClient(
                script_path=script_path,
                cwd=self._get_cwd(),
                timeout=self.timeout
            )
        return self._mcp_client

    async def execute(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        """Execute Claude Code via MCP using subprocess."""
        loop = asyncio.get_event_loop()

        def _run_mcp():
            try:
                client = self._get_mcp_client()
                result = client.call_tool("chat", {"content": content})
                return result
            except Exception as e:
                logger.error("MCP call failed: {}", e)
                return f"Error: {str(e)}"

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _run_mcp),
                timeout=self.timeout + 30
            )
            return result or "No response from Claude"
        except asyncio.TimeoutError:
            return "Error: Claude request timed out"
        except Exception as e:
            logger.error("Claude call failed: {}", e)
            return f"Error: {str(e)}"

    @property
    def name(self) -> str:
        return "claude"

    @property
    def description(self) -> str:
        return "Execute Claude Code for coding tasks"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The coding task or question for Claude"
                }
            },
            "required": ["content"]
        }
