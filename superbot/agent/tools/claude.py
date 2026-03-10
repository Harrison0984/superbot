"""Claude tool - MCP client for Claude Code execution."""

import asyncio
import os
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from superbot.agent.tools.base import Tool

if TYPE_CHECKING:
    from mcp import ClientSession

try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class ClaudeTool(Tool):
    """Tool for executing Claude Code via MCP.

    This tool provides direct access to Claude Code for coding tasks,
    code review, and programming-related requests.
    """

    def __init__(self, config: Any = None):
        self.config = config

        # MCP client configuration
        self.workdir = ""
        self.script_path = "superbot/scripts/fastmcp_server.py"
        self.auto_start = True
        self.retry_count = 3
        self.retry_delay = 1.0
        self.retry_backoff = 2.0
        self.timeout = 120

        # Parse config
        if config:
            self.workdir = getattr(config, 'workdir', '') or ""
            self.script_path = getattr(config, 'script_path', '') or "superbot/scripts/fastmcp_server.py"
            self.auto_start = getattr(config, 'auto_start', True)
            self.retry_count = getattr(config, 'retry_count', 3)
            self.retry_delay = getattr(config, 'retry_delay', 1.0)
            self.retry_backoff = getattr(config, 'retry_backoff', 2.0)

        # Persistent session state
        self._process: Optional[asyncio.subprocess.Process] = None
        self._session: Optional["ClientSession"] = None
        self._read = None
        self._write = None
        self._lock = asyncio.Lock()

    def initialize(self, bus: Any) -> None:
        """Initialize and start MCP client."""
        super().initialize(bus)
        if self.config and getattr(self.config, 'enabled', False):
            logger.info("ClaudeTool MCP client initialized: workdir={}, script={}", self.workdir, self.script_path)

    @property
    def name(self) -> str:
        return "claude"

    @property
    def description(self) -> str:
        return "Execute Claude Code for coding tasks, code review, or programming assistance. Use this for writing, debugging, or analyzing code."

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

    # ==================== MCP Client Methods ====================

    def _get_cwd(self) -> str:
        """Get the working directory."""
        return self.workdir or os.getcwd()

    def _get_env(self) -> dict:
        """Get clean environment."""
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        return env

    async def _ensure_connection(self) -> bool:
        """Ensure we have a valid connection, reconnect if needed."""
        async with self._lock:
            if self._session is not None:
                try:
                    return True
                except Exception:
                    await self._disconnect()
            return await self._connect()

    async def _connect(self) -> bool:
        """Establish a new connection to MCP server."""
        if not MCP_AVAILABLE:
            return False

        try:
            # Start server if not running
            if self._process is None or self._process.returncode is not None:
                cwd = self._get_cwd()
                logger.info("Starting MCP server: python {} in {}", self.script_path, cwd)

                self._process = await asyncio.create_subprocess_exec(
                    "python",
                    self.script_path,
                    cwd=cwd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._get_env(),
                )

                # Wait for server to start
                await asyncio.sleep(2)

                if self._process.returncode is not None:
                    logger.error("MCP server failed to start, return code: {}", self._process.returncode)
                    self._process = None
                    return False

                logger.info("MCP server started, PID: {}", self._process.pid)

            # Create stdio connection
            server_params = StdioServerParameters(
                command="python",
                args=[self.script_path],
                cwd=self._get_cwd(),
                env=self._get_env(),
            )

            self._read, self._write = await stdio_client(server_params).__aenter__()
            self._session = ClientSession(self._read, self._write)
            await self._session.initialize()

            logger.info("MCP session established")
            return True

        except Exception as e:
            logger.error("Failed to connect to MCP: {}", e)
            await self._disconnect()
            return False

    async def _disconnect(self) -> None:
        """Disconnect and cleanup."""
        try:
            if self._session:
                try:
                    await self._session.close()
                except Exception:
                    pass
                self._session = None

            if self._read:
                try:
                    await self._read.aclose()
                except Exception:
                    pass
                self._read = None

            if self._write:
                try:
                    await self._write.aclose()
                except Exception:
                    pass
                self._write = None

        except Exception as e:
            logger.warning("Error disconnecting: {}", e)

        finally:
            await self._stop_mcp_server()

    async def _stop_mcp_server(self) -> None:
        """Stop the MCP server process."""
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    self._process.kill()
                except Exception:
                    pass
            except Exception as e:
                logger.warning("Error stopping MCP server: {}", e)
            finally:
                self._process = None

    async def call_tool(self, tool_name: str, arguments: dict) -> Optional[dict]:
        """Call an MCP tool with retry mechanism."""
        if not MCP_AVAILABLE:
            logger.error("MCP SDK not installed")
            return None

        delay = self.retry_delay

        for attempt in range(1, self.retry_count + 1):
            try:
                if not await self._ensure_connection():
                    raise Exception("Failed to establish MCP connection")

                result = await asyncio.wait_for(
                    self._session.call_tool(tool_name, arguments),
                    timeout=self.timeout
                )
                return result

            except Exception as e:
                logger.warning("MCP call failed (attempt {}/{}): {}", attempt, self.retry_count, e)
                await self._disconnect()

                if attempt < self.retry_count:
                    await asyncio.sleep(delay)
                    delay *= self.retry_backoff

        logger.error("MCP call failed after {} attempts", self.retry_count)
        return None

    async def chat(self, content: str) -> str:
        """Call Claude Code chat via MCP."""
        result = await self.call_tool("chat", {"content": content})
        if result and result.content:
            return result.content[0].text
        return "Error: Failed to execute content"

    async def ensure_available(self) -> bool:
        """Ensure MCP is available, start if needed."""
        return await self._ensure_connection()

    async def close(self) -> None:
        """Close the MCP client and stop server."""
        await self._disconnect()

    # ==================== Tool execute ====================

    async def execute(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        """Execute Claude Code via MCP."""
        if not MCP_AVAILABLE:
            return '{"_tool_error": {"type": "not_configured", "message": "MCP SDK not installed"}}'

        if not self.config or not getattr(self.config, 'enabled', False):
            return '{"_tool_error": {"type": "not_configured", "message": "Claude not configured"}}'

        try:
            result = await self.chat(content)
            return result or "No response from Claude"
        except Exception as e:
            logger.error("Claude call failed: {}", e)
            return f'{{"_tool_error": {{"type": "execution_error", "message": "Claude failed: {str(e)}"}}}}'


# Convenience function for simple usage
async def call_mcp_tool(
    tool_name: str = "chat",
    arguments: dict = None,
    retry_count: int = 3,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    auto_start: bool = True,
) -> Optional[dict]:
    """Call an MCP tool via stdio transport with retry mechanism."""
    client = ClaudeTool(config=type('Config', (), {
        'enabled': True,
        'workdir': '',
        'script_path': 'superbot/scripts/fastmcp_server.py',
        'auto_start': auto_start,
        'retry_count': retry_count,
        'retry_delay': retry_delay,
        'retry_backoff': retry_backoff,
    })())

    if arguments is None:
        arguments = {}

    if not await client.ensure_available():
        return None

    return await client.call_tool(tool_name, arguments)
