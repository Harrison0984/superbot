"""Claude MCP client using stdio transport with auto-start and retry mechanism."""

import asyncio
import os
from typing import Optional
from loguru import logger

try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class ClaudeMCPClient:
    """MCP client using stdio transport with persistent session support."""

    def __init__(
        self,
        workdir: str = "",
        script_path: str = "superbot/scripts/fastmcp_server.py",
        auto_start: bool = True,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        timeout: int = 120,
    ):
        self.workdir = workdir
        self.script_path = script_path
        self.auto_start = auto_start
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.timeout = timeout

        # Persistent session state
        self._process: Optional[asyncio.subprocess.Process] = None
        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._lock = asyncio.Lock()

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
            # Check if existing session is still valid
            if self._session is not None:
                try:
                    # Try a simple operation to check if session is alive
                    return True
                except Exception:
                    # Session is dead, need to reconnect
                    await self._disconnect()

            # Need to establish new connection
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

    async def is_available(self) -> bool:
        """Check if MCP server is available."""
        return await self._ensure_connection()

    async def ensure_available(self) -> bool:
        """Ensure MCP is available, start if needed."""
        return await self._ensure_connection()

    async def call_tool(self, tool_name: str, arguments: dict) -> Optional[dict]:
        """Call an MCP tool with retry mechanism."""
        if not MCP_AVAILABLE:
            logger.error("MCP SDK not installed")
            return None

        delay = self.retry_delay

        for attempt in range(1, self.retry_count + 1):
            try:
                # Ensure we have a valid connection
                if not await self._ensure_connection():
                    raise Exception("Failed to establish MCP connection")

                # Use the persistent session
                result = await asyncio.wait_for(
                    self._session.call_tool(tool_name, arguments),
                    timeout=self.timeout
                )
                return result

            except Exception as e:
                logger.warning("MCP call failed (attempt {}/{}): {}", attempt, self.retry_count, e)

                # Disconnect to force reconnection on next attempt
                await self._disconnect()

                if attempt < self.retry_count:
                    await asyncio.sleep(delay)
                    delay *= self.retry_backoff

        logger.error("MCP call failed after {} attempts", self.retry_count)
        return None

    async def call(self, prompt: str) -> str:
        """Call Claude Code chat via MCP."""
        result = await self.call_tool("chat", {"prompt": prompt})
        if result and result.content:
            return result.content[0].text
        return "Error: Failed to execute prompt"

    async def close(self) -> None:
        """Close the MCP client and stop server."""
        await self._disconnect()


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
    client = ClaudeMCPClient(
        auto_start=auto_start,
        retry_count=retry_count,
        retry_delay=retry_delay,
        retry_backoff=retry_backoff,
    )

    if arguments is None:
        arguments = {}

    # Ensure available
    if not await client.ensure_available():
        return None

    return await client.call_tool(tool_name, arguments)
