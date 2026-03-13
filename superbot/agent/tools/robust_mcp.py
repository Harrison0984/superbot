"""
Robust MCP Client - Uses synchronous I/O to avoid asyncio cancellation issues

This implementation:
1. Uses subprocess with synchronous stdin/stdout
2. Implements its own JSON-RPC protocol
3. Keeps connection alive between calls
4. Handles reconnection gracefully
"""

import json
import logging
import os
import select
import subprocess
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RobustMCPClient:
    """MCP client using synchronous I/O to avoid asyncio issues."""

    def __init__(self, script_path: str, cwd: str, timeout: int = 120):
        self.script_path = script_path
        self.cwd = cwd
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._initialized = False
        self._request_id = 0

    def _get_env(self) -> dict:
        """Get clean environment."""
        env = os.environ.copy()
        # Remove known Claude environment variables (exact matches only)
        for key in ["CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_ENTRYPOINT"]:
            env.pop(key, None)
        return env

    def _start_process(self) -> bool:
        """Start the MCP subprocess."""
        if self._process and self._process.poll() is None:
            logger.debug("MCP process already running, reusing")
            return True

        logger.info("Starting new MCP subprocess")
        try:
            self._process = subprocess.Popen(
                ["python", self.script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                text=True,
                bufsize=1,
                env=self._get_env(),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start MCP process: {e}")
            return False

    def _send_json(self, method: str, params: dict = None) -> dict:
        """Send JSON-RPC request and return response."""
        if not self._process or self._process.poll() is not None:
            if not self._start_process():
                raise Exception("Failed to start MCP server")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {}
        }

        request_str = json.dumps(request) + "\n"
        logger.debug(f"MCP sending: {request_str[:200]}")
        self._process.stdin.write(request_str)
        self._process.stdin.flush()

        # Read response with timeout
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            ready, _, _ = select.select([self._process.stdout], [], [], 1)
            if ready:
                line = self._process.stdout.readline()
                if line:
                    logger.debug(f"MCP received: {line[:200]}")
                    try:
                        response = json.loads(line)
                        if "error" in response:
                            raise Exception(f"MCP error: {response['error']}")
                        return response.get("result", {})
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON: {line[:100]}")
                        continue
        raise Exception("MCP request timeout")

    def initialize(self) -> bool:
        """Initialize the MCP connection."""
        logger.info("MCP initializing connection...")
        try:
            result = self._send_json("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "superbot", "version": "1.0"}
            })
            self._initialized = True

            # Send initialized notification
            self._process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "initialized"}) + "\n")
            self._process.stdin.flush()

            logger.info("MCP initialized successfully")
            return True
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
            return False

    def ensure_initialized(self) -> bool:
        """Ensure MCP is initialized and process is healthy."""
        # Check if process is still running (handle crash case)
        if self._initialized and self._process and self._process.poll() is not None:
            logger.warning("MCP process crashed, resetting initialized state")
            self._initialized = False
            self._process = None

        if self._initialized:
            return True

        if not self._start_process():
            return False

        return self.initialize()

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool."""
        logger.info(f"MCP call_tool: {tool_name} (initialized={self._initialized})")
        with self._lock:
            try:
                if not self.ensure_initialized():
                    return "Error: Failed to initialize MCP"

                # Map 'content' to 'prompt' for chat tool
                if tool_name == "chat" and "content" in arguments:
                    arguments = {"prompt": arguments["content"]}

                result = self._send_json("tools/call", {
                    "name": tool_name,
                    "arguments": arguments
                })

                # Parse result
                if result and "content" in result:
                    content = result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        return content[0].get("text", "")

                return "No response from tool"

            except Exception as e:
                logger.error(f"MCP tool call failed: {e}")
                # Try to reconnect on next call
                self._initialized = False
                if self._process:
                    try:
                        self._process.terminate()
                    except Exception:
                        pass
                    self._process = None
                return f"Error: {str(e)[:200]}"

    def close(self):
        """Close the MCP connection."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
            self._initialized = False
