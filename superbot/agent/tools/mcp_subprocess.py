"""
Subprocess-based MCP Client - bypasses asyncio issues with the MCP library
"""

import asyncio
import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SubprocessMCPClient:
    """MCP client using subprocess to avoid asyncio cancellation issues."""

    def __init__(self, script_path: str, cwd: str, timeout: int = 120):
        self.script_path = script_path
        self.cwd = cwd
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._initialized = False

    def _get_clean_env(self) -> dict:
        """Get environment - remove CLAUDECODE to inherit terminal auth."""
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE", None)
        return env

    def _ensure_process(self) -> bool:
        """Ensure the MCP subprocess is running."""
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                ["python", self.script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                text=True,
                env=self._get_clean_env(),
            )
            # Initialize the connection
            if not self._initialize():
                return False
        return True

    def _send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request and get response."""
        if not self._ensure_process():
            raise Exception("Failed to start MCP server")

        request = {
            "jsonrpc": "2.0",
            "id": id(self) % 10000,
            "method": method,
            "params": params or {}
        }

        request_str = json.dumps(request) + "\n"
        self._process.stdin.write(request_str)
        self._process.stdin.flush()

        # Read response
        import select
        while True:
            ready = select.select([self._process.stdout], [], [], self.timeout)[0]
            if ready:
                line = self._process.stdout.readline()
                if line:
                    response = json.loads(line)
                    if "error" in response:
                        raise Exception(f"MCP error: {response['error']}")
                    return response.get("result", {})
        return {}

    def _initialize(self) -> bool:
        """Initialize the MCP connection."""
        try:
            result = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "superbot", "version": "1.0"}
            })
            self._initialized = True

            # Send initialized notification
            self._process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "initialized"}) + "\n")
            self._process.stdin.flush()

            return True
        except Exception as e:
            logger.error("MCP initialization failed: {}", e)
            return False

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return the result."""
        with self._lock:
            if not self._initialized:
                self._ensure_process()

            # Map 'content' to 'prompt' for chat tool
            if tool_name == "chat" and "content" in arguments:
                arguments = {"prompt": arguments["content"]}

            result = self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })

            # Parse the result
            if result and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", "")

            return "No response from MCP tool"

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
