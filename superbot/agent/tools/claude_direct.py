"""
ClaudeTool - Direct subprocess-based Claude Code integration

This implementation bypasses the MCP library to avoid asyncio cancellation issues.
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from superbot.agent.tools.base import Tool

logger = logging.getLogger(__name__)


class ClaudeToolDirect(Tool):
    """Direct Claude Code CLI integration without MCP."""

    def __init__(self, config: Any = None):
        self.config = config
        self.workdir = ""
        self.timeout = 120

        # Parse config
        if config:
            self.workdir = getattr(config, 'workdir', '') or ""
            self.timeout = getattr(config, 'timeout', 120)

        # Session file path for persistence
        self._session_file = Path.home() / ".claude" / "superbot_session"

    def _get_cwd(self) -> str:
        """Get the working directory."""
        return self.workdir or os.getcwd()

    def _get_clean_env(self) -> dict:
        """Get environment - just remove CLAUDECODE to inherit terminal auth."""
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE", None)
        return env

    def _get_session_id(self) -> str:
        """Get or create a session ID for conversation context."""
        try:
            if self._session_file.exists():
                return self._session_file.read_text().strip()
        except Exception:
            pass

        # Create new session and save to file
        session_id = f"superbot_{os.urandom(8).hex()}"
        try:
            self._session_file.parent.mkdir(parents=True, exist_ok=True)
            self._session_file.write_text(session_id)
        except Exception:
            pass
        return session_id

    async def execute(self, content: str) -> str:
        """Execute Claude Code directly via subprocess."""
        loop = asyncio.get_event_loop()

        def _run_subprocess():
            session_id = self._get_session_id()
            env = self._get_clean_env()
            cwd = self._get_cwd()

            try:
                result = subprocess.run(
                    ["claude", "--print", f"--session-id={session_id}", "--dangerously-skip-permissions"],
                    input=content,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=cwd,
                    timeout=self.timeout
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout
                    return f"Error: {error_msg[:500]}"

                output = result.stdout.strip()
                if not output:
                    return "No response from Claude"

                return output

            except subprocess.TimeoutExpired:
                return "Error: Claude request timed out"
            except FileNotFoundError:
                return "Error: Claude CLI not found"
            except Exception as e:
                return f"Error: {str(e)}"

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _run_subprocess),
                timeout=self.timeout + 10  # Add extra time for executor overhead
            )
            return result
        except asyncio.TimeoutError:
            return "Error: Claude request timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    async def execute(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        """Execute Claude Code via subprocess."""
        return await self.execute(content)

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
