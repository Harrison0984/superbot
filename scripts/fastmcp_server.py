"""
FastMCP Server - Claude Code CLI integration
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Add logging
import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr, format='%(name)s - %(message)s')
logger = logging.getLogger("fastmcp_server")

# Create FastMCP server (stdio transport)
mcp = FastMCP("claude-code-server")

# Conversation history file
_history_file = Path.home() / ".claude" / "superbot_history"
MAX_HISTORY_SIZE = 20  # Keep last 20 messages


def _load_history() -> list:
    """Load conversation history from file."""
    global _history_file
    try:
        if _history_file.exists():
            return json.loads(_history_file.read_text())
    except Exception:
        pass
    return []


def _save_history(history: list) -> None:
    """Save conversation history to file (truncated to max size)."""
    global _history_file, MAX_HISTORY_SIZE
    # Truncate to last MAX_HISTORY_SIZE messages
    if len(history) > MAX_HISTORY_SIZE:
        history = history[-MAX_HISTORY_SIZE:]
    try:
        _history_file.parent.mkdir(parents=True, exist_ok=True)
        _history_file.write_text(json.dumps(history))
    except Exception as e:
        logger.warning(f"Failed to save history: {e}")


def _get_clean_env() -> dict:
    """Get environment - remove Claude vars to avoid nested session error."""
    env = os.environ.copy()
    # Remove known Claude environment variables (exact matches only)
    claude_vars = ["CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_ENTRYPOINT"]
    for key in claude_vars:
        env.pop(key, None)
    return env


@mcp.tool()
def chat(prompt: str) -> str:
    """Send a message to Claude Code and get response. Maintains conversation context.

    Args:
        prompt: The message to send to Claude

    Returns:
        Claude's response
    """
    try:
        # Load conversation history
        history = _load_history()
        logger.info(f"Chat request with {len(history)} previous messages")

        env = _get_clean_env()

        # Build the full prompt with conversation history (JSON format for robustness)
        full_prompt = prompt
        if history:
            # Use JSON format to safely encode messages with newlines/special chars
            history_json = json.dumps(history, ensure_ascii=False)
            full_prompt = f"<conversation_history>{history_json}</conversation_history>\n\nCurrent message: {prompt}"

        # Use --print with --no-session-persistence to avoid session issues
        # We maintain our own history instead
        result = subprocess.run(
            ["claude", "--print", "--no-session-persistence", "--dangerously-skip-permissions"],
            input=full_prompt,
            capture_output=True,
            text=True,
            env=env,
            timeout=120
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            return f"Error: {error_msg[:500]}"

        output = result.stdout.strip()
        if not output:
            return "No response from Claude"

        # Save to conversation history
        history.append({"user": prompt, "claude": output})
        _save_history(history)
        logger.info(f"Saved response, history now has {len(history)} messages")

        return output

    except subprocess.TimeoutExpired:
        return "Error: Claude request timed out"
    except FileNotFoundError:
        return "Error: Claude CLI not found"
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@mcp.tool()
def clear_history() -> str:
    """Clear conversation history."""
    global _history_file
    try:
        if _history_file.exists():
            _history_file.unlink()
        return "Conversation history cleared"
    except Exception as e:
        return f"Error clearing history: {e}"


if __name__ == "__main__":
    # Run with stdio transport
    mcp.run(transport="stdio")
