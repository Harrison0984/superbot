"""
FastMCP Server - Claude Code CLI integration
"""
import os
import uuid
import subprocess
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Create FastMCP server (stdio transport)
mcp = FastMCP("claude-code-server")

# Session file path for persistence
_session_file = Path.home() / ".claude" / "superbot_session"


def _get_session_id() -> str:
    """Get or create a session ID for conversation context (persisted to file)."""
    global _session_file
    try:
        # Try to load existing session from file
        if _session_file.exists():
            return _session_file.read_text().strip()
    except Exception:
        pass

    # Create new session and save to file
    session_id = str(uuid.uuid4())
    try:
        _session_file.parent.mkdir(parents=True, exist_ok=True)
        _session_file.write_text(session_id)
    except Exception:
        pass
    return session_id


def _clear_session() -> None:
    """Clear the session ID (on error)."""
    global _session_file
    try:
        if _session_file.exists():
            _session_file.unlink()
    except Exception:
        pass


def _get_clean_env() -> dict:
    """Get environment - just remove CLAUDECODE to inherit terminal auth."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE", None)
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
        # Get session ID for conversation continuity
        session_id = _get_session_id()

        env = _get_clean_env()

        # Use --print with --session-id to maintain context across requests
        result = subprocess.run(
            ["claude", "--print", f"--session-id={session_id}", "--dangerously-skip-permissions"],
            input=prompt,
            capture_output=True,
            text=True,
            env=env,
            timeout=120
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            # If session error, try creating new session
            if "session" in error_msg.lower():
                _clear_session()  # Clear bad session
                session_id = _get_session_id()
                result = subprocess.run(
                    ["claude", "--print", f"--session-id={session_id}", "--dangerously-skip-permissions"],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=120
                )
                if result.returncode != 0:
                    return f"Error: {result.stderr or result.stdout}"
            else:
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
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    # Run with stdio transport
    mcp.run(transport="stdio")
