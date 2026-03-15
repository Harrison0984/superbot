"""Web tools: web_search and web_fetch using Chrome MCP."""

import json
import os
import re
import select
import subprocess
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

from superbot.agent.tools.base import Tool, tool_error
from superbot.agent.tools.travel.logger import get_logger

logger = get_logger(__name__)

# Chrome DevTools MCP - configurable via environment
CHROME_MCP_COMMAND = [
    "npx", "-y", "chrome-devtools-mcp@latest",
    "--browserUrl", os.environ.get("CHROME_DEBUG_URL", "http://localhost:9222")
]


class ChromeMCPClient:
    """MCP client for Chrome DevTools MCP."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._initialized = False

    def _get_clean_env(self) -> dict:
        """Get environment."""
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE", None)
        return env

    def _ensure_process(self) -> bool:
        """Ensure the MCP subprocess is running."""
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                CHROME_MCP_COMMAND,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd(),
                text=True,
                env=self._get_clean_env(),
            )
            if not self._initialize():
                return False
        return True

    def _send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request and get response."""
        if not self._ensure_process():
            raise Exception(
                "Failed to start Chrome MCP server. Make sure Chrome is running "
                "with remote debugging on port 9222: "
                "open Chrome with --remote-debugging-port=9222"
            )

        request = {
            "jsonrpc": "2.0",
            "id": id(self) % 10000,
            "method": method,
            "params": params or {}
        }

        request_str = json.dumps(request) + "\n"
        self._process.stdin.write(request_str)
        self._process.stdin.flush()

        # Read response with timeout
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise Exception(f"MCP request timeout after {self.timeout}s")

            ready = select.select([self._process.stdout], [], [], 1)[0]
            if ready:
                line = self._process.stdout.readline()
                if line:
                    response = json.loads(line)
                    if "error" in response:
                        raise Exception(f"MCP error: {response['error']}")
                    return response.get("result", {})

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
            logger.error("MCP initialization failed: " + str(e))
            return False

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return the result."""
        with self._lock:
            if not self._initialized:
                self._ensure_process()

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


class ChromeMCPTool(Tool):
    """Base class for Chrome MCP based tools.

    Uses a class-level singleton MCP client for efficiency.
    All tool instances share the same browser connection.
    """

    _mcp_client: Optional[ChromeMCPClient] = None

    def __init__(self):
        """Initialize with Chrome MCP client."""
        self._cwd = os.getcwd()

    @classmethod
    async def _get_mcp_client(cls) -> ChromeMCPClient:
        """Get or create MCP client (singleton).

        Singleton pattern ensures all Chrome MCP tools share
        the same browser connection for efficiency.
        """
        if cls._mcp_client is None:
            cls._mcp_client = ChromeMCPClient(timeout=60)
        return cls._mcp_client

    @classmethod
    async def _mcp_call(cls, tool_name: str, **kwargs) -> str:
        """Call an MCP tool."""
        client = await cls._get_mcp_client()
        return client.call_tool(tool_name, kwargs)

    @classmethod
    async def _close(cls):
        """Close MCP client."""
        if cls._mcp_client:
            cls._mcp_client.close()
            cls._mcp_client = None

    @staticmethod
    def _parse_search_results(snapshot: str) -> list:
        """Parse search results from snapshot.

        Extracts URLs from Chrome DevTools accessibility tree snapshot.
        """
        results = []
        lines = snapshot.split('\n')

        for line in lines:
            line = line.strip()

            # Extract URLs from href/ url patterns
            if 'url="' in line:
                url_match = re.search(r'url="(https?://[^"]+)"', line)
                if url_match:
                    url = url_match.group(1)
                    # Skip search engine internal URLs
                    if any(x in url for x in ['google.com/search', 'bing.com/search', 'webhp', 'support.google']):
                        continue
                    # Skip if too many query params (probably internal)
                    if url.count('&') > 3:
                        continue
                    if url.startswith('http'):
                        results.append({
                            "title": "",
                            "url": url,
                            "description": ""
                        })

        return results[:10]

    @staticmethod
    def _clean_snapshot(snapshot: str, mode: str) -> str:
        """Clean up snapshot text for readability.

        Args:
            snapshot: Raw snapshot text from Chrome DevTools
            mode: 'markdown' or 'text'

        Returns:
            Cleaned text content
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', snapshot)
        text = re.sub(r' {2,}', ' ', text)

        if mode == "text":
            text = re.sub(r'<[^>]+>', '', text)

        return text.strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


class WebSearchTool(ChromeMCPTool):
    """Search the web using Google via Chrome MCP."""

    SEARCH_URL = "https://www.google.com/search"

    name = "web_search"
    description = "Search the web using Google. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        super().__init__()

    async def execute(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        query = kwargs.get("query", content)
        count = kwargs.get("count", self.max_results)
        n = min(max(count, 1), 10)

        try:
            # Build search URL
            search_url = f"{self.SEARCH_URL}?q={query}&num={n}"
            logger.info("Searching: " + query)

            # Navigate to search page
            await self._mcp_call("navigate_page", url=search_url)

            # Wait for results using MCP
            await self._mcp_call("wait_for", text=["搜索结果", "results"])

            # Take snapshot
            snapshot = await self._mcp_call("take_snapshot")

            # Parse results from snapshot
            results = self._parse_search_results(snapshot)

            if not results:
                return f"No results for: {query}"

            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results[:n], 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if desc := item.get("description"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)

        except Exception as e:
            import traceback
            logger.error("WebSearch error: " + str(e))
            return tool_error("search_error", str(e), trace=traceback.format_exc())
        finally:
            await self._close()


class WebFetchTool(ChromeMCPTool):
    """Fetch and extract content from a URL using Chrome MCP."""

    name = "web_fetch"
    description = "Fetch URL and extract readable content using browser."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100}
        },
        "required": ["url"]
    }

    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars
        super().__init__()

    async def execute(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        url = kwargs.get("url", content)
        extractMode = kwargs.get("extractMode", "markdown")
        maxChars = kwargs.get("maxChars")
        max_chars = maxChars or self.max_chars

        # Validate URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return tool_error("invalid_url", f"URL validation failed: {error_msg}", url=url)

        try:
            logger.info("Fetching URL: " + url)

            # Navigate to the URL
            await self._mcp_call("navigate_page", url=url)

            # Wait for page to load
            await self._mcp_call("wait_for", text=["</body>", "Loading"])

            # Take snapshot
            snapshot = await self._mcp_call("take_snapshot")

            # Clean up the snapshot text
            text = self._clean_snapshot(snapshot, extractMode)

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps({
                "content": json.dumps({
                    "url": url,
                    "extractor": "chrome_mcp",
                    "truncated": truncated,
                    "length": len(text),
                    "text": text
                }, ensure_ascii=False),
                "media": []
            })

        except Exception as e:
            import traceback
            logger.error("WebFetch error for " + url + ": " + str(e))
            return tool_error("fetch_error", str(e), url=url)
        finally:
            await self._close()
