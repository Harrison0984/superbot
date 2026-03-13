"""Shared browser and session manager for travel tools."""

import asyncio
from typing import Optional

from superbot.agent.tools.travel.browser import StealthBrowser
from superbot.agent.tools.travel.session import get_session_manager

# Globally shared browser instance
_browser: Optional[StealthBrowser] = None
_browser_lock = asyncio.Lock()


async def get_shared_browser() -> StealthBrowser:
    """Get shared browser instance."""
    global _browser
    async with _browser_lock:
        if _browser is None:
            _browser = StealthBrowser()
            await _browser.initialize()
        return _browser


async def close_shared_browser():
    """Close shared browser instance."""
    global _browser
    async with _browser_lock:
        if _browser:
            await _browser.close()
            _browser = None
