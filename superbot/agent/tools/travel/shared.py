"""Shared browser and session manager for travel tools."""

import asyncio
from typing import Optional

from superbot.agent.tools.travel.browser import StealthBrowser
from superbot.agent.tools.travel.session import get_session_manager

# 全局共享的 browser 实例
_browser: Optional[StealthBrowser] = None
_browser_lock = asyncio.Lock()


async def get_shared_browser() -> StealthBrowser:
    """获取共享的浏览器实例"""
    global _browser
    async with _browser_lock:
        if _browser is None:
            _browser = StealthBrowser()
            await _browser.initialize()
        return _browser


async def close_shared_browser():
    """关闭共享的浏览器实例"""
    global _browser
    async with _browser_lock:
        if _browser:
            await _browser.close()
            _browser = None
