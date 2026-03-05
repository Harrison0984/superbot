"""Session management for travel tools - login persistence."""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from playwright.async_api import Page

from superbot.agent.tools.travel.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Manage Ctrip login session with cookie persistence."""

    def __init__(self, session_name: str = "ctrip"):
        self.session_name = session_name
        self.session_dir = Path.home() / ".superbot" / "sessions" / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.cookie_path = self.session_dir / "cookies.json"

    async def save_cookies(self, context) -> bool:
        """Save cookies from browser context."""
        try:
            cookies = await context.cookies()
            with open(self.cookie_path, "w") as f:
                json.dump(cookies, f)
            logger.info(f"Cookies saved to {self.cookie_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")
            return False

    def load_cookies(self) -> list:
        """Load cookies from file."""
        if self.cookie_path.exists():
            try:
                with open(self.cookie_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cookies: {e}")
        return []

    def has_session(self) -> bool:
        """Check if session file exists."""
        return self.cookie_path.exists()

    async def check_login(self, page: Page) -> bool:
        """Check if user is logged in by checking for login QR code on current page."""
        try:
            # 检查当前页面是否有登录二维码/登录弹窗
            # 如果没有二维码，说明已登录（或者页面不需要登录）

            qr_selectors = [
                '[class*="qrcode"]',
                '[class*="ercode"]',
                '[class*="login-mask"]',
                '[class*="login-modal"]',
                '.lg_ercode'
            ]

            for sel in qr_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el and await el.is_visible():
                        logger.info(f"Found login QR code: {sel}, not logged in")
                        return False
                except:
                    pass

            # 没有找到登录二维码，认为已登录
            logger.info("No login QR code found, assuming logged in")
            return True
        except Exception as e:
            logger.warning(f"Error checking login: {e}")
            return True  # 出错时默认已登录

    async def generate_qr_code(self, page: Page) -> tuple[bool, Path]:
        """Generate QR code for login without waiting (non-blocking).

        Returns:
            (success, qr_screenshot_path): Tuple of generation status and QR code screenshot path
        """
        screenshot_path = self.session_dir / "login_qr.png"

        try:
            # Navigate to login page
            await page.goto("https://passport.ctrip.com/user/login", wait_until="domcontentloaded")
            await asyncio.sleep(3)

            # Click scan login
            scan_login = await page.query_selector('a:has-text("扫码登录"), a:has-text("扫码")')
            if scan_login:
                await scan_login.click()
                await asyncio.sleep(5)

            # Screenshot QR code area
            qr_element = await page.query_selector('.lg_ercode, [class*="ercode"]')
            if qr_element:
                await qr_element.screenshot(path=str(screenshot_path))
                logger.info(f"📱 二维码截图已保存到: {screenshot_path}")
            else:
                await page.screenshot(path=str(screenshot_path), full_page=True)
                logger.info(f"📱 登录页面截图已保存到: {screenshot_path}")

            logger.info(f"请扫码登录: {screenshot_path}")
            return True, screenshot_path

        except Exception as e:
            logger.error(f"生成二维码出错: {e}")
            return False, screenshot_path

    async def apply_cookies_async(self, context) -> bool:
        """Apply saved cookies to browser context (async)."""
        cookies = self.load_cookies()
        if cookies:
            try:
                await context.add_cookies(cookies)
                logger.info("Cookies applied to context")
                return True
            except Exception as e:
                logger.error(f"Failed to apply cookies: {e}")
        return False

    def apply_cookies(self, context) -> bool:
        """Apply saved cookies to browser context (sync, deprecated)."""
        cookies = self.load_cookies()
        if cookies:
            try:
                # Note: This may not work correctly, use apply_cookies_async instead
                logger.warning("Using sync apply_cookies, prefer async version")
                return True
            except Exception as e:
                logger.error(f"Failed to apply cookies: {e}")
        return False


# Global session manager
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
