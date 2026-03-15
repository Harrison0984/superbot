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
            # Check if current page has login QR code/login popup
            # If no QR code, already logged in (or page doesn't require login)

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

            # No login QR code found, assume logged in
            logger.info("No login QR code found, assuming logged in")
            return True
        except Exception as e:
            logger.warning(f"Error checking login: {e}")
            return True  # Default to logged in on error

    async def check_login_async(self, page: Page) -> bool:
        """Check if user is logged in (standalone, doesn't require page navigation)."""
        try:
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
                        return False
                except:
                    pass

            return True
        except Exception as e:
            logger.warning(f"Error checking login: {e}")
            return True

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
                logger.info(f"QR code screenshot saved to: {screenshot_path}")
            else:
                await page.screenshot(path=str(screenshot_path), full_page=True)
                logger.info(f"Login page screenshot saved to: {screenshot_path}")

            logger.info(f"Please scan QR code to login: {screenshot_path}")
            return True, screenshot_path

        except Exception as e:
            logger.error(f"Error generating QR code: {e}")
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


async def verify_login(page: Page, url: str) -> bool:
    """Verify login status.

    Flow:
    1. Navigate to specified page
    2. Check if page header has "尊敬的xxx" or "我的订单"
    3. Return True(logged in)/False(not logged in)
    """
    try:
        # Navigate to specified page
        await page.goto(url, timeout=15000, wait_until="domcontentloaded")
        await asyncio.sleep(2)  # Wait for dynamic content

        # Detect user name
        user_selectors = [
            '.user-name',
            '.username',
            '.nick-name',
            '[class*="user-name"]',
            '[class*="userName"]',
            '[id*="userName"]',
            '.user-avatar',
            '.avatar',
            '[class*="avatar"]',
            '.hp_user',
            '.header-user',
            '.login-user',
            '[class*="header_user"]',
            '[class*="login_wrapper"]',
            'a:has-text("我的订单")',
            'a:has-text("您好")',
        ]

        for sel in user_selectors:
            try:
                el = await page.query_selector(sel)
                if el and await el.is_visible():
                    text = await el.inner_text()
                    if text and text.strip():
                        # Has "尊敬的" or "我的订单" -> logged in
                        if "尊敬的" in text or "我的订单" in text:
                            logger.info("Logged in, detected user: %s", text[:30])
                            return True
            except Exception:
                continue

        logger.info("No user element detected, considered not logged in")
        return False
    except Exception as e:
        logger.warning("Login verification failed: %s", e)
        return False


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
