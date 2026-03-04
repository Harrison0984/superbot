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
        """Check if user is logged in by visiting a protected page."""
        try:
            await page.goto("https://my.ctrip.com", wait_until="domcontentloaded", timeout=10000)
            await asyncio.sleep(2)

            # Check for user info element (indicates logged in)
            user_elements = await page.query_selector_all(
                '[class*="user-name"], [class*="username"], [class*="user-info"], [class*="login-success"]'
            )
            for el in user_elements:
                if await el.is_visible():
                    logger.info("User is logged in")
                    return True

            # Also check if login popup is NOT present
            login_popup = await page.query_selector('[class*="login-mask"], [class*="login-modal"]')
            if not login_popup or not await login_popup.is_visible():
                logger.info("No login popup, assuming logged in")
                return True

            return False
        except Exception as e:
            logger.warning(f"Error checking login: {e}")
            return False

    async def wait_for_login(self, page: Page, timeout: int = 120) -> bool:
        """Wait for user to scan QR code and login.

        Returns:
            True if login successful, False if timeout
        """
        # Screenshot for QR code
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

            logger.info(f"请扫码登录... (截图路径: {screenshot_path})")
            logger.info(f"等待登录... (超时 {timeout} 秒)")

            # Wait for login - check if QR code is gone (means logged in)
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                # Check if QR code / login area is still visible
                qr_element = await page.query_selector('.lg_ercode, [class*="ercode"], [class*="login-mask"]')
                is_logged_in = False

                if qr_element:
                    is_visible = await qr_element.is_visible()
                    if not is_visible:
                        # QR code gone, likely logged in
                        is_logged_in = True
                        logger.info("检测到二维码消失，可能已登录")

                # Also check for user info elements
                user_elements = await page.query_selector_all(
                    '[class*="user-name"], [class*="username"], [class*="user-info"], [class*="user"], a[href*="profile"]'
                )
                for el in user_elements:
                    try:
                        if await el.is_visible():
                            is_logged_in = True
                            break
                    except:
                        pass

                if is_logged_in:
                    logger.info("✅ 登录成功!")
                    # Save cookies
                    context = page.context
                    await self.save_cookies(context)
                    return True

                # Check URL change - if navigated away from login page, likely logged in
                current_url = page.url
                if 'passport' not in current_url and 'login' not in current_url:
                    logger.info(f"URL changed to {current_url}, assuming logged in")
                    context = page.context
                    await self.save_cookies(context)
                    return True

                await asyncio.sleep(3)

            logger.warning("登录超时")
            return False

        except Exception as e:
            logger.error(f"登录过程出错: {e}")
            return False

    def apply_cookies(self, context) -> bool:
        """Apply saved cookies to browser context."""
        cookies = self.load_cookies()
        if cookies:
            try:
                context.add_cookies(cookies)
                logger.info("Cookies applied to context")
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
