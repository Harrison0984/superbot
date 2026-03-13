"""Flight search tool for superbot."""

import asyncio
import json
import uuid
from typing import Any
from playwright.async_api import Page

from superbot.agent.tools.base import Tool, tool_error
from superbot.agent.tools.travel.shared import get_shared_browser
from superbot.agent.tools.travel.ctrip import ctrip_monitor
from superbot.agent.tools.travel.session import get_session_manager, verify_login
from superbot.agent.tools.travel.logger import get_logger

logger = get_logger(__name__)


class FlightTool(Tool):
    """Search flights using Ctrip/Trip.com."""

    # Config
    WAIT_TIMEOUT = 300  # 5 minutes
    POLL_INTERVAL = 5   # 5 seconds

    @property
    def name(self) -> str:
        return "flight_search"

    @property
    def description(self) -> str:
        return "搜索航班价格，必须使用机场代码（如SHA、PEK、SYX）而非城市名称"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "from_city": {
                    "type": "string",
                    "description": "出发机场代码（必须），如：上海浦东=SHA，北京首都=PEK，三亚=SYX，广州CAN，深圳SZX，成都CTU，杭州HGH，南京NKG，重庆CKG，西安XIY，厦门XMN，昆明KMG"
                },
                "to_city": {
                    "type": "string",
                    "description": "目的机场代码（必须），如：上海浦东=SHA，北京首都=PEK，三亚=SYX，广州CAN，深圳SZX，成都CTU，杭州HGH，南京NKG，重庆CKG，西安XIY，厦门XMN，昆明KMG"
                },
                "date": {
                    "type": "string",
                    "description": "出发日期，格式：YYYY-MM-DD，如：2026-03-15"
                }
            },
            "required": ["from_city", "to_city", "date"]
        }

    def __init__(self):
        self.session = get_session_manager()

    async def _get_browser(self):
        return await get_shared_browser()

    async def execute(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        """Execute flight search."""
        from_city = kwargs.get("from_city", "")
        to_city = kwargs.get("to_city", "")
        date = kwargs.get("date", "")
        task_id = str(uuid.uuid4())

        try:
            browser = await self._get_browser()
            page = await browser.new_page()

            # 1. Check if there's a saved Cookie
            has_saved_session = self.session.has_session()
            is_logged_in = False

            if has_saved_session:
                logger.info("Loading saved Cookie...")
                await self.session.apply_cookies_async(page.context)

            # 2. Must check page login status (cookie may be invalid)
            logger.info("Checking page login status...")
            is_logged_in = await verify_login(page, "https://flights.ctrip.com/online/list")

            if is_logged_in:
                logger.info("Already logged in, searching directly...")
            elif has_saved_session:
                logger.info("Cookie invalid, need to re-login...")
            else:
                logger.info("Not logged in, need to login...")

            if not is_logged_in:
                # 3. If not logged in, generate QR code
                logger.info("Not logged in, generating QR code...")
                try:
                    success, qr_path = await self.session.generate_qr_code(page)
                except Exception as e:
                    logger.warning("QR code generation error: %s", e)
                    # If QR code generation fails, page may have auto-redirected (already logged in)
                    # Check login status again
                    logger.info("Current page URL: %s", page.url)
                    is_logged_in = await verify_login(page, "https://flights.ctrip.com/online/list")
                    logger.info("Checking login status: %s", is_logged_in)
                    if is_logged_in:
                        logger.info("Page already logged in, searching directly")
                        result = await self._do_search(from_city, to_city, date, page)
                        return result
                    success, qr_path = False, None

                if success:
                    # Start background task
                    asyncio.create_task(
                        self._wait_for_login_and_search(from_city, to_city, date, page)
                    )

                    # Return immediately
                    return json.dumps({
                        "content": "请扫码登录，登录后自动开始搜索",
                        "media": [str(qr_path)] if qr_path else []
                    })
                else:
                    return tool_error("qr_generation_failed", "生成登录二维码失败，请重试")

            # 4. Already logged in, search directly
            return await self._do_search(from_city, to_city, date, page)

        except Exception as e:
            import traceback
            return tool_error("internal_error", str(e), trace=traceback.format_exc())

    async def _wait_for_login_and_search(
        self,
        from_city: str,
        to_city: str,
        date: str,
        page
    ):
        """Wait for login in background and search."""
        async def _check_login_popup(page: Page) -> bool:
            """Check if there's a login popup/QR code (not logged in)."""
            # First try clicking login button to show popup
            try:
                login_link = await page.query_selector('text=登录')
                if login_link:
                    await login_link.click()
                    await asyncio.sleep(1)
            except:
                pass

            # Login popup/QR code selector
            popup_selectors = [
                '[class*="login-modal"]',
                '[class*="login-dialog"]',
                '.login-dialog',
                '#login-modal',
                '[class*="qrcode"]',
                '[class*="ercode"]',
                '[class*="login-mask"]',
                '.lg_ercode',
                '#qrcode',
                '.qr-code',
                '[class*="login-overlay"]',
                '.popup-login',
                '[id*="login-popup"]',
                '.lg_loginbox_modal',
                '[id*="maskLogin"]',
                '[id*="accounts-pc"]',
            ]

            for selector in popup_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        is_visible = await element.is_visible()
                        if is_visible:
                            return True
                except:
                    continue

            # Check for QR login prompts (indicates login popup)
            qr_prompts = await page.query_selector_all('text=扫码登录, text=请扫码, text=手机号登录, text=账号密码登录')
            if qr_prompts:
                for prompt in qr_prompts:
                    try:
                        if await prompt.is_visible():
                            return True
                    except:
                        continue

            return False

        async def _check_logged_in(page: Page) -> bool:
            """Check if logged in (has user info elements)."""
            logged_in_indicators = [
                '[class*="user-name"]',
                '[class*="user-info"]',
                '[class*="username"]',
                '[id*="user-name"]',
                '[class*="login-success"]',
                '.user-info',
                '.username',
                '[class*="avatar"]',
                '#userName',
                '.hp_user'
            ]

            for selector in logged_in_indicators:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        is_visible = await element.is_visible()
                        if is_visible:
                            return True
                except:
                    continue

            return False

        async def check_login_status(page: Page) -> tuple[bool, str]:
            """Check login status.

            Returns:
                (is_logged_in, status): Login status and status description
            """
            # If page redirects to passport.ctrip.com, user is logging in
            if "passport.ctrip.com" in page.url:
                # QR scan successful, page redirected! Return True means login success, need to go back to search page
                return True, "login_redirect"

            # Page redirected to other page (like home), login successful
            if "flights.ctrip.com" not in page.url and page.url != "about:blank":
                return True, "page_redirect"

            # First check if there's a login popup/QR code
            has_popup = await _check_login_popup(page)
            if has_popup:
                return False, "login_popup"

            # Check if already logged in
            is_logged_in = await _check_logged_in(page)
            if is_logged_in:
                return True, "logged_in"

            # No popup or login elements, may not need login
            return True, "unknown"

        for elapsed in range(0, self.WAIT_TIMEOUT, self.POLL_INTERVAL):
            await asyncio.sleep(self.POLL_INTERVAL)
            
            # Check login status
            try:
                is_logged_in, status = await check_login_status(page)
                logger.info("Login status: is_logged_in=%s, status=%s", is_logged_in, status)

                if is_logged_in:
                    # Page redirected, QR scan successful, go back to search page
                    if status in ("login_redirect", "page_redirect"):
                        logger.info("Page redirect detected, QR scan successful! Redirecting back to search page...")
                        # Go back to search page
                        search_url = f"https://flights.ctrip.com/online/listonline/list/oneway-{from_city}-{to_city}?depdate={date}"
                        await page.goto(search_url)
                        await page.wait_for_load_state("networkidle")

                    # Save cookie and search
                    await self.session.save_cookies(page.context)
                    result = await self._do_search(from_city, to_city, date, page)
                    self.send_message(content=result, to="llm")
                    return
            except Exception as e:
                logger.warning("Failed to check login status: %s", e)

        # Timeout
        await page.close()
        self.send_message(content="扫码超时，请重试", to="")

    async def _do_search(self, from_city: str, to_city: str, date: str, page) -> str:
        """Execute flight search."""
        try:
            from_code = from_city.upper() if len(from_city) == 3 else from_city
            to_code = to_city.upper() if len(to_city) == 3 else to_city

            result = await ctrip_monitor.search_flight(
                page,
                from_code,
                to_code,
                date
            )

            if result:
                return json.dumps({
                    "content": json.dumps(result, ensure_ascii=False),
                    "media": []
                })
            else:
                return tool_error("no_results", f"未找到航班: {from_city} -> {to_city}, {date}")

        finally:
            await page.close()
