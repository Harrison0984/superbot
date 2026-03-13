"""Hotel search tool for superbot."""

import asyncio
import json
import uuid
from typing import Any

from superbot.agent.tools.base import Tool, tool_error
from superbot.agent.tools.travel.shared import get_shared_browser
from superbot.agent.tools.travel.session import get_session_manager, verify_login
from superbot.agent.tools.travel.logger import get_logger

logger = get_logger(__name__)


class HotelTool(Tool):
    """Search hotels using Ctrip/Trip.com."""

    # Config
    WAIT_TIMEOUT = 300  # 5 minutes
    POLL_INTERVAL = 5   # 5 seconds

    @property
    def name(self) -> str:
        return "hotel_search"

    @property
    def description(self) -> str:
        return "搜索酒店价格，支持城市，入住日期，退房日期"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如：上海"
                },
                "checkin": {
                    "type": "string",
                    "description": "入住日期，格式：YYYY-MM-DD，如：2026-03-15"
                },
                "checkout": {
                    "type": "string",
                    "description": "退房日期，格式：YYYY-MM-DD，如：2026-03-18"
                },
                "keywords": {
                    "type": "string",
                    "description": "关键词（可选），如：浦东"
                }
            },
            "required": ["city", "checkin", "checkout"]
        }

    def __init__(self):
        self.session = get_session_manager()

    async def _get_browser(self):
        return await get_shared_browser()

    async def _extract_hotels(self, page) -> list:
        """Extract hotel data from page."""
        js_code = """() => {
            var hotels = [];
            var listContainer = document.querySelector('div.hotel-list');
            if (!listContainer) return hotels;
            var items = listContainer.children;
            for (var i = 0; i < items.length; i++) {
                var item = items[i];
                var text = item.innerText;
                if (!text) continue;
                var priceMatch = text.match(/¥(\\d+)/);
                var price = priceMatch ? parseInt(priceMatch[1]) : null;
                var lines = text.split('\\n').filter(function(l) { return l.trim(); });
                var name = lines[0] || '';
                name = name.replace(/¥\\d+/, '').trim().substring(0, 50);
                if (name && price) {
                    hotels.push({name: name, price: price, rating: null});
                }
            }
            return hotels;
        }"""
        return await page.evaluate(js_code)

    async def execute(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        """Execute hotel search."""
        city = kwargs.get("city", "")
        checkin = kwargs.get("checkin", "")
        checkout = kwargs.get("checkout", "")
        keywords = kwargs.get("keywords", "")
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
            is_logged_in = await verify_login(page, "https://hotels.ctrip.com/hotels/list")

            if is_logged_in:
                logger.info("Already logged in, searching directly...")
            elif has_saved_session:
                logger.info("Cookie invalid, need to re-login...")
            else:
                logger.info("Not logged in, need to login...")

            if not is_logged_in:
                # 3. If not logged in, generate QR code
                logger.info("Not logged in, generating QR code...")
                success, qr_path = await self.session.generate_qr_code(page)
                if success:
                    # Start background task
                    asyncio.create_task(
                        self._wait_for_login_and_search(city, checkin, checkout, keywords, page)
                    )

                    # Return immediately
                    return json.dumps({
                        "content": "请扫码登录，登录后自动开始搜索",
                        "media": [str(qr_path)] if qr_path else []
                    })
                else:
                    return tool_error("qr_generation_failed", "生成登录二维码失败，请重试")

            # 4. Already logged in, search directly
            return await self._do_search(city, checkin, checkout, keywords, page)

        except Exception as e:
            import traceback
            return tool_error("internal_error", str(e), trace=traceback.format_exc())

    async def _wait_for_login_and_search(
        self,
        city: str,
        checkin: str,
        checkout: str,
        keywords: str,
        page
    ):
        """Wait for login in background and search."""
        for elapsed in range(0, self.WAIT_TIMEOUT, self.POLL_INTERVAL):
            await asyncio.sleep(self.POLL_INTERVAL)

            # Check login (need to refresh page to check if QR code disappeared)
            try:
                await page.reload()
                is_logged_in = await verify_login(page, "https://hotels.ctrip.com/hotels/list")
                if is_logged_in:
                    # Login successful, continue searching
                    result = await self._do_search(city, checkin, checkout, keywords, page)
                    self.send_message(content=result, to="llm")
                    return
            except Exception as e:
                logger.warning(f"Failed to check login status: {e}")

        # Timeout
        await page.close()
        self.send_message(content="扫码超时，请重试", to="")

    async def _do_search(self, city: str, checkin: str, checkout: str, keywords: str, page) -> str:
        """Execute hotel search."""
        try:
            base_url = "https://hotels.ctrip.com/hotels/list"
            params = f"?city={city}&checkin={checkin}&checkout={checkout}"
            if keywords:
                params += f"&keyword={keywords}"
            search_url = base_url + params

            await page.goto("https://www.ctrip.com", wait_until="domcontentloaded")
            await page.goto(search_url, wait_until="networkidle")
            hotels = await self._extract_hotels(page)

            if hotels:
                return json.dumps({
                    "content": json.dumps({
                        "city": city,
                        "checkin": checkin,
                        "checkout": checkout,
                        "hotels": hotels
                    }, ensure_ascii=False),
                    "media": []
                })
            else:
                return tool_error("no_results", f"未找到酒店: {city}")

        finally:
            await page.close()
