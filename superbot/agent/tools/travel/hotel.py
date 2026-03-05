"""Hotel search tool for superbot."""

import asyncio
import json
from typing import Any

from superbot.agent.tools.base import Tool, tool_error
from superbot.agent.tools.travel.shared import get_shared_browser
from superbot.agent.tools.travel.session import get_session_manager
from superbot.agent.tools.travel.logger import get_logger

logger = get_logger(__name__)


class HotelTool(Tool):
    """Search hotels using Ctrip/Trip.com."""

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

    async def execute(self, **kwargs: Any) -> str:
        """Execute hotel search."""
        city = kwargs.get("city", "")
        checkin = kwargs.get("checkin", "")
        checkout = kwargs.get("checkout", "")
        keywords = kwargs.get("keywords", "")

        try:
            browser = await self._get_browser()
            page = await browser.new_page()

            # 1. 检查是否有保存的 Cookie
            has_saved_session = self.session.has_session()
            if has_saved_session:
                logger.info("加载已保存的 Cookie...")
                await self.session.apply_cookies_async(page.context)

            # 2. 检查是否已登录（只有保存过 Cookie 才检查，否则需要登录）
            if has_saved_session:
                is_logged_in = await self.session.check_login(page)
                if is_logged_in:
                    logger.info("已登录，直接搜索...")
                else:
                    # Cookie 失效，需要重新登录
                    logger.info("Cookie 失效，需要重新登录...")
            else:
                # 没有保存过 Cookie，需要登录
                is_logged_in = False

            if not is_logged_in:
                # 3. 未登录则生成二维码
                logger.info("未登录，生成二维码...")
                success, qr_path = await self.session.generate_qr_code(page)
                if success:
                    # 统一的错误返回结构
                    return json.dumps({
                        "_tool_error": {
                            "type": "login_required",
                            "feedback_to": "user",
                            "message": "需要登录才能搜索酒店，请扫码登录后重试",
                            "media": [str(qr_path)]
                        }
                    }, ensure_ascii=False)
                else:
                    return json.dumps({
                        "_tool_error": {
                            "type": "qr_generation_failed",
                            "feedback_to": "user",
                            "message": "生成登录二维码失败，请重试"
                        }
                    }, ensure_ascii=False)

            # 4. 搜索酒店
            try:
                base_url = "https://hotels.ctrip.com/hotels/list"
                params = f"?city={city}&checkin={checkin}&checkout={checkout}"
                if keywords:
                    params += f"&keyword={keywords}"
                search_url = base_url + params

                logger.info(f"搜索酒店: {search_url}")
                await page.goto("https://www.ctrip.com", wait_until="domcontentloaded")
                await asyncio.sleep(3)
                await page.goto(search_url, wait_until="networkidle")
                await asyncio.sleep(10)

                hotels = await self._extract_hotels(page)

                if hotels:
                    return json.dumps({
                        "city": city,
                        "checkin": checkin,
                        "checkout": checkout,
                        "hotels": hotels
                    }, ensure_ascii=False, indent=2)
                else:
                    return tool_error("no_results", f"未找到酒店: {city}")

            finally:
                await page.close()

        except Exception as e:
            import traceback
            return tool_error("internal_error", str(e), trace=traceback.format_exc())
