"""Hotel search tool for superbot."""

import asyncio
import json
from typing import Any

from superbot.agent.tools.base import Tool
from superbot.agent.tools.travel.browser import StealthBrowser


class HotelTool(Tool):
    """Search hotels using Ctrip/Trip.com."""

    @property
    def name(self) -> str:
        return "hotel_search"

    @property
    def description(self) -> str:
        return "搜索酒店价格，支持城市、入住日期、退房日期"

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
        self.browser = None

    async def _get_browser(self):
        if self.browser is None:
            self.browser = StealthBrowser()
            await self.browser.initialize()
        return self.browser

    async def _check_login(self, page) -> bool:
        """Check if login is required."""
        try:
            login_indicators = await page.query_selector_all(
                '[class*="login"], [class*="qrcode"], #login-modal'
            )
            for el in login_indicators:
                if await el.is_visible():
                    return True
            return False
        except:
            return False

    async def _extract_hotels(self, page) -> list:
        """Extract hotel data from page."""
        return await page.evaluate(r"""
            () => {
                const hotels = [];
                const items = document.querySelectorAll('[class*="hotel-item"], [class*="hotel-list"] li');

                items.forEach((item, idx) => {
                    try {
                        const name = item.querySelector('[class*="name"], [class*="title"]')?.innerText?.trim();
                        const priceEl = item.querySelector('[class*="price"]');
                        const price = priceEl?.innerText?.match(/¥(\d+)/)?.[1];
                        const rating = item.querySelector('[class*="rating"]')?.innerText?.trim();

                        if (name || price) {
                            hotels.push({
                                name: name || `酒店${idx + 1}`,
                                price: price ? parseInt(price) : null,
                                rating: rating
                            });
                        }
                    } catch (e) {}
                });

                return hotels;
            }
        """)

    async def execute(self, **kwargs: Any) -> str:
        """Execute hotel search."""
        city = kwargs.get("city", "")
        checkin = kwargs.get("checkin", "")
        checkout = kwargs.get("checkout", "")
        keywords = kwargs.get("keywords", "")

        try:
            browser = await self._get_browser()
            page = await browser.new_page()

            try:
                base_url = "https://hotels.ctrip.com/hotels/list"
                params = f"?city={city}&checkin={checkin}&checkout={checkout}"
                if keywords:
                    params += f"&keyword={keywords}"

                search_url = base_url + params
                await page.goto(search_url, wait_until="domcontentloaded")
                await asyncio.sleep(5)

                if await self._check_login(page):
                    return json.dumps({
                        "error": "login_required",
                        "message": "需要登录，请先登录携程"
                    }, ensure_ascii=False)

                hotels = await self._extract_hotels(page)

                if hotels:
                    return json.dumps({
                        "city": city,
                        "checkin": checkin,
                        "checkout": checkout,
                        "hotels": hotels
                    }, ensure_ascii=False, indent=2)
                else:
                    return json.dumps({
                        "error": "no_results",
                        "message": "未找到酒店",
                        "city": city
                    }, ensure_ascii=False)

            finally:
                await page.close()

        except Exception as e:
            import traceback
            return json.dumps({
                "error": str(e),
                "trace": traceback.format_exc()
            }, ensure_ascii=False)
