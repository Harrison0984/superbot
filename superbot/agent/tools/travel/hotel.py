"""Hotel search tool for superbot."""

import asyncio
import json
from typing import Any

from superbot.agent.tools.base import Tool
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
        self.session = get_session_manager()

    async def _get_browser(self):
        return await get_shared_browser()

    async def _extract_hotels(self, page) -> list:
        """Extract hotel data from page."""
        return await page.evaluate(r"""
            () => {
                const hotels = [];
                // 查找所有酒店项
                const items = document.querySelectorAll('[class*="hotel-item"], .hotel-list-item, li[data-hotelid]');

                items.forEach((item, idx) => {
                    try {
                        // 查找酒店名称 - 从截图看是 h2 或 .name
                        const nameEl = item.querySelector('h2, .name, [class*="title"] a, [class*="hotel"] a');
                        const name = nameEl?.innerText?.trim() || item.innerText.split('¥')[0]?.trim();

                        // 查找价格 - 从截图看价格直接在文本中
                        const text = item.innerText;
                        const priceMatch = text.match(/¥(\d+)/);
                        const price = priceMatch ? parseInt(priceMatch[1]) : null;

                        // 查找评分
                        const ratingEl = item.querySelector('[class*="score"], [class*="rating"], [class*="star"]');
                        const rating = ratingEl?.innerText?.trim();

                        if (name || price) {
                            hotels.push({
                                name: name ? name.substring(0, 50) : `酒店${idx + 1}`,
                                price: price,
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

            # 严格流程：
            # 1. 加载保存的 Cookie
            # 2. 检查登录状态
            # 3. 未登录则扫码登录
            # 4. 登录成功后保存 Cookie
            # 5. 进行搜索

            # 1. 加载保存的 Cookie
            if self.session.has_session():
                logger.info("加载已保存的 Cookie...")
                await self.session.apply_cookies_async(page.context)

            # 2. 检查是否已登录
            is_logged_in = await self.session.check_login(page)
            if is_logged_in:
                logger.info("已登录，直接搜索...")

            # 3. 未登录则扫码登录
            if not is_logged_in:
                logger.info("未登录，开始扫码登录流程...")
                login_success = await self.session.wait_for_login(page, timeout=120)
                if not login_success:
                    return json.dumps({
                        "error": "login_required",
                        "message": "需要登录才能搜索酒店，请扫码登录后重试。截图已保存到 ~/.superbot/sessions/ctrip/login_qr.png"
                    }, ensure_ascii=False)
                # 4. 登录成功后，保存 Cookie
                await self.session.save_cookies(page.context)

            # 5. 搜索酒店
            try:
                base_url = "https://hotels.ctrip.com/hotels/list"
                params = f"?city={city}&checkin={checkin}&checkout={checkout}"
                if keywords:
                    params += f"&keyword={keywords}"

                search_url = base_url + params
                logger.info(f"搜索酒店: {search_url}")

                # 先访问首页，再访问酒店列表（避免反爬）
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
