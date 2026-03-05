"""Hotel search tool for superbot."""

import asyncio
import json
import uuid
from typing import Any

from superbot.agent.tools.base import Tool, tool_error
from superbot.agent.tools.travel.shared import get_shared_browser
from superbot.agent.tools.travel.session import get_session_manager, verify_login
from superbot.agent.tools.travel.logger import get_logger
from superbot.bus.events import ToolEvent

logger = get_logger(__name__)


class HotelTool(Tool):
    """Search hotels using Ctrip/Trip.com."""

    # 配置
    WAIT_TIMEOUT = 300  # 5分钟
    POLL_INTERVAL = 5   # 5秒
    PROGRESS_INTERVAL = 30  # 30秒

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
        session_key = kwargs.get("_session_key", "cli:direct")
        task_id = str(uuid.uuid4())

        try:
            browser = await self._get_browser()
            page = await browser.new_page()

            # 1. 检查是否有保存的 Cookie
            has_saved_session = self.session.has_session()
            is_logged_in = False

            if has_saved_session:
                logger.info("加载已保存的 Cookie...")
                await self.session.apply_cookies_async(page.context)

            # 2. 必须检查页面登录状态（有cookie也可能失效）
            logger.info("检查页面登录状态...")
            is_logged_in = await verify_login(page, "https://hotels.ctrip.com/hotels/list")

            if is_logged_in:
                logger.info("已登录，直接搜索...")
            elif has_saved_session:
                logger.info("Cookie 失效，需要重新登录...")
            else:
                logger.info("未登录，需要登录...")

            if not is_logged_in:
                # 3. 未登录则生成二维码
                logger.info("未登录，生成二维码...")
                success, qr_path = await self.session.generate_qr_code(page)
                if success:
                    # 发送 waiting 事件（带二维码）
                    self._emit_event(ToolEvent(
                        session_key=session_key,
                        task_id=task_id,
                        tool_name="hotel_search",
                        event_type="waiting",
                        content="请扫码登录，后台任务正在等待...",
                        media=[str(qr_path)]
                    ))

                    # 启动后台任务
                    asyncio.create_task(
                        self._wait_for_login_and_search(task_id, session_key, city, checkin, checkout, keywords, page)
                    )

                    # 立即返回
                    return json.dumps({"_tool_background": True, "task_id": task_id})
                else:
                    return tool_error("qr_generation_failed", "生成登录二维码失败，请重试")

            # 4. 已登录，直接搜索
            return await self._do_search(city, checkin, checkout, keywords, page)

        except Exception as e:
            import traceback
            return tool_error("internal_error", str(e), trace=traceback.format_exc())

    async def _wait_for_login_and_search(
        self,
        task_id: str,
        session_key: str,
        city: str,
        checkin: str,
        checkout: str,
        keywords: str,
        page
    ):
        """后台等待登录并搜索"""
        last_progress = 0

        for elapsed in range(0, self.WAIT_TIMEOUT, self.POLL_INTERVAL):
            await asyncio.sleep(self.POLL_INTERVAL)

            # 定时发送 progress
            if elapsed - last_progress >= self.PROGRESS_INTERVAL:
                self._emit_event(ToolEvent(
                    session_key=session_key,
                    task_id=task_id,
                    tool_name="hotel_search",
                    event_type="progress",
                    content=f"等待扫码中... ({elapsed // 60}分)"
                ))
                last_progress = elapsed

            # 检查登录（需要刷新页面检查二维码是否消失）
            try:
                await page.reload()
                await asyncio.sleep(2)
                is_logged_in = await verify_login(page, "https://hotels.ctrip.com/hotels/list")
                if is_logged_in:
                    logger.info("扫码成功，开始搜索...")
                    # 登录成功，继续搜索
                    result = await self._do_search(city, checkin, checkout, keywords, page)
                    self._emit_event(ToolEvent(
                        session_key=session_key,
                        task_id=task_id,
                        tool_name="hotel_search",
                        event_type="complete",
                        content=result
                    ))
                    return
            except Exception as e:
                logger.warning(f"检查登录状态失败: {e}")

        # 超时
        await page.close()
        self._emit_event(ToolEvent(
            session_key=session_key,
            task_id=task_id,
            tool_name="hotel_search",
            event_type="error",
            content="扫码超时，请重试"
        ))

    async def _do_search(self, city: str, checkin: str, checkout: str, keywords: str, page) -> str:
        """执行酒店搜索"""
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
