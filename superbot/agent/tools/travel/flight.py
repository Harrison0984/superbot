"""Flight search tool for superbot."""

import json
from typing import Any

from superbot.agent.tools.base import Tool, tool_error
from superbot.agent.tools.travel.shared import get_shared_browser
from superbot.agent.tools.travel.ctrip import ctrip_monitor
from superbot.agent.tools.travel.session import get_session_manager
from superbot.agent.tools.travel.logger import get_logger

logger = get_logger(__name__)


class FlightTool(Tool):
    """Search flights using Ctrip/Trip.com."""

    @property
    def name(self) -> str:
        return "flight_search"

    @property
    def description(self) -> str:
        return "搜索航班价格，支持出发城市、目的地和日期"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "from_city": {
                    "type": "string",
                    "description": "出发城市或机场代码，如：上海 或 SHA"
                },
                "to_city": {
                    "type": "string",
                    "description": "目的城市或机场代码，如：北京 或 PEK"
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

    async def execute(self, **kwargs: Any) -> str:
        """Execute flight search."""
        from_city = kwargs.get("from_city", "")
        to_city = kwargs.get("to_city", "")
        date = kwargs.get("date", "")

        try:
            browser = await self._get_browser()
            page = await browser.new_page()

            # 严格流程：
            # 1. 加载保存的 Cookie
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
                            "message": "需要登录才能搜索航班，请扫码登录后重试",
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

            # 5. 登录成功后，进行搜索
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
                    return json.dumps(result, ensure_ascii=False, indent=2)
                else:
                    return tool_error("no_results", f"未找到航班: {from_city} -> {to_city}, {date}")

            finally:
                await page.close()

        except Exception as e:
            import traceback
            return tool_error("internal_error", str(e), trace=traceback.format_exc())
