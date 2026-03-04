"""Flight search tool for superbot."""

import json
from typing import Any

from superbot.agent.tools.base import Tool
from superbot.agent.tools.travel.browser import StealthBrowser
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
        self.browser = None
        self.session = get_session_manager()

    async def _get_browser(self):
        if self.browser is None:
            self.browser = StealthBrowser()
            await self.browser.initialize()
        return self.browser

    async def execute(self, **kwargs: Any) -> str:
        """Execute flight search."""
        from_city = kwargs.get("from_city", "")
        to_city = kwargs.get("to_city", "")
        date = kwargs.get("date", "")

        try:
            browser = await self._get_browser()
            page = await browser.new_page()

            # Apply saved cookies if available
            if self.session.has_session():
                self.session.apply_cookies(page.context)

            # Check login status first
            if not await self.session.check_login(page):
                logger.info("User not logged in, waiting for login...")
                login_success = await self.session.wait_for_login(page)
                if not login_success:
                    return json.dumps({
                        "error": "login_required",
                        "message": "需要登录才能搜索航班，请扫码登录后重试"
                    }, ensure_ascii=False)

            try:
                # Convert city name to airport code if needed (3-letter uppercase = airport code)
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
                    return json.dumps({"error": "未找到航班", "from": from_city, "to": to_city, "date": date})

            finally:
                await page.close()

        except Exception as e:
            import traceback
            return json.dumps({"error": str(e), "trace": traceback.format_exc()})
