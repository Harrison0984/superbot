"""Flight search tool for superbot."""

import asyncio
import json
import uuid
from typing import Any

from superbot.agent.tools.base import Tool, tool_error
from superbot.agent.tools.travel.shared import get_shared_browser
from superbot.agent.tools.travel.ctrip import ctrip_monitor
from superbot.agent.tools.travel.session import get_session_manager
from superbot.agent.tools.travel.logger import get_logger
from superbot.bus.events import ToolEvent

logger = get_logger(__name__)


class FlightTool(Tool):
    """Search flights using Ctrip/Trip.com."""

    # 配置
    WAIT_TIMEOUT = 300  # 5分钟
    POLL_INTERVAL = 5   # 5秒
    PROGRESS_INTERVAL = 30  # 30秒

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

    async def execute(self, from_city: str, to_city: str, date: str, **kwargs: Any) -> str:
        """Execute flight search."""
        session_key = kwargs.get("_session_key", "cli:direct")
        task_id = str(uuid.uuid4())

        try:
            browser = await self._get_browser()
            page = await browser.new_page()

            # 1. 检查是否有保存的 Cookie
            has_saved_session = self.session.has_session()
            if has_saved_session:
                logger.info("加载已保存的 Cookie...")
                await self.session.apply_cookies_async(page.context)

            # 2. 检查是否已登录
            if has_saved_session:
                is_logged_in = await self.session.check_login(page)
                if is_logged_in:
                    logger.info("已登录，直接搜索...")
                else:
                    logger.info("Cookie 失效，需要重新登录...")
            else:
                is_logged_in = False

            if not is_logged_in:
                # 3. 未登录则生成二维码
                logger.info("未登录，生成二维码...")
                success, qr_path = await self.session.generate_qr_code(page)
                if success:
                    # 发送 waiting 事件（带二维码）
                    self._emit_event(ToolEvent(
                        session_key=session_key,
                        task_id=task_id,
                        tool_name="flight_search",
                        event_type="waiting",
                        content="请扫码登录，后台任务正在等待...",
                        media=[str(qr_path)]
                    ))

                    # 启动后台任务
                    asyncio.create_task(
                        self._wait_for_login_and_search(task_id, session_key, from_city, to_city, date, page)
                    )

                    # 立即返回
                    return json.dumps({"_tool_background": True, "task_id": task_id})
                else:
                    return tool_error("qr_generation_failed", "生成登录二维码失败，请重试")

            # 4. 已登录，直接搜索
            return await self._do_search(from_city, to_city, date, page)

        except Exception as e:
            import traceback
            return tool_error("internal_error", str(e), trace=traceback.format_exc())

    async def _wait_for_login_and_search(
        self,
        task_id: str,
        session_key: str,
        from_city: str,
        to_city: str,
        date: str,
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
                    tool_name="flight_search",
                    event_type="progress",
                    content=f"等待扫码中... ({elapsed // 60}分)"
                ))
                last_progress = elapsed

            # 检查登录（需要刷新页面检查二维码是否消失）
            try:
                await page.reload()
                await asyncio.sleep(2)
                is_logged_in = await self.session.check_login_async(page)
                if is_logged_in:
                    logger.info("扫码成功，开始搜索...")
                    # 登录成功，继续搜索
                    result = await self._do_search(from_city, to_city, date, page)
                    self._emit_event(ToolEvent(
                        session_key=session_key,
                        task_id=task_id,
                        tool_name="flight_search",
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
            tool_name="flight_search",
            event_type="error",
            content="扫码超时，请重试"
        ))

    async def _do_search(self, from_city: str, to_city: str, date: str, page) -> str:
        """执行航班搜索"""
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
