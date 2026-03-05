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

    async def execute(self, from_city: str, to_city: str, date: str, **kwargs: Any) -> str:
        """Execute flight search."""
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
            is_logged_in = await verify_login(page, "https://flights.ctrip.com/online/list")

            if is_logged_in:
                logger.info("已登录，直接搜索...")
            elif has_saved_session:
                logger.info("Cookie 失效，需要重新登录...")
            else:
                logger.info("未登录，需要登录...")

            if not is_logged_in:
                # 3. 未登录则生成二维码
                logger.info("未登录，生成二维码...")
                try:
                    success, qr_path = await self.session.generate_qr_code(page)
                except Exception as e:
                    logger.warning("生成二维码出错: %s", e)
                    # 如果生成二维码出错，可能是页面自动跳转了（已登录）
                    # 再次检查登录状态
                    logger.info("当前页面URL: %s", page.url)
                    is_logged_in = await verify_login(page, "https://flights.ctrip.com/online/list")
                    logger.info("检查登录状态: %s", is_logged_in)
                    if is_logged_in:
                        logger.info("页面已登录，直接搜索")
                        result = await self._do_search(from_city, to_city, date, page)
                        return result
                    success, qr_path = False, None

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

        async def _check_login_popup(page: Page) -> bool:
            """检查是否有登录弹窗/二维码（未登录）"""
            # 先尝试点击登录按钮，让弹窗显示出来
            try:
                login_link = await page.query_selector('text=登录')
                if login_link:
                    await login_link.click()
                    await asyncio.sleep(1)
            except:
                pass

            # 登录弹窗/二维码选择器
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

            # 检查是否有"扫码登录"等文字提示（说明是登录弹窗）
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
            """检查是否已登录（有用户信息元素）"""
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
            """检查登录状态

            Returns:
                (is_logged_in, status): 登录状态和状态描述
            """
            # 如果页面跳转到 passport.ctrip.com，说明用户正在登录
            if "passport.ctrip.com" in page.url:
                # 扫码成功，页面跳转了！返回True表示登录成功，需要跳回搜索页面
                return True, "login_redirect"

            # 页面跳转到了其他页面（比如首页），说明登录成功
            if "flights.ctrip.com" not in page.url and page.url != "about:blank":
                return True, "page_redirect"

            # 先检查是否有登录弹窗/二维码
            has_popup = await _check_login_popup(page)
            if has_popup:
                return False, "login_popup"

            # 检查是否已登录
            is_logged_in = await _check_logged_in(page)
            if is_logged_in:
                return True, "logged_in"

            # 没有弹窗也没有登录元素，可能不需要登录
            return True, "unknown"

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

            # 检查登录状态
            try:
                is_logged_in, status = await check_login_status(page)
                logger.info("登录状态: is_logged_in=%s, status=%s", is_logged_in, status)

                if is_logged_in:
                    # 页面跳转了，说明扫码成功，跳回搜索页面
                    if status in ("login_redirect", "page_redirect"):
                        logger.info("检测到页面跳转，扫码成功！正在跳转回搜索页面...")
                        # 跳回搜索页面
                        search_url = f"https://flights.ctrip.com/online/listonline/list/oneway-{from_city}-{to_city}?depdate={date}"
                        await page.goto(search_url)
                        await page.wait_for_load_state("networkidle")
                        await asyncio.sleep(2)

                    # 验证是否真正登录成功
                    await asyncio.sleep(2)  # 等待页面过渡

                    # 保存cookie并搜索
                    logger.info("扫码成功，开始搜索...")
                    await self.session.save_cookies(page.context)
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
                logger.warning("检查登录状态失败: %s", e)

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
