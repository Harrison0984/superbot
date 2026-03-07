"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import weakref
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from superbot.agent.context import ContextBuilder
from superbot.agent.memory import MemoryStore
from superbot.agent.subagent import SubagentManager
from superbot.agent.tools.cron import CronTool
from superbot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from superbot.agent.tools.message import MessageTool
from superbot.agent.tools.registry import ToolRegistry
from superbot.agent.tools.shell import ExecTool
from superbot.agent.tools.spawn import SpawnTool
from superbot.agent.tools.web import WebFetchTool, WebSearchTool
from superbot.agent.tools.travel.flight import FlightTool
from superbot.agent.tools.travel.hotel import HotelTool
from superbot.agent.tools.feishu_doc import FeishuDocTool
from superbot.bus.events import InboundMessage, OutboundMessage, ToolEvent
from superbot.bus.queue import MessageBus
from superbot.providers.base import LLMProvider
from superbot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from superbot.config.schema import ChannelsConfig, ExecToolConfig, ProxyConfig, WebToolsConfig
    from superbot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_config: "WebToolsConfig | None" = None,
        proxy_config: "ProxyConfig | None" = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from superbot.config.schema import ExecToolConfig, ProxyConfig, WebToolsConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_config = web_config
        self.proxy_config = proxy_config or ProxyConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            brave_api_key=brave_api_key,
            web_config=self.web_config,
            proxy_config=self.proxy_config,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()

        # Initialize Claude MCP client if enabled
        self._claude_mcp = None
        self._claude_channel = None  # Reusable ClaudeChannel instance
        if channels_config and getattr(channels_config, "claude", None):
            claude_config = channels_config.claude
            if claude_config.enabled:
                from superbot.channels.claude_mcp import ClaudeMCPClient
                from superbot.channels.claude_channel import ClaudeChannel
                self._claude_mcp = ClaudeMCPClient(
                    workdir=claude_config.workdir,
                    script_path=claude_config.script_path,
                    auto_start=claude_config.auto_start,
                    retry_count=claude_config.retry_count,
                    retry_delay=claude_config.retry_delay,
                    retry_backoff=claude_config.retry_backoff,
                )
                # Create reusable ClaudeChannel instance
                self._claude_channel = ClaudeChannel(
                    config=None,
                    channels={},
                    mcp_client=self._claude_mcp,
                    bus=self.bus,
                )
                logger.info("Claude MCP client initialized: workdir={}, script={}", claude_config.workdir, claude_config.script_path)

        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(
            api_key=self.brave_api_key,
            config=self.web_config,
            proxy_config=self.proxy_config,
        ))
        self.tools.register(WebFetchTool(
            config=self.web_config,
            proxy_config=self.proxy_config,
        ))
        self.tools.register(FlightTool())
        self.tools.get("flight_search").set_bus(self.bus)

        self.tools.register(HotelTool())
        self.tools.get("hotel_search").set_bus(self.bus)
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(FeishuDocTool(config=self.channels_config.feishu if self.channels_config else None))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from superbot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _check_user_error(result: str) -> dict | None:
        """Check if tool result requires user interaction.

        Returns:
            dict with keys: message, media (optional), type, trace (optional)
            Or None if no user interaction required.
        """
        try:
            data = json.loads(result)
            if isinstance(data, dict) and "_tool_error" in data:
                error = data["_tool_error"]
                if error.get("feedback_to") == "user":
                    return {
                        "type": error.get("type"),
                        "message": error.get("message", ""),
                        "media": error.get("media", []),
                        "trace": error.get("trace", "")
                    }
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        session_key: str,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict], list[str]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages, user_media)."""
        # Limit messages to prevent token overflow
        max_history = self.memory_window
        if len(initial_messages) > max_history:
            initial_messages = initial_messages[-max_history:]

        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        user_media: list[str] = []  # Media files to send to user

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    if response.tool_calls:
                        await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                if not response.tool_calls:
                    logger.warning("has_tool_calls is True but tool_calls is None")
                    response.has_tool_calls = False
                    clean = self._strip_think(response.content)
                    messages = self.context.add_assistant_message(
                        messages, clean, reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    )
                    final_content = clean
                    break

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    # 在 tool_call.arguments 复制一份，添加 session_key
                    args = dict(tool_call.arguments)
                    args["_session_key"] = session_key
                    result = await self.tools.execute(tool_call.name, args)

                    # 检查是否是用户交互错误（需要直接反馈给用户）
                    user_feedback = self._check_user_error(result)
                    if user_feedback:
                        # 直接返回给用户，包含消息和媒体
                        logger.info("Tool requires user interaction: {}", user_feedback["message"])
                        # 将工具结果添加到消息历史（不打断循环）
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )
                        # 返回用户反馈内容作为最终回复
                        media = user_feedback.get("media", [])
                        return user_feedback["message"], tools_used, messages, media

                    # 检查是否是后台任务（工具正在等待用户交互）
                    try:
                        result_data = json.loads(result)
                        if isinstance(result_data, dict) and result_data.get("_tool_background"):
                            # 后台任务正在运行，需要添加工具结果到消息历史
                            logger.info("Tool {} running in background", tool_call.name)
                            messages = self.context.add_tool_result(
                                messages, tool_call.id, tool_call.name, result
                            )
                            return "正在处理中，请稍候...", tools_used, messages, []
                    except (json.JSONDecodeError, TypeError):
                        pass

                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages, user_media

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                # 同时监听 inbound 和 tool_events
                tasks = [
                    asyncio.create_task(self.bus.consume_inbound()),
                    asyncio.create_task(self.bus.tool_events.get())
                ]
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # 取消未完成的任务
                for t in pending:
                    t.cancel()

                for task in done:
                    try:
                        result = task.result()
                    except asyncio.CancelledError:
                        continue

                    if isinstance(result, ToolEvent):
                        # 处理工具事件
                        asyncio.create_task(self._handle_tool_event(result))
                    else:
                        # 处理用户消息
                        msg = result
                        if msg.content.strip().lower() == "/stop":
                            await self._handle_stop(msg)
                        else:
                            task = asyncio.create_task(self._dispatch(msg))
                            self._active_tasks.setdefault(msg.session_key, []).append(task)
                            task.add_done_callback(
                                lambda t, k=msg.session_key:
                                self._active_tasks.get(k, []) and
                                self._active_tasks[k].remove(t)
                                if t in self._active_tasks.get(k, []) else None
                            )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in agent loop")

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_tool_event(self, event: ToolEvent) -> None:
        """Handle tool events from background tasks."""
        if event.event_type == "waiting":
            # 发送等待消息给用户，包含二维码
            logger.info("Tool {} waiting: {}", event.tool_name, event.content)
            await self.bus.publish_outbound(OutboundMessage(
                channel=event.session_key.split(":")[0] if ":" in event.session_key else "feishu",
                chat_id=event.session_key.split(":")[1] if ":" in event.session_key else event.session_key,
                content=event.content,
                media=event.media
            ))

        elif event.event_type == "progress":
            # 附加到会话 metadata，不单独发消息给用户
            session = self.sessions.get_or_create(event.session_key)
            session.metadata.setdefault("_tool_progress", {})[event.task_id] = {
                "tool": event.tool_name,
                "message": event.content,
                "timestamp": datetime.now().isoformat()
            }
            logger.debug("Tool progress: {} - {}", event.tool_name, event.content)

        elif event.event_type in ("complete", "error"):
            # 直接处理工具结果，而不是通过 bus
            logger.info("Tool {} event: {}", event.event_type, event.tool_name)
            await self._handle_tool_result(event)

    async def _handle_tool_result(self, event: ToolEvent) -> None:
        """Handle tool result and continue conversation with LLM."""
        # 解析 session_key 获取 channel 和 chat_id
        parts = event.session_key.split(":", 1) if ":" in event.session_key else ("cli", event.session_key)
        channel, chat_id = parts[0], parts[1] if len(parts) > 1 else event.session_key

        try:
            # 获取会话
            session = self.sessions.get_or_create(event.session_key)

            # 添加工具结果到会话
            session.add_message(
                role="system",
                content=f"[{event.tool_name}]: {event.content}"
            )

            # 获取历史并构建消息
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=f"工具 {event.tool_name} 已完成: {event.content}",
                channel=channel,
                chat_id=chat_id,
            )

            # 设置工具上下文
            self._set_tool_context(channel, chat_id)

            # 运行 agent 获取回复
            final_content, _, all_msgs, _ = await self._run_agent_loop(messages, event.session_key)

            # 保存到会话
            self._save_turn(session, all_msgs, len(history))
            self.sessions.save(session)

            # 发送回复给用户
            await self.bus.publish_outbound(OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "任务已完成"
            ))
        except Exception as e:
            logger.error("Error handling tool result: %s", e)
            # 发送错误消息给用户
            await self.bus.publish_outbound(OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=f"处理工具结果时出错: {str(e)}"
            ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs, _ = await self._run_agent_loop(messages, key)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # Check for @claude request
        if self._claude_mcp:
            from superbot.channels.claude_channel import ClaudeChannel
            prompt = ClaudeChannel.detect_claude_request(msg.content)
            if prompt:
                logger.info("Detected @claude request: {}", prompt[:50])
                # Ensure MCP is available
                if await self._claude_mcp.ensure_available():
                    # Get feishu config if available
                    feishu_config = None
                    if self.channels_config:
                        feishu_config = getattr(self.channels_config, "feishu", None)

                    # Use reusable ClaudeChannel instance for preprocessing
                    processed_prompt = await self._claude_channel.preprocess_content(
                        prompt, msg.media, feishu_config
                    )
                    # Call Claude MCP
                    try:
                        result = await self._claude_mcp.call(processed_prompt)
                        logger.info("Claude MCP response: {}", result[:100] if result else "empty")
                        # Return result to original channel
                        return OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=result or "No response from Claude",
                        )
                    except Exception as e:
                        logger.error("Claude MCP call failed: {}", e)
                        # Fall through to normal LLM processing
                else:
                    logger.warning("Claude MCP unavailable after retry, falling back to default LLM")

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 superbot commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/help — Show available commands")

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs, user_media = await self._run_agent_loop(
            initial_messages, key, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            media=user_media, metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results.

        All messages are kept in memory for the current conversation (including tool
        results). Tool messages are filtered out only when persisting to disk.
        """
        from datetime import datetime

        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")

            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            archive_all=archive_all, memory_window=self.memory_window,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
