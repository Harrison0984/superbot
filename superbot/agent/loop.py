"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from superbot.agent.context import ContextBuilder
from superbot.agent.subagent import SubagentManager
from superbot.agent.tools.cron import CronTool
from superbot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from superbot.agent.tools.registry import ToolRegistry
from superbot.agent.tools.shell import ExecTool
from superbot.agent.tools.spawn import SpawnTool
from superbot.agent.tools.web import WebFetchTool, WebSearchTool
from superbot.agent.tools.bookmark import BookmarkTool
from superbot.agent.tools.travel.flight import FlightTool
from superbot.agent.tools.travel.hotel import HotelTool
from superbot.agent.tools.feishu_doc import FeishuDocTool
from superbot.agent.tools.email import EmailTool
from superbot.bus.events import InboundMessage, OutboundMessage
from superbot.bus.queue import MessageBus
from superbot.providers.base import LLMProvider
from superbot.session.manager import SessionManager

if TYPE_CHECKING:
    from superbot.agent.idle_task import IdleTask
    from superbot.config.schema import BookmarkConfig, ChannelsConfig, ExecToolConfig, ProxyConfig, WebToolsConfig
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
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_config: "WebToolsConfig | None" = None,
        bookmark_config: "BookmarkConfig | None" = None,
        proxy_config: "ProxyConfig | None" = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        memory_provider: LLMProvider | None = None,
        memory_system: Any = None,
        embedding_config: Any = None,
    ):
        from superbot.config.schema import ExecToolConfig, ProxyConfig, WebToolsConfig
        self.bus = bus
        self.channels_config = channels_config
        self.memory_provider = memory_provider
        self.memory_system = memory_system
        self.provider = provider

        # Initialize experience store for tool execution history
        self.workspace = workspace

        self._experience_store = None
        self._init_experience_store()
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_config = web_config
        self.bookmark_config = bookmark_config
        self.proxy_config = proxy_config or ProxyConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        # Get channels that need build_messages (dynamically from config)
        self._channels_with_memory: set[str] = {"cli"}
        if channels_config:
            for name in ("feishu", "telegram", "whatsapp", "qq"):
                config = getattr(channels_config, name, None)
                if config and getattr(config, "enabled", False):
                    self._channels_with_memory.add(name)

        self.context = ContextBuilder(workspace, memory_system=memory_system)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.tools.set_bus(bus)
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
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()

        # Idle task system
        from superbot.agent.idle_task import DEFAULT_IDLE_THRESHOLD
        self._idle_tasks: dict[str, "IdleTask"] = {}  # task_type -> task
        self._running_idle_tasks: dict[str, asyncio.Task] = {}  # task_type -> running task
        self._last_task_end_time: float | None = None  # End time of last task
        self._idle_check_interval = 60  # Idle check interval in seconds
        self._idle_threshold = DEFAULT_IDLE_THRESHOLD  # Default idle threshold in seconds

        # Register default idle tasks
        self._register_default_idle_tasks()

        self._register_default_tools()

    def _register_default_idle_tasks(self) -> None:
        """Register default idle tasks."""
        # Import here to avoid circular imports
        from superbot.agent.idle_tasks import CleanupIdleTask

        # Register cleanup (default threshold)
        self.register_idle_task(CleanupIdleTask())

        # Register reflection task if memory system is available
        if self.memory_system is not None:
            from superbot.memory.facade.reflection_task import ReflectionIdleTask
            # Get internal MemorySystem from MemoryAdapter if needed
            memory_sys = self.memory_system
            if hasattr(self.memory_system, '_memory_system'):
                memory_sys = self.memory_system._memory_system
            if memory_sys is not None:
                self.register_idle_task(ReflectionIdleTask(memory_sys))
                logger.info("Registered reflection idle task")

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
        self.tools.register(WebSearchTool())
        self.tools.register(WebFetchTool())
        self.tools.register(BookmarkTool(
            bookmarks=self.bookmark_config.bookmarks if self.bookmark_config else {}
        ))
        self.tools.register(FlightTool())
        self.tools.get("flight_search").set_bus(self.bus)

        self.tools.register(HotelTool())
        self.tools.get("hotel_search").set_bus(self.bus)
        self.tools.register(FeishuDocTool(config=self.channels_config.feishu if self.channels_config else None))

        # Register Claude tool (invoked via @claude)
        # Use direct subprocess implementation to avoid MCP asyncio issues
        claude_config = getattr(self.channels_config, 'claude', None) if self.channels_config else None
        if claude_config and claude_config.enabled:
            from superbot.agent.tools.claude_subprocess import ClaudeToolDirect
            self.tools.register(ClaudeToolDirect(config=claude_config))

        self._email_tool: EmailTool | None = None
        if self.channels_config and hasattr(self.channels_config, 'email') and self.channels_config.email:
            self._email_tool = EmailTool(config=self.channels_config.email, proxy_config=self.proxy_config)
            self.tools.register(self._email_tool)
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    # Idle task methods

    def register_idle_task(
        self,
        task: "IdleTask",
        idle_threshold_seconds: int | None = None,
    ) -> None:
        """Register an idle task.

        Args:
            task: The task instance to register
            idle_threshold_seconds: Idle threshold in seconds, defaults to task's own idle_threshold_seconds
        """
        from superbot.agent.idle_task import DEFAULT_IDLE_THRESHOLD
        threshold = idle_threshold_seconds if idle_threshold_seconds is not None else task.idle_threshold_seconds
        # Store threshold information
        task._registered_threshold = threshold  # type: ignore[attr-defined]
        self._idle_tasks[task.task_type] = task
        logger.info(
            "Registered idle task: {} (type: {}, threshold: {}s)",
            task.name,
            task.task_type,
            threshold,
        )

    def unregister_idle_task(self, task_type: str) -> bool:
        """Unregister an idle task."""
        if task_type in self._idle_tasks:
            task = self._idle_tasks.pop(task_type)
            logger.info("Unregistered idle task: {} (type: {})", task.name, task_type)
            return True
        return False

    def list_idle_tasks(self) -> list[str]:
        """List registered idle task types."""
        return list(self._idle_tasks.keys())

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

    def _init_experience_store(self) -> None:
        """Initialize experience store for tool execution history."""
        try:
            from superbot.memory.storage.experience_store import ExperienceStore
            db_path = str(self.workspace / "memory" / "experience.db")
            self._experience_store = ExperienceStore(db_path=db_path)
            logger.info("Experience store initialized at {}", db_path)
        except Exception as e:
            logger.warning("Failed to initialize experience store: {}", e)
            self._experience_store = None

    def _rank_tool_calls(self, tool_calls: list) -> list:
        """Rank tool calls based on experience (success rate and recency)."""
        if not self._experience_store or not tool_calls:
            return tool_calls

        try:
            experiences = self._experience_store.get_all_success_rates()
            if not experiences:
                return tool_calls

            def _calc_score(tc):
                exp = experiences.get(tc.name)
                if not exp:
                    return 0.0  # No experience = 0

                # Base score from success rate
                success_rate = exp.get("success_rate", 0)
                if success_rate == 0:
                    return 0.0

                # Time factor: recent success = higher score
                # Today = 1.0, 30 days ago = 0.5, >30 days = 0.5
                time_factor = 0.5
                last_success = exp.get("last_success")
                if last_success:
                    try:
                        from datetime import datetime
                        last_dt = datetime.fromisoformat(last_success.replace('Z', '+00:00'))
                        days_since = (datetime.now() - last_dt.replace(tzinfo=None)).days
                        time_factor = max(0.5, 1.0 - days_since / 30)
                    except Exception:
                        pass

                return success_rate * time_factor

            # Sort by score descending
            scored = [(tc, _calc_score(tc)) for tc in tool_calls]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [tc for tc, _ in scored]
        except Exception as e:
            logger.warning("Failed to rank tool calls: {}", e)
            return tool_calls

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None


    def _record_action_result(
        self,
        tool_name: str,
        args: dict,
        result: str,
        time_cost: float,
    ) -> None:
        """Record tool execution result to pending actions."""
        if not self._experience_store:
            return

        try:
            # Determine method from args (common method fields)
            method = args.get("method") or args.get("source") or "default"

            # Determine if result is valid (not empty, no error)
            has_result = bool(result and len(result.strip()) > 0)
            has_error = result and ("error" in result.lower() or "failed" in result.lower())

            # Add to pending actions (success will be determined later)
            if not hasattr(self, '_pending_actions'):
                self._pending_actions = []

            self._pending_actions.append({
                "tool_name": tool_name,
                "method": method,
                "time_cost": time_cost,
                "has_result": has_result,
                "has_error": has_error,
                "args": args,
            })
        except Exception as e:
            logger.debug("Failed to record action result: {}", e)

    def _finalize_action_results(self) -> None:
        """Finalize pending action results - mark last one as success."""
        if not self._experience_store or not hasattr(self, '_pending_actions'):
            return

        if not self._pending_actions:
            return

        try:
            # Mark the last action as success (agent stopped here = satisfied)
            # Others are marked as intermediate/failed
            for i, action in enumerate(self._pending_actions):
                # Last one is success, others are intermediate
                is_last = (i == len(self._pending_actions) - 1)

                if is_last:
                    # Last tool call - agent stopped here = success
                    success = action["has_result"] and not action["has_error"]
                else:
                    # Previous tool calls - not the final answer
                    success = False

                self._experience_store.record_action(
                    action_type=action["tool_name"],
                    method=action["method"],
                    success=success,
                    quality=None,
                    time_cost=action["time_cost"],
                    context=action["args"],
                )

            logger.debug("Finalized {} action results", len(self._pending_actions))
        except Exception as e:
            logger.warning("Failed to finalize action results: {}", e)
        finally:
            # Clear pending actions
            self._pending_actions = []

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

    def _parse_at_tool(self, content: str) -> tuple[str, str] | None:
        """Parse @tool_name pattern from message content.

        Returns:
            (tool_name, remaining_content) if @tool detected, None otherwise
        """
        match = re.match(r"@(\w+)\s+(.+)", content, re.DOTALL)
        if match:
            tool_name = match.group(1).strip()
            remaining = match.group(2).strip()
            return tool_name, remaining
        return None

    async def _handle_at_tool(self, msg: InboundMessage) -> OutboundMessage | None:
        """Handle @tool_name direct tool invocation."""
        result = self._parse_at_tool(msg.content)
        if not result:
            return None

        tool_name, tool_input = result

        # Check if tool exists
        tool = self.tools.get(tool_name)
        if not tool:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.tool_names)}",
                to=msg.to,
            )

        try:
            # Prepare parameters
            params = {"media": msg.media} if msg.media else {}

            # Execute tool
            result = await tool.execute(
                msg.channel,
                msg.sender_id,
                msg.chat_id,
                tool_input,
                **params
            )

            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=result,
                to=msg.to,
            )
        except Exception as e:
            logger.error("Error executing @tool: {}", e)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Error executing {tool_name}: {str(e)}",
                to=msg.channel,
            )

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        channel: str,
        sender_id: str,
        chat_id: str,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str]]:
        """Run the agent iteration loop. Returns (final_content, user_media)."""
        # All context comes from vector memory system via system prompt
        messages = initial_messages
        iteration = 0
        final_content = None
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

                # Rank tool calls based on experience
                tool_calls = self._rank_tool_calls(response.tool_calls)

                # Initialize pending actions for this round
                self._pending_actions = []

                for tool_call in tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    args = dict(tool_call.arguments)
                    # Extract content from args
                    content = args.pop("content", "")

                    # Record start time for this tool execution
                    start_time = time.time()

                    result = await self.tools.execute(
                        tool_call.name, channel, sender_id, chat_id, content, args
                    )

                    # Calculate elapsed time
                    elapsed = time.time() - start_time

                    # Parse tool return value, delegate to LLM
                    try:
                        result_data = json.loads(result)
                        tool_content = result_data.get("content", result)
                        tool_media = result_data.get("media", [])
                    except (json.JSONDecodeError, AttributeError):
                        tool_content = result
                        tool_media = []

                    # Add media files returned by tool
                    user_media.extend(tool_media)

                    # Record action result to experience store (with time cost)
                    self._record_action_result(
                        tool_call.name, args, tool_content, elapsed
                    )

                    # Delegate to LLM
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, tool_content
                    )

                # Finalize action results after the loop (mark last as success)
                self._finalize_action_results()
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

        return final_content, user_media

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()

        while self._running:
            try:
                # Listen to both inbound and idle_timer simultaneously
                tasks = [
                    asyncio.create_task(self.bus.consume_inbound()),
                    asyncio.create_task(self._idle_timer()),
                ]
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel pending tasks
                for t in pending:
                    t.cancel()

                for task in done:
                    try:
                        result = task.result()
                    except asyncio.CancelledError:
                        continue

                    if result is None:
                        # idle_timer triggered, check idle tasks
                        await self._check_and_run_idle_tasks()
                    else:
                        # Process user message
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

    async def _idle_timer(self) -> None:
        """Idle timer, periodically triggers idle task check."""
        await asyncio.sleep(self._idle_check_interval)

    async def _check_and_run_idle_tasks(self) -> None:
        """Check and execute idle tasks (called by timer)."""
        if self._last_task_end_time is None:
            return

        idle_seconds = time.time() - self._last_task_end_time

        if idle_seconds >= self._idle_threshold:
            await self._run_idle_tasks(idle_seconds)

    async def _run_idle_tasks(self, idle_seconds: float) -> None:
        """Check and execute idle tasks.

        Args:
            idle_seconds: Current idle duration in seconds.
        """
        for task_type, task in list(self._idle_tasks.items()):
            # Check if already running (same type filter)
            if task_type in self._running_idle_tasks:
                continue

            # Get task's idle threshold
            threshold = getattr(task, "_registered_threshold", task.idle_threshold_seconds)

            # Check if idle time meets threshold
            if idle_seconds < threshold:
                continue

            # Check if task should run
            if not await task.should_run(self, idle_seconds):
                continue

            # Start task
            logger.info("Executing idle task: {} (type: {})", task.name, task_type)
            t = asyncio.create_task(self._execute_idle_task(task, task_type))
            self._running_idle_tasks[task_type] = t
            t.add_done_callback(lambda _, k=task_type: self._running_idle_tasks.pop(k, None))

    async def _execute_idle_task(self, task: "IdleTask", task_type: str) -> None:
        """Execute a single idle task."""
        try:
            await task.execute(self)
        except Exception:
            logger.exception("Error executing idle task: {}", task_type)
        finally:
            self._running_idle_tasks.pop(task_type, None)
            # Update last task end time to prevent repeated execution
            self._last_task_end_time = time.time()

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
            channel=msg.channel, chat_id=msg.chat_id, content=content, to=msg.channel,
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
                        content="", metadata=msg.metadata or {}, to=msg.channel,
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.", to=msg.to,
                ))
            finally:
                # Record task end time for idle duration calculation
                self._last_task_end_time = time.time()

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
        # Stop Email IMAP polling
        if self._email_tool:
            self._email_tool.stop_polling()
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        # System messages (from subagent) should skip memory retrieval
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            # Build messages WITHOUT memory for system messages (subagent results)
            runtime_ctx = self.context._build_runtime_context(channel, chat_id)
            messages = [
                {"role": "system", "content": self.context._get_identity(channel=channel)},
                {"role": "user", "content": f"{runtime_ctx}\n\n{msg.content}"},
            ]
            final_content, user_media = await self._run_agent_loop(
                messages, channel, msg.sender_id, chat_id
            )
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.", to=msg.to, media=user_media)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # Check for @tool_name direct invocation
        at_result = await self._handle_at_tool(msg)
        if at_result:
            return at_result

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Messages are already stored to vector memory in real-time, just clear session
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.", to=msg.to)
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 superbot commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/help — Show available commands", to=msg.to)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))

        # Only messages from channel or CLI execute build_messages
        # email is a forward source, no need for build_messages
        if msg.channel in self._channels_with_memory:
            initial_messages = self.context.build_messages(
                current_message=msg.content,
                media=msg.media if msg.media else None,
                channel=msg.channel, chat_id=msg.chat_id,
            )
        else:
            initial_messages = [
                {"role": "system", "content": self.context._get_identity(channel=msg.channel)},
                {"role": "user", "content": msg.content},
            ]

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta, to=msg.to,
            ))

        final_content, user_media = await self._run_agent_loop(
            initial_messages, msg.channel, msg.sender_id, msg.chat_id,
            on_progress=on_progress or _bus_progress,
        )

        logger.debug("final_content for {}:{}: {}", msg.channel, msg.sender_id, final_content)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Store to vector memory in real-time if enabled
        # Only messages from channel or CLI store memory
        if self.memory_system is not None and msg.channel in self._channels_with_memory:
            try:
                # Store assistant messages
                await self.memory_system.remember(final_content)
            except Exception as e:
                logger.error("Error storing to real-time memory: {}", e)

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            media=user_media, metadata=msg.metadata or {}, to=msg.to,
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
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content, to=channel)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
