"""Tests for idle task system."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestIdleTask:
    """Test IdleTask base class."""

    def test_default_threshold(self):
        """Test default idle threshold is 3 minutes (180 seconds)."""
        from superbot.agent.idle_task import DEFAULT_IDLE_THRESHOLD

        assert DEFAULT_IDLE_THRESHOLD == 180

    def test_idle_task_properties(self):
        """Test IdleTask abstract properties."""
        from superbot.agent.idle_task import IdleTask

        # Create a concrete implementation for testing
        class TestTask(IdleTask):
            @property
            def name(self):
                return "test_task"

            @property
            def task_type(self):
                return "test"

            @property
            def idle_threshold_seconds(self):
                return 60

            async def should_run(self, agent, idle_seconds):
                return idle_seconds >= 60

            async def execute(self, agent):
                pass

        task = TestTask()
        assert task.name == "test_task"
        assert task.task_type == "test"
        assert task.idle_threshold_seconds == 60
        assert task.enabled is True


class TestAgentLoopIdleTasks:
    """Test AgentLoop idle task methods."""

    def test_register_idle_task(self):
        """Test registering idle task."""
        from superbot.agent.loop import AgentLoop

        # Create a mock AgentLoop
        agent = MagicMock()
        agent._idle_tasks = {}
        agent._running_idle_tasks = {}
        agent._last_task_end_time = None
        agent._idle_check_interval = 60
        agent._idle_threshold = 180

        # Create a mock task
        task = MagicMock()
        task.name = "test_task"
        task.task_type = "test"
        task.idle_threshold_seconds = 60

        # Call the method
        AgentLoop.register_idle_task(agent, task)

        # Verify
        assert "test" in agent._idle_tasks
        assert task._registered_threshold == 60

    def test_register_idle_task_with_custom_threshold(self):
        """Test registering idle task with custom threshold."""
        from superbot.agent.loop import AgentLoop

        agent = MagicMock()
        agent._idle_tasks = {}
        agent._running_idle_tasks = {}
        agent._last_task_end_time = None
        agent._idle_check_interval = 60
        agent._idle_threshold = 180

        task = MagicMock()
        task.name = "test_task"
        task.task_type = "test"
        task.idle_threshold_seconds = 60

        # Call with custom threshold
        AgentLoop.register_idle_task(agent, task, idle_threshold_seconds=120)

        # Verify custom threshold is used
        assert task._registered_threshold == 120

    def test_unregister_idle_task(self):
        """Test unregistering idle task."""
        from superbot.agent.loop import AgentLoop

        task = MagicMock()
        task.name = "test_task"
        task.task_type = "test"

        agent = MagicMock()
        agent._idle_tasks = {"test": task}
        agent._running_idle_tasks = {}

        # Call the method
        result = AgentLoop.unregister_idle_task(agent, "test")

        # Verify
        assert result is True
        assert "test" not in agent._idle_tasks

    def test_unregister_nonexistent_task(self):
        """Test unregistering nonexistent task returns False."""
        from superbot.agent.loop import AgentLoop

        agent = MagicMock()
        agent._idle_tasks = {}
        agent._running_idle_tasks = {}

        result = AgentLoop.unregister_idle_task(agent, "nonexistent")

        assert result is False

    def test_list_idle_tasks(self):
        """Test listing idle tasks."""
        from superbot.agent.loop import AgentLoop

        task1 = MagicMock()
        task1.task_type = "task1"
        task2 = MagicMock()
        task2.task_type = "task2"

        agent = MagicMock()
        agent._idle_tasks = {"task1": task1, "task2": task2}

        result = AgentLoop.list_idle_tasks(agent)

        assert result == ["task1", "task2"]


class TestIdleTaskExecution:
    """Test idle task execution logic."""

    @pytest.mark.asyncio
    async def test_run_idle_tasks_skips_running_tasks(self):
        """Test that running tasks are skipped."""
        from superbot.agent.loop import AgentLoop

        # Create mock task
        task = MagicMock()
        task.task_type = "test"
        task._registered_threshold = 0
        task.should_run = AsyncMock(return_value=True)

        agent = MagicMock()
        agent._idle_tasks = {"test": task}
        agent._running_idle_tasks = {"test": MagicMock()}  # Already running
        agent._last_task_end_time = None

        await AgentLoop._run_idle_tasks(agent, idle_seconds=100)

        # Task should not be executed since it's already running
        task.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_idle_tasks_checks_threshold(self):
        """Test that threshold is checked before execution."""
        from superbot.agent.loop import AgentLoop

        task = MagicMock()
        task.task_type = "test"
        task._registered_threshold = 200  # Requires 200 seconds
        task.should_run = AsyncMock(return_value=True)

        agent = MagicMock()
        agent._idle_tasks = {"test": task}
        agent._running_idle_tasks = {}

        # Only 100 seconds idle - should not run
        await AgentLoop._run_idle_tasks(agent, idle_seconds=100)

        task.should_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_idle_tasks_executes_when_ready(self):
        """Test task is executed when conditions are met."""
        from superbot.agent.loop import AgentLoop

        task = MagicMock()
        task.task_type = "test"
        task._registered_threshold = 50
        task.should_run = AsyncMock(return_value=True)
        task.execute = AsyncMock()

        agent = MagicMock()
        agent._idle_tasks = {"test": task}
        agent._running_idle_tasks = {}

        # 100 seconds idle - meets threshold
        await AgentLoop._run_idle_tasks(agent, idle_seconds=100)

        task.should_run.assert_called_once()
        task.execute.assert_called_once()


class TestIdleTimer:
    """Test idle timer functionality."""

    @pytest.mark.asyncio
    async def test_idle_timer_waits_and_returns(self):
        """Test idle timer waits for interval then returns."""
        from superbot.agent.loop import AgentLoop
        import asyncio

        agent = MagicMock()
        agent._idle_check_interval = 1  # 1 second for test

        with patch.object(asyncio, 'sleep', return_value=None) as mock_sleep:
            await AgentLoop._idle_timer(agent)
            mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_check_and_run_idle_tasks_returns_early_if_no_end_time(self):
        """Test early return if no task has ended yet."""
        from superbot.agent.loop import AgentLoop

        agent = MagicMock()
        agent._last_task_end_time = None
        agent._run_idle_tasks = AsyncMock()

        await AgentLoop._check_and_run_idle_tasks(agent)

        agent._run_idle_tasks.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_and_run_idle_tasks_below_threshold(self):
        """Test tasks don't run if below threshold."""
        import time
        from superbot.agent.loop import AgentLoop

        agent = MagicMock()
        agent._last_task_end_time = time.time() - 10  # Only 10 seconds ago
        agent._idle_threshold = 180
        agent._run_idle_tasks = AsyncMock()

        await AgentLoop._check_and_run_idle_tasks(agent)

        agent._run_idle_tasks.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_and_run_idle_tasks_runs_when_idle(self):
        """Test tasks run when idle threshold is met."""
        import time
        from superbot.agent.loop import AgentLoop

        agent = MagicMock()
        agent._last_task_end_time = time.time() - 200  # 200 seconds ago
        agent._idle_threshold = 180
        agent._run_idle_tasks = AsyncMock()

        await AgentLoop._check_and_run_idle_tasks(agent)

        agent._run_idle_tasks.assert_called_once_with(200)
