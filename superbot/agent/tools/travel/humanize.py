"""Human behavior simulation for Ocular AI."""

import math
import random
import time
from typing import List, Tuple

from superbot.agent.tools.travel.config import config
from superbot.agent.tools.travel.logger import get_logger

logger = get_logger(__name__)


class HumanBehavior:
    """Simulates human behavior patterns for stealth."""

    def __init__(self):
        behavior_config = config.behavior

        # Delay settings
        delay_config = behavior_config.get("delay", {})
        self.delay_mean = delay_config.get("mean", 2.5)
        self.delay_std = delay_config.get("std", 0.8)
        self.delay_min = delay_config.get("min", 1.5)
        self.delay_max = delay_config.get("max", 4.5)

        # Mouse settings
        mouse_config = behavior_config.get("mouse", {})
        self.curve_type = mouse_config.get("curve_type", "bezier")
        self.mouse_points = mouse_config.get("points", 5)

        # Scroll settings
        scroll_config = behavior_config.get("scroll", {})
        self.scroll_stay_time = scroll_config.get("stay_time", 2.0)
        self.viewport_fraction = scroll_config.get("viewport_fraction", 0.5)

    def random_delay(self) -> float:
        """Generate a random delay using Gaussian distribution."""
        delay = random.gauss(self.delay_mean, self.delay_std)
        # Clamp to min/max range
        delay = max(self.delay_min, min(self.delay_max, delay))
        return delay

    def delay(self) -> None:
        """Execute a random delay."""
        delay_time = self.random_delay()
        logger.debug(f"Human delay: {delay_time:.2f}s")
        time.sleep(delay_time)

    def generate_bezier_curve(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        control_points: int = 3
    ) -> List[Tuple[int, int]]:
        """Generate a Bezier curve path for mouse movement.

        Args:
            start: Starting coordinates (x, y)
            end: Ending coordinates (x, y)
            control_points: Number of control points to generate

        Returns:
            List of coordinates representing the curved path
        """
        # Generate random control points
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        # Add random offset to create curve
        control_offset = random.randint(-100, 100)
        control_points_list = []

        for i in range(control_points):
            t = (i + 1) / (control_points + 1)
            x = mid_x + control_offset * math.sin(t * math.pi)
            y = mid_y + control_offset * math.cos(t * math.pi)
            control_points_list.append((int(x), int(y)))

        # Generate bezier curve points
        path = [start]
        num_steps = 20

        for i in range(num_steps):
            t = (i + 1) / num_steps
            point = self._bezier_point(start, end, control_points_list, t)
            path.append(point)

        return path

    def _bezier_point(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        control_points: List[Tuple[int, int]],
        t: float
    ) -> Tuple[int, int]:
        """Calculate a point on a Bezier curve."""
        # Cubic Bezier formula
        x = (1 - t) ** 3 * start[0]
        y = (1 - t) ** 3 * start[1]

        for i, cp in enumerate(control_points):
            coefficient = 3 * (t ** (i + 1)) * ((1 - t) ** (2 - i))
            x += coefficient * cp[0]
            y += coefficient * cp[1]

        x += t ** 3 * end[0]
        y += t ** 3 * end[1]

        return (int(x), int(y))

    def generate_arc_movement(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Generate an arc-shaped mouse movement path.

        Args:
            start: Starting coordinates (x, y)
            end: Ending coordinates (x, y)

        Returns:
            List of coordinates representing the arc path
        """
        path = []
        num_steps = 15

        # Calculate arc parameters
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # Random arc height
        arc_height = distance * random.uniform(0.2, 0.5)

        # Random direction (up or down)
        direction = 1 if random.random() > 0.5 else -1

        for i in range(num_steps):
            t = (i + 1) / num_steps

            # Linear interpolation
            x = start[0] + dx * t
            y = start[1] + dy * t

            # Add arc offset
            arc_offset = direction * arc_height * math.sin(t * math.pi)
            # Determine if we should offset in X or Y based on movement direction
            if abs(dx) > abs(dy):
                y += arc_offset
            else:
                x += arc_offset

            path.append((int(x), int(y)))

        return path

    def generate_mouse_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Generate a human-like mouse movement path.

        Args:
            start: Starting coordinates (x, y)
            end: Ending coordinates (x, y)

        Returns:
            List of coordinates representing the movement path
        """
        if self.curve_type == "bezier":
            return self.generate_bezier_curve(start, end, self.mouse_points)
        else:
            return self.generate_arc_movement(start, end)

    async def simulate_scroll(self, page) -> None:
        """Simulate human-like scrolling behavior.

        Args:
            page: Playwright page object
        """
        # Get viewport size
        viewport = await page.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")

        # Scroll to middle first
        mid_y = viewport["height"] * self.viewport_fraction / 2
        await page.evaluate(f"window.scrollTo(0, {mid_y})")

        # Stay for a while (simulating reading/looking)
        await page.wait_for_timeout(int(self.scroll_stay_time * 1000))

        logger.debug("Simulated scroll behavior")


# Global human behavior instance
human_behavior = HumanBehavior()
