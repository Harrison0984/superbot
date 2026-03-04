"""Stealth strategies for Ocular AI.

This module provides advanced anti-detection strategies.
"""

import asyncio
import random
from typing import Any, Callable, Dict, List, Optional

from playwright.async_api import Page

from superbot.agent.tools.travel.config import config
from superbot.agent.tools.travel.logger import get_logger
from superbot.agent.tools.travel.humanize import human_behavior

logger = get_logger(__name__)


class StealthStrategy:
    """Base class for stealth strategies."""

    def __init__(self):
        self.config = config.browser.get("stealth", {})

    async def apply(self, page: Page) -> None:
        """Apply stealth strategy to page. Must be implemented by subclasses."""
        raise NotImplementedError


class WebDriverHidden(StealthStrategy):
    """Hides the webdriver property."""

    async def apply(self, page: Page) -> None:
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        logger.debug("Applied webdriver hiding")


class CanvasFingerprintRandomizer(StealthStrategy):
    """Randomizes canvas fingerprint."""

    async def apply(self, page: Page) -> None:
        await page.add_init_script("""
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function(type) {
                const canvas = this;
                const ctx = canvas.getContext('2d');

                // Add noise to canvas
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;

                for (let i = 0; i < data.length; i += 4) {
                    if (Math.random() > 0.5) {
                        data[i] = Math.min(255, data[i] + Math.floor(Math.random() * 3));
                        data[i + 1] = Math.min(255, data[i + 1] + Math.floor(Math.random() * 3));
                        data[i + 2] = Math.min(255, data[i + 2] + Math.floor(Math.random() * 3));
                    }
                }

                ctx.putImageData(imageData, 0, 0);
                return originalToDataURL.apply(this, arguments);
            };
        """)
        logger.debug("Applied canvas fingerprint randomization")


class WebGLStealth(StealthStrategy):
    """Masks WebGL fingerprint."""

    async def apply(self, page: Page) -> None:
        await page.add_init_script("""
            const glParams = {
                37445: 'Intel Inc.',
                37446: 'Intel Iris OpenGL Engine',
                37487: 'WebGL GLSL ES 3.00',
            };

            const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (glParams.hasOwnProperty(parameter)) {
                    return glParams[parameter];
                }
                return originalGetParameter.apply(this, arguments);
            };
        """)
        logger.debug("Applied WebGL stealth")


class AudioContextStealth(StealthStrategy):
    """Masks AudioContext fingerprint."""

    async def apply(self, page: Page) -> None:
        await page.add_init_script("""
            const originalGetChannelData = AudioContext.prototype.getChannelData;
            AudioContext.prototype.getChannelData = function(channelData) {
                const result = originalGetChannelData.apply(this, arguments);

                // Add minimal noise
                const noise = () => (Math.random() - 0.5) * 0.00001;
                for (let i = 0; i < result.length; i += 100) {
                    result[i] += noise();
                }

                return result;
            };
        """)
        logger.debug("Applied AudioContext stealth")


class UserAgentSpoofing(StealthStrategy):
    """Spoofs user agent to match proxy location."""

    def __init__(self):
        super().__init__()
        self.user_agent = config.browser.get(
            "user_agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    async def apply(self, page: Page) -> None:
        # Set additional navigator properties
        await page.add_init_script(f"""
            Object.defineProperty(navigator, 'hardwareConcurrency', {{
                get: () => {random.randint(4, 8)}
            }});
            Object.defineProperty(navigator, 'deviceMemory', {{
                get: () => {random.choice([4, 8, 16])}
            }});
        """)
        logger.debug("Applied user agent spoofing")


class RetryStrategy:
    """Automatic retry strategy for failed operations."""

    def __init__(self):
        self.max_retries = config.proxy.get("failover", {}).get("max_retries", 3)
        self.base_delay = 1.0

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry on failure.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            Last exception if all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Operation succeeded on attempt {attempt + 1}")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)

        logger.error(f"All {self.max_retries} attempts failed")
        raise last_error


class StealthManager:
    """Manages all stealth strategies."""

    def __init__(self):
        self.strategies: List[StealthStrategy] = [
            WebDriverHidden(),
            CanvasFingerprintRandomizer(),
            WebGLStealth(),
            AudioContextStealth(),
            UserAgentSpoofing(),
        ]
        self.retry_strategy = RetryStrategy()

    async def apply_all(self, page: Page) -> None:
        """Apply all stealth strategies to a page."""
        for strategy in self.strategies:
            try:
                await strategy.apply(page)
            except Exception as e:
                logger.error(f"Failed to apply {strategy.__class__.__name__}: {e}")

        logger.info(f"Applied {len(self.strategies)} stealth strategies")

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry and stealth."""
        return await self.retry_strategy.execute(func, *args, **kwargs)


# Global stealth manager
stealth_manager = StealthManager()
