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
    """Hides the webdriver property and other automation features."""

    async def apply(self, page: Page) -> None:
        await page.add_init_script("""
            // Hide webdriver
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });

            // Hide automation flags
            Object.defineProperty(navigator, 'automation', {
                get: () => undefined
            });

            // Override chrome runtime
            if (window.chrome) {
                window.chrome.runtime = {
                    connect: () => {},
                    id: ''
                };
            }

            // Mask permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );

            // Mask plugins (return non-empty list)
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });

            // Mask languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['zh-CN', 'zh', 'en-US', 'en']
            });

            // Prevent CDP injection detection
            window.cdc_adoQpoasnfa76pfcZLmcfl_Array = true;
            window.cdc_adoQpoasnfa76pfcZLmcfl_Promise = true;
            window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol = true;
        """)
        logger.debug("Applied webdriver and automation feature hiding")


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
    """Masks and randomizes WebGL fingerprint."""

    async def apply(self, page: Page) -> None:
        # Randomize WebGL vendor and renderer
        vendors = ['Intel Inc.', 'NVIDIA Corporation', 'Apple Inc.', 'AMD']
        renderers = ['Intel Iris OpenGL Engine', 'NVIDIA GeForce GTX 1060', 'Apple M1', 'AMD Radeon Pro']
        vendor = random.choice(vendors)
        renderer = random.choice(renderers)

        await page.add_init_script("""
            const glParams = {
                37445: '""" + vendor + """',
                37446: '""" + renderer + """',
                37487: 'WebGL GLSL ES 3.00',
            };

            const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (glParams.hasOwnProperty(parameter)) {
                    return glParams[parameter];
                }
                return originalGetParameter.apply(this, arguments);
            };

            // Also mask WebGL2
            const originalGetParameter2 = WebGL2RenderingContext.prototype.getParameter;
            WebGL2RenderingContext.prototype.getParameter = function(parameter) {
                if (glParams.hasOwnProperty(parameter)) {
                    return glParams[parameter];
                }
                return originalGetParameter2.apply(this, arguments);
            }};
        """)
        logger.debug("Applied WebGL stealth with randomization")


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
    """Spoofs user agent and request headers."""

    def __init__(self):
        super().__init__()
        self.user_agent = config.browser.get(
            "user_agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    async def apply(self, page: Page) -> None:
        # Set additional navigator properties and request headers
        await page.add_init_script(f"""
            // Navigator properties
            Object.defineProperty(navigator, 'hardwareConcurrency', {{
                get: () => {random.randint(4, 8)}
            }});
            Object.defineProperty(navigator, 'deviceMemory', {{
                get: () => {random.choice([4, 8, 16])}
            }});
            Object.defineProperty(navigator, 'maxTouchPoints', {{
                get: () => 0
            }});

            // Platform
            Object.defineProperty(navigator, 'platform', {{
                get: () => 'MacIntel'
            }});

            // DoNotTrack
            Object.defineProperty(navigator, 'doNotTrack', {{
                get: () => '1'
            }});

            // Connection saveData
            if (navigator.connection) {{
                Object.defineProperty(navigator.connection, 'saveData', {{
                    get: () => false
                }});
            }}

            // Add sec-ch-ua headers via interceptor
            const originalFetch = window.fetch;
            window.fetch = async function(...args) {{
                const [resource, config] = args;
                const newConfig = config || {{}};
                newConfig.headers = newConfig.headers || {{}};

                // Add stealth headers
                newConfig.headers['sec-ch-ua'] = '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"';
                newConfig.headers['sec-ch-ua-mobile'] = '?0';
                newConfig.headers['sec-ch-ua-platform'] = '"macOS"';
                newConfig.headers['Sec-Fetch-Dest'] = 'document';
                newConfig.headers['Sec-Fetch-Mode'] = 'navigate';
                newConfig.headers['Sec-Fetch-Site'] = 'none';
                newConfig.headers['Sec-Fetch-User'] = '?1';
                newConfig.headers['Upgrade-Insecure-Requests'] = '1';

                return originalFetch.apply(this, [resource, newConfig]);
            }};

            const originalXHROpen = XMLHttpRequest.prototype.open;
            XMLHttpRequest.prototype.open = function(method, url, ...rest) {{
                this.setRequestHeader('sec-ch-ua', '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"');
                this.setRequestHeader('sec-ch-ua-mobile', '?0');
                this.setRequestHeader('sec-ch-ua-platform', '"macOS"');
                this.setRequestHeader('Sec-Fetch-Dest', 'document');
                this.setRequestHeader('Sec-Fetch-Mode', 'navigate');
                this.setRequestHeader('Sec-Fetch-Site', 'none');
                this.setRequestHeader('Sec-Fetch-User', '?1');
                this.setRequestHeader('Upgrade-Insecure-Requests', '1');
                return originalXHROpen.apply(this, [method, url, ...rest]);
            }};
        """)
        logger.debug("Applied user agent and request header spoofing")


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
