"""Browser management for Ocular AI with stealth and XVFB support."""

import asyncio
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from browser_use import Agent, Browser, Controller
from playwright.async_api import async_playwright, Page, Playwright, BrowserContext

from superbot.agent.tools.travel.config import config
# from src.core.proxy import proxy_manager
from superbot.agent.tools.travel.logger import get_logger

logger = get_logger(__name__)


class StealthBrowser:
    """Browser wrapper with stealth capabilities and XVFB support."""

    def __init__(self):
        self.browser_config = config.browser
        self.stealth_config = self.browser_config.get("stealth", {})

        # Browser settings
        self.headless = self.browser_config.get("headless", False)
        self.xvfb = self.browser_config.get("xvfb", True)

        # Viewport
        viewport = self.browser_config.get("viewport", {})
        self.viewport_width = viewport.get("width", 1920)
        self.viewport_height = viewport.get("height", 1080)

        # User data directory for session persistence
        self.user_data_dir = self.browser_config.get("user_data_dir", "./data/sessions")

        # Internal state
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

        # Create user data directory
        os.makedirs(self.user_data_dir, exist_ok=True)

    def _should_use_xvfb(self) -> bool:
        """Check if XVFB should be used."""
        if not self.xvfb:
            return False

        # Check if running in headless mode or if DISPLAY is not set
        if self.headless:
            return True

        # Check if display is available
        if os.environ.get("DISPLAY"):
            return False

        return True

    def _get_stealth_options(self) -> Dict[str, Any]:
        """Get stealth options for the browser context (valid Playwright options only)."""
        # Only return options that are valid for new_context()
        options = {
            "locale": self.stealth_config.get("languages", ["zh-CN", "zh"])[0],
            "timezone_id": "Asia/Shanghai",
        }

        return options

    def _get_user_agent(self) -> str:
        """Get User-Agent string, customized based on proxy location."""
        # In a real implementation, you'd get the proxy location and customize
        # For now, use the configured user agent
        return self.browser_config.get(
            "user_agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    async def initialize(self) -> None:
        """Initialize the browser with stealth settings."""
        logger.info("Initializing stealth browser...")

        # Determine if we should use xvfb
        use_xvfb = self._should_use_xvfb()
        if use_xvfb:
            logger.info("Using XVFB for headful mode simulation")

        # Start playwright
        self.playwright = await async_playwright().start()

        # Get stealth options
        stealth_options = self._get_stealth_options()

        # Configure browser args for stealth
        browser_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            # Disable webdriver flag
            "--disable-dev-shm-usage",
            "--disable-accelerated-2d-canvas",
            "--no-first-run",
            "--no-zygote",
            "--disable-gpu",
        ]

        # User agent
        user_agent = self._get_user_agent()

        # Proxy configuration
        # proxy_dict = proxy_manager.get_proxy_dict()
        proxy_dict = None

        # Launch browser with persistent context to save login session
        try:
            # Use launch_persistent_context to maintain login session
            self.context = await self.playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=self.headless,
                args=browser_args,
                proxy=proxy_dict,
                slow_mo=0,
                channel="chrome",
                viewport={"width": self.viewport_width, "height": self.viewport_height},
                user_agent=user_agent,
                permissions=["geolocation", "notifications"],
                ignore_https_errors=True,
                **stealth_options
            )

            # Get browser from context
            self.browser = self.context.browser

            # Try to load existing session if available
            if os.path.exists(self.user_data_dir):
                try:
                    storage_state = os.path.join(self.user_data_dir, "storage_state.json")
                    if os.path.exists(storage_state):
                        logger.info(f"Loading existing session from {storage_state}")
                        # Note: Playwright doesn't directly support loading storage state this way
                        # But keeping the user_data_dir means cookies/persisted state will be maintained
                except Exception as e:
                    logger.warning(f"Could not load existing session: {e}")

            # Apply additional stealth measures
            await self._apply_stealth(self.context)

            logger.info("Browser initialized successfully with stealth settings")

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise

    async def _apply_stealth(self, context: BrowserContext) -> None:
        """Apply additional stealth measures to the context."""
        # Execute stealth scripts in page
        page = await context.new_page()

        # Remove webdriver property
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        # Randomize canvas fingerprint
        if self.stealth_config.get("canvas_fingerprint_randomize", True):
            await page.add_init_script("""
                const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
                HTMLCanvasElement.prototype.toDataURL = function(type) {
                    const result = originalToDataURL.apply(this, arguments);
                    // Add slight noise to canvas data
                    return result;
                };
            """)

        # Randomize audio context
        await page.add_init_script("""
            const originalGetChannelData = AudioContext.prototype.getChannelData;
            AudioContext.prototype.getChannelData = function(channelData) {
                const result = originalGetChannelData.apply(this, arguments);
                // Add slight noise to audio
                for (let i = 0; i < result.length; i++) {
                    result[i] += (Math.random() - 0.5) * 0.0001;
                }
                return result;
            };
        """)

        # WebGL stealth
        await page.add_init_script("""
            const gl = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                // Return fake WebGL info
                if (parameter === 37445) {
                    return 'Intel Inc.';
                }
                if (parameter === 37446) {
                    return 'Intel Iris OpenGL Engine';
                }
                return gl.apply(this, arguments);
            };
        """)

        await page.close()

    def get_context(self) -> Optional[BrowserContext]:
        """Get the browser context."""
        return self.context

    async def new_page(self) -> Optional[Page]:
        """Create a new page in the context."""
        if not self.context:
            raise RuntimeError("Browser not initialized")

        return await self.context.new_page()

    async def close(self) -> None:
        """Close the browser and cleanup resources."""
        logger.info("Closing browser...")

        if self.context:
            await self.context.close()

        if self.browser:
            await self.browser.close()

        if self.playwright:
            await self.playwright.stop()

        logger.info("Browser closed")

    async def __aenter__(self) -> "StealthBrowser":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


class OcularAgent:
    """Browser Use agent wrapper for Ocular AI."""

    def __init__(self, browser: StealthBrowser):
        self.browser = browser
        self.controller = Controller()

        # Agent settings
        self.api_config = config.api

    async def run(
        self,
        task: str,
        max_steps: int = 10
    ) -> Dict[str, Any]:
        """Run a task using the browser agent.

        Args:
            task: The task description for the agent
            max_steps: Maximum number of steps to execute

        Returns:
            Result dictionary with final state and extracted data
        """
        if not self.browser.context:
            raise RuntimeError("Browser not initialized")

        # Create agent
        agent = Agent(
            task=task,
            browser=self.browser.browser,
            context=self.browser.context,
            controller=self.controller,
            max_steps=max_steps,
        )

        # Run agent
        logger.info(f"Running agent task: {task}")
        result = await agent.run()

        logger.info("Agent task completed")
        return {
            "success": result.done,
            "steps": result.steps,
            "final_result": result.final_result,
        }

    async def get_page_content(self, page: Page) -> str:
        """Get filtered page content (only interactive elements).

        Args:
            page: Playwright page

        Returns:
            Filtered HTML content
        """
        content = await page.evaluate("""
            () => {
                // Get all buttons and inputs
                const elements = document.querySelectorAll('button, input, select, textarea, a, [role="button"], [tabindex]');
                const result = [];
                elements.forEach(el => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        const tag = el.tagName.toLowerCase();
                        const text = el.innerText || el.value || '';
                        const type = el.getAttribute('type') || '';
                        const placeholder = el.getAttribute('placeholder') || '';
                        const ariaLabel = el.getAttribute('aria-label') || '';
                        if (text || placeholder || ariaLabel) {
                            result.push(`${tag}: ${text || placeholder || ariaLabel} (${type})`);
                        }
                    }
                });
                return result.join('\\n');
            }
        """)
        return content


# Global browser instance
_stealth_browser: Optional[StealthBrowser] = None


async def get_browser() -> StealthBrowser:
    """Get or create the global stealth browser instance."""
    global _stealth_browser

    if _stealth_browser is None:
        _stealth_browser = StealthBrowser()
        await _stealth_browser.initialize()

    return _stealth_browser


async def close_browser() -> None:
    """Close the global browser instance."""
    global _stealth_browser

    if _stealth_browser:
        await _stealth_browser.close()
        _stealth_browser = None
