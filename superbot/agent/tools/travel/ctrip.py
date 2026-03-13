"""Ctrip flight monitoring module for Ocular AI.

Monitors flight prices from Ctrip/Trip.com using browser automation.
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Page

from superbot.agent.tools.travel.config import config
from superbot.agent.tools.travel.logger import get_logger
from superbot.agent.tools.travel.humanize import human_behavior
from superbot.agent.tools.travel.stealth import stealth_manager

logger = get_logger(__name__)


class CtripMonitor:
    """Monitor flight prices from Ctrip/Trip.com."""

    def __init__(self):
        ctrip_config = config.get("ctrip", {})
        self.enabled = ctrip_config.get("enabled", False)
        self.watch_interval = ctrip_config.get("watch_interval", 3600)

        # Default routes
        self.routes = [("SHA", "YCU")]  # Shanghai -> Yuncheng

        # Load configured routes
        if "routes" in ctrip_config:
            self.routes = [(r.get("from"), r.get("to")) for r in ctrip_config["routes"]]

    def _format_date(self, date_str: str) -> str:
        """Format date for Ctrip URL (remove leading zeros).

        Args:
            date_str: Date string in format YYYY-MM-DD

        Returns:
            Formatted date string (YYYY-M-D)
        """
        # Parse the date
        from datetime import datetime
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            # Format without leading zeros: YYYY-M-D
            return f"{dt.year}-{dt.month}-{dt.day}"
        except:
            return date_str

    async def _check_login_popup(self, page: Page) -> bool:
        """Check if a login popup is displayed (not just the login link in header).

        Args:
            page: Playwright page

        Returns:
            True if login popup detected, False otherwise
        """
        try:
            # Check for actual login modal/popup (not just header link)
            popup_selectors = [
                # Login modal/dialog
                '[class*="login-modal"]',
                '[class*="login-dialog"]',
                '.login-dialog',
                '#login-modal',
                # QR code container (popup)
                '[class*="qrcode"][class*="popup"]',
                '[class*="qr-code"][class*="modal"]',
                # Login overlay
                '[class*="login-overlay"]',
                '[class*="login-mask"]',
                # Explicit popup containers
                '.popup-login',
                '[id*="login-popup"]',
            ]

            for selector in popup_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        is_visible = await element.is_visible()
                        if is_visible:
                            logger.info(f"Login popup detected: {selector}")
                            return True
                except:
                    continue

            # Check if user is already logged in (look for username/user info)
            logged_in_indicators = [
                '[class*="user-name"]',
                '[class*="user-info"]',
                '[class*="username"]',
                '[id*="user-name"]',
                '[class*="login-success"]',
            ]

            for selector in logged_in_indicators:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        is_visible = await element.is_visible()
                        if is_visible:
                            logger.info("User appears to be logged in")
                            return False  # Logged in, no popup
                except:
                    continue

            # Check for QR code scan prompt specifically (indicates login popup)
            qr_prompts = await page.query_selector_all('text=扫码登录, text=请扫码, text=手机号登录, text=账号密码登录')
            if qr_prompts:
                for prompt in qr_prompts:
                    try:
                        if await prompt.is_visible():
                            logger.info("Login QR code prompt detected")
                            return True
                    except:
                        continue

            return False

        except Exception as e:
            logger.error(f"Error checking login popup: {e}")
            return False

    async def _wait_for_login(self, page: Page, timeout: int = 60) -> bool:
        """Wait for user to manually login.

        Args:
            page: Playwright page
            timeout: Timeout in seconds

        Returns:
            True if login completed, False if timeout
        """
        logger.info("=" * 50)
        logger.info("⚠️  Detected login popup, please login manually...")
        logger.info("⚠️  Please login manually in the browser")
        logger.info("=" * 50)

        # Screenshot QR code area
        import os
        from pathlib import Path
        screenshot_dir = Path.home() / ".superbot" / "sessions"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = screenshot_dir / "ctrip_login.png"

        try:
            # First try clicking "scan login" button to show QR code
            scan_login = await page.query_selector('a:has-text("扫码登录"), a:has-text("扫码")')
            if scan_login:
                logger.info("Clicking scan login button...")
                await scan_login.click()
                await asyncio.sleep(5)  # Wait for QR code generation

            # Screenshot QR code area
            qr_element = await page.query_selector('.lg_ercode, [class*="ercode"]')
            if qr_element:
                await qr_element.screenshot(path=str(screenshot_path))
                logger.info(f"QR code screenshot saved to: {screenshot_path}")
            else:
                # If QR code area not found, screenshot entire page
                await page.screenshot(path=str(screenshot_path), full_page=True)
                logger.info(f"Login page screenshot saved to: {screenshot_path}")

            logger.info(f"Please scan QR code to login... (screenshot path: {screenshot_path})")
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check if login popup is gone
            has_popup = await self._check_login_popup(page)
            if not has_popup:
                # Also verify by checking for user info
                try:
                    user_elements = await page.query_selector_all('[class*="user-name"], [class*="username"], [class*="user-info"]')
                    for el in user_elements:
                        if await el.is_visible():
                            logger.info("Login successful, detected user info")
                            return True
                except:
                    pass

                # Also check if popup is gone
                logger.info("Login popup closed, continuing...")
                return True

            await asyncio.sleep(3)

        logger.warning("Login wait timeout")
        return False

    async def search_flight(
        self,
        page: Page,
        from_airport: str,
        to_airport: str,
        date: str
    ) -> Optional[Dict[str, Any]]:
        """Search for flights between two airports using Ctrip.

        Args:
            page: Playwright page
            from_airport: Departure airport code (e.g., "SHA")
            to_airport: Arrival airport code (e.g., "YCU")
            date: Departure date (e.g., "2026-02-24")

        Returns:
            Flight search results or None
        """
        try:
            # Format date for Ctrip URL (remove leading zeros)
            formatted_date = self._format_date(date)

            # Go directly to flight search results page
            search_url = f"https://flights.ctrip.com/online/list/oneway-{from_airport.lower()}-{to_airport.lower()}?depdate={formatted_date}"
            logger.info(f"Navigating to: {search_url}")

            await page.goto(search_url, wait_until="domcontentloaded")

            # Wait for JavaScript to fully load
            await asyncio.sleep(8)

            # Check for login popup
            has_login = await self._check_login_popup(page)
            if has_login:
                logger.warning("Login popup detected! Please login manually...")
                await self._wait_for_login(page, timeout=120)

            # Try to handle cookie popup if exists
            try:
                cookie_btn = await page.query_selector('button:has-text("接受"), button:has-text("同意"), .cookie-accept')
                if cookie_btn:
                    await cookie_btn.click()
                    await asyncio.sleep(1)
            except:
                pass

            logger.info("Ctrip search results loaded")

            # Extract prices
            prices = await self._extract_prices(page)

            logger.info(f"Found {len(prices)} flights for {from_airport} -> {to_airport}")

            return {
                "from": from_airport,
                "to": to_airport,
                "date": date,
                "prices": prices,
            }

        except Exception as e:
            logger.error(f"Failed to search flights: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _extract_prices(self, page: Page) -> List[Dict[str, Any]]:
        """Extract flight details from page including price, airline, times, duration, baggage.

        Args:
            page: Playwright page

        Returns:
            List of flight data with full details
        """
        prices = []

        try:
            # Extract using JavaScript - target flight list specifically
            flight_data = await page.evaluate("""
                () => {
                    const results = [];

                    // Find the flight list container
                    const flightList = document.querySelector('.flight-list, #flightList, [class*="flight-list"]');

                    if (!flightList) {
                        console.log('No flight list found');
                        return results;
                    }

                    // Find all flight items
                    const flightItems = flightList.querySelectorAll('[class*="flight-item"], [class*="flight-item"], .flight-list-item, li[class*="flight"]');

                    console.log(`Found ${flightItems.length} flight items`);

                    flightItems.forEach((item, index) => {
                        try {
                            // Extract flight number FIRST (before cleaning airline)
                            let flightNumber = null;
                            // Match patterns like: 9C7285, XJ761, TG663, MU5834
                            // Can start with letter OR digit followed by letter (to match 9C, 8L, etc)
                            const fnMatch = item.innerText.match(/((?:[A-Z]\d|\d[A-Z])[A-Z0-9]\d{3,4})/);
                            if (fnMatch) {
                                flightNumber = fnMatch[1];
                            } else {
                                // Fallback: try standard airline code pattern (e.g., MU, CA, CZ)
                                const stdMatch = item.innerText.match(/([A-Z]{2}\d{3,4})/);
                                if (stdMatch) {
                                    flightNumber = stdMatch[1];
                                }
                            }

                            // Extract airline name
                            let airline = null;
                            const airlineEl = item.querySelector('[class*="airline"], [class*="airline-name"], [class*="航空公司"], [class*="flight-airline"]');
                            if (airlineEl) {
                                airline = airlineEl.innerText.trim();
                            }
                            // Alternative: look for airline in text (first occurrence)
                            if (!airline) {
                                const text = item.innerText;
                                const airlineMatch = text.match(/(上海航空|春秋航空|东方航空|海南航空|南方航空|国航|泰国狮航|泰国亚航长途|泰国亚航|泰国国际航空|厦门航空|深圳航空|四川航空|山东航空|吉祥航空|首都航空|长龙航空|祥鹏航空|西部航空|奥凯航空|西藏航空|河北航空|江西航空|湖南航空|阿拉斯加航空|达美航空|联合航空|美国航空|西南航空|捷蓝航空|精神航空|边疆航空|夏威夷航空|香港航空|国泰航空|香港快运|澳门航空|新加坡航空|胜安航空|酷航|捷星亚洲|宿雾太平洋|菲律宾航空|亚航|马印航空|文莱航空)/);
                                if (airlineMatch) {
                                    airline = airlineMatch[1];
                                }
                            }
                            // Clean up airline name - use known airlines list to extract just the name
                            if (airline) {
                                const knownAirlines = ['上海航空', '春秋航空', '东方航空', '海南航空', '南方航空', '国航', '泰国狮航', '泰国亚航长途', '泰国亚航', '泰国国际航空', '厦门航空', '深圳航空', '四川航空', '山东航空', '吉祥航空', '首都航空', '长龙航空'];
                                for (const known of knownAirlines) {
                                    if (airline.includes(known)) {
                                        airline = known;
                                        break;
                                    }
                                }
                            }

                            // Extract times
                            let depTime = null;
                            let arrTime = null;
                            const timeEls = item.querySelectorAll('[class*="time"], [class*="dep-time"], [class*="arr-time"]');
                            const timeMatches = item.innerText.matchAll(/(\d{1,2}:\d{2})/g);
                            const times = [...timeMatches].map(m => m[1]);
                            if (times.length >= 2) {
                                depTime = times[0];
                                arrTime = times[1];
                            }

                            // Extract duration
                            let duration = null;
                            const durationMatch = item.innerText.match(/(\d+)小时(\d+)?分?/);
                            if (durationMatch) {
                                duration = durationMatch[0];
                            }

                            // Extract price
                            let price = null;
                            const priceEl = item.querySelector('[class*="price"], [class*="price-text"]');
                            if (priceEl) {
                                const priceText = priceEl.innerText.trim();
                                const priceMatch = priceText.match(/¥(\d+)/);
                                if (priceMatch) {
                                    price = parseInt(priceMatch[1]);
                                }
                            }
                            // Fallback: search for price in entire item text
                            if (!price) {
                                const allPriceMatches = item.innerText.matchAll(/¥(\d{3,4})/g);
                                const priceValues = [...allPriceMatches].map(m => parseInt(m[1])).filter(p => p >= 300 && p <= 10000);
                                if (priceValues.length > 0) {
                                    price = Math.min(...priceValues);
                                }
                            }

                            // Extract baggage info
                            let baggage = null;
                            const baggageKeywords = [
                                '免费20kg托运', '免费30kg托运', '免费40kg托运',
                                '20kg托运', '30kg托运', '40kg托运',
                                '托运行李', '免费托运', '免费行李',
                                '手提7kg', '手提5kg', '手提10kg',
                                '手提行李', '免费手提', '不含行李', '有行李', '行李额',
                                '免费行李额'
                            ];
                            for (const keyword of baggageKeywords) {
                                if (item.innerText.includes(keyword)) {
                                    baggage = keyword;
                                    break;
                                }
                            }
                            // Fallback: check if there's any baggage-related class
                            if (!baggage) {
                                const baggageEl = item.querySelector('[class*="baggage"], [class*="行李"], [class*="luggage"]');
                                if (baggageEl) {
                                    baggage = baggageEl.innerText.trim();
                                }
                            }
                            // Infer baggage policy based on airline type (since website doesn't show it directly)
                            if (!baggage && airline) {
                                // Budget airlines (廉航) typically don't include free baggage
                                const budgetAirlines = ['春秋航空', '吉祥航空', '西部航空', '奥凯航空', '祥鹏航空', '长龙航空', '湖南航空', '江西航空', '河北航空'];
                                const budgetAirlinesEn = ['AirAsia', 'Thai Lion', 'Thai AirAsia', 'Jetstar', 'Scoot', 'Cebu Pacific', 'Palawan', '亚航长途'];

                                const isBudget = budgetAirlines.some(a => airline.includes(a)) ||
                                                 budgetAirlinesEn.some(a => airline.includes(a));

                                if (isBudget) {
                                    baggage = '无免费托运行李';
                                } else {
                                    baggage = '免费托运行李';
                                }
                            }

                            // Only add if we have price
                            if (price && price >= 300) {
                                results.push({
                                    price: price,
                                    airline: airline,
                                    flightNumber: flightNumber,
                                    depTime: depTime,
                                    arrTime: arrTime,
                                    duration: duration,
                                    baggage: baggage,
                                    isLowest: false
                                });
                            }
                        } catch (e) {
                            console.log(`Error parsing flight item ${index}: ${e}`);
                        }
                    });

                    // If no flight items found, try extracting from page directly
                    if (results.length === 0) {
                        const allElements = document.querySelectorAll('div, span, p, li');
                        let pricesFound = [];
                        let airlinesFound = [];
                        let timesFound = [];

                        allElements.forEach(el => {
                            const text = el.innerText.trim();

                            // Price: ¥710, ¥1064 (only 3-4 digit prices)
                            const priceMatch = text.match(/^¥(\d{3,4})$/);
                            if (priceMatch) {
                                const price = parseInt(priceMatch[1]);
                                if (price >= 300 && price <= 10000) {
                                    pricesFound.push(price);
                                }
                            }

                            // Airlines
                            const airlineMatch = text.match(/(上海航空|春秋航空|东方航空|海南航空|南方航空|国航|泰国狮航|泰国亚航|泰国国际航空|厦门航空|深圳航空|四川航空|吉祥航空)/);
                            if (airlineMatch && text.length < 20) {
                                airlinesFound.push(airlineMatch[1]);
                            }

                            // Times
                            const timeMatch = text.match(/^(\d{1,2}:\d{2})$/);
                            if (timeMatch) {
                                timesFound.push(timeMatch[1]);
                            }
                        });

                        // Deduplicate
                        const uniquePrices = [...new Set(pricesFound)].sort((a, b) => a - b);
                        const uniqueAirlines = [...new Set(airlinesFound)];
                        const uniqueTimes = [...new Set(timesFound)].sort();

                        // Create entries for each price
                        uniquePrices.forEach((price, idx) => {
                            results.push({
                                price: price,
                                airline: uniqueAirlines[idx % uniqueAirlines.length] || null,
                                flightNumber: null,
                                depTime: uniqueTimes[idx * 2] || null,
                                arrTime: uniqueTimes[idx * 2 + 1] || null,
                                duration: null,
                                baggage: null,
                                isLowest: idx === 0
                            });
                        });
                    }

                    return results;
                }
            """)

            prices = flight_data if flight_data else []

            # Mark lowest price
            if prices:
                min_price = min(p.get("price", float('inf')) for p in prices if p.get("price"))
                for p in prices:
                    if p.get("price") == min_price:
                        p["isLowest"] = True

            logger.info(f"Extracted {len(prices)} flights with details")

            # Log sample data for debugging
            if prices:
                logger.info(f"Sample flight: {prices[0]}")

        except Exception as e:
            logger.error(f"Failed to extract prices: {e}")
            import traceback
            traceback.print_exc()

        return prices

    async def find_lowest_price(
        self,
        page: Page,
        from_airport: str,
        to_airport: str,
        start_date: str,
        days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """Find lowest price within date range.

        Args:
            page: Playwright page
            from_airport: Departure airport
            to_airport: Arrival airport
            start_date: Start date for search
            days: Number of days to search

        Returns:
            Lowest price data
        """
        lowest = None

        # Parse start date
        start = datetime.strptime(start_date, "%Y-%m-%d")

        for i in range(days):
            search_date = start + timedelta(days=i)
            date_str = search_date.strftime("%Y-%m-%d")

            logger.info(f"Searching {from_airport} -> {to_airport} for {date_str}")

            result = await self.search_flight(page, from_airport, to_airport, date_str)

            if result and result.get("prices"):
                for price_data in result["prices"]:
                    # Track lowest
                    if price_data.get("price"):
                        if lowest is None or price_data.get("price", 0) < lowest.get("price", 0):
                            lowest = {
                                "date": date_str,
                                **price_data
                            }

            # Human-like delay between searches
            import asyncio
            await asyncio.sleep(human_behavior.random_delay())

        return lowest

    async def monitor_routes(self, browser) -> List[Dict[str, Any]]:
        """Monitor all configured routes.

        Args:
            browser: Browser instance

        Returns:
            List of monitoring results
        """
        results = []

        for from_airport, to_airport in self.routes:
            logger.info(f"Monitoring route: {from_airport} -> {to_airport}")

            page = await browser.new_page()

            try:
                # Find lowest price for next 7 days
                lowest = await self.find_lowest_price(
                    page,
                    from_airport,
                    to_airport,
                    datetime.now().strftime("%Y-%m-%d"),
                    days=7
                )

                results.append({
                    "route": f"{from_airport}-{to_airport}",
                    "lowest_price": lowest
                })

            finally:
                await page.close()

        return results


# Global instance
ctrip_monitor = CtripMonitor()
