"""Bookmark tool - quick access to preset website URLs."""

import json
import os
from typing import Any

from superbot.agent.tools.base import Tool, tool_error
from superbot.agent.tools.travel.logger import get_logger

logger = get_logger(__name__)

from superbot.config.loader import get_config_path

CONFIG_PATH = get_config_path()


class BookmarkTool(Tool):
    """Quick access to preset website URLs.

    Returns the URL for known bookmarks. If not a known bookmark,
    suggests using web_search. Supports adding new bookmarks.
    """

    name = "bookmark"
    description = "Quickly get URL for preset websites. Use when user wants to visit a known site. Can also add new bookmarks."
    parameters = {
        "type": "object",
        "properties": {
            "site": {
                "type": "string",
                "description": "Site name to look up, add, or delete"
            },
            "url": {
                "type": "string",
                "description": "URL to add as bookmark (optional). If provided, adds the bookmark to config."
            },
            "action": {
                "type": "string",
                "enum": ["get", "add", "delete", "list"],
                "description": "Action: get (default), add, delete, or list bookmarks"
            }
        },
        "required": ["site"]
    }

    def __init__(self, bookmarks: dict[str, str] = None):
        self.bookmarks = bookmarks or {}

    async def execute(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        site = kwargs.get("site", content)
        url = kwargs.get("url")
        action = kwargs.get("action", "get")

        # Special action: list all bookmarks
        if action == "list":
            return json.dumps({
                "content": json.dumps({
                    "type": "bookmark_list",
                    "bookmarks": self.bookmarks
                }, ensure_ascii=False),
                "media": []
            })

        if not site:
            return tool_error("missing_site", "Site name is required")

        site_lower = site.lower().strip()

        # Delete bookmark
        if action == "delete":
            return await self._delete_bookmark(site, site_lower)

        # Add bookmark
        if action == "add" or url:
            return await self._add_bookmark(site, url)

        # Get bookmark (default)
        if site_lower in self.bookmarks:
            found_url = self.bookmarks[site_lower]
            return json.dumps({
                "content": json.dumps({
                    "site": site,
                    "url": found_url,
                    "type": "bookmark_found"
                }, ensure_ascii=False),
                "media": []
            })

        # Not a known bookmark
        return json.dumps({
            "content": json.dumps({
                "site": site,
                "type": "bookmark_not_found",
                "suggestion": "Not a known bookmark. Use action='add' with url to add it.",
                "known_bookmarks": list(self.bookmarks.keys())
            }, ensure_ascii=False),
            "media": []
        })

    async def _add_bookmark(self, site: str, url: str) -> str:
        """Add a new bookmark to config."""
        try:
            # Read current config
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)

            # Ensure bookmark section exists
            if "tools" not in config:
                config["tools"] = {}
            if "bookmark" not in config["tools"]:
                config["tools"]["bookmark"] = {"bookmarks": {}}

            # Add the new bookmark (use lowercase key for consistency)
            site_lower = site.lower().strip()
            config["tools"]["bookmark"]["bookmarks"][site_lower] = url

            # Update in-memory cache
            self.bookmarks[site_lower] = url

            # Write back
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info("Added bookmark: " + site + " -> " + url)

            return json.dumps({
                "content": json.dumps({
                    "site": site,
                    "url": url,
                    "type": "bookmark_added",
                    "message": f"Bookmark '{site}' added successfully"
                }, ensure_ascii=False),
                "media": []
            })

        except Exception as e:
            import traceback
            logger.error("Failed to add bookmark: " + str(e))
            return tool_error("add_bookmark_failed", str(e), trace=traceback.format_exc())

    async def _delete_bookmark(self, site: str, site_lower: str) -> str:
        """Delete a bookmark from config."""
        try:
            # Check if bookmark exists in current instance
            if site_lower not in self.bookmarks:
                return json.dumps({
                    "content": json.dumps({
                        "site": site,
                        "type": "bookmark_not_found",
                        "message": f"Bookmark '{site}' not found"
                    }, ensure_ascii=False),
                    "media": []
                })

            # Read current config
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)

            # Check if bookmark section exists
            if "tools" not in config or "bookmark" not in config.get("tools", {}):
                return json.dumps({
                    "content": json.dumps({
                        "site": site,
                        "type": "bookmark_not_found",
                        "message": f"Bookmark '{site}' not found in config"
                    }, ensure_ascii=False),
                    "media": []
                })

            # Delete the bookmark
            bookmarks = config["tools"]["bookmark"].get("bookmarks", {})
            if site_lower in bookmarks:
                del bookmarks[site_lower]
                config["tools"]["bookmark"]["bookmarks"] = bookmarks

                # Update in-memory cache
                del self.bookmarks[site_lower]

                # Write back
                with open(CONFIG_PATH, 'w') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

                logger.info("Deleted bookmark: " + site)

                return json.dumps({
                    "content": json.dumps({
                        "site": site,
                        "type": "bookmark_deleted",
                        "message": f"Bookmark '{site}' deleted successfully"
                    }, ensure_ascii=False),
                    "media": []
                })

            return json.dumps({
                "content": json.dumps({
                    "site": site,
                    "type": "bookmark_not_found",
                    "message": f"Bookmark '{site}' not found"
                }, ensure_ascii=False),
                "media": []
            })

        except Exception as e:
            import traceback
            logger.error("Failed to delete bookmark: " + str(e))
            return tool_error("delete_bookmark_failed", str(e), trace=traceback.format_exc())
