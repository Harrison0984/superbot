"""Claude channel for handling @claude messages via MCP."""

import re
from typing import Any

from loguru import logger

from superbot.bus.events import OutboundMessage
from superbot.bus.queue import MessageBus
from superbot.config.schema import FeishuConfig


class ClaudeChannel:
    """Channel for handling @claude messages via MCP."""

    name: str = "claude"

    def __init__(
        self,
        config: Any,
        channels: dict[str, Any],
        mcp_client: Any,
        bus: MessageBus,
    ):
        """Initialize Claude channel."""
        self.config = config
        self.channels = channels
        self.mcp_client = mcp_client
        self.bus = bus

    @staticmethod
    def detect_claude_request(content: str) -> str | None:
        """Detect @claude in message and extract prompt.

        Args:
            content: Message content

        Returns:
            Extracted prompt if @claude detected, None otherwise
        """
        # Match @claude followed by content
        match = re.search(r"@claude\s*(.+)", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    async def send(self, msg: OutboundMessage) -> None:
        """Handle Claude response - send to original channel.

        Args:
            msg: Outbound message to forward
        """
        original_channel = self.channels.get(msg.metadata.get("original_channel", "cli"))
        if original_channel:
            await original_channel.send(msg)
            logger.debug("Forwarded Claude response to {}", msg.metadata.get("original_channel"))
        else:
            logger.warning("Original channel not found: {}", msg.metadata.get("original_channel"))

    async def preprocess_content(
        self,
        content: str,
        media: list[str] | None = None,
        feishu_config: FeishuConfig | None = None,
    ) -> str:
        """Preprocess content - extract links/images/docs/feishu docs on demand.

        Args:
            content: Original message content
            media: List of media file paths
            feishu_config: Feishu configuration for document processing

        Returns:
            Processed content with link/image/doc content embedded
        """
        processed = content

        # Check for Feishu document links
        feishu_doc_links = re.findall(r"https?://[^\s]*feishu\.cn[^\s]*docx?[^\s]*", content)
        if feishu_doc_links and feishu_config and feishu_config.enabled:
            processed = await self._process_feishu_docs(processed, feishu_doc_links, feishu_config)

        # Check for HTTP links
        links = re.findall(r"https?://[^\s\)]+", content)

        if links:
            # Import here to avoid circular imports
            from superbot.agent.tools.web import WebFetchTool
            from superbot.config.schema import WebToolsConfig

            web_tool = WebFetchTool(WebToolsConfig())
            for link in links:
                if link.startswith("http"):
                    try:
                        result = await web_tool.execute(url=link)
                        if result and not result.startswith("Error"):
                            processed += f"\n\n--- Content from {link} ---\n{result[:5000]}"
                    except Exception as e:
                        logger.warning("Failed to fetch {}: {}", link, e)

        # Check for images in media
        if media:
            image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
            for file_path in media:
                if file_path.lower().endswith(image_extensions):
                    # For now, just append the file path
                    processed += f"\n\n[Image: {file_path}]"

        # Check for documents in media
        if media:
            doc_extensions = (".pdf", ".docx", ".xlsx", ".txt", ".md")
            for file_path in media:
                if any(file_path.lower().endswith(ext) for ext in doc_extensions):
                    processed += f"\n\n[Document: {file_path}]"

        return processed

    async def _process_feishu_docs(
        self,
        content: str,
        doc_links: list[str],
        feishu_config: FeishuConfig,
    ) -> str:
        """Process Feishu document links."""
        try:
            from superbot.agent.tools.feishu_doc import FeishuDocTool

            feishu_tool = FeishuDocTool(feishu_config)

            for link in doc_links:
                # Extract document token from URL
                # Format: https://xxx.feishu.cn/docx/xxx 或 https://xxx.feishu.cn/docs/xxx
                match = re.search(r"/(docx?|docs)/([A-Za-z0-9]+)", link)
                if match:
                    token = match.group(2)
                    logger.info("Reading Feishu doc: {}", token)
                    try:
                        result = await feishu_tool.execute(operation="read_document", token=token)
                        if result and not result.startswith("Error"):
                            content += f"\n\n--- Content from Feishu doc {link} ---\n{result[:10000]}"
                    except Exception as e:
                        logger.warning("Failed to read Feishu doc {}: {}", link, e)

        except Exception as e:
            logger.warning("Failed to process Feishu docs: {}", e)

        return content
