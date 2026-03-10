"""Feishu document and spreadsheet tool for reading and writing documents."""

import json
from typing import Any

import requests
from loguru import logger

from superbot.agent.tools.base import Tool, tool_error
from superbot.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.docx.v1 import (
        CreateDocumentRequest,
        CreateDocumentRequestBody,
        GetDocumentRequest,
        RawContentDocumentRequest,
    )
    # Sheets API imports
    from lark_oapi.api.sheets.v3 import CreateSpreadsheetRequest, GetSpreadsheetRequest
    from lark_oapi.api.sheets.v3.model import SpreadsheetBuilder
    FEISHU_AVAILABLE = True
except ImportError as e:
    FEISHU_AVAILABLE = False
    lark = None
    SpreadsheetBuilder = None
    import sys
    print(f"Feishu SDK import warning: {e}", file=sys.stderr)


def _extract_text_content(blocks: list) -> str:
    """Extract text content from document blocks."""
    if not blocks:
        return ""

    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type", "")
        block_data = block.get(block_type, {})

        if block_type == "text":
            text = block_data.get("text", {})
            if isinstance(text, dict):
                content = text.get("content", "")
                if content:
                    parts.append(content)
            elif isinstance(text, str):
                parts.append(text)

        elif block_type == "paragraph":
            elements = block_data.get("elements", [])
            for el in elements:
                if isinstance(el, dict):
                    el_type = el.get("type", "")
                    if el_type == "text":
                        text = el.get("text", {})
                        if isinstance(text, dict):
                            content = text.get("content", "")
                            if content:
                                parts.append(content)
                        elif isinstance(text, str):
                            parts.append(text)
                    elif el_type == "link":
                        content = el.get("content", {})
                        if isinstance(content, dict):
                            text = content.get("content", "") or content.get("text", "")
                            if text:
                                parts.append(text)
                        elif isinstance(content, str):
                            parts.append(content)

        elif block_type == "heading1":
            elements = block_data.get("elements", [])
            for el in elements:
                if isinstance(el, dict) and el.get("type") == "text":
                    text = el.get("text", {})
                    if isinstance(text, dict):
                        content = text.get("content", "")
                        if content:
                            parts.append(f"# {content}")
                    elif isinstance(text, str):
                        parts.append(f"# {text}")

        elif block_type == "heading2":
            elements = block_data.get("elements", [])
            for el in elements:
                if isinstance(el, dict) and el.get("type") == "text":
                    text = el.get("text", {})
                    if isinstance(text, dict):
                        content = text.get("content", "")
                        if content:
                            parts.append(f"## {content}")
                    elif isinstance(text, str):
                        parts.append(f"## {text}")

        elif block_type == "heading3":
            elements = block_data.get("elements", [])
            for el in elements:
                if isinstance(el, dict) and el.get("type") == "text":
                    text = el.get("text", {})
                    if isinstance(text, dict):
                        content = text.get("content", "")
                        if content:
                            parts.append(f"### {content}")
                    elif isinstance(text, str):
                        parts.append(f"### {text}")

        elif block_type == "list":
            items = block_data.get("elements", [])
            for item in items:
                if isinstance(item, dict):
                    content = item.get("content", {})
                    if isinstance(content, dict):
                        text = content.get("text", {})
                        if isinstance(text, dict):
                            text_content = text.get("content", "")
                            if text_content:
                                parts.append(f"- {text_content}")
                        elif isinstance(text, str):
                            parts.append(f"- {text}")

        elif block_type == "code":
            text = block_data.get("text", {})
            language = block_data.get("language", "")
            if isinstance(text, dict):
                content = text.get("content", "")
                if content:
                    parts.append(f"```{language}\n{content}\n```")
            elif isinstance(text, str):
                parts.append(f"```{language}\n{text}\n```")

    return "\n".join(parts)


def _convert_markdown_to_blocks(content: str) -> list:
    """Convert markdown content to Feishu document blocks."""
    import re

    blocks = []
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Heading 1: # Title
        if line.startswith("# ") and not line.startswith("## "):
            blocks.append({
                "block_type": 3,  # Heading 1
                "heading1": {
                    "elements": [{
                        "type": "text_run",
                        "text_run": {"content": line[2:]}
                    }]
                }
            })
        # Heading 2: ## Title
        elif line.startswith("## ") and not line.startswith("### "):
            blocks.append({
                "block_type": 4,  # Heading 2
                "heading2": {
                    "elements": [{
                        "type": "text_run",
                        "text_run": {"content": line[3:]}
                    }]
                }
            })
        # Heading 3: ### Title
        elif line.startswith("### "):
            blocks.append({
                "block_type": 5,  # Heading 3
                "heading3": {
                    "elements": [{
                        "type": "text_run",
                        "text_run": {"content": line[4:]}
                    }]
                }
            })
        # Bullet list: - item (block_type=12)
        elif line.startswith("- "):
            # Collect all consecutive list items
            list_items = []
            while i < len(lines) and lines[i].startswith("- "):
                list_items.append(lines[i][2:])
                i += 1
            i -= 1  # Back up one line since outer loop will increment

            # Add each list item as a bullet block
            for item in list_items:
                elements = _parse_inline_markdown(item)
                blocks.append({
                    "block_type": 12,  # Bullet list
                    "bullet": {
                        "elements": elements
                    }
                })
        # Code block: ```language
        elif line.startswith("```"):
            # Collect code block content
            language = line[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            code_content = "\n".join(code_lines)

            blocks.append({
                "block_type": 14,  # Code block
                "code": {
                    "language": language,
                    "elements": [
                        {"type": "text_run", "text_run": {"content": code_content}}
                    ]
                }
            })
        # Horizontal rule: --- or *** or ___ (skip for now, API doesn't support it)
        elif line.strip() in ["---", "***", "___"]:
            # Skip divider for now
            pass
        # Empty line - skip
        elif not line.strip():
            pass
        # Regular paragraph
        else:
            # Parse inline markdown (bold, italic, code)
            elements = _parse_inline_markdown(line)
            if elements:
                blocks.append({
                    "block_type": 2,
                    "text": {
                        "elements": elements
                    }
                })

        i += 1

    return blocks


def _parse_inline_markdown(text: str) -> list:
    """Parse inline markdown and return Feishu text elements."""
    import re

    elements = []
    pos = 0

    # Patterns for inline formatting
    patterns = [
        (r'\*\*(.+?)\*\*', 'bold'),       # **bold**
        (r'\*(.+?)\*', 'italic'),          # *italic*
        (r'__(.+?)__', 'bold'),           # __bold__
        (r'_(.+?)_', 'italic'),           # _italic_
        (r'`(.+?)`', 'inline_code'),       # `code`
    ]

    while pos < len(text):
        earliest_match = None
        earliest_pos = len(text)
        earliest_pattern = None
        earliest_style = None

        for pattern, style in patterns:
            match = re.search(pattern, text[pos:])
            if match and match.start() + pos < earliest_pos:
                earliest_match = match
                earliest_pos = match.start() + pos
                earliest_pattern = pattern
                earliest_style = style

        if earliest_match is None:
            # No more matches, add remaining text and break
            if pos < len(text):
                elements.append({
                    "type": "text_run",
                    "text_run": {"content": text[pos:]}
                })
            break

        # Add text before the match
        if earliest_pos > pos:
            elements.append({
                "type": "text_run",
                "text_run": {"content": text[pos:earliest_pos]}
            })

        # Add the formatted text
        element = {
            "type": "text_run",
            "text_run": {"content": earliest_match.group(1)}
        }
        if earliest_style == "bold":
            element["text_run"]["text_element_style"] = {"bold": True}
        elif earliest_style == "italic":
            element["text_run"]["text_element_style"] = {"italic": True}
        elif earliest_style == "inline_code":
            element["text_run"]["text_element_style"] = {"inline_code": True}

        elements.append(element)
        pos = earliest_pos + len(earliest_match.group(0))

    return elements if elements else [{"type": "text_run", "text_run": {"content": text}}]


class FeishuDocTool(Tool):
    """
    Tool for reading and writing Feishu documents and spreadsheets.

    Requires:
    - Feishu app_id and app_secret in configuration
    - Document permissions enabled in Feishu app
    """

    def __init__(self, config: FeishuConfig | None = None):
        self._config = config
        self._client: Any = None

    def _ensure_client(self) -> Any:
        """Ensure the Feishu client is initialized."""
        if not FEISHU_AVAILABLE:
            raise RuntimeError("Feishu SDK not installed. Run: pip install lark-oapi")

        if self._config is None or not self._config.app_id or not self._config.app_secret:
            raise RuntimeError("Feishu app_id and app_secret not configured")

        if self._client is None:
            self._client = lark.Client.builder() \
                .app_id(self._config.app_id) \
                .app_secret(self._config.app_secret) \
                .log_level(lark.LogLevel.INFO) \
                .build()

        return self._client

    @property
    def name(self) -> str:
        return "feishu_doc"

    @property
    def description(self) -> str:
        return """Read and write Feishu documents and spreadsheets.

Usage tips:
- Use token from document URL (e.g., https://feishu.cn/document/TOKEN or https://feishu.cn/wiki/TOKEN)
- For update_document, use markdown format: # Heading, ## Subheading, **bold**, *italic*, - list, ```code```
- Document must be shared with the app before writing (share document to the app)
- Use wiki space documents for shared access"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "read_document",
                        "create_document",
                        "update_document",
                        "create_sheet",
                        "get_sheet_info",
                    ],
                    "description": "Operation: read_document (read content), create_document (create new), update_document (append content with markdown), create_sheet (create spreadsheet), get_sheet_info (get sheet details)"
                },
                "token": {
                    "type": "string",
                    "description": "Document or spreadsheet token from URL (e.g., 'U5X6w2iO...' from https://feishu.cn/wiki/U5X6w2iO...)"
                },
                "space_id": {
                    "type": "string",
                    "description": "Space ID for creating documents in a specific space"
                },
                "title": {
                    "type": "string",
                    "description": "Title for new document or spreadsheet"
                },
                "content": {
                    "type": "string",
                    "description": "Content using markdown: # Heading1, ## Heading2, **bold**, *italic*, - list item, ```language for code blocks"
                },
            },
            "required": ["operation"]
        }

    async def execute(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        """Execute the requested operation."""
        operation = kwargs.get("operation", "")
        token = kwargs.get("token")
        space_id = kwargs.get("space_id")
        title = kwargs.get("title")
        doc_content = kwargs.get("content")

        try:
            client = self._ensure_client()
        except RuntimeError as e:
            return tool_error("not_configured", str(e))

        try:
            if operation == "read_document":
                return await self._read_document(client, token)
            elif operation == "create_document":
                return await self._create_document(client, space_id, title, doc_content or content)
            elif operation == "update_document":
                return await self._update_document(client, token, doc_content or content)
            elif operation == "create_sheet":
                return await self._create_sheet(client, space_id, title)
            elif operation == "get_sheet_info":
                return await self._get_sheet_info(client, token)
            else:
                return tool_error("invalid_params", f"Unknown operation: {operation}")
        except Exception as e:
            logger.error("Feishu doc operation {} failed: {}", operation, e)
            return tool_error("operation_failed", f"Error executing {operation}: {str(e)}")

    async def _read_document(self, client: Any, token: str | None) -> str:
        """Read document content by token."""
        if not token:
            return tool_error("invalid_params", "token is required for read_document")

        try:
            # Get document metadata
            request = GetDocumentRequest.builder().document_id(token).build()
            response = client.docx.v1.document.get(request)

            if not response.success():
                return tool_error("api_error", f"Failed to get document: {response.code} - {response.msg}")

            doc_title = response.data.document.title if response.data else "Untitled"

            # Get document raw content
            request = RawContentDocumentRequest.builder().document_id(token).build()
            response = client.docx.v1.document.raw_content(request)

            if not response.success():
                return tool_error("api_error", f"Failed to get document content: {response.code} - {response.msg}")

            # Extract content from response
            content = response.data.content if response.data else ""

            result = {
                "title": doc_title,
                "token": token,
                "content": content or "[Empty document]",
            }
            return json.dumps({
                "content": json.dumps(result, ensure_ascii=False),
                "media": []
            })

        except Exception as e:
            logger.error("Error reading document {}: {}", token, e)
            return tool_error("operation_failed", f"Error reading document: {str(e)}")

    async def _create_document(
        self, client: Any, space_id: str | None, title: str | None, content: str | None
    ) -> str:
        """Create a new document."""
        if not title:
            return tool_error("invalid_params", "title is required for create_document")

        try:
            # Build document content from markdown
            block_content = []
            if content:
                lines = content.split("\n")
                for line in lines:
                    if line.startswith("# "):
                        block_content.append({
                            "type": "heading1",
                            "heading1": {
                                "elements": [{
                                    "type": "text",
                                    "text": {"content": line[2:]}
                                }]
                            }
                        })
                    elif line.startswith("## "):
                        block_content.append({
                            "type": "heading2",
                            "heading2": {
                                "elements": [{
                                    "type": "text",
                                    "text": {"content": line[3:]}
                                }]
                            }
                        })
                    elif line.startswith("### "):
                        block_content.append({
                            "type": "heading3",
                            "heading3": {
                                "elements": [{
                                    "type": "text",
                                    "text": {"content": line[4:]}
                                }]
                            }
                        })
                    elif line.startswith("- "):
                        block_content.append({
                            "type": "list",
                            "list": {
                                "elements": [{
                                    "type": "list_item",
                                    "content": {
                                        "type": "text",
                                        "text": {"content": line[2:]}
                                    }
                                }]
                            }
                        })
                    elif line.strip():
                        block_content.append({
                            "type": "paragraph",
                            "paragraph": {
                                "elements": [{
                                    "type": "text",
                                    "text": {"content": line}
                                }]
                            }
                        })

            request = CreateDocumentRequest.builder() \
                .request_body(
                    CreateDocumentRequestBody.builder()
                    .title(title)
                    .folder_token(space_id)
                    .build()
                ).build()

            response = client.docx.v1.document.create(request)

            if not response.success():
                return tool_error("api_error", f"Failed to create document: {response.code} - {response.msg}")

            new_token = response.data.document.document_id

            # If content is provided, add blocks to the document
            result = {
                "title": title,
                "token": new_token,
                "url": f"https://feishu.cn/document/{new_token}",
            }
            return json.dumps({
                "content": json.dumps(result, ensure_ascii=False),
                "media": []
            })

        except Exception as e:
            logger.error("Error creating document: {}", e)
            return tool_error("operation_failed", f"Error creating document: {str(e)}")

    async def _update_document(self, client: Any, token: str | None, content: str | None) -> str:
        """Update document content by appending text blocks."""
        if not token:
            return tool_error("invalid_params", "token is required for update_document")

        if not content:
            return tool_error("invalid_params", "content is required for update_document")

        try:
            # Get access token
            token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            token_resp = requests.post(token_url, json={
                "app_id": self._config.app_id,
                "app_secret": self._config.app_secret
            }, timeout=10)
            token_data = token_resp.json()
            access_token = token_data.get("tenant_access_token")

            if not access_token:
                return tool_error("api_error", "Failed to get access token")

            # Get document blocks to find the root block
            list_url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{token}/blocks"
            headers = {"Authorization": f"Bearer {access_token}"}
            list_resp = requests.get(list_url, headers=headers, timeout=10)
            list_data = list_resp.json()

            if not list_data.get("success") and list_data.get("code") != 0:
                return tool_error("api_error", f"Failed to list blocks: {list_data.get('msg')}")

            # Find the root block (parent_id = "")
            root_block_id = None
            if list_data.get("data") and list_data["data"].get("items"):
                for item in list_data["data"]["items"]:
                    if item.get("parent_id") == "":
                        root_block_id = item.get("block_id")
                        break

            if not root_block_id:
                root_block_id = token  # Use document ID as fallback

            # Convert markdown to Feishu blocks
            children = _convert_markdown_to_blocks(content)

            if not children:
                return tool_error("invalid_params", "No content to add")

            # Add blocks to document
            add_url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{token}/blocks/{root_block_id}/children"
            add_resp = requests.post(add_url, headers=headers, json={"children": children}, timeout=10)
            add_data = add_resp.json()

            if not add_data.get("success") and add_data.get("code") != 0:
                return tool_error("api_error", f"Failed to add content: {add_data.get('msg')}")

            result = {
                "token": token,
                "message": f"Successfully added {len(children)} line(s) to document",
            }
            return json.dumps({
                "content": json.dumps(result, ensure_ascii=False),
                "media": []
            })

        except Exception as e:
            logger.error("Error updating document {}: {}", token, e)
            return tool_error("operation_failed", f"Error updating document: {str(e)}")

    async def _create_sheet(
        self, client: Any, space_id: str | None, title: str | None
    ) -> str:
        """Create a new spreadsheet."""
        if not title:
            return tool_error("invalid_params", "title is required for create_sheet")

        if SpreadsheetBuilder is None:
            return tool_error("not_configured", "Spreadsheet API not available")

        try:
            spreadsheet = SpreadsheetBuilder() \
                .title(title) \
                .folder_token(space_id) \
                .build()

            request = CreateSpreadsheetRequest.builder() \
                .request_body(spreadsheet) \
                .build()

            response = client.sheets.v3.spreadsheet.create(request)

            if not response.success():
                return tool_error("api_error", f"Failed to create spreadsheet: {response.code} - {response.msg}")

            token = response.data.spreadsheet.spreadsheet_token

            result = {
                "title": title,
                "token": token,
                "url": f"https://feishu.cn/spreadsheet/{token}",
            }
            return json.dumps({
                "content": json.dumps(result, ensure_ascii=False),
                "media": []
            })

        except Exception as e:
            logger.error("Error creating spreadsheet: {}", e)
            return tool_error("operation_failed", f"Error creating spreadsheet: {str(e)}")

    async def _get_sheet_info(self, client: Any, token: str | None) -> str:
        """Get spreadsheet information."""
        if not token:
            return tool_error("invalid_params", "token is required for get_sheet_info")

        try:
            # Get spreadsheet metadata
            request = GetSpreadsheetRequest.builder().spreadsheet_token(token).build()
            response = client.sheets.v3.spreadsheet.get(request)

            if not response.success():
                return tool_error("api_error", f"Failed to get spreadsheet: {response.code} - {response.msg}")

            spreadsheet = response.data.spreadsheet
            title = spreadsheet.title if hasattr(spreadsheet, 'title') else "Untitled"

            # Get sheets list
            from lark_oapi.api.sheets.v3 import QuerySpreadsheetSheetRequest
            sheets_request = QuerySpreadsheetSheetRequest.builder().spreadsheet_token(token).build()
            sheets_response = client.sheets.v3.spreadsheet_sheet.query(sheets_request)

            sheets = []
            if sheets_response.success() and sheets_response.data and sheets_response.data.sheets:
                for sheet in sheets_response.data.sheets:
                    sheets.append({
                        "sheet_id": sheet.sheet_id,
                        "title": sheet.title,
                    })

            result = {
                "token": token,
                "title": title,
                "sheets": sheets,
            }
            return json.dumps({
                "content": json.dumps(result, ensure_ascii=False),
                "media": []
            })

        except Exception as e:
            logger.error("Error getting sheet info {}: {}", token, e)
            return tool_error("operation_failed", f"Error getting sheet info: {str(e)}")
