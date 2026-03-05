"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from superbot.agent.memory import MemoryStore
from superbot.agent.skills import SkillsLoader

# Text file extensions that can be read as plain text
TEXT_FILE_EXTS = {".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".html", ".css", ".sh", ".bat", ".sql"}

# Try to import pypdf for PDF support
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    PdfReader = None


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        parts = [self._get_identity()]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return f"""# superbot 🐈

You are superbot, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## superbot Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        runtime_ctx = self._build_runtime_context(channel, chat_id)
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *history,
            {"role": "user", "content": merged},
        ]

    def _extract_pdf_text(self, path: Path) -> str | None:
        """Extract text content from PDF file."""
        if not PDF_AVAILABLE or PdfReader is None:
            logger.warning("pypdf not available, cannot extract PDF content")
            return None
        try:
            reader = PdfReader(str(path))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            content = "\n\n".join(text_parts)
            if not content.strip():
                return None
            return content[:50000]  # Limit to 50k chars
        except Exception as e:
            logger.error("Error extracting PDF text: {}", e)
            return None

    def _read_text_file(self, path: Path) -> str | None:
        """Read text file content."""
        # Try UTF-8 first
        try:
            content = path.read_text(encoding="utf-8")
            return content[:50000]  # Limit to 50k chars
        except UnicodeDecodeError:
            pass
        except Exception as e:
            logger.error("Error reading text file (utf-8): {}", e)
            return None

        # Try GBK encoding (common for Chinese text files)
        try:
            content = path.read_text(encoding="gbk")
            return content[:50000]
        except Exception as e:
            logger.error("Error reading text file (gbk): {}", e)
            return None

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images and file attachments."""
        if not media:
            return text

        content_parts = []
        attachments_info = []

        for path in media:
            p = Path(path)
            if not p.is_file():
                continue

            mime, _ = mimetypes.guess_type(str(p))
            ext = p.suffix.lower()

            # Handle image files - embed as base64
            if mime and mime.startswith("image/"):
                try:
                    b64 = base64.b64encode(p.read_bytes()).decode()
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    })
                except Exception as e:
                    logger.error("Error encoding image: {}", e)
                    attachments_info.append(f"[Image: {p.name}]")

            # Handle PDF files - extract text content
            elif ext == ".pdf":
                pdf_text = self._extract_pdf_text(p)
                if pdf_text:
                    content_parts.append({
                        "type": "text",
                        "text": f"--- PDF Content: {p.name} ---\n{pdf_text}"
                    })
                else:
                    attachments_info.append(f"[PDF: {p.name}]")

            # Handle text files - read content directly
            elif ext in TEXT_FILE_EXTS:
                file_text = self._read_text_file(p)
                if file_text:
                    content_parts.append({
                        "type": "text",
                        "text": f"--- File: {p.name} ---\n{file_text}"
                    })
                else:
                    attachments_info.append(f"[File: {p.name}]")

            # Other file types - just note that file was received
            else:
                attachments_info.append(f"[File: {p.name}]")

        # Append file attachment info to text
        if attachments_info:
            text = text + "\n\nReceived files: " + ", ".join(attachments_info)

        if not content_parts:
            return text

        return content_parts + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks
        messages.append(msg)
        return messages
