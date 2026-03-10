"""Email tool - both as tool for sending and as IMAP poller for receiving.

This class combines:
- Tool capability: Send emails via SMTP
- Background polling: Receive emails via IMAP
"""

import asyncio
import html
import imaplib
import json
import re
import smtplib
import socket
import ssl
import time
from datetime import date
from email import policy
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr
from pathlib import Path
from typing import Any

from loguru import logger

from superbot.agent.tools.base import Tool, tool_error
from superbot.bus.events import InboundMessage, OutboundMessage
from superbot.bus.queue import MessageBus
from superbot.config.schema import EmailConfig, ProxyConfig


class EmailTool(Tool):
    """Email tool - both for sending and receiving emails.

    As Tool:
    - send(to, subject, content): Send email via SMTP

    As Poller:
    - start_polling(bus): Start IMAP polling for inbound emails
    - stop_polling(): Stop IMAP polling
    """

    _IMAP_MONTHS = (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )

    def __init__(
        self,
        config: EmailConfig | None = None,
        proxy_config: ProxyConfig | None = None,
    ):
        self.config = config
        self.proxy_config = proxy_config or ProxyConfig()
        self._bus: MessageBus | None = None
        self._running = False

        # IMAP state
        self._last_subject_by_chat: dict[str, str] = {}
        self._last_message_id_by_chat: dict[str, str] = {}
        self._processed_uids: set[str] = set()
        self._MAX_PROCESSED_UIDS = 100000

        # Attachment settings
        self.MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024
        self.BLOCKED_EXTENSIONS = {
            ".exe", ".scr", ".bat", ".cmd", ".com", ".pif", ".msi",
            ".dll", ".vbs", ".js", ".jar", ".sh", ".ps1", ".deb", ".rpm"
        }

    # ==================== Tool interface ====================

    @property
    def name(self) -> str:
        return "email"

    @property
    def description(self) -> str:
        return "Send an email to a recipient. Use this to send emails via SMTP."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "content": {
                    "type": "string",
                    "description": "Email body content"
                },
                "in_reply_to": {
                    "type": "string",
                    "description": "Message-ID of the email being replied to (for threading)"
                },
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach to the email"
                }
            },
            "required": ["to", "content"]
        }

    async def execute(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        """Send an email via SMTP."""
        to = kwargs.get("to", "")
        subject = kwargs.get("subject", "superbot reply")
        in_reply_to = kwargs.get("in_reply_to")
        attachments = kwargs.get("attachments")

        if not self.config:
            return tool_error("not_configured", "Email tool not configured")

        # Validate config
        if not self.config.smtp_host:
            return tool_error("not_configured", "SMTP host not configured")
        if not self.config.smtp_username:
            return tool_error("not_configured", "SMTP username not configured")
        if not self.config.smtp_password:
            return tool_error("not_configured", "SMTP password not configured")

        to_addr = to.strip()
        if not to_addr:
            return tool_error("invalid_params", "No recipient address specified")

        # Get subject from last email if replying
        base_subject = self._last_subject_by_chat.get(to_addr, subject)
        if not subject or subject == "superbot reply":
            subject = self._reply_subject(base_subject)

        # Build email
        email_msg = EmailMessage()
        email_msg["From"] = self.config.from_address or self.config.smtp_username
        email_msg["To"] = to_addr
        email_msg["Subject"] = subject
        email_msg.set_content(content or "")

        # Handle threading
        reply_to = in_reply_to or self._last_message_id_by_chat.get(to_addr)
        if reply_to:
            email_msg["In-Reply-To"] = reply_to
            email_msg["References"] = reply_to

        # Add attachments
        attachment_files = attachments or []
        for file_path in attachment_files:
            try:
                self._add_attachment(email_msg, file_path)
            except Exception as e:
                logger.warning("Failed to add attachment {}: {}", file_path, e)

        # Send via SMTP
        try:
            await asyncio.to_thread(self._smtp_send, email_msg)
        except Exception as e:
            return tool_error("send_error", f"Failed to send email: {str(e)}")

        return json.dumps({
            "content": f"Email sent to {to_addr} with {len(attachment_files)} attachment(s)",
            "media": []
        })

    def initialize(self, bus: "MessageBus") -> None:
        """Initialize the tool with message bus and auto-start polling."""
        self._bus = bus
        # Auto-start IMAP polling if configured
        self.start_polling(bus)

    def _smtp_send(self, msg: EmailMessage) -> None:
        """Send email via SMTP."""
        timeout = 180
        use_proxy = self.config.use_proxy and self.proxy_config.enabled
        proxy_url = None

        if use_proxy:
            proxy_url = self.proxy_config.socks_proxy or self.proxy_config.https_proxy or self.proxy_config.http_proxy

        if use_proxy and proxy_url and proxy_url.startswith("socks"):
            try:
                import socks
                match = re.search(r"socks[45]?://([^:]+):(\d+)", proxy_url)
                if match:
                    proxy_host = match.group(1)
                    proxy_port = int(match.group(2))
                    proxy_type = socks.SOCKS5 if "5" in proxy_url else socks.SOCKS4
                    socks.set_default_proxy(proxy_type, proxy_host, proxy_port)
                    original_socket = socket.socket
                    socket.socket = socks.socksocket
                    try:
                        smtp = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=timeout)
                    finally:
                        socket.socket = original_socket
                        socks.set_default_proxy()
                else:
                    smtp = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=timeout)
            except ImportError:
                smtp = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=timeout)
            try:
                if self.config.smtp_use_tls:
                    smtp.starttls(context=ssl.create_default_context())
                smtp.login(self.config.smtp_username, self.config.smtp_password)
                smtp.send_message(msg)
                smtp.quit()
                return
            except Exception:
                smtp.quit()
                raise

        if self.config.smtp_use_ssl:
            with smtplib.SMTP_SSL(
                self.config.smtp_host,
                self.config.smtp_port,
                timeout=timeout,
            ) as smtp:
                smtp.login(self.config.smtp_username, self.config.smtp_password)
                smtp.send_message(msg)
            return

        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=timeout) as smtp:
            if self.config.smtp_use_tls:
                smtp.starttls(context=ssl.create_default_context())
            smtp.login(self.config.smtp_username, self.config.smtp_password)
            smtp.send_message(msg)

    def _reply_subject(self, base_subject: str) -> str:
        subject = (base_subject or "").strip() or "superbot reply"
        prefix = self.config.subject_prefix or "Re: "
        if subject.lower().startswith("re:"):
            return subject
        return f"{prefix}{subject}"

    def _add_attachment(self, msg: EmailMessage, file_path: str) -> None:
        """Add an attachment to the email message."""
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            logger.warning("Attachment file not found: {}", file_path)
            return

        filename = path.name
        ext = path.suffix.lower()

        # Check blocked extensions
        if ext in self.BLOCKED_EXTENSIONS:
            logger.warning("Blocked dangerous attachment: {}", filename)
            return

        # Check file size
        file_size = path.stat().st_size
        if file_size > self.MAX_ATTACHMENT_SIZE:
            logger.warning("Attachment too large ({} bytes): {}", file_size, filename)
            return

        # Read the file
        data = path.read_bytes()

        # Determine MIME type
        mime_type = self._get_mime_type(ext)
        maintype, subtype = mime_type.split("/", 1) if "/" in mime_type else ("application", "octet-stream")

        # Build attachment manually using MIMEBase
        from email.mime.base import MIMEBase
        from email import encoders

        part = MIMEBase(maintype, subtype)
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{filename}"')

        # If original message is not multipart, convert it
        if not msg.is_multipart():
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            # Get original content
            body = msg.get_body()
            original_content = ""
            if body:
                original_content = body.get_content()

            # Create new multipart message
            new_msg = MIMEMultipart('mixed')
            new_msg['From'] = msg['From']
            new_msg['To'] = msg['To']
            new_msg['Subject'] = msg['Subject']
            if msg.get('In-Reply-To'):
                new_msg['In-Reply-To'] = msg['In-Reply-To']
            if msg.get('References'):
                new_msg['References'] = msg['References']

            # Add original content as text part
            if original_content:
                text_part = MIMEText(original_content, 'plain', 'utf-8')
                new_msg.attach(text_part)

            # Attach the file
            new_msg.attach(part)

            # Copy back to original message
            msg._payload = new_msg._payload
            msg._headers = new_msg._headers
        else:
            msg.attach(part)

        logger.debug("Added attachment: {} ({} bytes)", filename, file_size)

    # ==================== MIME Types ====================

    _MIME_TYPES: dict[str, str] = {
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".ppt": "application/vnd.ms-powerpoint",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".txt": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".zip": "application/zip",
    }

    @staticmethod
    def _get_mime_type(ext: str) -> str:
        """Get MIME type for file extension."""
        return EmailTool._MIME_TYPES.get(ext, "application/octet-stream")

    # ==================== IMAP Polling ====================

    def start_polling(self, bus: MessageBus) -> None:
        """Start IMAP polling for inbound emails."""
        self._bus = bus
        if not self._validate_config():
            logger.error("Email tool: IMAP not configured, skipping polling")
            return
        asyncio.create_task(self._polling_loop())

    async def _polling_loop(self) -> None:
        """Main polling loop."""
        self._running = True
        logger.info("Starting Email tool polling (IMAP)...")

        poll_seconds = max(5, int(self.config.poll_interval_seconds))
        max_retries = 2

        while self._running:
            inbound_items = []
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    inbound_items = await asyncio.to_thread(self._fetch_new_messages)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning("Email polling attempt {} failed: {}, retrying...", attempt + 1, e)
                    else:
                        logger.error("Email polling failed after {} attempts: {}", max_retries + 1, e)

            if last_error and not inbound_items:
                await asyncio.sleep(poll_seconds)
                continue

            for item in inbound_items:
                sender = item["sender"]
                subject = item.get("subject", "")
                message_id = item.get("message_id", "")

                if subject:
                    self._last_subject_by_chat[sender] = subject
                if message_id:
                    self._last_message_id_by_chat[sender] = message_id

                # Send to agent via bus
                if self._bus:
                    msg = InboundMessage(
                        channel="email",
                        sender_id=sender,
                        chat_id=sender,
                        content=item["content"],
                        media=item.get("media", []),
                        metadata=item.get("metadata", {}),
                        to="",
                    )
                    await self._bus.publish_inbound(msg)

            await asyncio.sleep(poll_seconds)

    def stop_polling(self) -> None:
        """Stop IMAP polling."""
        self._running = False

    def _validate_config(self) -> bool:
        """Validate IMAP config for receiving emails."""
        if not self.config:
            return False
        missing = []
        if not self.config.imap_host:
            missing.append("imap_host")
        if not self.config.imap_username:
            missing.append("imap_username")
        if not self.config.imap_password:
            missing.append("imap_password")

        if missing:
            logger.error("Email tool IMAP not configured, missing: {}", ', '.join(missing))
            return False
        return True

    def _fetch_new_messages(self) -> list[dict[str, Any]]:
        """Poll IMAP and return parsed unread messages."""
        return self._fetch_messages(
            search_criteria=("UNSEEN",),
            mark_seen=self.config.mark_seen,
            dedupe=True,
            limit=0,
        )

    def _fetch_messages(
        self,
        search_criteria: tuple[str, ...],
        mark_seen: bool,
        dedupe: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch messages by arbitrary IMAP search criteria."""
        messages: list[dict[str, Any]] = []
        mailbox = self.config.imap_mailbox or "INBOX"
        imap_timeout = 60
        use_proxy = self.config.use_proxy and self.proxy_config.enabled
        proxy_url = None

        if use_proxy:
            proxy_url = self.proxy_config.socks_proxy or self.proxy_config.https_proxy or self.proxy_config.http_proxy
            if proxy_url:
                logger.info("Using proxy for IMAP: {}", proxy_url)

        if use_proxy and proxy_url and proxy_url.startswith("socks"):
            try:
                import socks
                match = re.search(r"socks[45]?://([^:]+):(\d+)", proxy_url)
                if match:
                    proxy_host = match.group(1)
                    proxy_port = int(match.group(2))
                    proxy_type = socks.SOCKS5 if "5" in proxy_url else socks.SOCKS4
                    socks.set_default_proxy(proxy_type, proxy_host, proxy_port)
                    original_socket = socket.socket
                    socket.socket = socks.socksocket
                    try:
                        client = imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port, timeout=imap_timeout) \
                            if self.config.imap_use_ssl else \
                            imaplib.IMAP4(self.config.imap_host, self.config.imap_port, timeout=imap_timeout)
                    finally:
                        socket.socket = original_socket
                        socks.set_default_proxy()
                else:
                    client = imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port, timeout=imap_timeout)
            except ImportError:
                client = imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port, timeout=imap_timeout)
        elif self.config.imap_use_ssl:
            client = imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port, timeout=imap_timeout)
        else:
            client = imaplib.IMAP4(self.config.imap_host, self.config.imap_port, timeout=imap_timeout)

        try:
            client.login(self.config.imap_username, self.config.imap_password)
            status, _ = client.select(mailbox)
            if status != "OK":
                return messages

            status, data = client.search(None, *search_criteria)
            if status != "OK" or not data:
                return messages

            ids = data[0].split()
            if limit > 0 and len(ids) > limit:
                ids = ids[-limit:]

            for imap_id in ids:
                status, fetched = client.fetch(imap_id, "(BODY.PEEK[] UID)")
                if status != "OK" or not fetched:
                    continue

                raw_bytes = self._extract_message_bytes(fetched)
                if raw_bytes is None:
                    continue

                uid = self._extract_uid(fetched)
                if dedupe and uid and uid in self._processed_uids:
                    continue

                parsed = BytesParser(policy=policy.default).parsebytes(raw_bytes)
                sender = parseaddr(parsed.get("From", ""))[1].strip().lower()
                if not sender:
                    continue

                subject = self._decode_header_value(parsed.get("Subject", ""))
                date_value = parsed.get("Date", "")
                message_id = parsed.get("Message-ID", "").strip()
                body = self._extract_text_body(parsed)
                media_paths, attachment_names = self._extract_attachments(parsed)

                if not body:
                    body = "(empty email body)"

                body = body[: self.config.max_body_chars]

                attachment_text = ""
                if attachment_names:
                    attachment_text = f"\n\nAttachments: {', '.join(attachment_names)}"

                content = (
                    f"Email received.\n"
                    f"From: {sender}\n"
                    f"Subject: {subject}\n"
                    f"Date: {date_value}{attachment_text}\n\n"
                    f"{body}"
                )

                metadata = {
                    "message_id": message_id,
                    "subject": subject,
                    "date": date_value,
                    "sender_email": sender,
                    "uid": uid,
                }

                messages.append({
                    "sender": sender,
                    "subject": subject,
                    "message_id": message_id,
                    "content": content,
                    "media": media_paths,
                    "metadata": metadata,
                })

                if dedupe and uid:
                    self._processed_uids.add(uid)
                    if len(self._processed_uids) > self._MAX_PROCESSED_UIDS:
                        self._processed_uids = set(list(self._processed_uids)[len(self._processed_uids) // 2:])

                if mark_seen:
                    try:
                        client.store(imap_id, "+FLAGS", "\\Seen")
                    except Exception as e:
                        logger.error("Failed to mark email as seen: {}", e)
        finally:
            try:
                client.logout()
            except Exception:
                pass

        return messages

    @staticmethod
    def _format_imap_date(value: date) -> str:
        month = EmailTool._IMAP_MONTHS[value.month - 1]
        return f"{value.day:02d}-{month}-{value.year}"

    @staticmethod
    def _extract_message_bytes(fetched: list[Any]) -> bytes | None:
        for item in fetched:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
                return bytes(item[1])
        return None

    @staticmethod
    def _extract_uid(fetched: list[Any]) -> str:
        for item in fetched:
            if isinstance(item, tuple) and item and isinstance(item[0], (bytes, bytearray)):
                head = bytes(item[0]).decode("utf-8", errors="ignore")
                m = re.search(r"UID\s+(\d+)", head)
                if m:
                    return m.group(1)
        return ""

    @staticmethod
    def _decode_header_value(value: str) -> str:
        if not value:
            return ""
        try:
            return str(make_header(decode_header(value)))
        except Exception:
            return value

    @classmethod
    def _extract_text_body(cls, msg: Any) -> str:
        if msg.is_multipart():
            plain_parts: list[str] = []
            html_parts: list[str] = []
            for part in msg.walk():
                if part.get_content_disposition() == "attachment":
                    continue
                content_type = part.get_content_type()
                try:
                    payload = part.get_content()
                except Exception:
                    payload_bytes = part.get_payload(decode=True) or b""
                    charset = part.get_content_charset() or "utf-8"
                    payload = payload_bytes.decode(charset, errors="replace")
                if not isinstance(payload, str):
                    continue
                if content_type == "text/plain":
                    plain_parts.append(payload)
                elif content_type == "text/html":
                    html_parts.append(payload)
            if plain_parts:
                return "\n\n".join(plain_parts).strip()
            if html_parts:
                return cls._html_to_text("\n\n".join(html_parts)).strip()
            return ""

        try:
            payload = msg.get_content()
        except Exception:
            payload_bytes = msg.get_payload(decode=True) or b""
            charset = msg.get_content_charset() or "utf-8"
            payload = payload_bytes.decode(charset, errors="replace")
        if not isinstance(payload, str):
            return ""
        if msg.get_content_type() == "text/html":
            return cls._html_to_text(payload).strip()
        return payload.strip()

    @staticmethod
    def _html_to_text(raw_html: str) -> str:
        text = re.sub(r"<\s*br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
        text = re.sub(r"<\s*/\s*p\s*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        return html.unescape(text)

    def _extract_attachments(self, msg: Any) -> tuple[list[str], list[str]]:
        """Extract attachments from email."""
        media_dir = Path.home() / ".superbot" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        media_paths: list[str] = []
        attachment_info: list[str] = []

        try:
            for part in msg.walk():
                content_disposition = part.get_content_disposition()
                if content_disposition != "attachment":
                    continue

                filename = part.get_filename()
                if not filename:
                    continue

                filename = self._decode_header_value(filename)
                ext = Path(filename).suffix.lower()
                if ext in self.BLOCKED_EXTENSIONS:
                    logger.warning("Blocked dangerous attachment: {}", filename)
                    attachment_info.append(f"{filename} (blocked)")
                    continue

                payload = part.get_payload(decode=True)
                if not payload or not isinstance(payload, bytes):
                    continue

                if len(payload) > self.MAX_ATTACHMENT_SIZE:
                    logger.warning("Attachment too large ({} bytes): {}", len(payload), filename)
                    attachment_info.append(f"{filename} (too large)")
                    continue

                file_path = media_dir / filename
                if file_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    file_path = media_dir / f"{stem}_{int(time.time())}{suffix}"

                file_path.write_bytes(payload)
                media_paths.append(str(file_path))
                attachment_info.append(filename)
                logger.debug("Saved email attachment: {}", filename)

        except Exception as e:
            logger.error("Error extracting email attachments: {}", e)

        return media_paths, attachment_info
