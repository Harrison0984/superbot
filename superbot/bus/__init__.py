"""Message bus module for decoupled channel-agent communication."""

from superbot.bus.events import InboundMessage, OutboundMessage
from superbot.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
