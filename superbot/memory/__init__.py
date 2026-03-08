"""Memmory - AI Context Memory Management System"""

__version__ = "0.1.0"

from .facade import MemorySystem
from .config import Config, config

__all__ = ["MemorySystem", "Config", "config"]
