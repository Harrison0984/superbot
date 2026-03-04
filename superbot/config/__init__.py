"""Configuration module for superbot."""

from superbot.config.loader import get_config_path, load_config
from superbot.config.schema import Config

__all__ = ["Config", "load_config", "get_config_path"]
