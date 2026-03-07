"""
Entry point for running superbot as a module: python -m superbot
"""
import warnings

# Suppress warnings from external packages
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")

from superbot.cli.commands import app

if __name__ == "__main__":
    app()
