"""Simplified config for travel tools."""

class Config:
    def __init__(self):
        self.browser = {
            "headless": False,
            "xvfb": True,
            "viewport": {"width": 1920, "height": 1080},
            "user_data_dir": "~/.superbot/sessions/ctrip",
            "stealth": {
                "canvas_fingerprint_randomize": True,
                "languages": ["en-US", "en"]
            },
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        self.proxy = {"failover": {"max_retries": 3}}
        self.behavior = {
            "delay": {"mean": 2.5, "std": 0.8, "min": 1.5, "max": 4.5}
        }

    def get(self, key, default=None):
        if "." in key:
            parts = key.split(".")
            val = self.__dict__
            for p in parts:
                val = val.get(p, {})
            return val if val else default
        return getattr(self, key, default)

config = Config()
