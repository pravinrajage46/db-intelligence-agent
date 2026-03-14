# config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration — reads from .env file."""

    DB_URL: str = os.getenv("DB_URL", "sqlite:///./data/olist.db")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    AI_MODEL: str = os.getenv("AI_MODEL", "claude-sonnet-4-20250514")
    SAMPLE_SIZE: int = int(os.getenv("SAMPLE_SIZE", "10000"))
    FRESHNESS_THRESHOLD_DAYS: int = int(os.getenv("FRESHNESS_THRESHOLD_DAYS", "30"))
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")

    @classmethod
    def update(cls, **kwargs):
        """Allow runtime overrides from CLI flags."""
        for k, v in kwargs.items():
            if v is not None:
                setattr(cls, k.upper(), v)


config = Config()