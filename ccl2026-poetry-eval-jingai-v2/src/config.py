"""Central configuration for paths and model settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    train_path: Path = DATA_DIR / "train.json"
    test_path: Path = DATA_DIR / "test.json"
    submission_path: Path = DATA_DIR / "submission.json"
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    debate_rounds: int = int(os.getenv("DEBATE_ROUNDS", "2"))
    dry_run: bool = os.getenv("DRY_RUN", "0") == "1"


settings = Settings()
