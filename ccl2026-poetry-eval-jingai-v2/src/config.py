"""Central configuration for paths and model settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "")
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "")
USE_DASHSCOPE = bool(DASHSCOPE_API_KEY or DASHSCOPE_BASE_URL or DASHSCOPE_MODEL)


@dataclass(frozen=True)
class Settings:
    """功能：集中保存项目路径、模型 API、推理参数和调试开关配置。

    字段：
        root_dir：项目根目录。
        data_dir：默认数据目录。
        train_path：默认训练集路径。
        test_path：默认测试集路径。
        submission_path：默认提交结果路径。
        api_key：模型 API Key，从 OPENAI_API_KEY 环境变量读取。
        base_url：OpenAI 兼容接口地址。
        model：调用的模型名称。
        timeout：单次模型请求超时时间，单位为秒。
        max_retries：模型请求失败后的最大重试次数。
        temperature：模型采样温度。
        max_tokens：单次模型响应最大 token 数。
        debate_rounds：Parser 与 Critic 的辩论轮数。
        dry_run：是否跳过真实模型调用并返回模拟结果。
        strict_json：是否偏向严格 JSON 输出约束。
        log_dir：日志目录。
    """

    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    train_path: Path = DATA_DIR / "train.json"
    test_path: Path = DATA_DIR / "test.json"
    submission_path: Path = DATA_DIR / "submission.json"
    api_key: str = DASHSCOPE_API_KEY or os.getenv("OPENAI_API_KEY", "")
    base_url: str = (
        DASHSCOPE_BASE_URL
        or os.getenv("OPENAI_BASE_URL")
        or ("https://dashscope.aliyuncs.com/compatible-mode/v1" if USE_DASHSCOPE else "https://api.openai.com/v1")
    )
    model: str = DASHSCOPE_MODEL or os.getenv("OPENAI_MODEL", "qwen-plus" if USE_DASHSCOPE else "gpt-4.1-mini")
    timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    debate_rounds: int = int(os.getenv("DEBATE_ROUNDS", "2"))
    dry_run: bool = os.getenv("DRY_RUN", "0") == "1"
    strict_json: bool = os.getenv("STRICT_JSON", "1") != "0"
    log_dir: Path = ROOT_DIR / os.getenv("LOG_DIR", "logs")


settings = Settings()
