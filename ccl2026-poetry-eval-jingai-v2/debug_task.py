"""Debug one official task sample and print readable Chinese JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_submission import load_env_file


def build_parser() -> argparse.ArgumentParser:
    """功能：构造单条任务调试脚本的命令行参数解析器。

    返回：
        argparse.ArgumentParser：包含任务名、样本序号、数据目录、环境文件和 dry-run 开关的解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["task1", "task2", "task3", "task4"], default="task1")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--test-dir", default="data")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def debug_one_task(
    task: str = "task1",
    index: int = 0,
    test_dir: str | Path = "data",
    env_file: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """功能：调试官方任务中的单条样本。

    参数：
        task：官方任务名，取值为 task1、task2、task3、task4。
        index：样本在该任务文件中的序号，从 0 开始。
        test_dir：官方测试数据目录，目录下应包含 task1.json 到 task4.json。
        env_file：可选环境变量文件路径，用于加载百炼 API Key 等配置。
        dry_run：是否跳过真实模型调用。

    返回：
        dict[str, Any]：包含 sample 原始样本和 result 官方格式结果的调试字典。
    """
    load_env_file(env_file)

    from src.bad_cases import BadCaseRecorder
    from src.config import settings
    from src.llm_client import LLMClient
    from src.official_data import OfficialDataIO
    from src.pipeline import OfficialPipeline

    rows = OfficialDataIO(test_dir=test_dir).load_task(test_dir, task)
    sample = rows[index]
    recorder = BadCaseRecorder()
    pipeline = OfficialPipeline(settings, client=LLMClient(settings, dry_run=dry_run), recorder=recorder)
    try:
        prompt_id = pipeline.TASK_TO_PROMPT[task]
        raw_internal = pipeline.core.run_sample_as(sample, index, prompt_id)
        result = pipeline.adapters.adapt(task, sample, raw_internal)
    except Exception as exc:  # noqa: BLE001
        recorder.record(task, sample.get("idx", index), type(exc).__name__, str(exc), sample=sample)
        raw_internal = {}
        result = pipeline.run_official_sample(task, sample, index)
    return {
        "settings": {
            "base_url": settings.base_url,
            "model": settings.model,
            "has_api_key": bool(settings.api_key),
            "dry_run": dry_run,
        },
        "sample": sample,
        "raw_internal": raw_internal,
        "result": result,
        "bad_cases": recorder.rows,
    }


def main() -> None:
    """功能：脚本入口，运行单条样本并用中文可读 JSON 打印结果。"""
    args = build_parser().parse_args()
    payload = debug_one_task(
        task=args.task,
        index=args.index,
        test_dir=args.test_dir,
        env_file=args.env_file,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
