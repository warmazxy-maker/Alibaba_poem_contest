"""Run the official CCPA2026 submission pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def build_parser() -> argparse.ArgumentParser:
    """功能：构造官方提交脚本的命令行参数解析器。

    返回：
        argparse.ArgumentParser：包含测试集目录、训练集目录、输出路径、坏例路径、任务筛选、样本数量限制、dry-run 开关和环境变量文件路径的解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default=str(WORKSPACE / "CCPA2026-test_data"))
    parser.add_argument("--train-dir", default=str(WORKSPACE / "CCPA2026-train_data"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "submission.json"))
    parser.add_argument("--bad-cases", default=str(ROOT / "outputs" / "bad_cases.jsonl"))
    parser.add_argument("--task", choices=["all", "task1", "task2", "task3", "task4"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-file", default="", help="Optional .env or PowerShell env script.")
    return parser


def load_env_file(path: str) -> None:
    """功能：从 .env 或 PowerShell 环境变量脚本中加载模型配置。

    参数：
        path：环境变量文件路径；为空字符串时不加载任何内容。
    """
    if not path:
        return
    env_path = Path(path)
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        match = re.match(r'^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\']?(.*?)["\']?$', text)
        if not match and "=" in text:
            key, value = text.split("=", 1)
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)$", key.strip())
            if match:
                os.environ[match.group(1)] = value.strip().strip('"').strip("'")
            continue
        if match:
            os.environ[match.group(1)] = match.group(2)


def main() -> None:
    """功能：官方提交脚本入口，读取参数后运行完整提交流水线并打印摘要。"""
    args = build_parser().parse_args()
    load_env_file(args.env_file)

    from src.pipeline import run_official_submission

    result = run_official_submission(
        test_dir=args.test_dir,
        train_dir=args.train_dir,
        output_path=args.output,
        bad_cases_path=args.bad_cases,
        task=args.task,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    submission = result["submission"]
    summary = {name: len(rows) for name, rows in submission.items()}
    print(json.dumps({"rows": summary, "errors": result["errors"]}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    print(f"bad cases {len(result['bad_cases'])}: {args.bad_cases}")


if __name__ == "__main__":
    main()
