"""Small JSONL bad-case recorder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BadCaseRecorder:
    """功能：在流水线运行过程中收集异常样本，并可保存为 JSONL 文件。"""

    def __init__(self, path: str | Path = "") -> None:
        """功能：初始化坏例记录器。

        参数：
            path：坏例 JSONL 输出路径；为空时只在内存中记录，不落盘。
        """
        self.path = Path(path) if path else None
        self.rows: list[dict[str, Any]] = []

    def record(
        self,
        task: str,
        idx: Any,
        error_type: str,
        message: str,
        sample: Any = None,
        raw_output: Any = None,
    ) -> None:
        """功能：记录一条坏例或质量告警。

        参数：
            task：坏例所属任务名，如 task1、task2、task3、task4 或 schema。
            idx：样本编号，通常对应官方数据中的 idx。
            error_type：错误类型名称，如 ValidationError、EmptyAnswer。
            message：错误或告警的文字说明。
            sample：触发错误的原始样本；没有时可为 None。
            raw_output：模型或流水线的原始输出；没有时可为 None。
        """
        self.rows.append(
            {
                "task": task,
                "idx": idx,
                "error_type": error_type,
                "message": message,
                "sample": sample,
                "raw_output": raw_output,
            }
        )

    def save(self) -> None:
        """功能：把已记录的坏例写入 JSONL 文件；未配置输出路径时不执行写入。"""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for row in self.rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
