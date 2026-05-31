"""Small JSONL bad-case recorder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BadCaseRecorder:
    def __init__(self, path: str | Path = "") -> None:
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
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for row in self.rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
