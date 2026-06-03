"""Strict format checks for the official CCPA2026 submission."""

from __future__ import annotations

from typing import Any

from .official_data import TASK_NAMES


class OfficialSubmissionValidator:
    """功能：校验最终 submission 是否满足官方 task1-task4 的基础格式要求。"""

    def validate(
        self,
        submission: dict[str, Any],
        test_data: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        """功能：校验完整官方提交文件。

        参数：
            submission：待校验的官方提交字典，顶层键应为 task1-task4。
            test_data：官方测试集数据，用于检查长度和 idx 是否匹配。

        返回：
            list[str]：错误信息列表；为空表示基础格式检查通过。
        """
        errors: list[str] = []
        submitted_tasks = set(submission)
        if not submitted_tasks or not submitted_tasks.issubset(set(TASK_NAMES)):
            errors.append("Top-level keys must be a non-empty subset of task1, task2, task3, task4.")
            return errors
        for task_name in TASK_NAMES:
            if task_name not in submitted_tasks:
                continue
            rows = submission.get(task_name)
            source = test_data.get(task_name, [])
            if not isinstance(rows, list):
                errors.append(f"{task_name} must be a list.")
                continue
            if len(rows) != len(source):
                errors.append(f"{task_name} length mismatch: {len(rows)} != {len(source)}.")
            source_by_idx: dict[int, dict[str, Any]] = {}
            for item in source:
                try:
                    source_by_idx[int(item["idx"])] = item
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{task_name} source row has invalid idx.")
            for row in rows:
                self._validate_row(task_name, row, source_by_idx, errors)
        return errors

    def _validate_row(
        self,
        task_name: str,
        row: Any,
        source_by_idx: dict[int, dict[str, Any]],
        errors: list[str],
    ) -> None:
        """功能：校验单条提交行并按任务分派到具体校验函数。

        参数：
            task_name：官方任务名。
            row：待校验的单条提交行。
            source_by_idx：按 idx 建立索引的官方源数据。
            errors：错误信息列表，函数会就地追加错误。
        """
        if not isinstance(row, dict) or "idx" not in row:
            errors.append(f"{task_name} row must be an object with idx.")
            return
        try:
            idx = int(row["idx"])
        except (TypeError, ValueError):
            errors.append(f"{task_name} row has invalid idx: {row.get('idx')}.")
            return
        source = source_by_idx.get(idx)
        if source is None:
            errors.append(f"{task_name} idx {idx} is not in source data.")
            return
        if task_name == "task1":
            self._task1(row, source, errors)
        elif task_name == "task2":
            self._task2(row, errors)
        elif task_name == "task3":
            self._task3(row, errors)
        elif task_name == "task4":
            self._task4(row, errors)

    def _task1(self, row: dict[str, Any], source: dict[str, Any], errors: list[str]) -> None:
        """功能：校验 task1 提交行。

        参数：
            row：task1 单条提交行。
            source：对应的官方原始样本。
            errors：错误信息列表，函数会就地追加错误。
        """
        words = row.get("ans_qa_words", {})
        sents = row.get("ans_qa_sents", {})
        if not isinstance(words, dict):
            errors.append(f"task1 idx {row['idx']} ans_qa_words must be object.")
            words = {}
        if not isinstance(sents, dict):
            errors.append(f"task1 idx {row['idx']} ans_qa_sents must be object.")
            sents = {}
        if set(words) != set(source.get("qa_words", [])):
            errors.append(f"task1 idx {row['idx']} word keys mismatch.")
        if set(sents) != set(source.get("qa_sents", [])):
            errors.append(f"task1 idx {row['idx']} sentence keys mismatch.")
        if row.get("choose_id") not in {"A", "B", "C", "D"}:
            errors.append(f"task1 idx {row['idx']} invalid choose_id.")

    def _task2(self, row: dict[str, Any], errors: list[str]) -> None:
        """功能：校验 task2 提交行。

        参数：
            row：task2 单条提交行。
            errors：错误信息列表，函数会就地追加错误。
        """
        if row.get("flag") not in {0, 1}:
            errors.append(f"task2 idx {row['idx']} invalid flag.")
        if not isinstance(row.get("answer"), str):
            errors.append(f"task2 idx {row['idx']} answer must be string.")

    def _task3(self, row: dict[str, Any], errors: list[str]) -> None:
        """功能：校验 task3 提交行。

        参数：
            row：task3 单条提交行。
            errors：错误信息列表，函数会就地追加错误。
        """
        if not isinstance(row.get("answer"), list):
            errors.append(f"task3 idx {row['idx']} answer must be list.")

    def _task4(self, row: dict[str, Any], errors: list[str]) -> None:
        """功能：校验 task4 提交行。

        参数：
            row：task4 单条提交行。
            errors：错误信息列表，函数会就地追加错误。
        """
        if row.get("answer") not in {"A", "B", "C", "D"}:
            errors.append(f"task4 idx {row['idx']} invalid answer.")
