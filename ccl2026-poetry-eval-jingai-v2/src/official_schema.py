"""Strict format checks for the official CCPA2026 submission."""

from __future__ import annotations

from typing import Any

from .official_data import TASK_NAMES


class OfficialSubmissionValidator:
    def validate(
        self,
        submission: dict[str, Any],
        test_data: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        errors: list[str] = []
        if set(submission) != set(TASK_NAMES):
            errors.append("Top-level keys must be task1, task2, task3, task4.")
            return errors
        for task_name in TASK_NAMES:
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
        if row.get("flag") not in {0, 1}:
            errors.append(f"task2 idx {row['idx']} invalid flag.")
        if not isinstance(row.get("answer"), str):
            errors.append(f"task2 idx {row['idx']} answer must be string.")

    def _task3(self, row: dict[str, Any], errors: list[str]) -> None:
        if not isinstance(row.get("answer"), list):
            errors.append(f"task3 idx {row['idx']} answer must be list.")

    def _task4(self, row: dict[str, Any], errors: list[str]) -> None:
        if row.get("answer") not in {"A", "B", "C", "D"}:
            errors.append(f"task4 idx {row['idx']} invalid answer.")
