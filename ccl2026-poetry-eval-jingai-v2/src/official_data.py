"""Official CCPA2026 data loading and output adapters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TASK_NAMES = ("task1", "task2", "task3", "task4")


class OfficialDataIO:
    def __init__(self, train_dir: str | Path = "", test_dir: str | Path = "") -> None:
        self.train_dir = Path(train_dir) if train_dir else Path()
        self.test_dir = Path(test_dir) if test_dir else Path()

    def load_task(self, directory: str | Path, task_name: str) -> list[dict[str, Any]]:
        path = Path(directory) / f"{task_name}.json"
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise TypeError(f"{path} must contain a JSON array.")
        return data

    def load_all_test(self) -> dict[str, list[dict[str, Any]]]:
        return {name: self.load_task(self.test_dir, name) for name in TASK_NAMES}

    def load_all_train(self) -> dict[str, list[dict[str, Any]]]:
        return {name: self.load_task(self.train_dir, name) for name in TASK_NAMES}

    def save_submission(self, output_path: str | Path, submission: dict[str, Any]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(submission, file, ensure_ascii=False, indent=2)
            file.write("\n")


class OfficialAdapters:
    def adapt(self, task_name: str, sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        if task_name == "task1":
            return self.adapt_task1(sample, result)
        if task_name == "task2":
            return self.adapt_task2(sample, result)
        if task_name == "task3":
            return self.adapt_task3(sample, result)
        if task_name == "task4":
            return self.adapt_task4(sample, result)
        raise ValueError(f"Unknown task: {task_name}")

    def adapt_task1(self, sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        words = sample.get("qa_words", [])
        sents = sample.get("qa_sents", [])
        raw_words = _as_dict(result.get("ans_qa_words"))
        raw_sents = _as_dict(result.get("ans_qa_sents"))
        text = _main_text(result)
        return {
            "idx": int(sample["idx"]),
            "ans_qa_words": {word: str(raw_words.get(word) or text) for word in words},
            "ans_qa_sents": {sent: str(raw_sents.get(sent) or text) for sent in sents},
            "choose_id": _choice_from_sample(sample, result.get("choose_id") or result.get("prediction"), "A"),
        }

    def adapt_task2(self, sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        flag = result.get("flag")
        text = _main_text(result)
        if flag not in (0, 1, "0", "1"):
            flag = 0 if _means_no_allusion(text) else 1 if text else 0
        answer = "" if int(flag) == 0 else str(result.get("answer") or text)
        return {
            "idx": int(sample["idx"]),
            "flag": int(flag),
            "answer": answer,
        }

    def adapt_task3(self, sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        answer = result.get("answer") or result.get("prediction") or []
        if isinstance(answer, str):
            answer = _split_answer(answer)
        if not isinstance(answer, list):
            answer = [str(answer)]
        need = max(1, len(re.findall(r"_{4,}", str(sample.get("que", "")))))
        clean = [str(item).strip() for item in answer if str(item).strip()]
        while len(clean) < need:
            clean.append("")
        return {"idx": int(sample["idx"]), "answer": clean[:need]}

    def adapt_task4(self, sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        return {
            "idx": int(sample["idx"]),
            "answer": _choice(result.get("answer") or result.get("prediction"), "A"),
        }


def fallback_row(task_name: str, sample: dict[str, Any]) -> dict[str, Any]:
    return OfficialAdapters().adapt(task_name, sample, {})


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _main_text(result: dict[str, Any]) -> str:
    for key in (
        "prediction",
        "answer",
        "output",
        "reason",
        "revised_prediction",
        "candidate_prediction",
        "explanation",
    ):
        value = result.get(key)
        if value:
            return str(value).strip()
    return ""


def _choice(value: Any, default: str) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = re.search(r"(?:答案|选项|选择|选|ANSWER|OPTION|CHOOSE_ID|CHOOSE)\s*[:：]?\s*([ABCD])\s*(?:项)?", text)
    return match.group(1) if match else default


def _choice_from_sample(sample: dict[str, Any], value: Any, default: str) -> str:
    choice = _choice(value, "")
    if choice:
        return choice
    text = str(value or "")
    options = sample.get("choose", {})
    if isinstance(options, dict):
        for key, option_text in options.items():
            if str(option_text) and str(option_text) in text:
                return str(key)
        best_key = ""
        best_score = 0
        for key, option_text in options.items():
            score = _overlap_score(text, str(option_text))
            if score > best_score:
                best_key = str(key)
                best_score = score
        if best_score >= 2:
            return best_key
    return default


def _means_no_allusion(text: str) -> bool:
    if text.strip() in {"无", "没有", "未见"}:
        return True
    return any(
        word in text
        for word in ("无典故", "没有典故", "不含典故", "未见典故", "无明显典故", "没有用典", "未用典", "无出处")
    )


def _overlap_score(left: str, right: str) -> int:
    useful = set(ch for ch in right if "\u4e00" <= ch <= "\u9fff")
    return sum(1 for ch in useful if ch in left)


def _split_answer(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return [part.strip() for part in re.split(r"[，,；;、\n]", stripped) if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, dict):
        value = parsed.get("answer") or parsed.get("prediction") or []
        return value if isinstance(value, list) else [str(value)]
    return [str(parsed)]
