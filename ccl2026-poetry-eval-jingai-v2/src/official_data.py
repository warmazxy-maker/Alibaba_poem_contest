"""Official CCPA2026 data loading and output adapters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TASK_NAMES = ("task1", "task2", "task3", "task4")


class OfficialDataIO:
    """功能：负责读取官方 task1-task4 数据文件，并保存官方提交 JSON。"""

    def __init__(self, train_dir: str | Path = "", test_dir: str | Path = "") -> None:
        """功能：初始化官方数据读写器。

        参数：
            train_dir：官方训练集目录；为空时使用空 Path 占位。
            test_dir：官方测试集目录；为空时使用空 Path 占位。
        """
        self.train_dir = Path(train_dir) if train_dir else Path()
        self.test_dir = Path(test_dir) if test_dir else Path()

    def load_task(self, directory: str | Path, task_name: str) -> list[dict[str, Any]]:
        """功能：读取某个官方任务的 JSON 数组文件。

        参数：
            directory：任务文件所在目录。
            task_name：任务名称，如 task1、task2、task3、task4。

        返回：
            list[dict[str, Any]]：该任务的样本列表。
        """
        path = Path(directory) / f"{task_name}.json"
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise TypeError(f"{path} must contain a JSON array.")
        for index, row in enumerate(data):
            if isinstance(row, dict) and "idx" not in row:
                row["idx"] = index
        return data

    def load_all_test(self) -> dict[str, list[dict[str, Any]]]:
        """功能：读取全部官方测试集任务。

        返回：
            dict[str, list[dict[str, Any]]]：键为 task1-task4，值为对应测试样本列表。
        """
        return {name: self.load_task(self.test_dir, name) for name in TASK_NAMES}

    def load_all_train(self) -> dict[str, list[dict[str, Any]]]:
        """功能：读取全部官方训练集任务。

        返回：
            dict[str, list[dict[str, Any]]]：键为 task1-task4，值为对应训练样本列表。
        """
        return {name: self.load_task(self.train_dir, name) for name in TASK_NAMES}

    def save_submission(self, output_path: str | Path, submission: dict[str, Any]) -> None:
        """功能：保存官方提交 JSON。

        参数：
            output_path：提交文件输出路径。
            submission：官方提交格式字典，顶层键应为 task1、task2、task3、task4。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(submission, file, ensure_ascii=False, indent=2)
            file.write("\n")


class OfficialAdapters:
    """功能：把 PoetryEvalPipeline 的内部结果转换成官方提交格式。"""

    def adapt(self, task_name: str, sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """功能：根据任务名分派到对应的官方格式适配函数。

        参数：
            task_name：官方任务名，如 task1、task2、task3、task4。
            sample：官方原始样本字典。
            result：核心流水线返回的内部结果字典。

        返回：
            dict[str, Any]：单条样本的官方提交格式结果。
        """
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
        """功能：适配 task1 字词/句意理解任务输出。

        参数：
            sample：官方 task1 原始样本，使用 idx、qa_words、qa_sents、choose 等字段。
            result：内部结果，优先使用 ans_qa_words、ans_qa_sents、choose_id 或 prediction。

        返回：
            dict[str, Any]：包含 idx、ans_qa_words、ans_qa_sents、choose_id 的官方 task1 提交行。
        """
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
        """功能：适配 task2 典故识别任务输出。

        参数：
            sample：官方 task2 原始样本，至少包含 idx。
            result：内部结果，优先使用 flag 和 answer；缺失时根据文本判断是否无典故。

        返回：
            dict[str, Any]：包含 idx、flag、answer 的官方 task2 提交行。
        """
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
        """功能：适配 task3 类比推理填空任务输出。

        参数：
            sample：官方 task3 原始样本，使用 idx 和 que 中的空格数量判断答案个数。
            result：内部结果，优先使用 answer；也可从 prediction 字符串拆分。

        返回：
            dict[str, Any]：包含 idx 和 answer 列表的官方 task3 提交行。
        """
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
        """功能：适配 task4 诗词辨析选择题输出。

        参数：
            sample：官方 task4 原始样本，至少包含 idx。
            result：内部结果，优先使用 answer 或 prediction 中的 A/B/C/D。

        返回：
            dict[str, Any]：包含 idx 和 answer 的官方 task4 提交行。
        """
        return {
            "idx": int(sample["idx"]),
            "answer": _choice(result.get("answer") or result.get("prediction"), "A"),
        }


def fallback_row(task_name: str, sample: dict[str, Any]) -> dict[str, Any]:
    """功能：当核心流水线异常时生成该任务的兜底提交行。

    参数：
        task_name：官方任务名，如 task1、task2、task3、task4。
        sample：官方原始样本字典。

    返回：
        dict[str, Any]：使用空结果适配出的官方提交行。
    """
    return OfficialAdapters().adapt(task_name, sample, {})


def _as_dict(value: Any) -> dict[str, Any]:
    """功能：把值安全转换为字典。

    参数：
        value：待检查的任意值。

    返回：
        dict[str, Any]：如果 value 是字典则原样返回，否则返回空字典。
    """
    return value if isinstance(value, dict) else {}


def _main_text(result: dict[str, Any]) -> str:
    """功能：从内部结果中抽取最适合作为文本答案的字段。

    参数：
        result：内部结果字典。

    返回：
        str：第一个非空的文本字段；没有时返回空字符串。
    """
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
    """功能：从任意文本中抽取 A/B/C/D 选项。

    参数：
        value：可能包含选项的任意值。
        default：无法抽取选项时返回的默认值。

    返回：
        str：抽取到的 A/B/C/D，或 default。
    """
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = re.search(r"(?:答案|选项|选择|选|ANSWER|OPTION|CHOOSE_ID|CHOOSE)\s*[:：]?\s*([ABCD])\s*(?:项)?", text)
    return match.group(1) if match else default


def _choice_from_sample(sample: dict[str, Any], value: Any, default: str) -> str:
    """功能：结合官方样本选项文本推断 A/B/C/D。

    参数：
        sample：官方原始样本，可能包含 choose 字段。
        value：模型或内部结果中的选项文本。
        default：无法推断时返回的默认值。

    返回：
        str：推断出的选项字母，或 default。
    """
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
    """功能：判断文本是否表示“没有典故”。

    参数：
        text：待判断的答案文本。

    返回：
        bool：文本表达无典故时返回 True，否则返回 False。
    """
    if text.strip() in {"无", "没有", "未见"}:
        return True
    return any(
        word in text
        for word in ("无典故", "没有典故", "不含典故", "未见典故", "无明显典故", "没有用典", "未用典", "无出处")
    )


def _overlap_score(left: str, right: str) -> int:
    """功能：计算两个中文文本的简单字符重合分。

    参数：
        left：待匹配文本。
        right：选项或参考文本。

    返回：
        int：right 中出现在 left 里的中文字符数量。
    """
    useful = set(ch for ch in right if "\u4e00" <= ch <= "\u9fff")
    return sum(1 for ch in useful if ch in left)


def _split_answer(text: str) -> list[str]:
    """功能：把 task3 的字符串答案拆成列表。

    参数：
        text：模型返回的答案字符串，可以是 JSON、逗号/顿号/分号分隔文本。

    返回：
        list[str]：拆分后的答案列表。
    """
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
