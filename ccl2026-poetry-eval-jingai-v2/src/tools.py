"""Shared utilities plus the core Tools functions from the workflow spec."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .prompts import TASK_ALIASES, get_task_prompt


ID_KEYS = ("id", "qid", "question_id", "sample_id")
LABEL_KEYS = ("label", "answer", "gold", "target")
PREDICTION_KEYS = ("prediction", "answer", "output")
TASK_KEYS = ("task_id", "subtask", "category", "task", "task_type", "type")
QUESTION_KEYS = ("question", "query", "prompt", "instruction", "stem")
OPTION_KEYS = ("choose", "options", "choices", "candidates")

TASK_REQUIRED_FIELDS = {
    "1-1": ("prediction", "reason", "evidence", "ans_qa_words", "ans_qa_sents", "choose_id"),
    "1-2": ("prediction", "reason", "evidence", "ans_qa_words", "ans_qa_sents", "choose_id"),
    "1-3": ("prediction", "reason", "evidence", "ans_qa_words", "ans_qa_sents", "choose_id"),
    "1-4": ("prediction", "reason", "evidence", "flag", "answer"),
    "2-1": ("prediction", "reason", "analogy_matrix", "answer"),
    "2-2": ("prediction", "reason", "votes", "answer"),
}


def load_json(path: str | Path) -> Any:
    """功能：读取 JSON 文件。

    参数：
        path：JSON 文件路径。

    返回：
        Any：反序列化后的 JSON 数据。
    """
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path) -> None:
    """功能：把数据保存为 UTF-8 JSON 文件。

    参数：
        data：需要序列化的数据。
        path：输出 JSON 文件路径。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def ensure_list(payload: Any) -> list[dict[str, Any]]:
    """功能：把输入 JSON 数据规范成样本字典列表。

    参数：
        payload：JSON 数据，可以是列表，也可以是包含 data/items/examples/samples 的字典。

    返回：
        list[dict[str, Any]]：样本字典列表。
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "examples", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise TypeError("Expected a JSON array or an object containing a data array.")


def pick_first(mapping: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    """功能：按候选键顺序从字典中取第一个非 None 值。

    参数：
        mapping：待查询的字典。
        keys：候选字段名序列。
        default：所有候选字段都不存在或为 None 时返回的默认值。

    返回：
        Any：第一个命中的字段值，或 default。
    """
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def get_sample_id(sample: dict[str, Any], index: int) -> str:
    """功能：获取样本 id。

    参数：
        sample：当前样本字典。
        index：样本序号，用于缺省 id。

    返回：
        str：样本 id 字符串。
    """
    return str(pick_first(sample, ID_KEYS, index))


def normalize_prediction(raw: Any) -> str:
    """功能：把任意预测结果规范成字符串。

    参数：
        raw：模型或工具返回的原始预测，可以是 None、字典、JSON 字符串或普通文本。

    返回：
        str：规范化后的预测字符串。
    """
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return str(pick_first(raw, PREDICTION_KEYS, raw))

    text = str(raw).strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(parsed, dict):
        return str(pick_first(parsed, PREDICTION_KEYS, text)).strip()
    return str(parsed).strip()


def build_submission_row(sample: dict[str, Any], raw_prediction: Any, index: int) -> dict[str, str]:
    """功能：构造普通格式提交行。

    参数：
        sample：当前样本字典。
        raw_prediction：原始预测结果。
        index：样本序号，用于缺省 id。

    返回：
        dict[str, str]：包含 id 和 prediction 的提交行。
    """
    return {
        "id": get_sample_id(sample, index),
        "prediction": normalize_prediction(raw_prediction),
    }


def extract_label(sample: dict[str, Any]) -> str:
    """功能：从样本中提取标准答案字段。

    参数：
        sample：当前样本字典。

    返回：
        str：标准答案字符串；不存在时返回空字符串。
    """
    return str(pick_first(sample, LABEL_KEYS, "")).strip()


def extract_prediction(row: dict[str, Any]) -> str:
    """功能：从预测行中提取预测答案字段。

    参数：
        row：预测结果字典。

    返回：
        str：规范化后的预测字符串。
    """
    return normalize_prediction(pick_first(row, PREDICTION_KEYS, ""))


def dispatch_sub_task_prompt(
    sample: dict[str, Any],
    task_id: str | None = None,
) -> dict[str, Any]:
    """功能：Tool 1，动态选择子任务 prompt 和执行轨道。

    参数：
        sample：当前题目的原始样本字典。
        task_id：指定的内部任务编号；为 None 时根据样本内容自动推断。

    返回：
        dict[str, Any]：任务计划字典，包含 task_id、track、task_name、stages、output_schema 等字段。
    """
    resolved_task_id = task_id or infer_task_id(sample)
    task = get_task_prompt(resolved_task_id)
    output_schema = dict(task.output_contract)
    output_schema["task_id"] = task.task_id
    output_schema["required"] = list(TASK_REQUIRED_FIELDS.get(task.task_id, tuple(output_schema.get("required", []))))
    return {
        "task_id": task.task_id,
        "track": task.track,
        "task_name": task.name_zh,
        "task_name_en": task.name_en,
        "task_description": task.description,
        "stages": list(task.stages),
        "output_schema": output_schema,
    }


def execute_debate_loop(
    parser_agent: Any,
    critic_agent: Any,
    sample: dict[str, Any],
    task_plan: dict[str, Any],
    context: dict[str, Any],
    rounds: int = 2,
) -> dict[str, Any]:
    """功能：Tool 2，控制 Parser 和 Critic 的多轮辩论。

    参数：
        parser_agent：Parser Agent 实例，需要支持 raw_extract 或 option_analysis。
        critic_agent：Critic Agent 实例，需要支持 debate 或 option_scoring。
        sample：当前题目的原始样本字典。
        task_plan：任务计划字典。
        context：当前流水线上下文字典。
        rounds：辩论轮数，最少执行 1 轮。

    返回：
        dict[str, Any]：包含 history 和 context 的辩论结果；history 保存每轮 Parser/Critic 输出。
    """
    history: list[dict[str, Any]] = []
    debate_context = dict(context)
    parser_stage = (
        "parser_option_analysis"
        if task_plan.get("task_id") == "2-2"
        else "parser_raw_extract"
    )
    critic_stage = (
        "critic_option_scoring"
        if task_plan.get("task_id") == "2-2"
        else "critic_debate"
    )

    for round_index in range(max(rounds, 1)):
        if parser_stage == "parser_option_analysis" and hasattr(parser_agent, "option_analysis"):
            parser_result = parser_agent.option_analysis(sample, debate_context, task_plan)
        elif hasattr(parser_agent, "raw_extract"):
            parser_result = parser_agent.raw_extract(sample, debate_context, task_plan)
        else:
            parser_result = parser_agent.run_stage(sample, parser_stage, debate_context, task_plan)
        debate_context[f"parser_round_{round_index + 1}"] = parser_result.content

        if critic_stage == "critic_option_scoring" and hasattr(critic_agent, "option_scoring"):
            critic_result = critic_agent.option_scoring(sample, debate_context, task_plan)
        elif hasattr(critic_agent, "debate"):
            critic_result = critic_agent.debate(sample, debate_context, task_plan)
        else:
            critic_result = critic_agent.run_stage(sample, critic_stage, debate_context, task_plan)
        debate_context[f"critic_round_{round_index + 1}"] = critic_result.content

        history.append(
            {
                "round": round_index + 1,
                "parser": parser_result.data or parser_result.content,
                "critic": critic_result.data or critic_result.content,
            }
        )

    return {"history": history, "context": debate_context}


def parse_analogy_matrix(raw: Any) -> list[dict[str, Any]]:
    """功能：Tool 3，解析并规范化 2-1 类比矩阵。

    参数：
        raw：模型或 Parser 返回的原始类比矩阵，可以是字典、列表、JSON 字符串或普通文本。

    返回：
        list[dict[str, Any]]：规范化后的矩阵列表，每项包含 index、source、target、relation、evidence、confidence。
    """
    payload = _coerce_json_payload(raw)
    if isinstance(payload, dict):
        matrix = payload.get("analogy_matrix") or payload.get("matrix") or payload.get("relations")
    else:
        matrix = payload

    if isinstance(matrix, list):
        normalized = []
        for index, item in enumerate(matrix):
            if isinstance(item, dict):
                normalized.append(
                    {
                        "index": item.get("index", index + 1),
                        "source": item.get("source", item.get("left", "")),
                        "target": item.get("target", item.get("right", "")),
                        "relation": item.get("relation", item.get("type", "")),
                        "evidence": item.get("evidence", ""),
                        "confidence": item.get("confidence", item.get("score", "")),
                    }
                )
            else:
                normalized.append({"index": index + 1, "source": "", "target": "", "relation": str(item)})
        return normalized

    if isinstance(payload, str):
        lines = [line.strip() for line in payload.splitlines() if line.strip()]
        return [{"index": index + 1, "source": "", "target": "", "relation": line} for index, line in enumerate(lines)]

    return []


def eliminate_and_vote(
    option_scores: Any,
    options: list[dict[str, Any]] | None = None,
    question_text: str = "",
) -> dict[str, Any]:
    """功能：Tool 4，根据选项分数淘汰并投票得到唯一选项。

    参数：
        option_scores：模型或 Critic 返回的原始选项分数，可以是列表、字典或 JSON 字符串。
        options：标准化选项列表，每项包含 option 和 text；用于补齐未打分选项。
        question_text：题目文本，用于识别“最合理”还是“不正确”类选择目标。

    返回：
        dict[str, Any]：投票结果，包含 prediction、answer、votes、reason、target。
    """
    scores = _normalize_option_scores(option_scores, options)
    if not scores:
        return {"prediction": "", "votes": [], "reason": "No option scores were available."}

    active_scores = [item for item in scores if not item.get("eliminated")]
    candidates = active_scores or scores
    target = _question_target(question_text)
    if target == "incorrect" and any(item.get("error_score") is not None for item in candidates):
        winner = max(candidates, key=lambda item: (_safe_float(item.get("error_score")), str(item.get("option", ""))))
    elif target == "incorrect" and any(item.get("correctness_score") is not None for item in candidates):
        winner = min(candidates, key=lambda item: (_safe_float(item.get("correctness_score")), str(item.get("option", ""))))
    else:
        winner = max(candidates, key=lambda item: (_safe_float(item.get("score")), str(item.get("option", ""))))

    sorted_votes = sorted(scores, key=lambda item: _safe_float(item.get("score")), reverse=True)
    choice = _choice(winner.get("option") or winner.get("answer") or winner.get("prediction"), "")
    return {
        "prediction": choice,
        "answer": choice,
        "votes": sorted_votes,
        "reason": str(winner.get("reason", "")),
        "target": target,
    }


def enforce_json_schema(
    payload: Any,
    schema: dict[str, Any] | None = None,
    sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """功能：Tool 5，对最终结果做 JSON 兜底和轻量 schema 约束。

    参数：
        payload：Agent 或工具返回的原始结果，可以是字典、列表、JSON 字符串或普通文本。
        schema：任务输出契约，包含 task_id 和 required 字段。
        sample：当前题目的原始样本字典，用于补齐 task1 词句键、task3 答案个数等。

    返回：
        dict[str, Any]：补齐必要字段后的结构化结果。
    """
    data = _coerce_json_payload(payload)
    if not isinstance(data, dict):
        data = {"prediction": normalize_prediction(data), "reason": str(data)}

    if "prediction" not in data:
        data["prediction"] = normalize_prediction(pick_first(data, PREDICTION_KEYS, ""))
    data["prediction"] = normalize_prediction(data.get("prediction", ""))

    required = (schema or {}).get("required", ["prediction", "reason"])
    for field in required:
        if field not in data:
            data[field] = [] if field in {"evidence", "votes", "analogy_matrix"} else ""

    task_id = (schema or {}).get("task_id", "")
    if task_id in {"1-1", "1-2", "1-3"}:
        _enforce_task1_like(data, sample or {})
    elif task_id == "1-4":
        _enforce_allusion(data)
    elif task_id == "2-1":
        _enforce_analogy_answer(data, sample or {})
    elif task_id == "2-2":
        _enforce_choice_answer(data)

    return data


def infer_task_id(sample: dict[str, Any]) -> str:
    """功能：根据样本文本和字段自动推断内部任务编号。

    参数：
        sample：当前题目的原始样本字典。

    返回：
        str：内部任务编号，如 1-1、1-4、2-1、2-2。
    """
    raw_fields = [
        str(pick_first(sample, TASK_KEYS, "")),
        str(pick_first(sample, QUESTION_KEYS, "")),
        json.dumps(sample, ensure_ascii=False),
    ]
    raw_text = " ".join(raw_fields).lower()

    for alias, task_id in TASK_ALIASES.items():
        if alias.lower() in raw_text:
            return task_id

    if extract_options(sample):
        return "2-2"
    if any(keyword in raw_text for keyword in ("类比", "对应", "关系", "填空")):
        return "2-1"
    if any(keyword in raw_text for keyword in ("典故", "出处")):
        return "1-4"
    if any(keyword in raw_text for keyword in ("情感", "感情", "情绪", "心境")):
        return "1-3"
    if any(keyword in raw_text for keyword in ("句意", "诗句", "意思")):
        return "1-2"
    return "1-1"


def extract_options(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """功能：从官方或普通样本中提取并标准化选项。

    参数：
        sample：当前题目的原始样本字典，可能包含 choose、options、choices 或 candidates。

    返回：
        list[dict[str, Any]]：标准化选项列表，每项包含 option 和 text。
    """
    options = pick_first(sample, OPTION_KEYS, [])
    if not options and isinstance(sample.get("questions"), list) and sample["questions"]:
        first_question = sample["questions"][0]
        if isinstance(first_question, dict):
            options = pick_first(first_question, OPTION_KEYS, [])
    if isinstance(options, dict):
        return [{"option": _choice(key, str(key)), "text": str(value)} for key, value in options.items()]
    if isinstance(options, list):
        normalized = []
        for index, option in enumerate(options):
            if isinstance(option, dict):
                label = pick_first(option, ("option", "label", "key"), chr(65 + index))
                text = pick_first(option, ("text", "content", "value"), option)
                normalized.append({"option": _choice(label, str(label)), "text": str(text)})
            else:
                normalized.append({"option": chr(65 + index), "text": str(option)})
        return normalized
    return []


def _coerce_json_payload(raw: Any) -> Any:
    """功能：把任意原始输出尽量解析成 JSON 数据。

    参数：
        raw：待解析的任意值，可以是字典、列表、Markdown JSON 代码块、混合文本或普通文本。

    返回：
        Any：解析出的 JSON 数据；解析失败时返回原始文本字符串。
    """
    if isinstance(raw, (dict, list)):
        return raw

    text = str(raw).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.extend([text, _first_balanced_json(text)])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return text


def _normalize_option_scores(
    raw: Any,
    options: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """功能：把多种选项打分格式规范成统一 votes 列表。

    参数：
        raw：原始选项分数，可以是字典、列表、JSON 字符串或包含 option_scores/option_analysis/votes 的字典。
        options：标准化选项列表，用于补齐没有被模型打分的选项文本。

    返回：
        list[dict[str, Any]]：规范化选项分数列表，每项包含 option、text、score、reason、eliminated 等字段。
    """
    payload = _coerce_json_payload(raw)
    if isinstance(payload, dict):
        payload = payload.get("option_scores") or payload.get("option_analysis") or payload.get("votes") or payload

    option_text_by_label = {
        str(item.get("option", "")).strip(): str(item.get("text", ""))
        for item in (options or [])
        if item.get("option")
    }
    score_by_label: dict[str, dict[str, Any]] = {}

    if isinstance(payload, dict):
        for option, score in payload.items():
            label = _choice(option, str(option))
            score_by_label[label] = {
                "option": label,
                "text": option_text_by_label.get(label, ""),
                "score": _safe_float(score),
                "reason": "",
                "eliminated": False,
            }
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                label = _choice(pick_first(item, ("option", "label", "key"), chr(65 + index)), chr(65 + index))
                score_by_label[label] = {
                    "option": label,
                    "text": str(pick_first(item, ("text", "content"), option_text_by_label.get(label, ""))),
                    "score": _safe_float(pick_first(item, ("score", "vote", "confidence"), 0)),
                    "correctness_score": _optional_float(
                        pick_first(item, ("correctness_score", "correct_score", "support_score"), None)
                    ),
                    "error_score": _optional_float(
                        pick_first(item, ("error_score", "wrong_score", "contradiction_score"), None)
                    ),
                    "reason": str(pick_first(item, ("reason", "evidence", "comment"), "")),
                    "eliminated": bool(item.get("eliminated", False)),
                }

    for label, text in option_text_by_label.items():
        score_by_label.setdefault(
            label,
            {"option": label, "text": text, "score": 0.0, "reason": "No score returned.", "eliminated": False},
        )
    return list(score_by_label.values())


def _safe_float(value: Any) -> float:
    """功能：安全地把值转换成浮点数。

    参数：
        value：待转换的任意值。

    返回：
        float：转换成功的浮点数；失败时返回 0.0。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    """功能：把可选值安全转换为浮点数。

    参数：
        value：待转换的任意值；None 表示缺失。

    返回：
        float | None：转换成功时返回浮点数；值缺失或转换失败时返回 None。
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_balanced_json(text: str) -> str:
    """功能：从混合文本中截取第一个花括号平衡的 JSON 对象字符串。

    参数：
        text：可能包含 JSON 对象的文本。

    返回：
        str：第一个平衡 JSON 对象字符串；没有找到时返回空字符串。
    """
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _question_target(question_text: str) -> str:
    """功能：判断选择题目标是选最佳项还是错误项。

    参数：
        question_text：题面文本或样本 JSON 字符串。

    返回：
        str："incorrect" 表示选择不正确/错误项；"best" 表示选择最佳/合理项。
    """
    text = question_text.lower()
    negative_markers = ("不正确", "不恰当", "不符合", "错误", "有误", "incorrect", "not true", "false")
    return "incorrect" if any(marker in text for marker in negative_markers) else "best"


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
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else default


def _main_text(data: dict[str, Any]) -> str:
    """功能：从结构化结果中挑选最适合作为解释文本的字段。

    参数：
        data：结构化结果字典。

    返回：
        str：优先级最高的非空文本；没有时返回空字符串。
    """
    for key in ("answer", "candidate_answer", "revised_prediction", "output", "reason", "prediction"):
        value = data.get(key)
        if value:
            text = str(value).strip()
            if key == "prediction" and _choice(text, "") and data.get("reason"):
                continue
            if _looks_like_meta_answer(text):
                continue
            return text
    return ""


def _enforce_task1_like(data: dict[str, Any], sample: dict[str, Any]) -> None:
    """功能：为 task1/1-1/1-2/1-3 类型结果补齐词语、句子和选择题字段。

    参数：
        data：待补齐的结构化结果字典，函数会就地修改。
        sample：当前题目的原始样本，用于读取 qa_words、qa_sents 和选项。
    """
    text = _main_text(data)
    words = sample.get("qa_words", [])
    sents = sample.get("qa_sents", [])
    if not isinstance(data.get("ans_qa_words"), dict):
        data["ans_qa_words"] = {}
    if not isinstance(data.get("ans_qa_sents"), dict):
        data["ans_qa_sents"] = {}
    for word in words if isinstance(words, list) else []:
        data["ans_qa_words"].setdefault(str(word), text)
    for sent in sents if isinstance(sents, list) else []:
        data["ans_qa_sents"].setdefault(str(sent), text)
    data["choose_id"] = _choice(data.get("choose_id") or data.get("prediction") or data.get("answer"), "")
    if not data["choose_id"] and extract_options(sample):
        data["choose_id"] = "A"


def _looks_like_meta_answer(text: str) -> bool:
    """功能：判断文本是否像任务说明或格式分析，而不是实际答案。

    参数：
        text：待判断文本。

    返回：
        bool：像任务说明/格式分析时返回 True，否则返回 False。
    """
    markers = ("题干任务", "任务说明", "输出字段", "必须输出", "JSON", "符合任务本质", "要求解释词语")
    return any(marker in text for marker in markers)


def _enforce_allusion(data: dict[str, Any]) -> None:
    """功能：为典故识别结果补齐 flag 和 answer。

    参数：
        data：待补齐的结构化结果字典，函数会就地修改。
    """
    text = _main_text(data)
    flag = data.get("flag")
    if flag not in (0, 1, "0", "1"):
        flag = 0 if _means_no_allusion(text) else 1 if text else 0
    data["flag"] = int(flag)
    data["answer"] = "" if data["flag"] == 0 else str(data.get("answer") or text)


def _enforce_analogy_answer(data: dict[str, Any], sample: dict[str, Any]) -> None:
    """功能：为类比推理结果补齐 answer 列表和 prediction。

    参数：
        data：待补齐的结构化结果字典，函数会就地修改。
        sample：当前题目的原始样本，用于根据 que 中的空格数量判断答案个数。
    """
    answer = data.get("answer") or data.get("prediction") or []
    if isinstance(answer, str):
        answer = _split_answer(answer)
    if not isinstance(answer, list):
        answer = [str(answer)]
    need = max(1, len(re.findall(r"_{4,}", str(sample.get("que", "")))))
    clean = [str(item).strip() for item in answer if str(item).strip()]
    while len(clean) < need:
        clean.append("")
    data["answer"] = clean[:need]
    data.setdefault("prediction", "；".join(clean[:need]))


def _enforce_choice_answer(data: dict[str, Any]) -> None:
    """功能：为选择题结果补齐唯一 A/B/C/D 答案。

    参数：
        data：待补齐的结构化结果字典，函数会就地修改。
    """
    choice = _choice(data.get("answer") or data.get("prediction"), "")
    if not choice and isinstance(data.get("votes"), list) and data["votes"]:
        choice = _choice(data["votes"][0].get("option"), "")
    data["answer"] = choice or "A"
    data["prediction"] = data["answer"]


def _means_no_allusion(text: str) -> bool:
    """功能：判断文本是否表达“没有典故”。

    参数：
        text：待判断文本。

    返回：
        bool：表示无典故时返回 True，否则返回 False。
    """
    stripped = text.strip()
    if stripped in {"无", "没有", "未见", "无典故"}:
        return True
    return any(
        word in stripped
        for word in ("没有典故", "不含典故", "未见典故", "无明显典故", "没有用典", "未用典", "无出处")
    )


def _split_answer(text: str) -> list[str]:
    """功能：把类比推理的字符串答案拆成列表。

    参数：
        text：待拆分答案，可以是 JSON、逗号/顿号/分号/换行分隔文本。

    返回：
        list[str]：拆分后的答案列表。
    """
    stripped = text.strip()
    if not stripped:
        return []
    parsed = _coerce_json_payload(stripped)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, dict):
        value = parsed.get("answer") or parsed.get("prediction") or []
        return value if isinstance(value, list) else [str(value)]
    return [part.strip() for part in re.split(r"[，,、;\n]", stripped) if part.strip()]
