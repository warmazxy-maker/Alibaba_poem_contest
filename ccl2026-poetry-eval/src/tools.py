"""Shared utilities plus the core Tools functions from the workflow spec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .prompts import TASK_ALIASES, get_task_prompt


ID_KEYS = ("id", "qid", "question_id", "sample_id")
LABEL_KEYS = ("label", "answer", "gold", "target")
PREDICTION_KEYS = ("prediction", "answer", "output")
TASK_KEYS = ("task_id", "subtask", "category", "task", "task_type", "type")
QUESTION_KEYS = ("question", "query", "prompt", "instruction", "stem")
OPTION_KEYS = ("options", "choices", "candidates")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def ensure_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "examples", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise TypeError("Expected a JSON array or an object containing a data array.")


def pick_first(mapping: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def get_sample_id(sample: dict[str, Any], index: int) -> str:
    return str(pick_first(sample, ID_KEYS, index))


def normalize_prediction(raw: Any) -> str:
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
    return {
        "id": get_sample_id(sample, index),
        "prediction": normalize_prediction(raw_prediction),
    }


def extract_label(sample: dict[str, Any]) -> str:
    return str(pick_first(sample, LABEL_KEYS, "")).strip()


def extract_prediction(row: dict[str, Any]) -> str:
    return normalize_prediction(pick_first(row, PREDICTION_KEYS, ""))


def dispatch_sub_task_prompt(
    sample: dict[str, Any],
    task_id: str | None = None,
) -> dict[str, Any]:
    """Tool 1: dynamically choose the task prompt and workflow route."""
    resolved_task_id = task_id or infer_task_id(sample)
    task = get_task_prompt(resolved_task_id)
    return {
        "task_id": task.task_id,
        "track": task.track,
        "task_name": task.name_zh,
        "task_name_en": task.name_en,
        "task_description": task.description,
        "stages": list(task.stages),
        "output_schema": task.output_contract,
    }


def execute_debate_loop(
    parser_agent: Any,
    critic_agent: Any,
    sample: dict[str, Any],
    task_plan: dict[str, Any],
    context: dict[str, Any],
    rounds: int = 2,
) -> dict[str, Any]:
    """Tool 2: control multi-round Parser/Critic debate."""
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
        parser_result = parser_agent.run_stage(sample, parser_stage, debate_context, task_plan)
        debate_context[f"parser_round_{round_index + 1}"] = parser_result.content

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
    """Tool 3: parse and normalize the 2-1 analogy alignment matrix."""
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


def eliminate_and_vote(option_scores: Any) -> dict[str, Any]:
    """Tool 4: eliminate weak options and choose the highest-scoring one."""
    scores = _normalize_option_scores(option_scores)
    if not scores:
        return {"prediction": "", "votes": [], "reason": "No option scores were available."}

    active_scores = [item for item in scores if not item.get("eliminated")]
    candidates = active_scores or scores
    winner = max(candidates, key=lambda item: (float(item.get("score", 0)), str(item.get("option", ""))))
    sorted_votes = sorted(scores, key=lambda item: float(item.get("score", 0)), reverse=True)
    return {
        "prediction": str(winner.get("option", "")).strip(),
        "votes": sorted_votes,
        "reason": str(winner.get("reason", "")),
    }


def enforce_json_schema(payload: Any, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tool 5: final JSON bottoming and light schema enforcement."""
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

    return data


def infer_task_id(sample: dict[str, Any]) -> str:
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
    options = pick_first(sample, OPTION_KEYS, [])
    if isinstance(options, dict):
        return [{"option": str(key), "text": str(value)} for key, value in options.items()]
    if isinstance(options, list):
        normalized = []
        for index, option in enumerate(options):
            if isinstance(option, dict):
                label = pick_first(option, ("option", "label", "key"), chr(65 + index))
                text = pick_first(option, ("text", "content", "value"), option)
                normalized.append({"option": str(label), "text": str(text)})
            else:
                normalized.append({"option": chr(65 + index), "text": str(option)})
        return normalized
    return []


def _coerce_json_payload(raw: Any) -> Any:
    if isinstance(raw, dict | list):
        return raw

    text = str(raw).strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _normalize_option_scores(raw: Any) -> list[dict[str, Any]]:
    payload = _coerce_json_payload(raw)
    if isinstance(payload, dict):
        payload = payload.get("option_scores") or payload.get("option_analysis") or payload.get("votes") or payload

    if isinstance(payload, dict):
        return [
            {
                "option": str(option),
                "score": _safe_float(score),
                "reason": "",
                "eliminated": False,
            }
            for option, score in payload.items()
        ]

    if isinstance(payload, list):
        scores = []
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                scores.append(
                    {
                        "option": str(pick_first(item, ("option", "label", "key"), chr(65 + index))).strip(),
                        "score": _safe_float(pick_first(item, ("score", "vote", "confidence"), 0)),
                        "reason": str(pick_first(item, ("reason", "evidence", "comment"), "")),
                        "eliminated": bool(item.get("eliminated", False)),
                    }
                )
        return scores

    return []


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
