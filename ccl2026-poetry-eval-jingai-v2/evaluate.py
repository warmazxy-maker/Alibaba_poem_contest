"""Local lightweight evaluator for generated official-format submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import settings
from src.official_data import OfficialDataIO, TASK_NAMES
from src.tools import (
    ensure_list,
    extract_label,
    extract_prediction,
    get_sample_id,
    load_json,
    save_json,
)


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return settings.root_dir / resolved


def evaluate(gold_items: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_by_id = {
        get_sample_id(row, index): extract_prediction(row)
        for index, row in enumerate(predictions)
    }

    total = 0
    correct = 0
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []

    for index, gold in enumerate(gold_items):
        sample_id = get_sample_id(gold, index)
        label = extract_label(gold)
        if not label:
            continue

        total += 1
        prediction = prediction_by_id.get(sample_id, "")
        if not prediction:
            missing.append(sample_id)
        if prediction == label:
            correct += 1
        else:
            mismatches.append({"id": sample_id, "gold": label, "prediction": prediction})

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "missing": missing,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def evaluate_official(gold_data: dict[str, list[dict[str, Any]]], pred: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for task_name in TASK_NAMES:
        gold_rows = gold_data[task_name]
        pred_rows = {}
        for row in pred.get(task_name, []):
            try:
                pred_rows[int(row["idx"])] = row
            except (KeyError, TypeError, ValueError):
                continue
        total = 0
        correct = 0
        for gold in gold_rows:
            try:
                idx = int(gold.get("idx", total))
            except (TypeError, ValueError):
                continue
            row = pred_rows.get(idx)
            if row is None:
                continue
            if task_name == "task1":
                total += 1
                if row.get("choose_id") == gold.get("choose_id"):
                    correct += 1
            elif task_name == "task2":
                continue
            elif task_name == "task3":
                total += 1
                if row.get("answer") == gold.get("answer"):
                    correct += 1
            elif task_name == "task4":
                questions = gold.get("questions") or [{}]
                answer = questions[0].get("answer")
                total += 1
                if row.get("answer") == answer:
                    correct += 1
        metrics[task_name] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else None,
        }
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate submission exact-match accuracy.")
    parser.add_argument("--gold", default="data/train.json", help="Gold data path.")
    parser.add_argument("--pred", default="data/submission.json", help="Prediction JSON path.")
    parser.add_argument("--output", default="", help="Optional metrics output path.")
    parser.add_argument("--official", action="store_true", help="Gold is an official task directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.official:
        gold = OfficialDataIO(train_dir=args.gold).load_all_train()
        predictions = load_json(resolve_project_path(args.pred))
        metrics = evaluate_official(gold, predictions)
    else:
        gold = ensure_list(load_json(resolve_project_path(args.gold)))
        predictions = ensure_list(load_json(resolve_project_path(args.pred)))
        metrics = evaluate(gold, predictions)

    if args.output:
        save_json(metrics, resolve_project_path(args.output))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
