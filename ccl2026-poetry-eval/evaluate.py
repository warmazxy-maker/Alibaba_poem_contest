"""Local exact-match evaluator for generated submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import settings
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
            mismatches.append(
                {
                    "id": sample_id,
                    "gold": label,
                    "prediction": prediction,
                }
            )

    accuracy = correct / total if total else None
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "missing": missing,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate submission exact-match accuracy.")
    parser.add_argument("--gold", default="data/train.json", help="Gold data path.")
    parser.add_argument("--pred", default="data/submission.json", help="Prediction JSON path.")
    parser.add_argument("--output", default="", help="Optional metrics output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gold = ensure_list(load_json(resolve_project_path(args.gold)))
    predictions = ensure_list(load_json(resolve_project_path(args.pred)))
    metrics = evaluate(gold, predictions)

    if args.output:
        save_json(metrics, resolve_project_path(args.output))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
