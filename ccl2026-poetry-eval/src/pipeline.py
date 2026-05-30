"""Main control pipeline for generating poetry evaluation submissions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .agents import AgentMatrix, default_agent_matrix
from .config import Settings, settings
from .llm_client import LLMClient
from .tools import (
    dispatch_sub_task_prompt,
    eliminate_and_vote,
    enforce_json_schema,
    ensure_list,
    execute_debate_loop,
    extract_options,
    get_sample_id,
    load_json,
    parse_analogy_matrix,
    save_json,
)


class PoetryEvalPipeline:
    def __init__(
        self,
        cfg: Settings = settings,
        client: LLMClient | None = None,
        agents: AgentMatrix | None = None,
    ) -> None:
        self.cfg = cfg
        self.client = client or LLMClient(cfg)
        self.agents = agents or default_agent_matrix(self.client)

    def run_sample(self, sample: dict[str, Any], index: int) -> dict[str, Any]:
        task_plan = dispatch_sub_task_prompt(sample)
        context: dict[str, Any] = {
            "sample_id": get_sample_id(sample, index),
            "task_plan": task_plan,
            "options": extract_options(sample),
        }

        if task_plan["track"] == "basic_understanding":
            payload = self._run_basic_understanding(sample, task_plan, context)
        elif task_plan["track"] == "analogy_reasoning":
            payload = self._run_analogy_reasoning(sample, task_plan, context)
        elif task_plan["track"] == "critical_analysis":
            payload = self._run_critical_analysis(sample, task_plan, context)
        else:
            raise ValueError(f"Unsupported task track: {task_plan['track']}")

        final_payload = enforce_json_schema(payload, task_plan["output_schema"])
        final_payload["id"] = get_sample_id(sample, index)
        final_payload["task_id"] = task_plan["task_id"]
        return final_payload

    def run(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        source = resolve_project_path(input_path or self.cfg.test_path, self.cfg)
        target = resolve_project_path(output_path or self.cfg.submission_path, self.cfg)
        samples = ensure_list(load_json(source))
        if limit is not None:
            samples = samples[:limit]

        submission = [
            self.run_sample(sample, index)
            for index, sample in enumerate(samples)
        ]
        save_json(submission, target)
        return submission

    def _run_basic_understanding(
        self,
        sample: dict[str, Any],
        task_plan: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        evoked = self.agents.evoker.background(sample, context, task_plan)
        context["evoker_background"] = evoked.data or evoked.content

        parsed = self.agents.parser.run_stage(sample, "parser_raw_extract", context, task_plan)
        context["parser_raw_extract"] = parsed.data or parsed.content

        debate = execute_debate_loop(
            self.agents.parser,
            self.agents.critic,
            sample,
            task_plan,
            context,
            rounds=self.cfg.debate_rounds,
        )
        context["debate"] = debate["history"]

        judged = self.agents.umpire.run_stage(sample, "umpire_json", context, task_plan)
        return judged.data or {"prediction": judged.content, "reason": judged.content}

    def _run_analogy_reasoning(
        self,
        sample: dict[str, Any],
        task_plan: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        evoked = self.agents.evoker.background(sample, context, task_plan)
        context["evoker_bidirectional"] = evoked.data or evoked.content

        parsed = self.agents.parser.run_stage(sample, "parser_relation", context, task_plan)
        matrix = parse_analogy_matrix(parsed.data or parsed.content)
        context["analogy_matrix"] = matrix

        judged = self.agents.umpire.run_stage(sample, "umpire_matrix_align", context, task_plan)
        payload = judged.data or {"prediction": judged.content, "reason": judged.content}
        payload.setdefault("analogy_matrix", matrix)
        return payload

    def _run_critical_analysis(
        self,
        sample: dict[str, Any],
        task_plan: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        evoked = self.agents.evoker.background(sample, context, task_plan)
        context["evoker_background"] = evoked.data or evoked.content

        debate = execute_debate_loop(
            self.agents.parser,
            self.agents.critic,
            sample,
            task_plan,
            context,
            rounds=self.cfg.debate_rounds,
        )
        context["option_debate"] = debate["history"]

        option_scores = []
        for round_item in debate["history"]:
            critic_payload = round_item.get("critic")
            if isinstance(critic_payload, dict):
                option_scores.extend(critic_payload.get("option_scores", []))

        vote = eliminate_and_vote(option_scores)
        context["tool_vote"] = vote

        judged = self.agents.umpire.run_stage(sample, "umpire_vote", context, task_plan)
        payload = judged.data or vote
        payload.setdefault("prediction", vote.get("prediction", ""))
        payload.setdefault("votes", vote.get("votes", []))
        return payload


def resolve_project_path(path: str | Path, cfg: Settings = settings) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return cfg.root_dir / resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the poetry evaluation pipeline.")
    parser.add_argument("--input", default="data/test.json", help="Path to test data.")
    parser.add_argument("--output", default="data/submission.json", help="Path for submission JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--dry-run", action="store_true", help="Skip real model calls.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = LLMClient(settings, dry_run=args.dry_run)
    pipeline = PoetryEvalPipeline(settings, client=client)
    submission = pipeline.run(args.input, args.output, args.limit)
    print(f"Wrote {len(submission)} rows to {resolve_project_path(args.output)}")


if __name__ == "__main__":
    main()
