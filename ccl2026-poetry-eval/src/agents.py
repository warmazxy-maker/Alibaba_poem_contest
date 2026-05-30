"""Agent matrix definitions: Evoker, Parser, Critic, and Umpire."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm_client import LLMClient
from .prompts import render_prompt


@dataclass(frozen=True)
class AgentResult:
    name: str
    stage: str
    content: str
    data: dict[str, Any]
    metadata: dict[str, Any]


class BaseAgent:
    role_name = "Base Agent"

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    @property
    def name(self) -> str:
        return self.role_name

    def run_stage(
        self,
        sample: dict[str, Any],
        stage: str,
        context: dict[str, Any] | None = None,
        task_plan: dict[str, Any] | None = None,
    ) -> AgentResult:
        plan = task_plan or {}
        messages = render_prompt(
            stage,
            sample=json.dumps(sample, ensure_ascii=False, indent=2),
            context=json.dumps(context or {}, ensure_ascii=False, indent=2),
            task_id=str(plan.get("task_id", "")),
            task_name=str(plan.get("task_name", "")),
            task_description=str(plan.get("task_description", "")),
        )
        content = self.client.chat(messages)
        return AgentResult(
            name=self.name,
            stage=stage,
            content=content,
            data=_parse_json_object(content),
            metadata={"task_id": plan.get("task_id", ""), "track": plan.get("track", "")},
        )

    def run(
        self,
        sample: dict[str, Any],
        context: dict[str, Any] | None = None,
        task_plan: dict[str, Any] | None = None,
    ) -> AgentResult:
        stages = (task_plan or {}).get("stages") or ()
        if not stages:
            raise ValueError(f"{self.name} needs an explicit stage or task plan.")
        return self.run_stage(sample, stages[0], context, task_plan)


class EvokerAgent(BaseAgent):
    role_name = "Evoker Agent"

    def background(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        stage = "evoker_bidirectional" if task_plan.get("task_id") == "2-1" else "evoker_background"
        return self.run_stage(sample, stage, context, task_plan)


class ParserAgent(BaseAgent):
    role_name = "Parser Agent"


class CriticAgent(BaseAgent):
    role_name = "Critic Agent"


class UmpireAgent(BaseAgent):
    role_name = "Umpire Agent"


@dataclass(frozen=True)
class AgentMatrix:
    evoker: EvokerAgent
    parser: ParserAgent
    critic: CriticAgent
    umpire: UmpireAgent


def default_agent_matrix(client: LLMClient | None = None) -> AgentMatrix:
    shared_client = client or LLMClient()
    return AgentMatrix(
        evoker=EvokerAgent(shared_client),
        parser=ParserAgent(shared_client),
        critic=CriticAgent(shared_client),
        umpire=UmpireAgent(shared_client),
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
