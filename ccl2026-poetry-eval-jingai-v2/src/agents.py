"""Agent matrix definitions: Evoker, Parser, Critic, and Umpire."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .llm_client import LLMClient
from .prompts import render_prompt


@dataclass(frozen=True)
class AgentResult:
    """功能：保存一次 Agent 调用的完整结果。

    字段：
        name：Agent 角色名称，如 Evoker Agent、Parser Agent。
        stage：本次调用使用的 prompt 阶段名称。
        content：模型返回的原始文本。
        data：从 content 中解析出的 JSON 字典；解析失败时为空字典。
        metadata：附加元数据，当前包含 task_id 和 track。
    """

    name: str
    stage: str
    content: str
    data: dict[str, Any]
    metadata: dict[str, Any]


class BaseAgent:
    """功能：所有 Agent 的基类，负责渲染 prompt、调用模型并解析 JSON 结果。"""

    role_name = "Base Agent"

    def __init__(self, client: LLMClient | None = None) -> None:
        """功能：初始化 Agent。

        参数：
            client：模型客户端；为 None 时创建默认 LLMClient。
        """
        self.client = client or LLMClient()

    @property
    def name(self) -> str:
        """功能：读取 Agent 的角色名称。

        返回：
            str：Agent 角色名称。
        """
        return self.role_name

    def run_stage(
        self,
        sample: dict[str, Any],
        stage: str,
        context: dict[str, Any] | None = None,
        task_plan: dict[str, Any] | None = None,
    ) -> AgentResult:
        """功能：按指定阶段执行一次 Agent 推理。

        参数：
            sample：当前题目的原始样本字典。
            stage：prompt 阶段名称，用于从 prompts.py 中选择模板。
            context：当前流水线上下文字典，包含背景、解析、辩论等中间结果。
            task_plan：任务计划字典，包含 task_id、track、task_name、task_description 等。

        返回：
            AgentResult：包含模型原文、解析后的 JSON 数据和任务元信息。
        """
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
        """功能：按任务计划中的第一个阶段执行 Agent，主要作为通用兜底入口。

        参数：
            sample：当前题目的原始样本字典。
            context：当前流水线上下文字典。
            task_plan：任务计划字典，必须包含 stages。

        返回：
            AgentResult：第一个阶段的 Agent 调用结果。
        """
        stages = (task_plan or {}).get("stages") or ()
        if not stages:
            raise ValueError(f"{self.name} needs an explicit stage or task plan.")
        return self.run_stage(sample, stages[0], context, task_plan)


class EvokerAgent(BaseAgent):
    """功能：背景生成 Agent，负责补充作者、时代、语境、意象、典故等作答背景。"""

    role_name = "Evoker Agent"

    def background(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：根据任务类型生成单向或双向背景信息。

        参数：
            sample：当前题目的原始样本字典。
            context：当前流水线上下文字典。
            task_plan：任务计划字典；task_id 为 2-1 时使用双向背景阶段。

        返回：
            AgentResult：背景生成结果，data 通常包含 background、keywords、risks 或 relation_hints。
        """
        stage = "evoker_bidirectional" if task_plan.get("task_id") == "2-1" else "evoker_background"
        return self.run_stage(sample, stage, context, task_plan)


class ParserAgent(BaseAgent):
    """功能：解析 Agent，负责题面抽取、关系解析、选项分析，并给出第一版候选答案。"""

    role_name = "Parser Agent"

    def raw_extract(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：执行基础理解类任务的原题抽取与候选答案生成。

        参数：
            sample：当前题目的原始样本字典。
            context：当前流水线上下文字典。
            task_plan：任务计划字典。

        返回：
            AgentResult：解析结果，data 期望包含 poem、question、target_span、options、candidate_answer、evidence 等字段。
        """
        return self.run_stage(sample, "parser_raw_extract", context, task_plan)

    def official_task1(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：执行官方 task1 的逐词、逐句和情感选项解析。

        参数：
            sample：当前 task1 官方样本字典。
            context：当前流水线上下文字典。
            task_plan：任务计划字典。

        返回：
            AgentResult：task1 专用解析结果，data 期望包含 ans_qa_words、ans_qa_sents、choose_id。
        """
        return self.run_stage(sample, "task1_official_parse", context, task_plan)

    def relation(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：执行 2-1 类比推理任务的语义关系解析。

        参数：
            sample：当前题目的原始样本字典。
            context：当前流水线上下文字典。
            task_plan：任务计划字典。

        返回：
            AgentResult：关系解析结果，data 期望包含 analogy_matrix、answer、candidate_prediction 等字段。
        """
        return self.run_stage(sample, "parser_relation", context, task_plan)

    def option_analysis(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：执行 2-2 选择题辨析任务的逐项分析。

        参数：
            sample：当前题目的原始样本字典。
            context：当前流水线上下文字典，通常包含标准化后的 options。
            task_plan：任务计划字典。

        返回：
            AgentResult：选项分析结果，data 期望包含 option_analysis、candidate_answer、prediction 等字段。
        """
        return self.run_stage(sample, "parser_option_analysis", context, task_plan)


class CriticAgent(BaseAgent):
    """功能：批判 Agent，负责检查 Parser 的候选答案、指出漏洞并给出修正建议。"""

    role_name = "Critic Agent"

    def debate(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：对基础理解类任务的 Parser 结果进行辩论复核。

        参数：
            sample：当前题目的原始样本字典。
            context：当前流水线上下文字典，包含 Parser 的候选答案。
            task_plan：任务计划字典。

        返回：
            AgentResult：批判结果，data 期望包含 objections、supporting_evidence、revised_prediction 等字段。
        """
        return self.run_stage(sample, "critic_debate", context, task_plan)

    def option_scoring(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：对 2-2 选择题的各选项进行复核打分。

        参数：
            sample：当前题目的原始样本字典。
            context：当前流水线上下文字典，包含 Parser 的选项分析。
            task_plan：任务计划字典。

        返回：
            AgentResult：选项打分结果，data 期望包含 option_scores 列表。
        """
        return self.run_stage(sample, "critic_option_scoring", context, task_plan)


class UmpireAgent(BaseAgent):
    """功能：裁判 Agent，负责综合背景、解析和批判结果，输出最终 JSON。"""

    role_name = "Umpire Agent"

    def finalize_basic(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：裁决基础理解类任务的最终答案。

        参数：
            sample：当前题目的原始样本字典。
            context：当前流水线上下文字典，包含背景、解析和辩论历史。
            task_plan：任务计划字典。

        返回：
            AgentResult：最终结果，data 期望包含 prediction、reason、evidence，以及 task1 所需的 ans_qa_words、ans_qa_sents、choose_id。
        """
        return self.run_stage(sample, "umpire_json", context, task_plan)

    def finalize_official_task1(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：裁决官方 task1 的最终逐词、逐句和情感选项答案。

        参数：
            sample：当前 task1 官方样本字典。
            context：当前流水线上下文字典，包含 task1 专用解析和辩论结果。
            task_plan：任务计划字典。

        返回：
            AgentResult：task1 专用最终结果，data 期望包含 ans_qa_words、ans_qa_sents、choose_id。
        """
        return self.run_stage(sample, "task1_official_final", context, task_plan)

    def finalize_matrix(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：裁决 2-1 类比矩阵对齐任务的最终答案。

        参数：
            sample：当前题目的原始样本字典。
            context：当前流水线上下文字典，包含 analogy_matrix。
            task_plan：任务计划字典。

        返回：
            AgentResult：最终结果，data 期望包含 prediction、answer、reason、analogy_matrix。
        """
        return self.run_stage(sample, "umpire_matrix_align", context, task_plan)

    def finalize_vote(
        self,
        sample: dict[str, Any],
        context: dict[str, Any],
        task_plan: dict[str, Any],
    ) -> AgentResult:
        """功能：裁决 2-2 选择题辨析任务的唯一选项。

        参数：
            sample：当前题目的原始样本字典。
            context：当前流水线上下文字典，包含 tool_vote 和选项辩论历史。
            task_plan：任务计划字典。

        返回：
            AgentResult：最终结果，data 期望包含 prediction、answer、reason、votes。
        """
        return self.run_stage(sample, "umpire_vote", context, task_plan)


@dataclass(frozen=True)
class AgentMatrix:
    """功能：集中保存四个 Agent 实例，方便 Pipeline 统一调度。

    字段：
        evoker：背景生成 Agent。
        parser：题面解析和候选答案 Agent。
        critic：批判复核 Agent。
        umpire：最终裁决 Agent。
    """

    evoker: EvokerAgent
    parser: ParserAgent
    critic: CriticAgent
    umpire: UmpireAgent


def default_agent_matrix(client: LLMClient | None = None) -> AgentMatrix:
    """功能：使用同一个模型客户端创建默认 Agent 矩阵。

    参数：
        client：共享模型客户端；为 None 时创建默认 LLMClient。

    返回：
        AgentMatrix：包含 Evoker、Parser、Critic、Umpire 的默认矩阵。
    """
    shared_client = client or LLMClient()
    return AgentMatrix(
        evoker=EvokerAgent(shared_client),
        parser=ParserAgent(shared_client),
        critic=CriticAgent(shared_client),
        umpire=UmpireAgent(shared_client),
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    """功能：从模型文本中解析 JSON 对象。

    参数：
        text：模型返回的原始文本。

    返回：
        dict[str, Any]：解析出的 JSON 字典；解析失败或解析结果不是对象时返回空字典。
    """
    parsed = _extract_json_value(text)
    return parsed if isinstance(parsed, dict) else {}


def _extract_json_value(text: str) -> Any:
    """功能：从纯 JSON、Markdown JSON 代码块或混合文本中提取第一个 JSON 值。

    参数：
        text：待解析的模型返回文本。

    返回：
        Any：解析出的 JSON 值；解析失败时返回空字典。
    """
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(stripped)

    balanced = _first_balanced_json(stripped)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


def _first_balanced_json(text: str) -> str:
    """功能：在混合文本中查找第一个花括号平衡的 JSON 对象片段。

    参数：
        text：包含或可能包含 JSON 对象的文本。

    返回：
        str：第一个平衡 JSON 对象字符串；找不到时返回空字符串。
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
