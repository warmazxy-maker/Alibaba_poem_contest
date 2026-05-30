"""Prompt templates and task matrix used by the agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Any


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str

    def render(self, **kwargs: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": Template(self.system).safe_substitute(**kwargs)},
            {"role": "user", "content": Template(self.user).safe_substitute(**kwargs)},
        ]


@dataclass(frozen=True)
class TaskPrompt:
    task_id: str
    track: str
    name_zh: str
    name_en: str
    description: str
    stages: tuple[str, ...]
    output_contract: dict[str, Any]


COMMON_SYSTEM = (
    "你是严谨的中国古典诗词评测助手。请基于题目、诗句、选项与证据作答，"
    "避免编造；最终答案必须方便转成 JSON。"
)


TASK_PROMPTS: dict[str, TaskPrompt] = {
    "1-1": TaskPrompt(
        task_id="1-1",
        track="basic_understanding",
        name_zh="字词理解",
        name_en="Word-level Understanding",
        description="解释古诗词中词语或短语的含义。",
        stages=("evoker_background", "parser_raw_extract", "critic_debate", "umpire_json"),
        output_contract={"required": ["prediction", "reason", "evidence"]},
    ),
    "1-2": TaskPrompt(
        task_id="1-2",
        track="basic_understanding",
        name_zh="诗句理解",
        name_en="Sentence-level Understanding",
        description="解释整句诗的语义以及表达内容。",
        stages=("evoker_background", "parser_raw_extract", "critic_debate", "umpire_json"),
        output_contract={"required": ["prediction", "reason", "evidence"]},
    ),
    "1-3": TaskPrompt(
        task_id="1-3",
        track="basic_understanding",
        name_zh="情感理解",
        name_en="Emotion Recognition",
        description="判断诗人通过诗句表达的情感倾向。",
        stages=("evoker_background", "parser_raw_extract", "critic_debate", "umpire_json"),
        output_contract={"required": ["prediction", "reason", "evidence"]},
    ),
    "1-4": TaskPrompt(
        task_id="1-4",
        track="basic_understanding",
        name_zh="典故识别",
        name_en="Allusion Identification",
        description="识别诗句中的典故，并进行解释。",
        stages=("evoker_background", "parser_raw_extract", "critic_debate", "umpire_json"),
        output_contract={"required": ["prediction", "reason", "evidence"]},
    ),
    "2-1": TaskPrompt(
        task_id="2-1",
        track="analogy_reasoning",
        name_zh="古诗词类比",
        name_en="Analogy Reasoning",
        description="从诗句中抽取隐含关系或语义对应，完成类比推理或填空。",
        stages=("evoker_bidirectional", "parser_relation", "umpire_matrix_align"),
        output_contract={"required": ["prediction", "reason", "analogy_matrix"]},
    ),
    "2-2": TaskPrompt(
        task_id="2-2",
        track="critical_analysis",
        name_zh="古诗词辨析",
        name_en="Critical Analysis",
        description="根据诗词内容与语境，对多个选项进行分析，选择最合理或不正确的一项。",
        stages=("evoker_background", "parser_option_analysis", "critic_option_scoring", "umpire_vote"),
        output_contract={"required": ["prediction", "reason", "votes"]},
    ),
}


TASK_ALIASES: dict[str, str] = {
    "1-1": "1-1",
    "word": "1-1",
    "word-level": "1-1",
    "字词": "1-1",
    "词语": "1-1",
    "短语": "1-1",
    "1-2": "1-2",
    "sentence": "1-2",
    "sentence-level": "1-2",
    "诗句": "1-2",
    "句意": "1-2",
    "1-3": "1-3",
    "emotion": "1-3",
    "情感": "1-3",
    "情绪": "1-3",
    "1-4": "1-4",
    "allusion": "1-4",
    "典故": "1-4",
    "2-1": "2-1",
    "analogy": "2-1",
    "类比": "2-1",
    "推理": "2-1",
    "关系": "2-1",
    "2-2": "2-2",
    "critical": "2-2",
    "analysis": "2-2",
    "辨析": "2-2",
    "选择题": "2-2",
    "单选": "2-2",
    "选项": "2-2",
}


TEMPLATES: dict[str, PromptTemplate] = {
    "evoker_background": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "任务：$task_id $task_name\n"
            "任务说明：$task_description\n\n"
            "请为下面题目生成必要背景，包括作者、时代、诗词语境、关键意象、"
            "可能涉及的典故或常识。只写与作答相关的信息。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "请输出 JSON，字段包含 background、keywords、risks。"
        ),
    ),
    "evoker_bidirectional": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "任务：$task_id $task_name\n"
            "任务说明：$task_description\n\n"
            "这是类比推理任务。请分别从左到右、从右到左生成双向背景，"
            "关注两首诗、两个短语或两个对象之间的语义关系。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "请输出 JSON，字段包含 source_background、target_background、relation_hints。"
        ),
    ),
    "parser_raw_extract": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "任务：$task_id $task_name\n"
            "任务说明：$task_description\n\n"
            "请从原题中抽取诗词文本、问题目标、关键词、候选答案与直接证据。"
            "不要急于最终作答。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "请输出 JSON，字段包含 poem、question、target_span、options、evidence。"
        ),
    ),
    "parser_relation": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "任务：$task_id $task_name\n"
            "任务说明：$task_description\n\n"
            "请解构两诗、两词或两个对象之间的语义关系，形成 2-1 类比矩阵。"
            "关注对象、属性、动作、情感、意象、修辞、位置等可对齐维度。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "请输出 JSON，字段包含 analogy_matrix 和 candidate_prediction。"
        ),
    ),
    "parser_option_analysis": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "任务：$task_id $task_name\n"
            "任务说明：$task_description\n\n"
            "请逐一分析单选题选项，判断每个选项与诗词内容、语境、情感、"
            "典故或修辞是否一致。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "请输出 JSON，字段包含 option_analysis，每项包含 option、score、reason、eliminated。"
        ),
    ),
    "critic_debate": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "任务：$task_id $task_name\n"
            "任务说明：$task_description\n\n"
            "请作为 Critic Agent 复核 Parser 的抽取与推理，指出漏洞、"
            "歧义和反证，并给出是否支持当前候选答案。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "请输出 JSON，字段包含 objections、supporting_evidence、revised_prediction。"
        ),
    ),
    "critic_option_scoring": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "任务：$task_id $task_name\n"
            "任务说明：$task_description\n\n"
            "请作为 Critic Agent 检查 Parser 对各选项的分析，给每个选项打分。"
            "分数越高表示越可能是最终答案；明显错误的选项应标记 eliminated=true。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "请输出 JSON，字段包含 option_scores，每项包含 option、score、reason、eliminated。"
        ),
    ),
    "umpire_json": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "任务：$task_id $task_name\n"
            "任务说明：$task_description\n\n"
            "请作为 Umpire Agent 综合背景、抽取结果与辩论意见，做最终裁决。"
            "必须输出合法 JSON。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "输出字段：prediction、reason、evidence。"
        ),
    ),
    "umpire_matrix_align": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "任务：$task_id $task_name\n"
            "任务说明：$task_description\n\n"
            "请作为 Umpire Agent 根据类比矩阵完成对齐裁决，给出最终答案。"
            "必须输出合法 JSON。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "输出字段：prediction、reason、analogy_matrix。"
        ),
    ),
    "umpire_vote": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "任务：$task_id $task_name\n"
            "任务说明：$task_description\n\n"
            "请作为 Umpire Agent 根据各轮选项分析、打分和淘汰结果投票决策。"
            "最终 prediction 只能是唯一选项，如 A、B、C、D。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "输出字段：prediction、reason、votes。"
        ),
    ),
}


def get_task_prompt(task_id: str) -> TaskPrompt:
    if task_id not in TASK_PROMPTS:
        raise KeyError(f"Unknown task id: {task_id}")
    return TASK_PROMPTS[task_id]


def render_prompt(name: str, **kwargs: str) -> list[dict[str, str]]:
    if name not in TEMPLATES:
        raise KeyError(f"Unknown prompt template: {name}")
    return TEMPLATES[name].render(**kwargs)
