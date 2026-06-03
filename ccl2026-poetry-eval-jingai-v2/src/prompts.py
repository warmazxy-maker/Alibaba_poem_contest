"""Prompt templates and task matrix used by the agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Any


@dataclass(frozen=True)
class PromptTemplate:
    """功能：保存一个阶段 prompt 的 system 和 user 模板。

    字段：
        system：系统提示词模板。
        user：用户提示词模板，可使用 string.Template 风格变量。
    """

    system: str
    user: str

    def render(self, **kwargs: str) -> list[dict[str, str]]:
        """功能：把模板变量替换为具体内容，并生成 OpenAI chat 消息列表。

        参数：
            **kwargs：模板变量键值对，如 sample、context、task_id、task_name、task_description。

        返回：
            list[dict[str, str]]：包含 system 和 user 两条消息的 chat 格式列表。
        """
        return [
            {"role": "system", "content": Template(self.system).safe_substitute(**kwargs)},
            {"role": "user", "content": Template(self.user).safe_substitute(**kwargs)},
        ]


@dataclass(frozen=True)
class TaskPrompt:
    """功能：保存一个内部任务的元信息、执行轨道和输出契约。

    字段：
        task_id：内部任务编号，如 1-1、1-4、2-1、2-2。
        track：任务轨道，如 basic_understanding、analogy_reasoning、critical_analysis。
        name_zh：中文任务名。
        name_en：英文任务名。
        description：任务说明。
        stages：该任务默认执行的 prompt 阶段序列。
        output_contract：该任务期望输出的字段约束。
    """

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
            "请从原题中抽取诗词文本、问题目标、关键词、候选答案与直接证据，"
            "并给出第一版候选答案，供 Critic 复核。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "请输出 JSON，字段包含 poem、question、target_span、options、"
            "ans_qa_words、ans_qa_sents、choose_id、candidate_answer、evidence、confidence。"
        ),
    ),
    "task1_official_parse": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "你正在处理官方 task1。必须直接完成题目，不要解释任务规则。\n\n"
            "输入样本字段说明：\n"
            "- title：诗题\n"
            "- author：作者\n"
            "- content：诗歌全文\n"
            "- qa_words：需要解释的词语列表\n"
            "- qa_sents：需要解释的诗句列表\n"
            "- choose：情感理解选择题选项\n\n"
            "样本：\n$sample\n\n"
            "请输出合法 JSON，且只输出 JSON：\n"
            "{\n"
            "  \"ans_qa_words\": {\"词语原文\": \"该词在本诗语境中的具体含义\"},\n"
            "  \"ans_qa_sents\": {\"诗句原文\": \"该诗句的现代汉语意思\"},\n"
            "  \"choose_id\": \"A/B/C/D 中最符合全诗情感的一项\",\n"
            "  \"reason\": \"选择该情感选项的简短依据\",\n"
            "  \"evidence\": [\"诗中可作为依据的原句\"]\n"
            "}\n\n"
            "硬性要求：ans_qa_words 的键必须完全等于 qa_words 中的词；"
            "ans_qa_sents 的键必须完全等于 qa_sents 中的句子；"
            "不要把任务说明、评分标准、背景分析写进词义或句意。"
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
            "请输出 JSON，字段包含 analogy_matrix、answer、candidate_prediction、confidence。"
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
            "请输出 JSON，字段包含 option_analysis、candidate_answer、prediction，"
            "option_analysis 每项包含 option、score、reason、eliminated。"
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
            "请输出 JSON，字段包含 objections、supporting_evidence、revised_prediction、"
            "revised_ans_qa_words、revised_ans_qa_sents、revised_choose_id、confidence。"
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
            "输出字段：prediction、reason、evidence。若样本包含 qa_words、qa_sents 或 choose，"
            "还必须输出 ans_qa_words、ans_qa_sents、choose_id。"
        ),
    ),
    "task1_official_final": PromptTemplate(
        system=COMMON_SYSTEM,
        user=(
            "你是官方 task1 的最终裁判。请根据样本和上下文，给出最终答案。"
            "不要输出任务分析，不要复述规则，只输出合法 JSON。\n\n"
            "样本：\n$sample\n\n"
            "已有上下文：\n$context\n\n"
            "最终 JSON 格式必须是：\n"
            "{\n"
            "  \"ans_qa_words\": {\"词语原文\": \"词语在本诗中的具体含义\"},\n"
            "  \"ans_qa_sents\": {\"诗句原文\": \"诗句现代汉语意思\"},\n"
            "  \"choose_id\": \"A/B/C/D\",\n"
            "  \"reason\": \"不超过80字的理由\",\n"
            "  \"evidence\": [\"依据原句\"]\n"
            "}\n\n"
            "检查要求：\n"
            "1. ans_qa_words 必须逐个解释 qa_words，不允许所有词共用同一段泛化文字。\n"
            "2. ans_qa_sents 必须逐句翻译 qa_sents，不允许写任务说明。\n"
            "3. choose_id 只能是 A、B、C、D 中一个。\n"
            "4. 如果上下文里有泛化任务说明，应忽略它，按诗歌内容重新作答。"
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
            "输出字段：prediction、answer、reason、analogy_matrix。answer 必须是字符串数组。"
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
            "输出字段：prediction、answer、reason、votes。prediction 和 answer 必须是同一个唯一选项。"
        ),
    ),
}


def get_task_prompt(task_id: str) -> TaskPrompt:
    """功能：根据内部任务编号读取任务配置。

    参数：
        task_id：内部任务编号，如 1-1、1-4、2-1、2-2。

    返回：
        TaskPrompt：对应任务的配置对象。
    """
    if task_id not in TASK_PROMPTS:
        raise KeyError(f"Unknown task id: {task_id}")
    return TASK_PROMPTS[task_id]


def render_prompt(name: str, **kwargs: str) -> list[dict[str, str]]:
    """功能：根据阶段名称渲染 prompt。

    参数：
        name：阶段 prompt 名称，如 parser_raw_extract、umpire_vote。
        **kwargs：模板变量键值对。

    返回：
        list[dict[str, str]]：OpenAI chat 格式消息列表。
    """
    if name not in TEMPLATES:
        raise KeyError(f"Unknown prompt template: {name}")
    return TEMPLATES[name].render(**kwargs)
