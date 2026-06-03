"""Main control pipeline for generating poetry evaluation submissions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents import AgentMatrix, default_agent_matrix
from .bad_cases import BadCaseRecorder
from .config import Settings, settings
from .llm_client import LLMClient
from .official_data import OfficialAdapters, OfficialDataIO, TASK_NAMES, fallback_row
from .official_schema import OfficialSubmissionValidator
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
    """功能：核心诗词评测业务流水线，负责调度 Agent 矩阵和 Tools 工具。"""

    def __init__(
        self,
        cfg: Settings = settings,
        client: LLMClient | None = None,
        agents: AgentMatrix | None = None,
    ) -> None:
        """功能：初始化核心业务流水线。

        参数：
            cfg：项目配置对象，提供路径、模型参数和辩论轮数。
            client：模型客户端；为 None 时根据 cfg 创建默认 LLMClient。
            agents：Agent 矩阵；为 None 时创建默认 Evoker、Parser、Critic、Umpire。
        """
        self.cfg = cfg
        self.client = client or LLMClient(cfg)
        self.agents = agents or default_agent_matrix(self.client)

    def run_sample(self, sample: dict[str, Any], index: int) -> dict[str, Any]:
        """功能：处理单条样本，并自动推断内部任务类型。

        参数：
            sample：当前题目的原始样本字典。
            index：样本在当前任务列表中的序号，用作缺省样本 id。

        返回：
            dict[str, Any]：核心流水线内部结果，包含 prediction、task_id 以及对应任务需要的字段。
        """
        return self.run_sample_as(sample, index)

    def run_sample_as(
        self,
        sample: dict[str, Any],
        index: int,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """功能：按指定内部 task_id 处理单条样本。

        参数：
            sample：当前题目的原始样本字典。
            index：样本在当前任务列表中的序号，用作缺省样本 id。
            task_id：内部任务编号，如 1-1、1-4、2-1、2-2；为 None 时自动推断。

        返回：
            dict[str, Any]：经过 Agent 推理和 schema 兜底后的内部结构化结果。
        """
        task_plan = dispatch_sub_task_prompt(sample, task_id)
        context: dict[str, Any] = {
            "sample_id": get_sample_id(sample, index),
            "task_plan": task_plan,
            "options": extract_options(sample),
        }

        if task_plan["track"] == "basic_understanding":
            if task_plan["task_id"] == "1-1" and ("qa_words" in sample or "qa_sents" in sample):
                payload = self._run_official_task1(sample, task_plan, context)
            else:
                payload = self._run_basic_understanding(sample, task_plan, context)
        elif task_plan["track"] == "analogy_reasoning":
            payload = self._run_analogy_reasoning(sample, task_plan, context)
        elif task_plan["track"] == "critical_analysis":
            payload = self._run_critical_analysis(sample, task_plan, context)
        else:
            raise ValueError(f"Unsupported task track: {task_plan['track']}")

        final_payload = enforce_json_schema(payload, task_plan["output_schema"], sample=sample)
        final_payload["id"] = get_sample_id(sample, index)
        final_payload["task_id"] = task_plan["task_id"]
        return final_payload

    def run(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """功能：处理普通 JSON 数组格式的数据文件并写出结果。

        参数：
            input_path：输入 JSON 文件路径；为 None 时使用配置中的默认测试集路径。
            output_path：输出 JSON 文件路径；为 None 时使用配置中的默认提交路径。
            limit：最多处理的样本数；为 None 时处理全部样本。

        返回：
            list[dict[str, Any]]：所有样本的内部结构化结果列表。
        """
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
        """功能：执行基础理解轨道，包括 1-1、1-2、1-3、1-4。

        参数：
            sample：当前题目的原始样本字典。
            task_plan：任务计划字典，包含 task_id、track、stages 和 output_schema。
            context：流水线上下文字典，函数会逐步写入背景、解析和辩论历史。

        返回：
            dict[str, Any]：Umpire 裁决后的内部结果，通常包含 prediction、reason、evidence 和任务专属字段。
        """
        evoked = self.agents.evoker.background(sample, context, task_plan)
        context["evoker_background"] = evoked.data or evoked.content

        parsed = self.agents.parser.raw_extract(sample, context, task_plan)
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

        judged = self.agents.umpire.finalize_basic(sample, context, task_plan)
        return judged.data or {"prediction": judged.content, "reason": judged.content}

    def _run_official_task1(
        self,
        sample: dict[str, Any],
        task_plan: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """功能：执行官方 task1 专用轨道，逐个解释词语、翻译诗句并选择情感选项。

        参数：
            sample：当前 task1 官方样本字典。
            task_plan：任务计划字典，task_id 应为 1-1。
            context：流水线上下文字典，函数会写入背景和 task1 专用解析结果。

        返回：
            dict[str, Any]：官方 task1 内部结果，包含 ans_qa_words、ans_qa_sents、choose_id。
        """
        evoked = self.agents.evoker.background(sample, context, task_plan)
        context["evoker_background"] = evoked.data or evoked.content

        parsed = self.agents.parser.official_task1(sample, context, task_plan)
        context["task1_official_parse"] = parsed.data or parsed.content

        judged = self.agents.umpire.finalize_official_task1(sample, context, task_plan)
        payload = judged.data or parsed.data or {"prediction": judged.content, "reason": judged.content}
        return payload

    def _run_analogy_reasoning(
        self,
        sample: dict[str, Any],
        task_plan: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """功能：执行 2-1 类比推理轨道。

        参数：
            sample：当前题目的原始样本字典。
            task_plan：任务计划字典，track 应为 analogy_reasoning。
            context：流水线上下文字典，函数会写入双向背景和 analogy_matrix。

        返回：
            dict[str, Any]：Umpire 裁决后的类比推理结果，通常包含 answer、prediction、reason、analogy_matrix。
        """
        evoked = self.agents.evoker.background(sample, context, task_plan)
        context["evoker_bidirectional"] = evoked.data or evoked.content

        parsed = self.agents.parser.relation(sample, context, task_plan)
        matrix = parse_analogy_matrix(parsed.data or parsed.content)
        context["analogy_matrix"] = matrix

        judged = self.agents.umpire.finalize_matrix(sample, context, task_plan)
        payload = judged.data or {"prediction": judged.content, "reason": judged.content}
        payload.setdefault("analogy_matrix", matrix)
        return payload

    def _run_critical_analysis(
        self,
        sample: dict[str, Any],
        task_plan: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """功能：执行 2-2 诗词辨析选择题轨道。

        参数：
            sample：当前题目的原始样本字典。
            task_plan：任务计划字典，track 应为 critical_analysis。
            context：流水线上下文字典，函数会写入背景、选项辩论历史和工具投票结果。

        返回：
            dict[str, Any]：Umpire 裁决后的选择题结果，通常包含 prediction、answer、reason、votes。
        """
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

        vote = eliminate_and_vote(
            option_scores,
            options=context.get("options", []),
            question_text=json.dumps(sample, ensure_ascii=False),
        )
        context["tool_vote"] = vote

        judged = self.agents.umpire.finalize_vote(sample, context, task_plan)
        payload = judged.data or vote
        payload.setdefault("prediction", vote.get("prediction", ""))
        payload.setdefault("votes", vote.get("votes", []))
        return payload


class OfficialPipeline:
    """功能：官方数据外壳流水线，负责 task1-task4 批处理、格式适配、异常兜底和校验。"""

    TASK_TO_PROMPT = {
        "task1": "1-1",
        "task2": "1-4",
        "task3": "2-1",
        "task4": "2-2",
    }

    def __init__(
        self,
        cfg: Settings = settings,
        client: LLMClient | None = None,
        agents: AgentMatrix | None = None,
        recorder: BadCaseRecorder | None = None,
    ) -> None:
        """功能：初始化官方提交流水线。

        参数：
            cfg：项目配置对象。
            client：模型客户端；为 None 时核心流水线会创建默认客户端。
            agents：Agent 矩阵；为 None 时核心流水线会创建默认矩阵。
            recorder：坏例记录器；为 None 时创建默认 BadCaseRecorder。
        """
        self.core = PoetryEvalPipeline(cfg, client=client, agents=agents)
        self.adapters = OfficialAdapters()
        self.validator = OfficialSubmissionValidator()
        self.recorder = recorder or BadCaseRecorder()

    def run_all(
        self,
        test_data: dict[str, list[dict[str, Any]]],
        limit: int | None = None,
        task: str = "all",
    ) -> dict[str, list[dict[str, Any]]]:
        """功能：处理全部官方测试任务。

        参数：
            test_data：官方测试数据，键为 task1、task2、task3、task4。
            limit：每个任务最多处理的样本数；为 None 时处理全部样本。
            task：需要处理的任务名；"all" 表示处理 task1-task4 全部任务。

        返回：
            dict[str, list[dict[str, Any]]]：官方提交格式结果，键为 task1、task2、task3、task4。
        """
        submission: dict[str, list[dict[str, Any]]] = {}
        selected_tasks = TASK_NAMES if task == "all" else (task,)
        for task_name in selected_tasks:
            rows = test_data[task_name]
            if limit is not None:
                rows = rows[:limit]
            submission[task_name] = self.run_task(task_name, rows)
        return submission

    def run_task(self, task_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """功能：处理某一个官方任务的全部样本。

        参数：
            task_name：官方任务名，如 task1、task2、task3、task4。
            rows：该任务的官方原始样本列表。

        返回：
            list[dict[str, Any]]：该任务的官方提交格式结果列表。
        """
        return [self.run_official_sample(task_name, sample, index) for index, sample in enumerate(rows)]

    def run_official_sample(
        self,
        task_name: str,
        sample: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        """功能：处理一条官方样本，并转换成官方提交行。

        参数：
            task_name：官方任务名，用于映射内部 task_id。
            sample：当前官方原始样本字典。
            index：样本在当前任务列表中的序号。

        返回：
            dict[str, Any]：单条官方提交格式结果；异常时返回 fallback_row。
        """
        try:
            prompt_id = self.TASK_TO_PROMPT[task_name]
            raw = self.core.run_sample_as(sample, index, prompt_id)
            return self.adapters.adapt(task_name, sample, raw)
        except Exception as exc:  # noqa: BLE001
            self.recorder.record(
                task=task_name,
                idx=sample.get("idx", index),
                error_type=type(exc).__name__,
                message=str(exc),
                sample=sample,
            )
            return fallback_row(task_name, sample)

    def validate(self, submission: dict[str, Any], test_data: dict[str, list[dict[str, Any]]]) -> list[str]:
        """功能：校验完整官方提交结果。

        参数：
            submission：官方提交格式结果。
            test_data：官方测试数据，用于检查长度和 idx。

        返回：
            list[str]：错误信息列表；为空表示基础格式校验通过。
        """
        return self.validator.validate(submission, test_data)

    def record_quality_warnings(self, submission: dict[str, Any]) -> None:
        """功能：记录空答案等质量告警。

        参数：
            submission：官方提交格式结果，函数会扫描 task1、task2、task3 中的空答案。
        """
        for row in submission.get("task1", []):
            if any(not str(value).strip() for value in row.get("ans_qa_words", {}).values()):
                self.recorder.record("task1", row.get("idx"), "EmptyAnswer", "empty word answer")
            if any(not str(value).strip() for value in row.get("ans_qa_sents", {}).values()):
                self.recorder.record("task1", row.get("idx"), "EmptyAnswer", "empty sentence answer")
        for row in submission.get("task2", []):
            if row.get("flag") == 1 and not str(row.get("answer", "")).strip():
                self.recorder.record("task2", row.get("idx"), "EmptyAnswer", "empty allusion answer")
        for row in submission.get("task3", []):
            if any(not str(item).strip() for item in row.get("answer", [])):
                self.recorder.record("task3", row.get("idx"), "EmptyAnswer", "empty answer item")


def run_official_submission(
    test_dir: str | Path,
    output_path: str | Path,
    train_dir: str | Path = "",
    bad_cases_path: str | Path = "",
    task: str = "all",
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """功能：运行一次完整官方提交流程。

    参数：
        test_dir：官方测试集目录，目录下应包含 task1.json 到 task4.json。
        output_path：官方 submission.json 输出路径。
        train_dir：官方训练集目录；当前主要用于参数保留和评测扩展。
        bad_cases_path：坏例 JSONL 输出路径；为空时不落盘。
        task：需要处理的任务名；"all" 表示处理全部任务。
        limit：每个任务最多处理的样本数；为 None 时处理全部样本。
        dry_run：是否跳过真实模型调用并使用模拟响应。

    返回：
        dict[str, Any]：运行结果字典，包含 submission、errors 和 bad_cases。
    """
    io = OfficialDataIO(train_dir=train_dir, test_dir=test_dir)
    test_data = io.load_all_test()
    recorder = BadCaseRecorder(bad_cases_path)
    client = LLMClient(settings, dry_run=dry_run)
    pipeline = OfficialPipeline(settings, client=client, recorder=recorder)
    submission = pipeline.run_all(test_data, limit=limit, task=task)

    validation_data = test_data
    if limit is not None:
        validation_data = {name: rows[:limit] for name, rows in test_data.items()}
    if task != "all":
        validation_data = {task: validation_data[task]}
    errors = pipeline.validate(submission, validation_data)
    if not dry_run:
        pipeline.record_quality_warnings(submission)
    if errors:
        for message in errors:
            recorder.record("schema", "", "ValidationError", message)
    recorder.save()
    io.save_submission(output_path, submission)
    return {"submission": submission, "errors": errors, "bad_cases": recorder.rows}


def resolve_project_path(path: str | Path, cfg: Settings = settings) -> Path:
    """功能：把相对路径解析到项目根目录下。

    参数：
        path：待解析路径，可以是绝对路径或相对路径。
        cfg：项目配置对象，用于提供 root_dir。

    返回：
        Path：解析后的路径；绝对路径会原样返回。
    """
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return cfg.root_dir / resolved
