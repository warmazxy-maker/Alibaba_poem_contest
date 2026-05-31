# ccl2026-poetry-eval

Project scaffold for a CCL 2026 poetry evaluation workflow.

`src/pipeline.py` is the main controller. It dispatches each sample to the
right task track, calls the Agent matrix, invokes the core Tools functions, and
writes `data/submission.json`. `evaluate.py` is only a local evaluation script.

## Layout

```text
ccl2026-poetry-eval/
  data/
    train.json
    test.json
    submission.json
  src/
    __init__.py
    config.py
    llm_client.py
    agents.py      # Evoker / Parser / Critic / Umpire Agent matrix
    tools.py       # core control functions
    prompts.py     # 1-1 to 2-2 task prompts and stage prompts
    pipeline.py    # main controller
  evaluate.py      # local evaluator, not the controller
  requirements.txt
```

## Module ownership

- Student A: `src/config.py`, `src/llm_client.py`, `src/agents.py`, `src/tools.py`
- Student B: `src/prompts.py`
- Student C: `data/`, `src/pipeline.py`, `evaluate.py`

## Required workflow

Agent matrix:

- `Evoker Agent`: background generation.
- `Parser Agent`: raw extraction, relation parsing, option analysis.
- `Critic Agent`: debate control and option scoring.
- `Umpire Agent`: matrix alignment, voting, and final JSON formatting.

Core Tools functions in `src/tools.py`:

- `dispatch_sub_task_prompt`: dynamically loads the sub-task prompt.
- `execute_debate_loop`: controls multi-round debate.
- `parse_analogy_matrix`: normalizes the 2-1 analogy matrix.
- `eliminate_and_vote`: makes the 2-2 mathematical voting decision.
- `enforce_json_schema`: guarantees final JSON fields.

Task tracks:

- Track 1, basic understanding: `1-1` word, `1-2` sentence, `1-3` emotion, `1-4` allusion.
- Track 2, analogy reasoning: `2-1`.
- Track 3, critical analysis multiple-choice: `2-2`.

## Quick start

Run the pipeline in dry-run mode:

```bash
python -m src.pipeline --input data/test.json --output data/submission.json --dry-run
```

Run local exact-match evaluation:

```bash
python evaluate.py --gold data/train.json --pred data/submission.json
```

For real LLM calls, set these environment variables before running:

```bash
set OPENAI_API_KEY=your_api_key
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-4.1-mini
```

Expected data files are JSON arrays. Each test item should include an `id`
field when possible. Evaluation looks for gold labels in `label`, `answer`,
`gold`, or `target`, and predictions in `prediction`, `answer`, or `output`.
