# GuideQwen3-4B-Instruct-2507.md

## Purpose

This guide is a focused Codex task for optimizing the `generate_rationales` stage in the `tvquy85/testzxc` FIRE-Fin pipeline.

The current issue: rationale generation is slow when using `Qwen3-4B-Instruct-2507` with too many candidates and long outputs. The goal is to keep quality high enough for AAAI-grade downstream judges/flow-reward/DPO while reducing runtime on a single RTX 3090.

This task must be executed in the existing `testzxc` repository. Do not rewrite the whole project. Make minimal, safe, testable changes.

---

## External facts to respect

1. `Qwen/Qwen3-4B-Instruct-2507` is an instruction model with native long context, but this project must cap runtime context to 3072-4096 tokens for RTX 3090 throughput.
2. This model supports **non-thinking mode only** and does **not** generate `<think></think>` blocks. Do not add `enable_thinking=False`; it is not needed for this model.
3. vLLM is preferred for high-throughput offline generation if the user runs Linux/WSL2; otherwise keep a Transformers backend.
4. For this project, schema-valid short JSON rationales are more valuable than long free-form reasoning.

References:
- Qwen model card: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- vLLM project: https://github.com/vllm-project/vllm
- vLLM automatic prefix caching: https://docs.vllm.ai/en/v0.20.1/features/automatic_prefix_caching/

---

## Required model role

Use `Qwen3-4B-Instruct-2507` as the **main rationale generator**.

Use these local roles:

```yaml
models:
  main_explanation_llm: "E:/huggingface/models/Qwen/Qwen3-4B-Instruct-2507"
  finance_judge_llm: "E:/huggingface/models/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
  reasoning_judge_llm: "E:/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  nli_judge: "E:/huggingface/models/cross-encoder/nli-deberta-v3-small"
```

If the repository uses Linux/WSL paths, convert `E:/...` to `/mnt/e/...`.

Do not use `DeepSeek-R1-Distill-Qwen-1.5B` as the main generator. It is only a cheap reasoning judge or ablation target.

---

## Current bottleneck

Avoid this slow setting for bulk generation:

```bash
--limit 20000 --num-candidates 3 --max-new-tokens 512 --temperature 0.7
```

This may decode up to:

```text
20,000 × 3 × 512 = 30,720,000 output tokens
```

Use this bulk setting instead:

```bash
--limit 20000 --num-candidates 1 --max-new-tokens 256 --temperature 0.40
```

This decodes at most:

```text
20,000 × 1 × 256 = 5,120,000 output tokens
```

That is approximately 6x fewer output tokens.

---

## Required repository changes

### Change 1 — Add or update config file

Create or update:

```text
configs/rationale_generation_qwen3_fast.yaml
```

Content:

```yaml
rationale_generation:
  generator_model_key: "main_explanation_llm"
  generator_model_name: "Qwen3-4B-Instruct-2507"
  backend: "transformers"
  dtype: "float16"
  max_model_len: 4096
  max_input_tokens: 3072

  bulk:
    limit: 20000
    num_candidates: 1
    max_new_tokens: 256
    temperature: 0.40
    top_p: 0.85
    top_k: 40
    repetition_penalty: 1.05
    batch_size: 4
    sort_by_length: true
    save_every: 100
    resume: true
    output: "data/rationales/raw/train_qwen3_4b_bulk_h1.jsonl"

  conflict:
    input: "data/splits/train_conflict_subset_h1.parquet"
    limit: 5000
    num_candidates: 3
    max_new_tokens: 320
    temperature: 0.55
    top_p: 0.90
    top_k: 50
    repetition_penalty: 1.05
    batch_size: 4
    sort_by_length: true
    save_every: 50
    resume: true
    output: "data/rationales/raw/train_qwen3_4b_conflict_3cand_h1.jsonl"

  validation_gates:
    raw_parse_ok_rate_min: 0.85
    raw_schema_ok_rate_min: 0.80
    avg_output_tokens_max: 280
    invalid_json_rate_max: 0.15
    explicit_label_leak_rate_max: 0.03
```

Acceptance:
- File exists.
- YAML loads successfully.
- No local absolute path is hard-coded in this file except outputs inside the repo.

---

### Change 2 — Add short strict prompt

Create:

```text
prompts/rationale_generation_prompt_qwen3_fast_json.txt
```

Content:

```text
You are a financial rationale generator.

Task:
Given company news, technical event tokens, and market context, generate a short grounded rationale for short-horizon stock movement.

Rules:
- Return valid JSON only.
- Do not include markdown.
- Do not include explanations outside JSON.
- Do not reveal or quote the realized label.
- Maximum 2 news_rationale items.
- Maximum 2 technical_rationale items.
- conflict_resolution must be <= 35 words.
- risk_note must be <= 20 words.
- Each rationale item must be grounded in the provided context.

Output schema:
{
  "news_rationale": [
    {"factor": "...", "direction": "positive|negative|neutral", "strength": "weak|medium|strong"}
  ],
  "technical_rationale": [
    {"signal": "...", "direction": "positive|negative|neutral", "strength": "weak|medium|strong"}
  ],
  "conflict_resolution": "...",
  "forecast_distribution": {
    "Strong Down": 0.0,
    "Mild Down": 0.0,
    "Neutral": 0.0,
    "Mild Up": 0.0,
    "Strong Up": 0.0
  },
  "action": "long|short|hold",
  "risk_note": "..."
}

Context:
{context}
```

Acceptance:
- Prompt is concise.
- Prompt forces JSON only.
- Prompt does not ask for chain-of-thought.
- Prompt caps output length.

---

### Change 3 — Update `src/llm/generate_rationales.py`

Add these CLI flags if missing:

```text
--model-key
--batch-size
--sort-by-length
--save-every
--resume
--max-input-tokens
--top-p
--top-k
--repetition-penalty
--backend
--vllm-base-url
--strict-raw-output
```

Expected behavior:

1. `--model-key` reads the model path from `configs/local_paths.yaml`.
2. `--max-input-tokens` truncates input context before generation, preferably from the oldest/least important fields first.
3. `--batch-size` performs batched generation for Transformers backend.
4. `--sort-by-length` sorts prompts by approximate token length before batching to reduce padding waste.
5. `--save-every N` writes partial JSONL to disk every N generated records.
6. `--resume` skips sample IDs already present in the output file.
7. `--strict-raw-output` stores raw model output without auto-fixing forecast distributions.
8. If `--backend vllm`, call the local OpenAI-compatible vLLM server.
9. If `--backend transformers`, use local Transformers inference.

Do not call `parse_and_validate_rationale.py` auto-fix inside `generate_rationales.py`.

Each JSONL row must include:

```json
{
  "sample_id": "...",
  "ticker": "...",
  "timestamp": "...",
  "horizon": "h1",
  "generator_model": "Qwen3-4B-Instruct-2507",
  "candidate_id": 0,
  "prompt_tokens_est": 0,
  "output_tokens_est": 0,
  "raw_output": "...",
  "generation_config": {
    "temperature": 0.4,
    "top_p": 0.85,
    "top_k": 40,
    "max_new_tokens": 256,
    "backend": "transformers"
  }
}
```

Acceptance:
- Generation resumes correctly if interrupted.
- Existing outputs are not overwritten unless `--overwrite` is explicitly passed.
- Raw output is preserved.
- No auto-fix is applied during generation.

---

### Change 4 — Add strict raw validation script

Create:

```text
src/llm/validate_raw_rationales.py
```

This script validates raw generated rationales without repair.

CLI:

```bash
python -m src.llm.validate_raw_rationales \
  --input data/rationales/debug/qwen3_4b_200.jsonl \
  --output outputs/validation/qwen3_4b_200_validation.json \
  --strict-no-autofix
```

It must compute:

```text
num_rows
raw_parse_ok_rate
raw_schema_ok_rate
invalid_json_rate
explicit_label_leak_rate
avg_output_tokens_est
forecast_distribution_sum_error_mean
forecast_distribution_sum_error_p95
action_valid_rate
news_rationale_item_count_mean
technical_rationale_item_count_mean
```

Strict schema rules:

1. Output must parse as JSON.
2. Required keys must exist.
3. `forecast_distribution` must contain exactly:
   - `Strong Down`
   - `Mild Down`
   - `Neutral`
   - `Mild Up`
   - `Strong Up`
4. Distribution values must be numeric and finite.
5. Distribution sum must be in `[0.99, 1.01]`.
6. `action` must be one of `long`, `short`, `hold`.
7. No explicit ground-truth label leakage is allowed if the source row contains `label_5`.

Acceptance:
- Script exits nonzero if validation gates fail and `--enforce-gates` is passed.
- Script writes a JSON summary.
- Script never repairs outputs.

---

### Change 5 — Optional vLLM support

If vLLM is available, support generation through a local OpenAI-compatible server.

Example server command for WSL/Linux:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/e/huggingface/models/Qwen/Qwen3-4B-Instruct-2507 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --disable-log-requests \
  --host 127.0.0.1 \
  --port 8000
```

If OOM, reduce in this order:

```text
--max-num-seqs 32 -> 16
--max-num-batched-tokens 8192 -> 4096
--gpu-memory-utilization 0.90 -> 0.85
--max-model-len 4096 -> 3072
```

Client command:

```bash
python -m src.llm.generate_rationales \
  --backend vllm \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --samples data/splits/train_samples_h1.parquet \
  --prompt prompts/rationale_generation_prompt_qwen3_fast_json.txt \
  --config configs/local_paths.yaml \
  --model-key main_explanation_llm \
  --limit 20000 \
  --num-candidates 1 \
  --max-new-tokens 256 \
  --temperature 0.40 \
  --top-p 0.85 \
  --top-k 40 \
  --save-every 100 \
  --resume \
  --strict-raw-output \
  --output data/rationales/raw/train_qwen3_4b_bulk_h1.jsonl
```

Acceptance:
- vLLM backend is optional.
- Transformers backend must still work if vLLM is not installed.
- Do not make vLLM a hard dependency.

---

## Execution plan

### Phase A — smoke test

Run this first:

```bash
python -m src.llm.generate_rationales \
  --backend transformers \
  --samples data/splits/train_samples_h1.parquet \
  --prompt prompts/rationale_generation_prompt_qwen3_fast_json.txt \
  --config configs/local_paths.yaml \
  --model-key main_explanation_llm \
  --limit 200 \
  --num-candidates 1 \
  --max-new-tokens 256 \
  --temperature 0.40 \
  --top-p 0.85 \
  --top-k 40 \
  --batch-size 4 \
  --sort-by-length \
  --save-every 50 \
  --resume \
  --strict-raw-output \
  --output data/rationales/debug/qwen3_4b_200.jsonl
```

Validate:

```bash
python -m src.llm.validate_raw_rationales \
  --input data/rationales/debug/qwen3_4b_200.jsonl \
  --output outputs/validation/qwen3_4b_200_validation.json \
  --strict-no-autofix \
  --enforce-gates
```

Smoke test gates:

```text
raw_parse_ok_rate >= 0.85
raw_schema_ok_rate >= 0.80
avg_output_tokens_est <= 280
invalid_json_rate <= 0.15
explicit_label_leak_rate <= 0.03
```

If gates fail, do not run full generation. Improve prompt or lower temperature.

---

### Phase B — bulk generation

Run for normal train samples:

```bash
python -m src.llm.generate_rationales \
  --backend transformers \
  --samples data/splits/train_samples_h1.parquet \
  --prompt prompts/rationale_generation_prompt_qwen3_fast_json.txt \
  --config configs/local_paths.yaml \
  --model-key main_explanation_llm \
  --limit 20000 \
  --num-candidates 1 \
  --max-new-tokens 256 \
  --temperature 0.40 \
  --top-p 0.85 \
  --top-k 40 \
  --batch-size 4 \
  --sort-by-length \
  --save-every 100 \
  --resume \
  --strict-raw-output \
  --output data/rationales/raw/train_qwen3_4b_bulk_h1.jsonl
```

Validate:

```bash
python -m src.llm.validate_raw_rationales \
  --input data/rationales/raw/train_qwen3_4b_bulk_h1.jsonl \
  --output outputs/validation/train_qwen3_4b_bulk_h1_validation.json \
  --strict-no-autofix
```

---

### Phase C — conflict subset generation

Only generate 3 candidates on hard/conflict samples.

Input expected:

```text
data/splits/train_conflict_subset_h1.parquet
```

If this file does not exist, create it by selecting training samples with at least one of:

```text
bullish news + bearish technical tokens
bearish news + bullish technical tokens
high volatility regime
earnings/guidance events
strong_up/strong_down label
baseline confidence below threshold
```

Run:

```bash
python -m src.llm.generate_rationales \
  --backend transformers \
  --samples data/splits/train_conflict_subset_h1.parquet \
  --prompt prompts/rationale_generation_prompt_qwen3_fast_json.txt \
  --config configs/local_paths.yaml \
  --model-key main_explanation_llm \
  --limit 5000 \
  --num-candidates 3 \
  --max-new-tokens 320 \
  --temperature 0.55 \
  --top-p 0.90 \
  --top-k 50 \
  --batch-size 4 \
  --sort-by-length \
  --save-every 50 \
  --resume \
  --strict-raw-output \
  --output data/rationales/raw/train_qwen3_4b_conflict_3cand_h1.jsonl
```

Validate:

```bash
python -m src.llm.validate_raw_rationales \
  --input data/rationales/raw/train_qwen3_4b_conflict_3cand_h1.jsonl \
  --output outputs/validation/train_qwen3_4b_conflict_3cand_h1_validation.json \
  --strict-no-autofix
```

---

## Performance rules

Do not use these defaults for bulk generation:

```text
num_candidates = 3
max_new_tokens = 512
temperature = 0.7
max_model_len > 4096
batch_size = 1 unless OOM
```

Use these defaults:

```text
num_candidates = 1
max_new_tokens = 256
temperature = 0.40
top_p = 0.85
top_k = 40
batch_size = 4
max_input_tokens = 3072
max_model_len = 4096
```

If OOM with Transformers:

```text
batch_size 4 -> 2 -> 1
max_input_tokens 3072 -> 2048
max_new_tokens 256 -> 224
```

If still too slow:

```text
Reduce normal-sample generation first; keep Qwen3-4B-Instruct-2507 for conflict/hard samples.
```

Hybrid fallback:

```text
Normal samples:
- model: an explicitly available local fallback only after cache verification
- num_candidates: 1
- max_new_tokens: 224
- temperature: 0.35

Hard/conflict samples:
- model: Qwen3-4B-Instruct-2507
- num_candidates: 3
- max_new_tokens: 320
- temperature: 0.55
```

---

## Final acceptance criteria

This Codex task is complete only if all conditions hold:

1. `configs/rationale_generation_qwen3_fast.yaml` exists and loads.
2. `prompts/rationale_generation_prompt_qwen3_fast_json.txt` exists.
3. `src/llm/generate_rationales.py` supports the requested fast-generation flags.
4. `src/llm/validate_raw_rationales.py` exists and validates raw outputs without repair.
5. Smoke test with 200 samples passes validation gates or reports exact failing gates.
6. Generation supports resume and save-every behavior.
7. No auto-fix is applied during raw generation.
8. The output JSONL includes generator model, candidate ID, raw output and generation config.
9. The implementation does not make vLLM mandatory.
10. A final status JSON is written to:

```text
outputs/status/guide_qwen3_4b_instruct_2507_status.json
```

Status JSON schema:

```json
{
  "task": "GuideQwen3-4B-Instruct-2507",
  "status": "PASS|FAIL|PARTIAL",
  "files_created": [],
  "files_modified": [],
  "smoke_test_command": "...",
  "smoke_test_output": "...",
  "validation_summary_path": "outputs/validation/qwen3_4b_200_validation.json",
  "blocking_errors": [],
  "notes": []
}
```

---

## Notes for Codex

- Do not optimize by hiding invalid outputs. Measure raw generation quality.
- Do not overwrite existing generated rationale files unless explicitly requested.
- Keep all changes small and reversible.
- Do not train models in this task.
- This task only improves rationale generation speed, prompt discipline, raw validation, checkpointing and resume behavior.
