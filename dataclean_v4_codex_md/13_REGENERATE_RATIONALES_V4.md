# 13 — Regenerate Rationales on Evidence Contexts V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Generate train-only rationales with Qwen3-4B using evidence prompt.

## Inputs
- `data/processed/current_clean_train_pool_v4.parquet`
- `prompts/rationale_generation_prompt_evidence_v4.txt`

## Outputs
- `data/rationales/raw/current_clean_train_qwen3_4b_v4.jsonl`
- `data/rationales/parsed/current_clean_train_qwen3_4b_v4.parquet`
- `outputs/status/13_REGENERATE_RATIONALES_V4.status.json`

## Recommended config
`num_candidates=1`, `max_new_tokens=256`, `max_input_tokens=3072`, `temperature=0.35`, `top_p=0.85`, `top_k=40`, `batch_size=4`, `sort_by_length`, `save_every=100`, `resume`.

## Command
```bash
python -m src.llm.generate_rationales \
  --input data/processed/current_clean_train_pool_v4.parquet \
  --prompt prompts/rationale_generation_prompt_evidence_v4.txt \
  --config configs/default_paths.yaml \
  --split train \
  --model qwen3_4b \
  --num-samples 6000 \
  --num-candidates 1 \
  --max-new-tokens 256 \
  --max-input-tokens 3072 \
  --temperature 0.35 \
  --top-p 0.85 \
  --top-k 40 \
  --batch-size 4 \
  --sort-by-length \
  --save-every 100 \
  --resume \
  --stage current_clean_v4 \
  --raw-output data/rationales/raw/current_clean_train_qwen3_4b_v4.jsonl \
  --parsed-output data/rationales/parsed/current_clean_train_qwen3_4b_v4.parquet \
  --status outputs/status/13_REGENERATE_RATIONALES_V4.status.json
```

## Gates
- train split only.
- parse OK >= 0.95.
- schema OK >= 0.95.
- avg output tokens <= 300.
