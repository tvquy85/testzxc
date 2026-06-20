# 06 — Regenerate Rationales on Clean Contexts

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Generate fewer but cleaner train-only rationales from aggregated contexts.

## Inputs
- `data/processed/ticker_date_contexts_h1_v2_targets.parquet`
- `data/indicators/technical_event_tokens_h1_v2.parquet`

## Outputs
- `prompts/rationale_generation_prompt_clean_context_v3.txt`
- `data/rationales/raw/current_clean_train_qwen3_4b_v3.jsonl`
- `data/rationales/parsed/current_clean_train_qwen3_4b_v3.parquet`
- `outputs/status/06_RATIONALE_PROMPT_CLEAN_CONTEXT.status.json`

## Prompt rules
- JSON only.
- Max 2 news rationale items.
- Max 2 technical rationale items.
- `conflict_resolution` <= 35 words.
- `risk_note` <= 20 words.
- Do not reveal realized label.
- Do not invent unsupported technical claims.

## Run
```bash
python -m src.llm.generate_rationales \
  --input data/processed/ticker_date_contexts_h1_v2_targets.parquet \
  --tokens data/indicators/technical_event_tokens_h1_v2.parquet \
  --prompt prompts/rationale_generation_prompt_clean_context_v3.txt \
  --config configs/default_paths.yaml \
  --split train \
  --model qwen3_4b \
  --num-samples 6000 \
  --num-candidates 1 \
  --max-new-tokens 256 \
  --max-input-tokens 3072 \
  --temperature 0.40 \
  --top-p 0.85 \
  --top-k 40 \
  --batch-size 4 \
  --sort-by-length \
  --save-every 100 \
  --resume \
  --stage current_clean_v3 \
  --raw-output data/rationales/raw/current_clean_train_qwen3_4b_v3.jsonl \
  --parsed-output data/rationales/parsed/current_clean_train_qwen3_4b_v3.parquet \
  --status outputs/status/06_RATIONALE_PROMPT_CLEAN_CONTEXT.status.json
```

## Verify
```bash
python - <<'PY'
import pandas as pd
df=pd.read_parquet("data/rationales/parsed/current_clean_train_qwen3_4b_v3.parquet")
assert len(df)>1000
assert df["split"].eq("train").all()
assert df["parse_ok"].mean() >= 0.95
assert df["schema_ok"].mean() >= 0.90
print(len(df), df["parse_ok"].mean(), df["schema_ok"].mean())
PY
```

## Acceptance
- Parse rate >= 0.95.
- Schema rate >= 0.90.
- Train-only.
- Avg output tokens <= 280.
