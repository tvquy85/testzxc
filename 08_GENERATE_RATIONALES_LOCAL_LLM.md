# Step 08 — Generate Candidate Rationales with Local LLM

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Generate candidate structured rationales for a bounded sample set using a local LLM. Do not train yet.

## Inputs

```text
data/labels/aligned_samples_h1.parquet
data/indicators/technical_event_tokens_h1.parquet
prompts/rationale_generation_prompt.txt
configs/local_paths.yaml
```

## Preferred model
Use this priority order:

1. `Qwen2.5-3B-Instruct-cf-finflow` if present.
2. `Qwen3-4B-Instruct-2507` if present.
3. `Qwen2.5-3B-Instruct` if present.
4. `Phi-3.5-mini-instruct` fallback.

Load 4-bit if VRAM requires it.

## Outputs

```text
data/rationales/candidate_rationales_h1.jsonl
outputs/status/08_GENERATE_RATIONALES_LOCAL_LLM.status.json
```

Each line must contain `sample_id`, `candidate_id`, `generator_model`, `prompt_hash`, `rationale_json`, `raw_text`, `parse_ok`, `schema_ok`.

## Tasks

Create:

```text
src/llm/render_context.py
src/llm/generate_rationales.py
src/llm/parse_and_validate_rationale.py
```

Context must include ticker, event date, headline/body, technical tokens, regime, recent return summary. It must not include realized return, label, or future prices.

MVP command:

```bash
python src/llm/generate_rationales.py \
  --samples data/labels/aligned_samples_h1.parquet \
  --tech data/indicators/technical_event_tokens_h1.parquet \
  --prompt prompts/rationale_generation_prompt.txt \
  --config configs/local_paths.yaml \
  --limit 5000 \
  --num-candidates 3 \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --output data/rationales/candidate_rationales_h1.jsonl
```

## Verification
Run:

```bash
python - <<'PYCHECK'
import json
p='data/rationales/candidate_rationales_h1.jsonl'
rows=[json.loads(x) for x in open(p, encoding='utf-8')]
print(len(rows))
print(sum(r.get('schema_ok', False) for r in rows)/len(rows))
assert len(rows) >= 1000
assert sum(r.get('schema_ok', False) for r in rows)/len(rows) >= 0.70
PYCHECK
```

## Acceptance criteria
PASS only if at least 1,000 candidates exist, schema OK rate >= 70%, and no realized label appears in prompt.

## Status JSON

```json
{
  "step": "08_GENERATE_RATIONALES_LOCAL_LLM",
  "status": "PASS|FAIL",
  "candidate_count": 0,
  "unique_sample_count": 0,
  "schema_ok_rate": 0.0,
  "generator_model": "...",
  "notes": "..."
}
```
