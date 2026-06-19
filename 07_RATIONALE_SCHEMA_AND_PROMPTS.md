# Step 07 — Rationale Schema and Prompt Templates

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Define the structured rationale format and prompts for local LLM generation, proxy judging, and grounding. No model training in this step.

## Inputs

```text
data/labels/aligned_samples_h1.parquet
data/indicators/technical_event_tokens_h1.parquet
outputs/metrics/baseline_metrics_h1.json
```

## Outputs

```text
prompts/rationale_generation_prompt.txt
prompts/proxy_inferability_judge_prompt.txt
prompts/financial_soundness_judge_prompt.txt
prompts/counterfactual_prompt.txt
src/llm/rationale_schema.py
outputs/status/07_RATIONALE_SCHEMA_AND_PROMPTS.status.json
```

## Tasks

### 1. Create Pydantic schema
Create `src/llm/rationale_schema.py` with `ForecastDistribution` and `RationaleOutput`.

Required fields:

```text
news_rationale: list[str]
technical_rationale: list[str]
conflict_resolution: str
forecast_distribution: {strong_down, mild_down, neutral, mild_up, strong_up}
action: long | short | hold
risk_note: str
```

Validation rules:

- Distribution values must be between 0 and 1.
- Sum must be between 0.95 and 1.05.
- `action` must be consistent with up/down probabilities.

### 2. Create rationale generation prompt
The prompt must tell the LLM:

- Use news + technical tokens + market regime.
- Do not mention realized label.
- Do not use future price.
- Explain conflict between news and technical signals.
- Return strict JSON only.

### 3. Create proxy inferability prompt
The proxy judge sees `context + rationale` and infers one of:

```text
Strong Down, Mild Down, Neutral, Mild Up, Strong Up
```

with probabilities.

### 4. Create financial soundness prompt
The judge scores:

```text
financial_soundness: 0-1
groundedness: 0-1
overconfidence: 0-1
main_error: string
```

### 5. Create counterfactual prompt
Generate modified contexts by removing/neutralizing one important signal.

## Verification
Create `src/llm/test_rationale_schema.py` and run:

```bash
cd firefin
python src/llm/test_rationale_schema.py
```

It must validate one good JSON and reject one bad JSON.

## Acceptance criteria
PASS only if:

- All prompt files exist.
- Schema validation works.
- Example prompt can be rendered from one sample.
- Output schema is strict JSON, not free text.

## Status JSON

```json
{
  "step": "07_RATIONALE_SCHEMA_AND_PROMPTS",
  "status": "PASS|FAIL",
  "prompt_files": [],
  "schema_file": "src/llm/rationale_schema.py",
  "example_rendered": true,
  "notes": "..."
}
```
