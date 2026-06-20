# 07 — Independent Inferability Judge

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Build a judge that reads context + rationale and infers the return bucket. It must not read the generator's own forecast distribution as score.

## Inputs
- `data/processed/ticker_date_contexts_h1_v2_targets.parquet`
- `data/rationales/parsed/current_clean_train_qwen3_4b_v3.parquet`

## Outputs
- `data/judges/independent_inferability_v3.parquet`
- `outputs/metrics/independent_inferability_v3.json`
- `outputs/status/07_INDEPENDENT_INFERABILITY_JUDGE.status.json`

## Codex task
Create `src/judges/inferability_judge_independent_v3.py`.

## Non-negotiable
- Do not use `forecast_distribution` from rationale JSON.
- Do not include true label in prompt.
- After judge returns probabilities, compute `true_label_probability` from `target_label_5`.

## Required output columns
`sample_id`, `candidate_id`, `judge_model`, `p_strong_down`, `p_mild_down`, `p_neutral`, `p_mild_up`, `p_strong_up`, `predicted_label`, `true_label_probability`, `judge_parse_ok`, `judge_schema_ok`, `raw_judge_output`.

## Run
```bash
python -m src.judges.inferability_judge_independent_v3 \
  --contexts data/processed/ticker_date_contexts_h1_v2_targets.parquet \
  --rationales data/rationales/parsed/current_clean_train_qwen3_4b_v3.parquet \
  --limit 2000 \
  --judge-model qwen3_4b \
  --output data/judges/independent_inferability_v3.parquet \
  --metrics outputs/metrics/independent_inferability_v3.json \
  --status outputs/status/07_INDEPENDENT_INFERABILITY_JUDGE.status.json
```

## Verify
```bash
python - <<'PY'
import pandas as pd, json
df=pd.read_parquet("data/judges/independent_inferability_v3.parquet")
cols=["p_strong_down","p_mild_down","p_neutral","p_mild_up","p_strong_up"]
assert len(df)>0
assert ((df[cols].sum(axis=1)-1).abs()<1e-3).mean()>0.95
assert df["true_label_probability"].between(0,1).all()
print(json.load(open("outputs/metrics/independent_inferability_v3.json")))
PY
```

## Acceptance
- Judge parse/schema >= 0.90.
- Independent score stored.
