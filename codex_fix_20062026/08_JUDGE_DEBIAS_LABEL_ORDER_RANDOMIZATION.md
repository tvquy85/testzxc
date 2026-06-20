# 08 — Judge Debias via Label-Order Randomization

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Measure and reduce LLM judge position bias.

## Inputs
- `data/judges/independent_inferability_v3.parquet`
- contexts and rationales from prior steps

## Outputs
- `data/judges/independent_inferability_v3_debiased.parquet`
- `outputs/metrics/judge_debias_v3.json`
- `outputs/status/08_JUDGE_DEBIAS_LABEL_ORDER_RANDOMIZATION.status.json`

## Codex task
Create `src/judges/judge_debias_label_order_v3.py`.

## Required behavior
- Run judge twice with different label orders on 1,000 rows.
- Convert to canonical label order.
- Compute `argmax_consistency`, `mean_l1_probability_delta`, `true_label_prob_delta_mean`.
- Average the probability vectors if stability is acceptable.

## Run
```bash
python -m src.judges.judge_debias_label_order_v3 \
  --contexts data/processed/ticker_date_contexts_h1_v2_targets.parquet \
  --rationales data/rationales/parsed/current_clean_train_qwen3_4b_v3.parquet \
  --base-judge data/judges/independent_inferability_v3.parquet \
  --limit 1000 \
  --judge-model qwen3_4b \
  --output data/judges/independent_inferability_v3_debiased.parquet \
  --metrics outputs/metrics/judge_debias_v3.json \
  --status outputs/status/08_JUDGE_DEBIAS_LABEL_ORDER_RANDOMIZATION.status.json
```

## Verify
```bash
python - <<'PY'
import json, pandas as pd
m=json.load(open("outputs/metrics/judge_debias_v3.json"))
assert m["evaluated_rows"] >= 500
assert m["argmax_consistency"] >= 0.55
assert m["mean_l1_probability_delta"] <= 0.80
assert len(pd.read_parquet("data/judges/independent_inferability_v3_debiased.parquet"))>0
print(m)
PY
```

## Acceptance
- Argmax consistency >= 0.55 for current MVP.
- Debiased file becomes default judge target.
