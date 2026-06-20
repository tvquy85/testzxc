# 15 — Counterfactual Directional Evaluation V3

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Improve counterfactual evaluation and diagnose failures.

## Inputs
- clean contexts
- technical tokens
- aligned model checkpoint

## Outputs
- `data/counterfactual/current_cf_tasks_v3.parquet`
- `outputs/metrics/counterfactual_directional_current_v3.json`
- `outputs/data_samples/counterfactual_fail_examples_current_v3.json`
- `outputs/status/15_COUNTERFACTUAL_DIRECTIONAL_FIX_CURRENT_DATA.status.json`

## Task types
- remove_negative_news: P(down) should decrease
- remove_positive_news: P(up) should decrease
- neutralize_bearish_technical: P(down) should decrease
- neutralize_bullish_technical: P(up) should decrease

Pass if expected probability changes by at least 0.05.

## Codex tasks
Create:
- `src/eval/build_counterfactual_current_v3.py`
- `src/eval/evaluate_counterfactual_directional_v3.py`

## Run
```bash
python -m src.eval.build_counterfactual_current_v3 \
  --contexts data/processed/ticker_date_contexts_h1_v2_targets.parquet \
  --tokens data/indicators/technical_event_tokens_h1_v2.parquet \
  --output data/counterfactual/current_cf_tasks_v3.parquet \
  --limit 500

python -m src.eval.evaluate_counterfactual_directional_v3 \
  --tasks data/counterfactual/current_cf_tasks_v3.parquet \
  --checkpoint checkpoints/aligned/qwen3_4b/current_v3_dpo \
  --temperature 0.0 \
  --min-delta 0.05 \
  --metrics outputs/metrics/counterfactual_directional_current_v3.json \
  --fail-examples outputs/data_samples/counterfactual_fail_examples_current_v3.json \
  --status outputs/status/15_COUNTERFACTUAL_DIRECTIONAL_FIX_CURRENT_DATA.status.json
```

## Verify
```bash
python - <<'PY'
import json
m=json.load(open("outputs/metrics/counterfactual_directional_current_v3.json"))
assert m["num_tasks"] >= 100
assert "pass_rate" in m and "no_change_rate" in m
print(m)
PY
```

## Acceptance
- Report by task type.
- Save failure examples.
- Target: pass_rate > 0.16 OR no_change_rate < 0.696.
