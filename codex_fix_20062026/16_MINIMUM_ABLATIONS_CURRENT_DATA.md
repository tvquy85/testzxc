# 16 — Minimum Real Ablations on Current Data

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Replace NOT_RUN ablations with real current-data evidence.

## Required ablations
- Full_Current_V3
- No_Technical_Tokens
- No_News_Body
- No_Flow_Reward
- SFT_Only

If full retrain is too expensive, run inference ablations on 1,000 rows and mark `ablation_type=inference_ablation`.

## Outputs
- `outputs/tables/ablation_current_v3.csv`
- `outputs/metrics/ablation_current_v3.json`
- `outputs/status/16_MINIMUM_ABLATIONS_CURRENT_DATA.status.json`

## Codex task
Create `src/eval/run_current_ablation_suite_v3.py`.

## Run
```bash
python -m src.eval.run_current_ablation_suite_v3 \
  --contexts data/processed/ticker_date_contexts_h1_v2_targets.parquet \
  --checkpoint-root checkpoints/aligned/qwen3_4b \
  --max-eval-rows 1000 \
  --output-table outputs/tables/ablation_current_v3.csv \
  --metrics outputs/metrics/ablation_current_v3.json \
  --status outputs/status/16_MINIMUM_ABLATIONS_CURRENT_DATA.status.json
```

## Verify
```bash
python - <<'PY'
import pandas as pd, json
df=pd.read_csv("outputs/tables/ablation_current_v3.csv")
assert len(df)>=5
assert not df["status"].eq("NOT_RUN").any()
print(df)
print(json.load(open("outputs/metrics/ablation_current_v3.json")))
PY
```

## Acceptance
- At least 5 real ablations.
- No NOT_RUN rows.
- If inference-only, explicitly state it.
