# 17 — Paper Tables with Negative-Result Safety

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Build tables that never overclaim. If metrics are negative, tables should still be publishable as failure analysis, but not as method superiority evidence.

## Inputs
- metrics from Steps 07–16
- baseline tables already in `outputs/tables/`

## Outputs
- `outputs/tables/current_v3_table_prediction.csv`
- `outputs/tables/current_v3_table_explanation.csv`
- `outputs/tables/current_v3_table_flow_reward.csv`
- `outputs/tables/current_v3_table_backtest.csv`
- `outputs/tables/current_v3_table_ablation.csv`
- `outputs/metrics/current_v3_claim_matrix.json`
- `outputs/status/17_PAPER_TABLES_NEGATIVE_RESULTS_GATES.status.json`

## Claim matrix requirements
Create explicit booleans:
```json
{
  "prediction_improvement_claim_allowed": false,
  "flow_reward_improvement_claim_allowed": false,
  "counterfactual_faithfulness_claim_allowed": false,
  "trading_alpha_claim_allowed": false,
  "reproducibility_claim_allowed": true
}
```

## Codex task
Create `src/eval/build_current_v3_paper_tables.py`.

## Gate rules
- Trading alpha claim allowed only if Sharpe > baseline and > 0, test days >= 60, costs included.
- Flow claim allowed only if Flow V3 beats proxy by configured margins.
- Counterfactual claim allowed only if pass rate improves and no-change rate decreases.
- Prediction claim allowed only if macro-F1/MCC beats strongest baseline with bootstrap CI or paired test support.
- Reproducibility claim allowed if artifacts/manifests/status are complete.

## Run
```bash
python -m src.eval.build_current_v3_paper_tables \
  --output-dir outputs/tables \
  --claim-matrix outputs/metrics/current_v3_claim_matrix.json \
  --status outputs/status/17_PAPER_TABLES_NEGATIVE_RESULTS_GATES.status.json
```

## Verification
```bash
python - <<'PY'
import json, pathlib
cm=json.load(open("outputs/metrics/current_v3_claim_matrix.json"))
assert "trading_alpha_claim_allowed" in cm
for f in ["current_v3_table_prediction.csv","current_v3_table_explanation.csv","current_v3_table_flow_reward.csv","current_v3_table_backtest.csv","current_v3_table_ablation.csv"]:
    assert pathlib.Path("outputs/tables", f).exists(), f
print(cm)
PY
```

## Acceptance criteria
- Tables exist even if results are negative.
- Claim matrix blocks unsupported claims.
- No dummy zero-fill values.
- NOT_RUN rows are not used as evidence.
