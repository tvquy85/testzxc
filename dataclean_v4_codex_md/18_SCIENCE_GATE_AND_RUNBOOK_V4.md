# 18 — Science Gate and Runbook V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Create the final science gate and provide the concise runbook.

## Output
- `outputs/repro/currentdata_clean_v4_science_gate_report.json`
- `outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json`

## Claim matrix
```json
{
  "data_cleaning_improved_context_quality": "allowed|blocked",
  "counterfactual_faithfulness": "allowed|blocked",
  "flow_reward_improvement": "allowed|blocked",
  "trading_alpha": "allowed|blocked",
  "multimodal_news_technical_reasoning": "allowed|blocked",
  "aaai_main_ready": "allowed|blocked"
}
```

## Gate rules
- If Sharpe <= 0, block trading alpha.
- If flow does not beat proxy in 2/3 gates, block flow improvement.
- If news counterfactual does not improve vs V3, block full multimodal faithfulness.
- If ablations missing or dummy, block AAAI-ready.
- Pipeline can GO even if claims are restricted.

## Command
```bash
python -m src.repro.currentdata_clean_v4_science_gate \
  --status-dir outputs/status \
  --output outputs/repro/currentdata_clean_v4_science_gate_report.json \
  --status outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json
```

## Final verification
```bash
python -m pytest -q tests
python -m src.utils.verify_status --status outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json
```
