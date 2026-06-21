# 01 — Freeze Current V3 Baseline

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Freeze the current V3 metrics/artifacts before changing data cleaning.

## Inputs to verify
- `BaoCaoCodexFixSmallScale_20062026.md`
- `outputs/repro/currentdata_science_gate_report_v2.json`
- `outputs/metrics/flow_vs_proxy_v3_1_eval.json`
- `outputs/metrics/backtest_daily_portfolio_current_v3.json`
- `outputs/metrics/counterfactual_directional_current_v3.json`
- `data/processed/ticker_date_contexts_h1_v2_targets.parquet`

## Outputs
- `outputs/baseline_freeze/currentdata_v3_before_clean_v4/`
- `outputs/status/01_FREEZE_CURRENTDATA_V3_FOR_CLEAN_V4.status.json`
- `outputs/manifests/01_FREEZE_CURRENTDATA_V3_FOR_CLEAN_V4.manifest.json`

## Command
Implement `src.repro.freeze_currentdata_baseline` if missing, then run:

```bash
python -m src.repro.freeze_currentdata_baseline \
  --tag currentdata_v3_before_clean_v4 \
  --output-dir outputs/baseline_freeze/currentdata_v3_before_clean_v4 \
  --status outputs/status/01_FREEZE_CURRENTDATA_V3_FOR_CLEAN_V4.status.json \
  --manifest outputs/manifests/01_FREEZE_CURRENTDATA_V3_FOR_CLEAN_V4.manifest.json
```

## Verification
```bash
python -m src.utils.verify_status --status outputs/status/01_FREEZE_CURRENTDATA_V3_FOR_CLEAN_V4.status.json
```

## Acceptance criteria
- Status PASS.
- Manifest includes SHA256 and file sizes.
- No model weights/checkpoints are copied.
