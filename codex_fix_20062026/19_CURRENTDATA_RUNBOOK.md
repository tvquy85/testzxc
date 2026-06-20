# 19 — Current-Data Runbook

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Provide one concise runbook after all scripts are implemented.

## Run first-stage commands
```bash
python -m src.utils.freeze_current_baseline
python -m src.data.current_data_quality_audit --input data/labels/labels_h1_abnormal.parquet --splits data/processed/split_membership.parquet --output data/quality/current_data_quality_v2.parquet --metrics outputs/metrics/current_data_quality_v2.json --status outputs/status/02_CURRENT_DATA_QUALITY_AUDIT.status.json
python -m src.data.filter_current_samples_v2 --labels data/labels/labels_h1_abnormal.parquet --quality data/quality/current_data_quality_v2.parquet --splits data/processed/split_membership.parquet --min-quality 0.45 --output data/processed/current_filtered_samples_v2.parquet --metrics outputs/metrics/current_filtered_samples_v2.json --status outputs/status/03_ENTITY_EVENT_FILTER_CURRENT_DATA.status.json
python -m src.data.build_ticker_date_contexts_v2 --input data/processed/current_filtered_samples_v2.parquet --tokens data/indicators/technical_event_tokens_h1_v2.parquet --output data/processed/ticker_date_contexts_h1_v2.parquet --metrics outputs/metrics/ticker_date_contexts_h1_v2.json --status outputs/status/04_TICKER_DATE_CONTEXT_AGGREGATION.status.json
python -m src.data.ensure_abnormal_targets_v2 --contexts data/processed/ticker_date_contexts_h1_v2.parquet --labels data/labels/labels_h1_abnormal.parquet --output data/processed/ticker_date_contexts_h1_v2_targets.parquet --metrics outputs/metrics/target_integrity_h1_v2.json --status outputs/status/05_LABEL_BACKTEST_TARGET_FIX_ABNORMAL_RETURN.status.json
```

Then run rationale generation, judging, flow, alignment, backtest, counterfactual, ablation, and gate using files 06–18.

## First improvement target on current data
Do not demand final AAAI-quality numbers in the first run. Demand **directional improvement** over previous current metrics:
- independent inferability mean true-label probability > 0.214 or clearly explain why not;
- grounding supported rate improves over prior supported count/rate;
- Flow V3 either beats proxy or explicitly disables flow-improvement claim;
- counterfactual pass rate > 0.16 OR no-change rate < 0.696;
- backtest uses abnormal return and claim gate blocks alpha if Sharpe remains negative;
- ablations no longer NOT_RUN.

## Do not proceed to full scale until
- data filter and ticker-date aggregation produce stable contexts;
- independent judge is stable under label-order randomization;
- at least one current-data claim is allowed OR the project is reframed as failure analysis.
