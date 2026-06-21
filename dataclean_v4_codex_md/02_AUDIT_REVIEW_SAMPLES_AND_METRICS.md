# 02 — Audit Current V3 Samples and Metrics

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Create a diagnostic report explaining why the current pipeline is weak before data cleaning.

## Inputs
- `review_samples/codex_fix_20062026/06_rationale_generation_samples.jsonl`
- `review_samples/codex_fix_20062026/15_counterfactual_task_samples.jsonl`
- `review_samples/codex_fix_20062026/10_flow_dataset_target_samples.json`
- `outputs/metrics/flow_vs_proxy_v3_1_eval.json`
- `outputs/metrics/backtest_daily_portfolio_current_v3.json`
- `outputs/metrics/counterfactual_directional_current_v3.json`
- `data/quality/current_data_quality_v2.parquet`

## Outputs
- `outputs/metrics/current_v3_failure_diagnosis_for_clean_v4.json`
- `outputs/tables/current_v3_failure_diagnosis_for_clean_v4.csv`
- `outputs/status/02_AUDIT_CURRENT_V3_FOR_CLEAN_V4.status.json`

## Required diagnostics
Report:
- no-news rationale rate;
- empty headline/body rate;
- technical-template rate;
- multi-company-like rate;
- flow-vs-proxy gap;
- Sharpe and mean daily return;
- counterfactual pass/no-change by perturbation type.

## Code snippet
```python
def is_no_news_rationale(parsed_json: dict) -> bool:
    items = parsed_json.get('news_rationale', []) or []
    text = ' '.join(str(x) for x in items).lower()
    return (not items) or 'no significant news' in text or 'no company-specific' in text

def looks_multi_company(text: str) -> bool:
    import re
    tickers = set(re.findall(r'\$?[A-Z]{1,5}', text or ''))
    return len(tickers) >= 6
```

## Acceptance criteria
Status PASS if JSON has numeric values for all diagnostics. This task must not change data/model artifacts.
