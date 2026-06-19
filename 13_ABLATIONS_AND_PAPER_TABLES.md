# Step 13 — Ablations and Paper-Ready Tables

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Run compact ablations that directly support FIRE-Fin-Lite novelty claims.

## Inputs

```text
outputs/metrics/baseline_metrics_h1.json
outputs/metrics/final_prediction_metrics_h1.json
outputs/metrics/final_explanation_metrics_h1.json
outputs/metrics/final_backtest_h1.json
data/judge_outputs/flow_rewards_h1.parquet
```

## Outputs

```text
outputs/tables/table_1_prediction_main.csv
outputs/tables/table_2_explanation_quality.csv
outputs/tables/table_3_backtest.csv
outputs/tables/table_4_ablation.csv
outputs/figures/flow_vs_proxy_calibration.png
outputs/figures/conflict_subset_performance.png
outputs/status/13_ABLATIONS_AND_PAPER_TABLES.status.json
```

## Required ablations
Run only feasible ablations. Do not launch long retraining unless previous artifacts exist.

- A1: no technical indicators.
- A2: technical only.
- A3: no flow reward; use proxy score.
- A4: no grounding reward.
- A5: no regime-conditioned sigma; fixed sigma=1.0.
- A6: report metrics without counterfactual pass-rate.

## Tables

Table 1 prediction columns:

```text
method, input_modalities, accuracy, macro_f1, mcc, brier, ece
```

Table 2 explanation columns:

```text
method, inferability_acc, technical_grounding, news_entailment, contradiction_rate, json_valid_rate
```

Table 3 backtest columns:

```text
method, annual_return, sharpe, sortino, max_drawdown, turnover, coverage
```

Table 4 ablation columns:

```text
variant, removed_component, macro_f1, mcc, grounding_score, sharpe
```

## Scripts to create

```text
src/eval/run_ablation_from_existing_artifacts.py
src/eval/build_paper_tables.py
src/eval/plot_calibration.py
src/eval/plot_conflict_subset.py
```

## Verification
Run:

```bash
cd firefin
python src/eval/build_paper_tables.py --output-dir outputs/tables
python src/eval/plot_calibration.py --output outputs/figures/flow_vs_proxy_calibration.png
```

Then:

```bash
python - <<'PYCHECK'
from pathlib import Path
for p in [
 'outputs/tables/table_1_prediction_main.csv',
 'outputs/tables/table_2_explanation_quality.csv',
 'outputs/tables/table_3_backtest.csv',
 'outputs/tables/table_4_ablation.csv'
]:
    print(p, Path(p).exists(), Path(p).stat().st_size if Path(p).exists() else 0)
    assert Path(p).exists()
PYCHECK
```

## Acceptance criteria
PASS only if 4 CSV tables and at least one calibration figure are created, and missing ablations are marked `not_run` with reason.

## Status JSON

```json
{
  "step": "13_ABLATIONS_AND_PAPER_TABLES",
  "status": "PASS|FAIL",
  "tables_created": [],
  "figures_created": [],
  "ablations_completed": [],
  "ablations_not_run": {},
  "notes": "..."
}
```
