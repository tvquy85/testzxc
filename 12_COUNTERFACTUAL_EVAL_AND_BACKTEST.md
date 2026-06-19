# Step 12 — Counterfactual Evaluation and Backtest

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Evaluate forecasting quality, explanation faithfulness, and trading utility.

## Inputs

```text
outputs/checkpoints/rwsft_qwen_h1/
outputs/checkpoints/dpo_qwen_h1/   # optional
data/labels/aligned_samples_h1.parquet
data/indicators/technical_event_tokens_h1.parquet
data/indicators/technical_features_h1.parquet
```

## Outputs

```text
outputs/metrics/final_prediction_metrics_h1.json
outputs/metrics/final_explanation_metrics_h1.json
outputs/metrics/final_backtest_h1.json
outputs/tables/final_eval_summary_h1.csv
outputs/status/12_COUNTERFACTUAL_EVAL_AND_BACKTEST.status.json
```

## Tasks
Create:

```text
src/eval/generate_eval_predictions.py
src/eval/evaluate_prediction_metrics.py
src/eval/build_counterfactual_contexts.py
src/eval/evaluate_counterfactual_consistency.py
src/eval/evaluate_grounding_metrics.py
src/eval/backtest_long_short_hold.py
```

Use only the latest 15% dates as test.

Prediction metrics:

```text
accuracy, macro_f1, mcc, brier_score, ece, nll, confusion_matrix
```

Counterfactuals:

```text
CF_NEWS_REMOVE: remove strongest news factor if identifiable
CF_TECH_NEUTRALIZE: neutralize strongest technical token
CF_REGIME_FLIP: high_vol <-> low_vol if applicable
```

Backtest action rule:

```text
P_up = mild_up + strong_up
P_down = mild_down + strong_down
long if P_up > 0.60
short if P_down > 0.60
hold otherwise
```

Transaction cost: 5 bps per trade.

## Verification
Run:

```bash
cd firefin
python src/eval/generate_eval_predictions.py --checkpoint outputs/checkpoints/rwsft_qwen_h1 --limit 1000 --output outputs/metrics/test_predictions_h1.jsonl
python src/eval/evaluate_prediction_metrics.py --pred outputs/metrics/test_predictions_h1.jsonl --output outputs/metrics/final_prediction_metrics_h1.json
python src/eval/evaluate_grounding_metrics.py --pred outputs/metrics/test_predictions_h1.jsonl --output outputs/metrics/final_explanation_metrics_h1.json
python src/eval/backtest_long_short_hold.py --pred outputs/metrics/test_predictions_h1.jsonl --output outputs/metrics/final_backtest_h1.json
```

## Acceptance criteria
PASS only if test predictions, classification metrics, grounding metrics, and backtest metrics exist; no train/val sample may appear in test.

## Status JSON

```json
{
  "step": "12_COUNTERFACTUAL_EVAL_AND_BACKTEST",
  "status": "PASS|FAIL",
  "test_prediction_rows": 0,
  "macro_f1": 0.0,
  "mcc": 0.0,
  "sharpe": 0.0,
  "max_drawdown": 0.0,
  "counterfactual_eval_done": true,
  "notes": "..."
}
```
