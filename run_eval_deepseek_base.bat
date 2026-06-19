@echo off
echo Running generate_eval_predictions for DeepSeek Base...
python src/eval/generate_eval_predictions.py --checkpoint e:/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 --limit 100 --output outputs/metrics/test_predictions_deepseek_base.jsonl

echo Running evaluate_prediction_metrics for DeepSeek Base...
python src/eval/evaluate_prediction_metrics.py --pred outputs/metrics/test_predictions_deepseek_base.jsonl --output outputs/metrics/final_prediction_metrics_deepseek_base.json

echo Running backtest_long_short_hold for DeepSeek Base...
python src/eval/backtest_long_short_hold.py --pred outputs/metrics/test_predictions_deepseek_base.jsonl --output outputs/metrics/final_backtest_deepseek_base.json

echo Running evaluate_counterfactual_consistency for DeepSeek Base...
python src/eval/evaluate_counterfactual_consistency.py --checkpoint e:/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 --input outputs/metrics/test_cf_contexts_h1.jsonl --output outputs/metrics/final_cf_metrics_deepseek_base.json

echo ALL EVALUATIONS COMPLETED.
