@echo off
echo Running generate_eval_predictions for DeepSeek...
python src/eval/generate_eval_predictions.py --checkpoint outputs/checkpoints/dpo_deepseek_h1_dryrun --limit 100 --output outputs/metrics/test_predictions_deepseek.jsonl

echo Running evaluate_prediction_metrics for DeepSeek...
python src/eval/evaluate_prediction_metrics.py --pred outputs/metrics/test_predictions_deepseek.jsonl --output outputs/metrics/final_prediction_metrics_deepseek.json

echo Running evaluate_grounding_metrics for DeepSeek...
python src/eval/evaluate_grounding_metrics.py --pred outputs/metrics/test_predictions_deepseek.jsonl --output outputs/metrics/final_explanation_metrics_deepseek.json

echo Running backtest_long_short_hold for DeepSeek...
python src/eval/backtest_long_short_hold.py --pred outputs/metrics/test_predictions_deepseek.jsonl --output outputs/metrics/final_backtest_deepseek.json

echo Running evaluate_counterfactual_consistency for DeepSeek...
python src/eval/evaluate_counterfactual_consistency.py --checkpoint outputs/checkpoints/dpo_deepseek_h1_dryrun --input outputs/metrics/test_cf_contexts_h1.jsonl --output outputs/metrics/final_cf_metrics_deepseek.json

echo ALL EVALUATIONS COMPLETED.
