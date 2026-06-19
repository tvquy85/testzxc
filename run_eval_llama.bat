@echo off
echo Running generate_eval_predictions for Llama...
python src/eval/generate_eval_predictions.py --checkpoint outputs/checkpoints/dpo_llama_h1_dryrun --limit 100 --output outputs/metrics/test_predictions_llama.jsonl

echo Running evaluate_prediction_metrics for Llama...
python src/eval/evaluate_prediction_metrics.py --pred outputs/metrics/test_predictions_llama.jsonl --output outputs/metrics/final_prediction_metrics_llama.json

echo Running evaluate_grounding_metrics for Llama...
python src/eval/evaluate_grounding_metrics.py --pred outputs/metrics/test_predictions_llama.jsonl --output outputs/metrics/final_explanation_metrics_llama.json

echo Running backtest_long_short_hold for Llama...
python src/eval/backtest_long_short_hold.py --pred outputs/metrics/test_predictions_llama.jsonl --output outputs/metrics/final_backtest_llama.json

echo Running evaluate_counterfactual_consistency for Llama...
python src/eval/evaluate_counterfactual_consistency.py --checkpoint outputs/checkpoints/dpo_llama_h1_dryrun --input outputs/metrics/test_cf_contexts_h1.jsonl --output outputs/metrics/final_cf_metrics_llama.json

echo ALL EVALUATIONS COMPLETED.
