from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_manifest, write_status

STEP = "20_RUNBOOK_AND_NEGATIVE_RESULT_FALLBACK"

NEGATIVE_FALLBACK = (
    "The current-data V6 pipeline improves reproducibility and evidence grounding, "
    "but does not yet establish strong forecast or trading improvement. Flow reward "
    "should be interpreted as a diagnostic experiment until it beats proxy. Repaired "
    "DPO now beats RWSFT on point metrics, but still does not beat the technical "
    "baseline under strict validation."
)


def read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def markdown_claim_table(claims: pd.DataFrame) -> str:
    if claims.empty:
        return "No claim table rows found.\n"
    def cell(value: Any) -> str:
        return "" if pd.isna(value) else str(value)

    rows = ["| Claim | Allowed | Block Reason |", "|---|---:|---|"]
    for _, row in claims.iterrows():
        rows.append(
            f"| {cell(row.get('claim', ''))} | {cell(row.get('claim_allowed', ''))} | {cell(row.get('claim_block_reason', ''))} |"
        )
    return "\n".join(rows) + "\n"


def build_runbook(gate: dict[str, Any], claims: pd.DataFrame) -> str:
    blocked = claims[claims["claim_allowed"].astype(str).str.lower().isin(["false", "0"])] if not claims.empty else pd.DataFrame()
    allowed = claims[claims["claim_allowed"].astype(str).str.lower().isin(["true", "1"])] if not claims.empty else pd.DataFrame()
    return f"""# RUNBOOK Current-Data V6

## Scope
Current-data only. Do not use SN2. Do not expand full FNSPID for this V6 recovery run.

## Environment And Model Paths
- Working directory: `D:\\Conferences\\firefin`
- Python used for verified commands: `D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe`
- Local model/cache root: `E:\\huggingface`
- Main local LLM: `E:/huggingface/models/Qwen3-4B-Instruct-2507`
- DPO adapter: `outputs/models/qwen3_current_v6_dpo_adapter`
- RWSFT adapter: `outputs/models/qwen3_current_v6_rwsft_adapter`

## Final Gate Decision
- pipeline_decision: `{gate.get('pipeline_decision')}`
- claim_decision: `{gate.get('claim_decision')}`
- strong_accept_ready: `{gate.get('strong_accept_ready')}`

{NEGATIVE_FALLBACK}

## Command Order
Follow `antigravity_currentdata_v6_md/testzxc_antigravity_currentdata_v6_md/00_MASTER_ANTIGRAVITY_ORDER.md`.

Final verified command pattern:
```powershell
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m pytest -q tests
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.utils.verify_status --status outputs/status/<STEP>.status.json
```

Key terminal commands for final stages:
```powershell
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.llm.audit_rationale_template_decomposition_v6 --input data/rationales/parsed/current_v6_train_qwen3_1000x3_repaired.parquet --metrics outputs/metrics/07_5_v6_rationale_template_decomposition.json --table outputs/tables/07_5_v6_rationale_template_decomposition.csv --samples review_samples/currentdata_v6/07_5_rationale_decomposition_examples.jsonl --status outputs/status/07_5_RATIONALE_TEMPLATE_DECOMPOSITION_V6.status.json --manifest outputs/manifests/07_5_RATIONALE_TEMPLATE_DECOMPOSITION_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.reward.diagnose_flow_utility_v6 --predictions outputs/tables/11_v6_flow_predictions.csv --split val --metrics outputs/metrics/11_5_v6_flow_utility_diagnostic.json --summary-table outputs/tables/11_5_v6_flow_method_summary.csv --overlap-table outputs/tables/11_5_v6_flow_top_decile_overlap.csv --quantile-table outputs/tables/11_5_v6_flow_score_deciles.csv --examples review_samples/currentdata_v6/11_5_flow_pair_disagreement_examples.jsonl --status outputs/status/11_5_FLOW_UTILITY_SURFACE_DIAGNOSTIC_V6.status.json --manifest outputs/manifests/11_5_FLOW_UTILITY_SURFACE_DIAGNOSTIC_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.reward.flow_utility_reranker_probe_v6 --predictions outputs/tables/11_v6_flow_predictions.csv --summary-output outputs/tables/11_6_v6_flow_utility_reranker_summary.csv --predictions-output outputs/tables/11_6_v6_flow_utility_reranker_predictions.csv --metrics outputs/metrics/11_6_v6_flow_utility_reranker_probe.json --status outputs/status/11_6_FLOW_UTILITY_RERANKER_PROBE_V6.status.json --manifest outputs/manifests/11_6_FLOW_UTILITY_RERANKER_PROBE_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.reward.flow_reranker_ablation_attribution_v6 --predictions outputs/tables/11_v6_flow_predictions.csv --output outputs/tables/11_7_v6_flow_reranker_ablation_attribution.csv --grid-output outputs/tables/11_7_v6_flow_reranker_ablation_grid.csv --metrics outputs/metrics/11_7_v6_flow_reranker_ablation_attribution.json --status outputs/status/11_7_FLOW_RERANKER_ABLATION_ATTRIBUTION_V6.status.json --manifest outputs/manifests/11_7_FLOW_RERANKER_ABLATION_ATTRIBUTION_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.trading_policy_variant_probe_v6 --contexts data/processed/current_v6_prediction_contexts.parquet --dpo-predictions outputs/predictions/current_v6_dpo_predictions.parquet --hybrid-predictions outputs/predictions/current_v6_validation_calibrated_hybrid_predictions.parquet --stacked-predictions outputs/predictions/current_v6_validation_stacked_forecast_predictions.parquet --supervised-predictions outputs/predictions/current_v6_supervised_signal_ceiling_predictions.parquet --summary-output outputs/tables/15_5_v6_trading_policy_variant_summary.csv --daily-output outputs/tables/15_5_v6_trading_policy_variant_daily_returns.csv --metrics outputs/metrics/15_5_v6_trading_policy_variant_probe.json --status outputs/status/15_5_TRADING_POLICY_VARIANT_PROBE_V6.status.json --manifest outputs/manifests/15_5_TRADING_POLICY_VARIANT_PROBE_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.validation_selected_trading_policy_v6 --val-contexts data/processed/current_v6_validation_contexts.parquet --val-predictions outputs/predictions/current_v6_dpo_val_predictions_repaired.parquet --test-contexts data/processed/current_v6_prediction_contexts.parquet --test-predictions outputs/predictions/current_v6_dpo_predictions_repaired.parquet --thresholds 0.00:0.90:0.01 --position-caps 1,2,3,5,10,20 --min-val-nonzero-days 5 --grid-output outputs/tables/15_6_v6_validation_selected_trading_grid.csv --summary-output outputs/tables/15_6_v6_validation_selected_trading_summary.csv --daily-output outputs/tables/15_6_v6_validation_selected_trading_daily_returns.csv --metrics outputs/metrics/15_6_v6_validation_selected_trading_policy.json --status outputs/status/15_6_VALIDATION_SELECTED_TRADING_POLICY_V6.status.json --manifest outputs/manifests/15_6_VALIDATION_SELECTED_TRADING_POLICY_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.audit_counterfactual_eligibility_v6 --tasks data/eval/current_v6_counterfactual_tasks.jsonl --predictions outputs/predictions/current_v6_dpo_predictions.parquet --breakdown outputs/tables/16_v6_counterfactual_breakdown.csv --output outputs/tables/16_5_v6_counterfactual_eligibility_by_type.csv --task-output outputs/tables/16_5_v6_counterfactual_task_eligibility.csv --metrics outputs/metrics/16_5_v6_counterfactual_eligibility_audit.json --status outputs/status/16_5_COUNTERFACTUAL_ELIGIBILITY_AUDIT_V6.status.json --manifest outputs/manifests/16_5_COUNTERFACTUAL_ELIGIBILITY_AUDIT_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.build_counterfactual_quality_filtered_v6 --contexts data/processed/current_v6_prediction_contexts.parquet --output data/eval/current_v6_counterfactual_quality_filtered_tasks.jsonl --candidate-output outputs/tables/16_6_v6_counterfactual_quality_candidates.csv --selected-output outputs/tables/16_6_v6_counterfactual_quality_selected.csv --summary-output outputs/tables/16_6_v6_counterfactual_quality_by_type.csv --metrics outputs/metrics/16_6_v6_counterfactual_quality_filtered_tasks.json --status outputs/status/16_6_COUNTERFACTUAL_QUALITY_FILTERED_TASKS_V6.status.json --manifest outputs/manifests/16_6_COUNTERFACTUAL_QUALITY_FILTERED_TASKS_V6.manifest.json --per-type-limit 30 --min-per-type 8
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.evaluate_counterfactual_directional_v4 --tasks data/eval/current_v6_counterfactual_quality_filtered_tasks.jsonl --adapter outputs/models/qwen3_current_v6_dpo_adapter --config configs/local_paths.yaml --model-key main_explanation_llm --hf-home E:/huggingface --output outputs/metrics/16_6_v6_counterfactual_quality_filtered_eval.json --breakdown-output outputs/tables/16_6_v6_counterfactual_quality_filtered_breakdown.csv --failures-output review_samples/currentdata_v6/16_6_counterfactual_quality_filtered_failures.jsonl --status outputs/status/16_6_COUNTERFACTUAL_QUALITY_FILTERED_EVAL_V6.status.json --manifest outputs/manifests/16_6_COUNTERFACTUAL_QUALITY_FILTERED_EVAL_V6.manifest.json --step-name 16_6_COUNTERFACTUAL_QUALITY_FILTERED_EVAL_V6 --max-tasks 192 --batch-size 8 --max-new-tokens 128 --min-schema-ok-rate 0.90 --min-pass-rate 0.50 --max-no-change-rate 0.35
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.build_counterfactual_quality_filtered_v6 --contexts data/processed/current_v6_prediction_contexts.parquet --output data/eval/current_v6_counterfactual_semantic_neutralized_tasks.jsonl --candidate-output outputs/tables/16_7_v6_counterfactual_semantic_candidates.csv --selected-output outputs/tables/16_7_v6_counterfactual_semantic_selected.csv --summary-output outputs/tables/16_7_v6_counterfactual_semantic_by_type.csv --metrics outputs/metrics/16_7_v6_counterfactual_semantic_neutralized_tasks.json --status outputs/status/16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_TASKS_V6.status.json --manifest outputs/manifests/16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_TASKS_V6.manifest.json --step-name 16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_TASKS_V6 --per-type-limit 30 --min-per-type 8 --semantic-neutralization
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.evaluate_counterfactual_directional_v4 --tasks data/eval/current_v6_counterfactual_semantic_neutralized_tasks.jsonl --adapter outputs/models/qwen3_current_v6_dpo_adapter --config configs/local_paths.yaml --model-key main_explanation_llm --hf-home E:/huggingface --output outputs/metrics/16_7_v6_counterfactual_semantic_eval.json --breakdown-output outputs/tables/16_7_v6_counterfactual_semantic_breakdown.csv --failures-output review_samples/currentdata_v6/16_7_counterfactual_semantic_failures.jsonl --rows-output outputs/tables/16_7_v6_counterfactual_semantic_rows.csv --status outputs/status/16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_EVAL_V6.status.json --manifest outputs/manifests/16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_EVAL_V6.manifest.json --step-name 16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_EVAL_V6 --max-tasks 192 --batch-size 8 --max-new-tokens 128 --min-schema-ok-rate 0.90 --min-pass-rate 0.50 --max-no-change-rate 0.35
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.baselines.run_v6_comparable_baselines --contexts data/processed/current_v6_prediction_contexts.parquet --dpo-predictions outputs/predictions/current_v6_dpo_predictions.parquet --rwsft-predictions outputs/predictions/current_v6_rwsft_predictions.parquet --output outputs/tables/17_v6_comparable_baselines.csv --metrics outputs/metrics/17_v6_baseline_comparison.json --status outputs/status/17_BASELINES_SEP_POLICY_TECH_RULE.status.json --manifest outputs/manifests/17_BASELINES_SEP_POLICY_TECH_RULE.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.repair_forecast_predictions_v6 --predictions outputs/predictions/current_v6_dpo_predictions.parquet --contexts data/processed/current_v6_prediction_contexts.parquet --output outputs/predictions/current_v6_dpo_predictions_repaired.parquet --metrics outputs/metrics/14_6_v6_dpo_forecast_distribution_repair.json --status outputs/status/14_6_DPO_FORECAST_DISTRIBUTION_REPAIR_V6.status.json --manifest outputs/manifests/14_6_DPO_FORECAST_DISTRIBUTION_REPAIR_V6.manifest.json --min-schema-ok-rate 0.99
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.repair_forecast_predictions_v6 --predictions outputs/predictions/current_v6_rwsft_predictions.parquet --contexts data/processed/current_v6_prediction_contexts.parquet --output outputs/predictions/current_v6_rwsft_predictions_repaired.parquet --metrics outputs/metrics/14_6_v6_rwsft_forecast_distribution_repair.json --status outputs/status/14_6_RWSFT_FORECAST_DISTRIBUTION_REPAIR_V6.status.json --manifest outputs/manifests/14_6_RWSFT_FORECAST_DISTRIBUTION_REPAIR_V6.manifest.json --min-schema-ok-rate 0.99
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.validation_calibrated_hybrid_v6 --val-contexts data/processed/current_v6_validation_contexts.parquet --val-predictions outputs/predictions/current_v6_dpo_val_predictions.parquet --test-contexts data/processed/current_v6_prediction_contexts.parquet --test-predictions outputs/predictions/current_v6_dpo_predictions.parquet --output outputs/tables/17_5_v6_validation_calibrated_hybrid.csv --threshold-table outputs/tables/17_5_v6_validation_threshold_search.csv --predictions-output outputs/predictions/current_v6_validation_calibrated_hybrid_predictions.parquet --metrics outputs/metrics/17_5_v6_validation_calibrated_hybrid.json --status outputs/status/17_5_VALIDATION_CALIBRATED_FORECAST_HYBRID_V6.status.json --manifest outputs/manifests/17_5_VALIDATION_CALIBRATED_FORECAST_HYBRID_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.validation_stacked_forecast_probe_v6 --val-contexts data/processed/current_v6_validation_contexts.parquet --val-predictions outputs/predictions/current_v6_dpo_val_predictions.parquet --test-contexts data/processed/current_v6_prediction_contexts.parquet --test-predictions outputs/predictions/current_v6_dpo_predictions.parquet --output outputs/tables/17_6_v6_validation_stacked_forecast_probe.csv --grid-output outputs/tables/17_6_v6_validation_stacked_grid.csv --predictions-output outputs/predictions/current_v6_validation_stacked_forecast_predictions.parquet --metrics outputs/metrics/17_6_v6_validation_stacked_forecast_probe.json --status outputs/status/17_6_VALIDATION_STACKED_FORECAST_PROBE_V6.status.json --manifest outputs/manifests/17_6_VALIDATION_STACKED_FORECAST_PROBE_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.supervised_signal_ceiling_probe_v6 --train-contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --val-contexts data/processed/current_v6_validation_contexts.parquet --test-contexts data/processed/current_v6_prediction_contexts.parquet --output outputs/tables/17_7_v6_supervised_signal_ceiling_probe.csv --grid-output outputs/tables/17_7_v6_supervised_signal_ceiling_grid.csv --predictions-output outputs/predictions/current_v6_supervised_signal_ceiling_predictions.parquet --metrics outputs/metrics/17_7_v6_supervised_signal_ceiling_probe.json --status outputs/status/17_7_SUPERVISED_SIGNAL_CEILING_PROBE_V6.status.json --manifest outputs/manifests/17_7_SUPERVISED_SIGNAL_CEILING_PROBE_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.repaired_forecast_baseline_probe_v6 --contexts data/processed/current_v6_prediction_contexts.parquet --dpo-original outputs/predictions/current_v6_dpo_predictions.parquet --dpo-repaired outputs/predictions/current_v6_dpo_predictions_repaired.parquet --rwsft-repaired outputs/predictions/current_v6_rwsft_predictions_repaired.parquet --output outputs/tables/17_8_v6_repaired_forecast_baseline_probe.csv --metrics outputs/metrics/17_8_v6_repaired_forecast_baseline_probe.json --status outputs/status/17_8_REPAIRED_FORECAST_BASELINE_PROBE_V6.status.json --manifest outputs/manifests/17_8_REPAIRED_FORECAST_BASELINE_PROBE_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.eval.run_v6_ablation_suite --contexts data/processed/current_v6_prediction_contexts.parquet --predictions outputs/predictions/current_v6_dpo_predictions.parquet --baselines outputs/tables/17_v6_comparable_baselines.csv --output outputs/tables/18_v6_ablation_results.csv --metrics outputs/metrics/18_v6_ablation_summary.json --status outputs/status/18_ABLATIONS_V6.status.json --manifest outputs/manifests/18_ABLATIONS_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.repro.run_v6_statistical_tests --daily-returns outputs/tables/15_v6_daily_returns.csv --contexts data/processed/current_v6_prediction_contexts.parquet --dpo-predictions outputs/predictions/current_v6_dpo_predictions.parquet --rwsft-predictions outputs/predictions/current_v6_rwsft_predictions.parquet --backtest-metrics outputs/metrics/15_v6_backtest_track_baseline.json --output outputs/tables/19_v6_statistical_tests.csv --daily-comparison-output outputs/tables/19_v6_backtest_daily_comparison.csv --metrics outputs/metrics/19_v6_statistical_tests.json --status outputs/status/18_5_STATISTICAL_TESTS_AND_CI_V6.status.json --manifest outputs/manifests/18_5_STATISTICAL_TESTS_AND_CI_V6.manifest.json
D:\\LOBProj\\LOBExp\\.venv\\Scripts\\python.exe -m src.repro.currentdata_v6_strong_accept_gate --metrics-dir outputs/metrics --tables-dir outputs/tables --status-dir outputs/status --output outputs/repro/currentdata_v6_strong_accept_gate.json --claim-table outputs/tables/19_v6_claim_matrix.csv --status outputs/status/19_STRICT_AAAI_STRONG_ACCEPT_GATE.status.json --manifest outputs/manifests/19_STRICT_AAAI_STRONG_ACCEPT_GATE.manifest.json
```

## Artifacts By Step
- Step 04: `data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet`
- Step 07.5: `outputs/metrics/07_5_v6_rationale_template_decomposition.json`, `outputs/tables/07_5_v6_rationale_template_decomposition.csv`
- Step 08: `data/judges/current_v6_independent_judge_ensemble.parquet`
- Step 11.5: `outputs/metrics/11_5_v6_flow_utility_diagnostic.json`, `outputs/tables/11_5_v6_flow_method_summary.csv`
- Step 11.6: `outputs/tables/11_6_v6_flow_utility_reranker_summary.csv`, `outputs/metrics/11_6_v6_flow_utility_reranker_probe.json`
- Step 11.7: `outputs/tables/11_7_v6_flow_reranker_ablation_attribution.csv`, `outputs/metrics/11_7_v6_flow_reranker_ablation_attribution.json`
- Step 12: `data/alignment/current_v6_rwsft.jsonl`, `data/alignment/current_v6_dpo_pairs.jsonl`
- Step 13: `outputs/models/qwen3_current_v6_rwsft_adapter`, `outputs/models/qwen3_current_v6_dpo_adapter`
- Step 14: `outputs/predictions/current_v6_dpo_predictions.parquet`, `outputs/predictions/current_v6_rwsft_predictions.parquet`
- Step 14.6: `outputs/predictions/current_v6_dpo_predictions_repaired.parquet`, `outputs/metrics/14_6_v6_dpo_forecast_distribution_repair.json`
- Step 14.5: `data/processed/current_v6_validation_contexts.parquet`, `outputs/predictions/current_v6_dpo_val_predictions.parquet`
- Step 15: `outputs/metrics/15_v6_backtest_track_baseline.json`, `outputs/tables/15_v6_daily_returns.csv`
- Step 15.5: `outputs/tables/15_5_v6_trading_policy_variant_summary.csv`, `outputs/metrics/15_5_v6_trading_policy_variant_probe.json`
- Step 15.6: `outputs/tables/15_6_v6_validation_selected_trading_summary.csv`, `outputs/metrics/15_6_v6_validation_selected_trading_policy.json`
- Step 16: `outputs/metrics/16_v6_counterfactual_news_evidence.json`
- Step 16.5: `outputs/metrics/16_5_v6_counterfactual_eligibility_audit.json`, `outputs/tables/16_5_v6_counterfactual_eligibility_by_type.csv`
- Step 16.6: `data/eval/current_v6_counterfactual_quality_filtered_tasks.jsonl`, `outputs/metrics/16_6_v6_counterfactual_quality_filtered_eval.json`
- Step 16.7: `data/eval/current_v6_counterfactual_semantic_neutralized_tasks.jsonl`, `outputs/metrics/16_7_v6_counterfactual_semantic_eval.json`
- Step 17: `outputs/tables/17_v6_comparable_baselines.csv`
- Step 17.5: `outputs/tables/17_5_v6_validation_calibrated_hybrid.csv`, `outputs/metrics/17_5_v6_validation_calibrated_hybrid.json`
- Step 17.6: `outputs/tables/17_6_v6_validation_stacked_forecast_probe.csv`, `outputs/metrics/17_6_v6_validation_stacked_forecast_probe.json`
- Step 17.7: `outputs/tables/17_7_v6_supervised_signal_ceiling_probe.csv`, `outputs/metrics/17_7_v6_supervised_signal_ceiling_probe.json`
- Step 17.8: `outputs/tables/17_8_v6_repaired_forecast_baseline_probe.csv`, `outputs/metrics/17_8_v6_repaired_forecast_baseline_probe.json`
- Step 18: `outputs/tables/18_v6_ablation_results.csv`
- Step 18.5: `outputs/tables/19_v6_statistical_tests.csv`, `outputs/metrics/19_v6_statistical_tests.json`
- Step 19: `outputs/repro/currentdata_v6_strong_accept_gate.json`, `outputs/tables/19_v6_claim_matrix.csv`

## Claim Matrix
{markdown_claim_table(claims)}

## Known Failures
{chr(10).join(f"- {row.get('claim')}: {row.get('claim_block_reason')}" for _, row in blocked.iterrows()) if not blocked.empty else "- None"}

## What Can Be Claimed
{chr(10).join(f"- {row.get('claim')}" for _, row in allowed.iterrows()) if not allowed.empty else "- Pipeline execution only"}

## What Cannot Be Claimed
- Strong-accept readiness.
- Flow reward improvement over proxy.
- Flow-attributed reranker improvement; Step 11.7 shows no-flow ablation matches/exceeds the full reranker feature set.
- Forecast superiority over Technical_Rule.
- Forecast superiority after DPO probability repair; Step 17.8 still trails Technical_Rule.
- Validation-stacked forecast superiority over Technical_Rule.
- Supervised signal-ceiling superiority over Technical_Rule.
- Paper-level trading alpha with confidence intervals that support positive alpha.
- Post-hoc best-of-many trading policy alpha from Step 15.5.
- Validation-selected trading alpha from Step 15.6 as paper-level evidence while absolute Sharpe/mean-return CIs cross zero.

## Table Reproduction
Use the CSV artifacts in `outputs/tables/`:
- `17_v6_comparable_baselines.csv`
- `18_v6_ablation_results.csv`
- `19_v6_statistical_tests.csv`
- `19_v6_claim_matrix.csv`

## Next Full-Scale Path
1. Promote Step 14.6 repaired predictions only if all downstream artifacts are rerun consistently; Step 17.8 shows repaired DPO beats RWSFT but still trails Technical_Rule, so the main forecast objective remains open.
2. Train/evaluate an official Flow/listwise reward checkpoint with top-decile utility terms and ablations; Step 11.6 shows a pairwise utility reranker can win rank/pair, but Step 11.7 shows that win is not attributable to Flow-specific signal.
3. Preserve the Step 16.7 semantic-neutralized counterfactual protocol in future reruns; do not revert to brittle token-level neutralization.
4. Validate the Step 15.6 preregistered DPO threshold/cap policy on a fresh out-of-sample horizon or larger current-data expansion before any paper-level alpha claim; validation selection improved point Sharpe but absolute CIs still cross zero.
5. Re-run Steps 14-19 after each gated fix.
"""


def build_negative_summary(gate: dict[str, Any], claims: pd.DataFrame) -> str:
    blocked = claims[claims["claim_allowed"].astype(str).str.lower().isin(["false", "0"])] if not claims.empty else pd.DataFrame()
    return f"""# Current-Data V6 Negative Result Summary

Final decision: `{gate.get('claim_decision')}`.

{NEGATIVE_FALLBACK}

Blocked claims:
{chr(10).join(f"- {row.get('claim')}: {row.get('claim_block_reason')}" for _, row in blocked.iterrows()) if not blocked.empty else "- None"}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument("--claim-table", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-summary", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.gate).exists():
        failures.append(f"gate missing: {args.gate}")
    if not Path(args.claim_table).exists():
        failures.append(f"claim table missing: {args.claim_table}")
    gate = read_json(args.gate)
    claims = pd.read_csv(args.claim_table) if Path(args.claim_table).exists() else pd.DataFrame()
    if gate.get("claim_decision") != "CLAIM_RESTRICTED":
        failures.append("runbook expected CLAIM_RESTRICTED gate for negative-result fallback")
    if claims.empty:
        failures.append("claim table is empty")

    Path(args.output).write_text(build_runbook(gate, claims), encoding="utf-8")
    Path(args.negative_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.negative_summary).write_text(build_negative_summary(gate, claims), encoding="utf-8")
    write_manifest(args.manifest, [args.gate, args.claim_table, args.output, args.negative_summary], STEP)
    metrics = {
        "claim_decision": gate.get("claim_decision"),
        "runbook_written": Path(args.output).exists(),
        "negative_summary_written": Path(args.negative_summary).exists(),
        "claim_rows": int(len(claims)),
        "pipeline_pass": not failures,
        "claim_allowed": False,
    }
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.gate, args.claim_table],
        [args.output, args.negative_summary, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
