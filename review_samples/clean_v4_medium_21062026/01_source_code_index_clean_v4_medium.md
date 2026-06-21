# Source Code Index - Clean V4 Medium 21/06/2026

| File | Vai trò | Tồn tại |
|---|---|---|
| `src/repro/freeze_clean_v4_small_baseline.py` | Freeze baseline Clean V4 small before medium changes. | True |
| `src/repro/audit_clean_v4_failure_modes.py` | Audit known weak points from small-scale Clean V4. | True |
| `src/data/select_medium_clean_v4_samples.py` | Select stratified current-data medium subset and render v4-compatible contexts. | True |
| `src/data/gate_evidence_pack_quality_v4_1.py` | Gate evidence pack quality before generation. | True |
| `src/llm/generate_rationales.py` | Generate train-only Qwen3 rationale candidates. | True |
| `src/llm/audit_rationale_diversity_v4.py` | Audit rationale diversity/template repetition. | True |
| `src/judges/independent_inferability_judge_v4.py` | Independent inferability judge with label-order variants. | True |
| `src/judges/judge_debias_multi_permutation_v5.py` | Aggregate/debias judge outputs across label orders. | True |
| `src/judges/claim_level_grounding_v4.py` | Claim-level grounding using local NLI via sentence-transformers/transformers. | True |
| `src/judges/news_evidence_direction_audit_v5.py` | Negative-news grounding audit support. | True |
| `src/features/build_semantic_embeddings_v5.py` | Build local semantic embeddings for flow reward conditioning. | True |
| `src/reward/build_flow_dataset_v5.py` | Build masked multi-target flow reward dataset. | True |
| `src/reward/train_flow_reward_v5.py` | Train medium flow reward model. | True |
| `src/reward/evaluate_flow_vs_proxy_v5.py` | Evaluate flow reward against proxy average on validation. | True |
| `src/alignment/build_alignment_medium_v5.py` | Build RWSFT/DPO datasets from scored candidates. | True |
| `src/alignment/train_rwsft_qlora.py` | RWSFT QLoRA medium adapter training wrapper. | True |
| `src/alignment/train_dpo_qlora.py` | DPO QLoRA medium adapter training wrapper. | True |
| `src/alignment/train_rwsft_v2.py` | Reusable RWSFT trainer utilities/model loading. | True |
| `src/alignment/train_dpo_v2.py` | DPO trainer, starting from RWSFT adapter. | True |
| `src/eval/forecast_prediction.py` | Forecast-only parser, action derivation, score helpers. | True |
| `src/eval/generate_test_predictions_v2.py` | Generate deterministic adapter-backed test predictions. | True |
| `src/eval/backtest_daily_portfolio_v3.py` | Daily one-position-per-ticker portfolio backtest with turnover costs. | True |
| `src/eval/build_counterfactual_clean_v4.py` | Build balanced evidence-level counterfactual tasks. | True |
| `src/eval/evaluate_counterfactual_directional_v4.py` | LLM counterfactual directional evaluation. | True |
| `src/baselines/run_reference_baselines_medium.py` | Medium reference baselines including technical/text/Qwen RWSFT/DPO and reference-only PEN/SEP/Policy. | True |
| `src/eval/run_clean_v4_ablation_suite.py` | Build negative-results-aware medium ablation table. | True |
| `src/repro/currentdata_clean_v4_medium_science_gate.py` | Strict science gate and claim matrix. | True |
| `prompts/rationale_generation_prompt_evidence_v4.txt` | Evidence-ID rationale generation prompt. | True |
| `prompts/forecast_prediction_prompt_qwen3_json.txt` | Forecast-only prediction prompt. | True |
