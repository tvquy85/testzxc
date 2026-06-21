# DataClean V4 Review Samples

Thư mục này chứa các file sample nhỏ sinh ra từ flow `dataclean_v4_codex_md`.
Mục tiêu là đưa lên GitHub để ChatGPT UI review chất lượng code, prompt, dữ liệu mẫu, rationale, judge, reward, alignment và evaluation mà không cần upload raw dataset hoặc checkpoint lớn.

Recommended review order:

1. `01_source_code_index_dataclean_v4.md`
2. `02_master_order_dataclean_v4.md`
3. `04_entity_event_scoring_samples.jsonl`, `05_article_type_noise_filter_samples.jsonl`, `06_dedup_news_samples.jsonl`
4. `07_evidence_pack_context_samples.jsonl`, `08_rendered_context_text_samples.jsonl`, `09_prompt_template_evidence_v4.md`
5. `13_rationale_generation_samples.jsonl`
6. `11_claim_grounding_nli_samples.jsonl`, `14_independent_judge_debias_samples.jsonl`
7. `15_flow_reward_dataset_samples.json`, `15_flow_vs_proxy_clean_v4_stage0_combined.csv`
8. `16_rwsft_alignment_samples.jsonl`, `16_dpo_alignment_pair_samples.jsonl`
9. `17_prediction_forecast_samples.jsonl`, `17_backtest_daily_returns_sample.csv`, `17_counterfactual_task_samples.jsonl`, `17_ablation_current_clean_v4.csv`
10. `18_science_gate_report.json`, `00_status_summary_dataclean_v4_01_18.json`

Files:

- `00_status_summary_dataclean_v4_01_18.json`
- `01_source_code_index_dataclean_v4.md`
- `02_master_order_dataclean_v4.md`
- `03_ticker_alias_map_sample.json`
- `04_entity_event_scoring_samples.jsonl`
- `05_article_type_noise_filter_samples.jsonl`
- `06_dedup_news_samples.jsonl`
- `07_evidence_pack_context_samples.jsonl`
- `08_rendered_context_text_samples.jsonl`
- `09_prompt_template_evidence_v4.md`
- `10_sample_parsed_rationale.json`
- `10_sample_raw_rationale.jsonl`
- `10_sample_technical_tokens.json`
- `11_claim_grounding_nli_samples.jsonl`
- `12_track_train_pool_samples.jsonl`
- `13_rationale_generation_samples.jsonl`
- `14_independent_judge_debias_samples.jsonl`
- `15_flow_reward_dataset_samples.json`
- `15_flow_vs_proxy_clean_v4_stage0_combined.csv`
- `16_dpo_alignment_pair_samples.jsonl`
- `16_rwsft_alignment_samples.jsonl`
- `16_scored_alignment_candidates_samples.jsonl`
- `17_ablation_current_clean_v4.csv`
- `17_backtest_daily_returns_sample.csv`
- `17_counterfactual_fail_examples.json`
- `17_counterfactual_task_samples.jsonl`
- `17_prediction_forecast_samples.jsonl`
- `18_science_gate_report.json`
- `README.md`
- `metrics_snapshot_dataclean_v4.json`
- `sample_manifest.json`
