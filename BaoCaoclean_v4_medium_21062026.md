# Báo cáo Clean V4 Medium 21/06/2026

## 1. Mục tiêu và phạm vi

Báo cáo này tổng hợp flow medium dựa trên `clean_v4_medium_recovery_codex_md/00_MASTER_CLEAN_V4_MEDIUM_ORDER.md`.

Phạm vi đã giữ đúng contract:
- Chỉ dùng current-data Clean V4, không dùng SN2.
- Không mở rộng full FNSPID.
- Chạy theo gate tuần tự, mỗi gate có status JSON và được verify.
- PASS pipeline không đồng nghĩa với claim khoa học.
- Kết luận cuối giữ `GO_MEDIUM` nhưng `CLAIM_RESTRICTED`.

## 2. Source code chính đã tạo/sửa

Xem file index đầy đủ: `review_samples/clean_v4_medium_21062026/01_source_code_index_clean_v4_medium.md`.

Các nhóm source chính:
- Repro/audit: `src/repro/freeze_clean_v4_small_baseline.py`, `src/repro/audit_clean_v4_failure_modes.py`, `src/repro/currentdata_clean_v4_medium_science_gate.py`.
- Data/evidence: `src/data/select_medium_clean_v4_samples.py`, `src/data/gate_evidence_pack_quality_v4_1.py`.
- Rationale: `src/llm/generate_rationales.py`, `src/llm/audit_rationale_diversity_v4.py`, prompt `prompts/rationale_generation_prompt_evidence_v4.txt`.
- Judge/grounding: `src/judges/independent_inferability_judge_v4.py`, `src/judges/judge_debias_multi_permutation_v5.py`, `src/judges/claim_level_grounding_v4.py`, `src/judges/news_evidence_direction_audit_v5.py`.
- Flow reward: `src/features/build_semantic_embeddings_v5.py`, `src/reward/build_flow_dataset_v5.py`, `src/reward/train_flow_reward_v5.py`, `src/reward/evaluate_flow_vs_proxy_v5.py`.
- Alignment/training: `src/alignment/build_alignment_medium_v5.py`, `src/alignment/train_rwsft_qlora.py`, `src/alignment/train_dpo_qlora.py`, `src/alignment/train_rwsft_v2.py`, `src/alignment/train_dpo_v2.py`.
- Evaluation: `src/eval/forecast_prediction.py`, `src/eval/generate_test_predictions_v2.py`, `src/eval/backtest_daily_portfolio_v3.py`, `src/eval/build_counterfactual_clean_v4.py`, `src/eval/evaluate_counterfactual_directional_v4.py`.
- Baseline/ablation: `src/baselines/run_reference_baselines_medium.py`, `src/eval/run_clean_v4_ablation_suite.py`.

## 3. Kết quả theo từng gate

### 03 - Medium sample selector

- Rows: `900`.
- Split counts: `{'train': 500, 'test': 300, 'val': 100}`.
- Test trading days: `173`.
- Ghi chú: dùng fallback current-data v2 contexts render sang V4-compatible evidence packs vì V4 small contexts không đủ số ngày test.

### 04 - Evidence pack quality

- Rows gated: `900`.
- Track distribution: `{'hard_event_news': 467, 'company_news_general': 393, 'soft_recommendation_news': 40}`.

### 05 - Rationale generation

- Rows: `1500`.
- Unique sample IDs: `500`.
- Parse OK: `1.0`.
- Schema OK: `1.0`.
- Avg output tokens: `222.55`.
- Attention backend: `flash_attention_2`.

### 06 - Rationale diversity audit

- Mean within-sample Jaccard: `0.7275193495724753`.
- Repeated template cluster rate: `0.3413333333333333`.
- Technical-only phrase rate: `0.19666666666666666`.
- News evidence citation rate: `1.0`.

Đánh giá: schema tốt, nhưng template repetition vẫn là rủi ro chất lượng cần review bằng sample.

### 07-08 - Independent judge và debias

- Judge rows: `1500`.
- Judge schema OK: `0.998`.
- Mean true-label probability: `0.20081666666666667`.
- Mean argmax consistency: `0.745`.
- Debias rows: `1500`.
- Debias argmax consistency multi: `1.0`.

Đánh giá: judge vượt ngưỡng stop rule rất sát; nên xem đây là tín hiệu yếu chứ chưa phải evidence mạnh.

### 09 - Claim grounding NLI

- NLI backend: `True`.
- NLI loader: `sentence_transformers_cross_encoder`.
- Total claims: `4199`.
- News claims: `1205`.
- Technical claims: `2994`.
- Negative news claim count: `686`.
- Unsupported news claim rate: `0.08049792531120332`.

Đánh giá: đã dùng local NLI qua `sentence_transformers.CrossEncoder`, không dùng lexical fallback làm evidence chính.

### 10-12 - Flow reward V5

- Embedding rows: `1500`, dim `384`, backend `sentence_transformers_local`.
- Flow dataset rows: `1500`.
- Target dim: `7`.
- Mask coverage: `0.9719047546386719`.
- Flow eval rows: `288`.
- Flow metric wins: `{'rank_correlation_with_realized_utility': False, 'preference_pair_accuracy': True, 'top_decile_realized_utility': False}`.
- Flow improvement claim: `False`.

Đánh giá: Flow V5 pipeline chạy được nhưng không được claim superiority vì chưa thắng proxy average trên đủ metric.

### 13-14 - Alignment dataset và training

- RWSFT examples: `1370`.
- DPO pairs: `451`.
- Mean reward gap: `0.03743029848421766`.
- Reward source: `proxy_average_independent`.
- RWSFT smoke pass: `True`.
- DPO smoke pass: `True`.
- RWSFT max steps: `800`.
- DPO max steps: `800`.

Đánh giá: adapter thật đã có cho RWSFT và DPO, nhưng DPO loss khá phẳng nên chưa coi là bằng chứng hiệu năng.

### 15 - Prediction bằng adapter

DPO prediction:
- Rows: `300`.
- Schema OK: `0.9666666666666667`.
- Trading days selected: `173`.
- Action distribution: `{'short': 139, 'long': 90, 'hold': 61, 'invalid': 10}`.
- Raw action consistency: `0.32666666666666666`.

RWSFT baseline prediction:
- Rows: `300`.
- Schema OK: `0.9966666666666667`.
- Action distribution: `{'short': 137, 'long': 88, 'hold': 74, 'invalid': 1}`.
- Raw action consistency: `0.49666666666666665`.

Đánh giá: prediction dùng forecast-only interface; action chính thức derive từ distribution để tránh raw `hold` sai với xác suất directional.

### 16 - Backtest daily portfolio

- Trading days: `146`.
- Sharpe annualized: `-0.43705988094406034`.
- Mean daily return: `-0.000717738432780799`.
- Total turnover: `207.0`.
- Alpha claim allowed: `False`.

Đánh giá: backtest hợp lệ về pipeline nhưng Sharpe âm nên không được claim trading alpha.

### 17 - Counterfactual evidence

- Tasks: `495`.
- Pass rate: `0.3838383838383838`.
- No-change rate: `0.33131313131313134`.
- Schema OK: `0.9575757575757575`.
- Claim allowed: `False`.
- Block reason: `general claim requires pass_rate >= 0.50 and no_change_rate <= 0.35; news claim requires remove_positive_evidence and remove_negative_evidence pass rates >= 0.35`.

Đánh giá: pipeline PASS nhưng claim bị chặn theo ngưỡng strict; news perturbation vẫn yếu.

### 18 - Baselines

- Baseline count: `9`.
- Comparable baseline count: `6`.
- Reference-only count: `3`.
- Best macro-F1 method: `Technical_Rule`.
- Best macro-F1: `0.2101004560087129`.

Đánh giá: có Qwen RWSFT baseline thật; PEN/SEP/Policy chỉ là `reference_only=true`, không claim outperforming.

### 19 - Ablation suite

- Ablation count: `16`.
- Pass count: `16`.
- NOT_RUN count: `0`.
- Required present: `True`.

### 20 - Strict science gate

- Pipeline decision: `GO_MEDIUM`.
- Claim decision: `CLAIM_RESTRICTED`.
- Allowed claim count: `5`.
- AAAI main ready: `False`.

## 4. Kết luận khoa học hiện tại

- Medium pipeline đã chạy end-to-end và verify được.
- Chưa claim AAAI-ready.
- Chưa claim trading alpha vì Sharpe âm.
- Chưa claim Flow Reward superiority vì Flow V5 chưa thắng proxy đủ metric.
- Chưa claim counterfactual faithfulness theo ngưỡng strict.
- Chưa claim outperform PEN/SEP/Policy vì các baseline này chỉ reference-only, chưa có comparable current-data medium run.

## 5. Bộ sample để đưa lên GitHub/ChatGPT UI review

Thư mục: `review_samples/clean_v4_medium_21062026/`

- `review_samples/clean_v4_medium_21062026/00_status_summary_clean_v4_medium_01_20.json` - rows: 24
- `review_samples/clean_v4_medium_21062026/01_source_code_index_clean_v4_medium.md`
- `review_samples/clean_v4_medium_21062026/02_master_order_clean_v4_medium.md`
- `review_samples/clean_v4_medium_21062026/03_medium_context_and_evidence_pack_samples.jsonl` - rows: 8
- `review_samples/clean_v4_medium_21062026/04_raw_rationale_generation_samples.jsonl` - rows: 6
- `review_samples/clean_v4_medium_21062026/05_parsed_rationale_generation_samples.jsonl` - rows: 8
- `review_samples/clean_v4_medium_21062026/06_independent_judge_samples.jsonl` - rows: 6
- `review_samples/clean_v4_medium_21062026/07_debias_multi_permutation_samples.jsonl` - rows: 6
- `review_samples/clean_v4_medium_21062026/08_claim_grounding_nli_samples.jsonl` - rows: 8
- `review_samples/clean_v4_medium_21062026/09_flow_embedding_index_samples.jsonl` - rows: 6
- `review_samples/clean_v4_medium_21062026/10_flow_dataset_target_samples.json` - rows: 1
- `review_samples/clean_v4_medium_21062026/11_scored_alignment_candidate_samples.jsonl` - rows: 8
- `review_samples/clean_v4_medium_21062026/12_rwsft_alignment_samples.jsonl` - rows: 6
- `review_samples/clean_v4_medium_21062026/13_dpo_alignment_pair_samples.jsonl` - rows: 6
- `review_samples/clean_v4_medium_21062026/14_dpo_prediction_forecast_samples.jsonl` - rows: 10
- `review_samples/clean_v4_medium_21062026/15_rwsft_prediction_baseline_samples.jsonl` - rows: 10
- `review_samples/clean_v4_medium_21062026/16_backtest_daily_returns_sample.csv` - rows: 30
- `review_samples/clean_v4_medium_21062026/17_track_breakdown.csv` - rows: 146
- `review_samples/clean_v4_medium_21062026/18_counterfactual_task_samples.jsonl` - rows: 8
- `review_samples/clean_v4_medium_21062026/19_counterfactual_failure_samples.jsonl` - rows: 8
- `review_samples/clean_v4_medium_21062026/20_counterfactual_breakdown.csv` - rows: 5
- `review_samples/clean_v4_medium_21062026/21_baseline_comparison.csv` - rows: 9
- `review_samples/clean_v4_medium_21062026/22_ablation_results.csv` - rows: 16
- `review_samples/clean_v4_medium_21062026/23_claim_matrix.csv` - rows: 9
- `review_samples/clean_v4_medium_21062026/24_science_gate_report.json` - rows: 1
- `review_samples/clean_v4_medium_21062026/25_metrics_snapshot_clean_v4_medium.json` - rows: 1
- `review_samples/clean_v4_medium_21062026/26_prompt_rationale_generation_evidence_v4.txt`
- `review_samples/clean_v4_medium_21062026/27_prompt_forecast_prediction_qwen3_json.txt`
- `review_samples/clean_v4_medium_21062026/README.md`
- `review_samples/clean_v4_medium_21062026/sample_manifest.json` - manifest hash/row-count cho các sample files.

## 6. Lệnh verify đã chạy

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest -q tests
```

Kết quả: `35 passed`.

Ngoài ra đã verify status cho các gate `01-20` bằng `src.utils.verify_status` và chạy custom audit contract/artifact/threshold.
