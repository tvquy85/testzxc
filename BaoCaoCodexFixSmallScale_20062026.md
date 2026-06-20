# Báo Cáo Kết Quả Codex Fix Current-Data V3.1 Small-Scale

Ngày cập nhật: 2026-06-21

## 1. Mục Tiêu Và Kết Luận Nhanh

Mục tiêu của đợt này là thực hiện đầy đủ các bước trong `codex_fix_20062026` trên quy mô nhỏ/current-data, sửa các điểm yếu đã review, kiểm chứng rõ ràng từng bước, và chỉ scale up sau khi flow có gate sạch.

Kết luận hiện tại:

- Pipeline current-data small-scale đã `GO` tại strict science gate.
- Tất cả status bắt buộc 01-18 đều verify PASS bằng `src.utils.verify_status`.
- Claim được phép hiện tại: `counterfactual_faithfulness`.
- Claim bị chặn: `trading_alpha` và `flow_reward_improvement`.
- Kết quả này là pipeline validation/small-scale evidence, chưa phải paper-scale performance claim.

## 2. Các Sửa Đổi Chính Đã Làm

### Source code

- `src/llm/render_context.py`: đọc được context aggregated `aggregated_headlines`, `aggregated_body`, giữ technical token của ticker-date context.
- `src/alignment/train_rwsft_v2.py`: expand `$HF_HOME` từ `--hf-home`/env để Qwen3 local load đúng.
- `src/eval/generate_test_predictions_v2.py`: forecast-only parser, date-aware selection, min trading days gate, ablation modes, checkpoint DPO gate linh hoạt cho `current_v3_dpo`.
- `src/judges/judge_debias_label_order_v3.py`: reversed label-order prompt canonical, strict JSON, manifest/status đầy đủ.
- `src/judges/claim_level_grounding_v3.py`: claim-level grounding nghiêm hơn, tách news/technical/regime/forecast/risk, không còn supported-rate giả 1.0.
- `src/reward/flow_dataset_v3.py`, `src/reward/train_flow_reward_v3.py`, `src/reward/evaluate_flow_vs_proxy_v3.py`: flow v3.1 với mask target, debias stability, train-only utility, proxy average thật.
- `src/eval/backtest_daily_portfolio_v3.py`: finance-valid test-only daily simulator, schema gate, turnover cost, deterministic top confidence selection.
- `src/eval/build_counterfactual_current_v3.py`, `src/eval/evaluate_counterfactual_directional_v3.py`: balanced applicable counterfactual tasks và forecast-only deterministic eval.
- `src/eval/run_current_ablation_suite_v3.py`: ablation runner thật, không chấp nhận `NOT_RUN`.
- `src/eval/build_current_v3_paper_tables.py`: table builder negative-results-aware, không zero-fill.
- `src/repro/currentdata_science_gate_v2.py`: strict science gate kiểm tra status 01-17, artifact, metrics, claim matrix.
- `src/alignment/build_alignment_current_v3.py`, `src/alignment/train_current_v3.py`: status/manifest đầy đủ hơn cho Step12/13.

### Paper stories

- Cập nhật `paper/stories/prompt_and_scoring_innovations.md` với các bài học:
  - forecast-only evaluation interface;
  - canonical reversed label-order debias;
  - giữ technical tokens ở context-level;
  - counterfactual task balancing;
  - negative-result gate là một phần của contribution.

## 3. Kết Quả Từng Bước Theo `codex_fix_20062026`

| Step | Trạng thái | Artifact/Status chính | Kết quả chính |
|---|---:|---|---|
| 01 Freeze baseline | PASS | `outputs/status/01_FREEZE_BASELINE_CURRENTDATA.status.json` | Freeze dir `outputs/baseline_freeze/currentdata_20260620_165904`, 169 files |
| 02 Data quality audit | PASS | `data/quality/current_data_quality_v2.parquet` | 156,098 rows; mean quality 0.6951; body empty 0.1840 |
| 03 Entity/event filter | PASS | `data/processed/current_filtered_samples_v2.parquet` | 135,393 kept rows; drop rate 0.1326; all splits remain |
| 04 Ticker-date aggregation | PASS | `data/processed/ticker_date_contexts_h1_v2.parquet` | 28,796 contexts; mean 4.70 news/context; missing tech token 0.0 |
| 05 Abnormal target fix | PASS | `data/processed/ticker_date_contexts_h1_v2_targets.parquet` | target missing 0.0; direction distribution neutral 13,991 / down 7,602 / up 7,203 |
| 06 Clean rationale generation | PASS | `data/rationales/parsed/current_clean_train_qwen3_4b_v3_fixed.parquet` | 2,292 train-only rows; parse 1.0; schema 0.9987; avg output tokens 152.88 |
| 07 Independent inferability judge | PASS | `data/judges/independent_inferability_v3.parquet` | 2,146 judged rows; schema 1.0; mean true-label probability 0.2232 |
| 08 Label-order debias | PASS | `data/judges/independent_inferability_v3_1_debiased_small.parquet` | 300 rows; argmax consistency 0.7196; reversed schema 0.9867 |
| 09 Claim grounding | PASS | `data/judges/claim_grounding_v3_1_small.parquet` | 500 rows; NLI backend true; supported rate 0.40; not_applicable 0.60; no fake 1.0 supported rate |
| 10 Flow dataset v3.1 | PASS | `data/reward/flow_v3_dataset.pt` | 500 train rows; target dim 7; mask coverage 0.8571; debias not used because consistency gate below threshold |
| 11 Flow train/eval | PASS | `outputs/metrics/flow_vs_proxy_v3_1_eval.json` | Pipeline PASS, but flow claim blocked: flow rank corr 0.0565 vs proxy 0.5514 |
| 12 RWSFT/DPO rebuild | PASS | `data/alignment/rwsft_current_v3.jsonl`, `data/alignment/dpo_current_v3.jsonl` | RWSFT 1,144; DPO 507; chosen reward 0.5792 > rejected 0.4528 |
| 13 Real alignment | PASS | `checkpoints/aligned/qwen3_4b/current_v3_rwsft`, `current_v3_dpo` | RWSFT 300 steps; DPO 300 steps; real adapter safetensors exist |
| 14 Daily backtest | PASS | `outputs/metrics/backtest_daily_portfolio_current_v3.json` | 62 trading days; turnover 136; Sharpe -1.3467; alpha claim blocked |
| 15 Counterfactual directional eval | PASS | `outputs/metrics/counterfactual_directional_current_v3.json` | 500 tasks; pass rate 0.422; no-change 0.446; schema 0.982 |
| 16 Minimum ablations | PASS | `outputs/tables/ablation_current_v3.csv` | 5 ablations PASS; no `NOT_RUN` row |
| 17 Paper tables / claim matrix | PASS | `outputs/metrics/current_v3_claim_matrix.json` | Tables built from real metrics; counterfactual + reproducibility allowed; alpha/flow blocked |
| 18 Strict science gate | PASS | `outputs/repro/currentdata_science_gate_report_v2.json` | `pipeline_decision=GO`; `claim_decision=CLAIM_ALLOWED`; allowed claim: counterfactual faithfulness |

## 4. Chi Tiết Các Gate Quan Trọng

### Prediction gate

- Artifact: `outputs/predictions/current_v3_1_test_predictions.parquet`
- Rows: 500
- Split: test-only
- Trading days: 63 selected, 62 in backtest after schema-valid merge
- Parse OK: 1.0
- Schema OK: 0.984
- Action distribution: hold 245, long 164, short 83, invalid 8

### Backtest gate

- Artifact: `outputs/tables/backtest_daily_returns_current_v3.csv`
- Trading days: 62
- Total turnover: 136
- Mean daily return: -0.000647
- Sharpe annualized: -1.3467
- Result: pipeline mechanics PASS, but trading alpha claim blocked.

### Counterfactual gate

- Artifact: `data/counterfactual/current_cf_tasks_v3_1_small.parquet`
- Task balance: 125 each for remove negative news, remove positive news, neutralize bearish technical, neutralize bullish technical.
- Pass rate: 0.422
- No-change rate: 0.446
- By type:
  - neutralize bullish technical: 0.808
  - neutralize bearish technical: 0.520
  - remove positive news: 0.240
  - remove negative news: 0.120
- Result: counterfactual faithfulness claim allowed at small-scale gate.

### Flow reward gate

- Artifact: `outputs/metrics/flow_vs_proxy_v3_1_eval.json`
- Flow rank correlation: 0.0565
- Proxy rank correlation: 0.5514
- Flow top-decile utility: -0.00057
- Proxy top-decile utility: 0.02309
- Result: flow pipeline PASS, but flow improvement claim blocked.

### Ablation gate

- Artifact: `outputs/tables/ablation_current_v3.csv`
- Required rows all PASS:
  - Full_Current_V3
  - No_Technical_Tokens
  - No_News_Body
  - SFT_Only
  - No_Flow_Reward
- No `NOT_RUN` evidence rows.

## 5. Verification Đã Chạy

### Unit tests

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest -q tests
```

Result:

```text
29 passed
```

### Status verification

Tất cả status bắt buộc 01-18 đã PASS:

```bash
python -m src.utils.verify_status --status outputs/status/<STEP>.status.json
```

Đã verify:

- `01_FREEZE_BASELINE_CURRENTDATA`
- `02_CURRENT_DATA_QUALITY_AUDIT`
- `03_ENTITY_EVENT_FILTER_CURRENT_DATA`
- `04_TICKER_DATE_CONTEXT_AGGREGATION`
- `05_LABEL_BACKTEST_TARGET_FIX_ABNORMAL_RETURN`
- `06_RATIONALE_PROMPT_CLEAN_CONTEXT`
- `07_INDEPENDENT_INFERABILITY_JUDGE`
- `08_JUDGE_DEBIAS_LABEL_ORDER_RANDOMIZATION`
- `09_CLAIM_EXTRACTION_GROUNDING_V2`
- `10_FLOW_DATASET_V3_TARGETS_AND_EMBEDDINGS`
- `11_FLOW_TRAIN_EVAL_FIX_VALID_SPLIT`
- `12_RWSFT_DPO_REBUILD_FROM_INDEPENDENT_REWARDS`
- `13_ALIGNMENT_REAL_RUN_CURRENT_DATA`
- `14_DAILY_BACKTEST_ABNORMAL_AND_COSTS`
- `15_COUNTERFACTUAL_DIRECTIONAL_FIX_CURRENT_DATA`
- `16_MINIMUM_ABLATIONS_CURRENT_DATA`
- `17_PAPER_TABLES_NEGATIVE_RESULTS_GATES`
- `18_AAAI_SCIENCE_GATE_STRICT`

## 6. Hạn Chế Còn Lại

- NLI `cross-encoder/nli-deberta-v3-small` đã load được hoàn toàn từ local cache bằng model id với `HF_HOME=E:/huggingface` và `local_files_only=True`; Step09 hiện ghi `nli_backend=true`, `nli_loader=transformers_model_id`.
- Flow reward v3.1 chưa beat proxy; không được claim improvement.
- Backtest Sharpe âm; không được claim trading alpha.
- Step13 adapter là real 300-step adapter, nhưng trainer hiện tại không lưu per-step loss trace trong artifact cũ; status đã ghi `loss_trace_available=false`.
- Kết quả này là small-scale validation. Chưa nên đưa làm final AAAI performance table nếu chưa scale medium/full và chạy statistics.

## 7. Khuyến Nghị Bước Tiếp Theo

Chỉ scale medium sau khi giữ nguyên các gate hiện tại:

1. Mở Step08/09 lên toàn bộ current rationale rows để có debias/grounding đầy đủ hơn.
2. Rerun flow v3.1 với target/conditioning tốt hơn, vì proxy hiện đang mạnh hơn flow.
3. Scale prediction/backtest lên 2k-5k rows test và giữ date-aware sampling.
4. Scale counterfactual lên 1k-2k tasks, giữ balanced applicable task generation.
5. Chỉ cập nhật paper claim khi strict gate tiếp tục tách rõ `pipeline_pass` và `claim_allowed`.

## 8. Sample Files Để Đưa Lên GitHub / ChatGPT UI Review

Thư mục sample mới:

```text
review_samples/codex_fix_20062026/
```

Script có thể rerun:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.repro.export_codex_fix_samples \
  --output-dir review_samples/codex_fix_20062026
```

Bảng đường dẫn sample:

| Mục review | File sample | Source artifact |
|---|---|---|
| Hướng dẫn review | `review_samples/codex_fix_20062026/README.md` | Generated index |
| Source code index | `review_samples/codex_fix_20062026/01_source_code_index.md` | Source modules trong `src/` và `tests/` |
| Status 01-18 | `review_samples/codex_fix_20062026/00_status_summary_01_18.json` | `outputs/status/*.status.json` |
| Rationale generation | `review_samples/codex_fix_20062026/06_rationale_generation_samples.jsonl` | `data/rationales/parsed/current_clean_train_qwen3_4b_v3_fixed.parquet` |
| Inferability/debias judge | `review_samples/codex_fix_20062026/07_08_inferability_debias_samples.jsonl` | `data/judges/independent_inferability_v3_1_debiased_small.parquet` |
| NLI claim grounding | `review_samples/codex_fix_20062026/09_claim_grounding_nli_samples.jsonl` | `data/judges/claim_grounding_v3_1_small.parquet` |
| Flow target/mask | `review_samples/codex_fix_20062026/10_flow_dataset_target_samples.json` | `data/reward/flow_v3_dataset.pt` |
| RWSFT data | `review_samples/codex_fix_20062026/12_rwsft_alignment_samples.jsonl` | `data/alignment/rwsft_current_v3_fixed.jsonl` |
| DPO pair data | `review_samples/codex_fix_20062026/12_dpo_alignment_pair_samples.jsonl` | `data/alignment/dpo_current_v3_fixed.jsonl` |
| Test prediction | `review_samples/codex_fix_20062026/14_prediction_forecast_samples.jsonl` | `outputs/predictions/current_v3_1_test_predictions.parquet` |
| Daily backtest | `review_samples/codex_fix_20062026/14_backtest_daily_returns_sample.csv` | `outputs/tables/backtest_daily_returns_current_v3.csv` |
| Counterfactual tasks | `review_samples/codex_fix_20062026/15_counterfactual_task_samples.jsonl` | `data/counterfactual/current_cf_tasks_v3_1_small.parquet` |
| Counterfactual failures | `review_samples/codex_fix_20062026/15_counterfactual_fail_examples.json` | `outputs/data_samples/counterfactual_fail_examples_current_v3.json` |
| Ablation results | `review_samples/codex_fix_20062026/16_ablation_results.csv` | `outputs/tables/ablation_current_v3.csv` |
| Paper table samples | `review_samples/codex_fix_20062026/17_table_prediction.csv`, `17_table_explanation.csv`, `17_table_flow_reward.csv`, `17_table_backtest.csv`, `17_table_ablation.csv` | `outputs/tables/current_v3_table_*.csv` |
| Claim/science gate | `review_samples/codex_fix_20062026/18_claim_matrix_and_science_gate.json` | `outputs/metrics/current_v3_claim_matrix.json`, `outputs/repro/currentdata_science_gate_report_v2.json` |
| Metrics snapshot | `review_samples/codex_fix_20062026/metrics_snapshot_current_v3_1.json` | Key metrics from Steps 09-16 |
| Sample manifest | `review_samples/codex_fix_20062026/sample_manifest.json` | Hash/row count/source map for all sample files |

Ghi chú: các sample này không include raw dataset lớn, model weights, hoặc checkpoint adapter; chỉ include record nhỏ để review schema, prompt/output quality, reward target, counterfactual logic, và negative-result gates.
