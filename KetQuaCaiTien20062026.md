# Kết Quả Cải Tiến FIRE-Fin AAAI Upgrade 20/06/2026

File này tổng hợp những phần đã làm theo `codex_upgrade_md/00_MASTER_CODEX_EXECUTION_ORDER.md` để đưa vào ChatGPT hoặc reviewer khác phân tích tiếp. Nội dung được lấy từ source code, status JSON, manifests, metrics và final artifacts hiện tại trong repo.

## 1. Executive Summary

Mục tiêu của đợt cải tiến là nâng prototype FIRE-Fin từ MVP nội bộ thành pipeline tái lập theo chuẩn AAAI: có locked split, leakage guard, strict rationale parsing, train-only rationale generation, judge/grounding, flow reward, RWSFT/DPO smoke, finance-valid evaluation, baselines, statistics, paper tables không dummy, và reproducibility package.

Trạng thái hiện tại:

| Hạng mục | Kết quả |
|---|---|
| Master steps | 01-21 đã PASS |
| Global verification | `src/utils/verify_status.py` PASS cho 21 status |
| Unit tests | `25 passed` |
| Final gate | `outputs/repro/aaai_gate_report.json` có `decision: GO` |
| Repro package | `outputs/repro/firefin_repro_package.zip` |
| Package size | 815,175 bytes |
| Package file count | 253 files |
| Raw dataset/model weights lớn | Không include |
| Known warning | Ablation table là registry `NOT_RUN`, không dùng làm evidence |

Kết quả khoa học cần đọc cẩn thận:

- Pipeline/reproducibility gate đã GO.
- Chưa nên claim trading alpha: Step16 medium Sharpe đang âm `-10.9598`.
- Chưa nên claim Flow Reward v2 thắng proxy: Step13 `claim_improvement=false`.
- Chưa nên claim full ablation evidence: A0-A8 chưa chạy, chỉ ghi registry `NOT_RUN`.

## 2. Non-Negotiable Rules Đã Được Xử Lý

Theo master order:

| Rule | Cách xử lý hiện tại |
|---|---|
| Không fabricate metrics/tables/status | Mỗi step ghi status JSON từ script; final tables lấy từ metrics/tables thật |
| Không PASS nếu thiếu output | `verify_status.py` kiểm status contract; Step20/21 gate fail nếu source thiếu |
| Không dùng test cho training/reward/alignment | Split locked: train <= 2021-12-31, val 2022, test >= 2023; alignment dataset train-only |
| Bảo tồn raw LLM outputs | Raw/parsed rationale tách riêng trong `data/rationales/raw` và `data/rationales/parsed` |
| Không dummy tables/neutral fallback | Strict parser giữ invalid là invalid; final tables không zero-fill dummy |
| Script runnable từ repo root | Entry points trong `src/...` nhận CLI path/config |
| Artifact có manifest | `outputs/manifests/*.manifest.json` ghi path, row_count, sha256, timestamp |

## 3. Execution Status 01-21

Nguồn: `outputs/status/<STEP>.status.json`

| Step | Status | Key metrics / outputs |
|---|---|---|
| 01 Repo audit and safe branch | PASS | Branch `upgrade-aaai-reproducibility`; tracked files 54; risk flags 7; dirty worktree true |
| 02 Config paths and reproducible env | PASS | Python `D:\LOBProj\LOBExp\.venv\Scripts\python.exe`; hard-coded local path files 0; missing required modules 0 |
| 03 Data scale and split lock | PASS | 156,098 rows; train 80,415; val 29,404; test 46,279 |
| 04 Leakage guards and unit tests | PASS | Current leaks 0; legacy alignment risk count 50 audited only |
| 05 Abnormal return labels | PASS | 156,098 rows; 5 classes |
| 06 Technical features v2 | PASS | Feature rows 156,098; max missing rate 0 |
| 07 Technical event tokens v2 | PASS | Rows 156,098; missing token rate 0 |
| 08 Strict rationale schema | PASS | Strict schema tests 4 |
| 09 Rationale generation train-only | PASS | 12,500 rows; 7,336 unique samples; parse-ok 0.99576; schema-ok 0.98352; stage `stage_1_small_scale` |
| 10 Multi-model inferability judge | PASS | 100,000 judge rows; parse-ok 0.99576; label-order consistency 1.0 |
| 11 Claim-level grounding | PASS | 85,084 claims; supported 19,615; unverified 28,128; not_applicable 37,341 |
| 12 Flow reward v2 train | PASS | 12,500 rows; target dim 7; train loss 2.3424 -> 1.0103 |
| 13 Flow vs proxy eval | PASS | 12,500 rows; eval rows 2,480; method count 4; claim_improvement false |
| 14 RWSFT/DPO dataset rebuild | PASS | RWSFT 7,336 examples; DPO 3,362 pairs; DPO unique samples 1,565 |
| 15 Alignment training reproducible | PASS | RWSFT smoke PASS; DPO smoke PASS; max steps 10; real adapters saved |
| 16 Daily portfolio backtest v2 | PASS | 33 trading days; Sharpe -10.9598; turnover 227.1351 |
| 17 Counterfactual directional eval v2 | PASS | 500 tasks; 1,000 generations; pass rate 0.16; no-change 0.696 |
| 18 Baselines expansion and seeds | PASS | B1-B4 run with seeds 11/22/33; 95,574 baseline prediction rows |
| 19 Ablation and statistical tests | PASS | 12 paired prediction bootstrap; 12 block bootstrap; ablations NOT_RUN not evidence |
| 20 Paper tables no dummy gates | PASS | 6 final tables; table manifest generated; no dummy zero-fill |
| 21 Repro package and AAAI gate | PASS | Gate decision GO; package built; blocker count 0 |

## 4. Source Code Đã Tạo / Cập Nhật

### 4.1 Config, status, artifact, environment

| File | Vai trò |
|---|---|
| `configs/default_paths.yaml` | Path config dùng `$HF_HOME`, không hard-code local absolute path trong source |
| `configs/local_paths.template.yaml` | Template local path cho user |
| `configs/local_paths.yaml` | Local config dùng env-expanded paths |
| `configs/rationale_generation_qwen3_fast.yaml` | Config generation fast cho Qwen3 |
| `src/utils/config.py` | Load/expand config, validate `$HF_HOME` |
| `src/utils/artifacts.py` | Write JSON, status, manifest, row_count, sha256 |
| `src/utils/verify_status.py` | Verify status JSON contract sau mỗi step |
| `src/utils/audit_repo_state.py` | Repo audit, hash files, risk flags |
| `src/utils/check_repro_env.py` | Repro env/module inventory |
| `src/utils/model_inventory.py` | Local model inventory |

### 4.2 Data, split, leakage

| File | Vai trò |
|---|---|
| `src/data/build_dataset_manifest.py` | Scan FNSPID local cache và ghi dataset manifest |
| `src/data/build_locked_splits.py` | Chronological split train/val/test |
| `src/data/build_abnormal_return_labels.py` | Tạo label abnormal return 5 lớp |
| `src/data/leakage_checks.py` | Kiểm leakage split/timestamp/prompt/alignment |
| `src/data/build_train_conflict_subset_v2.py` | Tạo train-only conflict/hard subset cho DPO candidates |
| `tests/test_leakage_checks.py` | Unit tests leakage synthetic |
| `tests/test_stage1_conflict_and_alignment.py` | Test conflict subset train-only và DPO pairing |

### 4.3 Technical features and event tokens

| File | Vai trò |
|---|---|
| `src/features/compute_technical_indicators.py` | Pure functions RSI, MACD, rolling, Bollinger, volume z-score, technical features |
| `src/features/compile_technical_event_tokens_v2.py` | Compile technical event tokens JSON có rule/direction/strength |
| `tests/test_technical_features.py` | Unit tests technical indicators |

### 4.4 LLM rationale generation and strict parser

| File | Vai trò |
|---|---|
| `src/llm/generate_rationales.py` | Generate train-only rationales bằng local Qwen3, raw/parsed tách riêng, stage-aware |
| `src/llm/validate_raw_rationales.py` | Validate raw generation schema quality |
| `src/llm/combine_stage1_rationales.py` | Combine bulk/conflict rationale outputs |
| `src/llm/parse_and_validate_rationale.py` | Strict JSON parsing, không autofix main metrics |
| `src/llm/rationale_schema.py` | Rationale schema contract |
| `src/llm/render_context.py` | Render news + technical tokens context |
| `tests/test_rationale_schema_strict.py` | Unit tests strict schema |
| `prompts/rationale_generation_prompt_qwen3_fast_json.txt` | Fast Qwen3 JSON rationale prompt |
| `prompts/rationale_generation_prompt.txt` | Original rationale prompt reference |

### 4.5 Judges and grounding

| File | Vai trò |
|---|---|
| `src/judges/inferability_judge_v2.py` | Deterministic inferability judge interface |
| `src/judges/run_multi_judge_inferability.py` | Multi-model inferability judge normal/reversed label order |
| `src/judges/extract_rationale_claims.py` | Extract news/technical/regime/forecast/risk claims |
| `src/judges/claim_level_grounding.py` | Claim-level grounding logic |
| `src/judges/run_claim_grounding.py` | Run grounding over rationale candidates |

### 4.6 Flow reward v2

| File | Vai trò |
|---|---|
| `src/reward/flow_dataset_v2.py` | Build multi-target reward dataset with masks |
| `src/reward/flow_model_v2.py` | FlowRewardV2 model và masked MSE |
| `src/reward/train_flow_reward_v2.py` | Train flow reward v2 |
| `src/reward/evaluate_flow_vs_proxy.py` | Real flow-vs-proxy evaluation trên deterministic train holdout |

Key code behavior của Step13:

```text
realized_utility = forecast_distribution[true_label]
methods = proxy_average_reward, single_best_judge_reward, flow_reward_v1, flow_reward_v2
metrics = rank correlation, calibration error, top-k quality, preference-pair accuracy, variance by regime
claim_improvement = true chỉ khi flow_v2 thắng proxy trên >= 2 metric
```

### 4.7 Alignment datasets and training

| File | Vai trò |
|---|---|
| `src/alignment/build_rwsft_dataset_v2.py` | Build RWSFT train-only examples |
| `src/alignment/build_dpo_pairs_v2.py` | Build DPO same-sample chosen/rejected pairs with reward gap |
| `src/alignment/run_alignment_dataset_rebuild_v2.py` | Rebuild RWSFT/DPO from Step12/flow/proxy scores |
| `src/alignment/train_rwsft_v2.py` | RWSFT QLoRA smoke training |
| `src/alignment/train_dpo_v2.py` | DPO QLoRA smoke training, real adapter output |

### 4.8 Finance evaluation, baselines, statistics

| File | Vai trò |
|---|---|
| `src/eval/forecast_prediction.py` | Forecast-only parser for Step16/17 |
| `src/eval/generate_test_predictions_v2.py` | Deterministic DPO adapter test predictions, date-aware selection |
| `src/eval/backtest_daily_portfolio_v2.py` | Daily portfolio simulator, one position per ticker-date, turnover-based cost |
| `src/eval/build_counterfactual_contexts_v2.py` | Build test-only counterfactual tasks |
| `src/eval/evaluate_counterfactual_directional_v2.py` | Adapter-backed counterfactual directional eval |
| `src/eval/run_baseline_suite.py` | B1-B4 baselines, 3 seeds, aggregate + prediction-level outputs |
| `src/eval/statistical_tests.py` | Paired prediction bootstrap + daily-return block bootstrap |
| `src/eval/run_ablation_suite.py` | A0-A8 registry, NOT_RUN if real ablations not run |
| `tests/test_step15_16_17_smoke_contracts.py` | Tests DPO pairs, prediction gates, backtest, counterfactual, Step18 NOT_RUN fail behavior |

### 4.9 Paper tables and reproducibility gate

| File | Vai trò |
|---|---|
| `src/eval/build_paper_tables_v2.py` | Build final paper tables from real metrics only, long-format with source mapping |
| `src/repro/aaai_gate_check.py` | Strict GO/NO_GO gate: statuses, paths, train-only alignment, daily backtest, flow eval, baselines |
| `src/repro/build_repro_package.py` | Build reproducibility zip excluding raw/model artifacts |
| `outputs/repro/README_REPRODUCE.md` | Smoke test and expected outputs |
| `outputs/repro/REPRODUCIBILITY_CHECKLIST_DRAFT.md` | Checklist draft |

## 5. Important Artifacts

### 5.1 Data and split

| Artifact | Mô tả |
|---|---|
| `outputs/manifests/fnspid_dataset_manifest.json` | FNSPID local cache/data manifest |
| `data/processed/split_membership.parquet` | Locked chronological split |
| `data/labels/labels_h1_abnormal.parquet` | Abnormal-return labels |
| `data/indicators/technical_features_h1_v2.parquet` | Technical features |
| `data/indicators/technical_event_tokens_h1_v2.parquet` | Technical event tokens |

Split:

| Split | Rows | Date range |
|---|---:|---|
| train | 80,415 | 2018-03-29 to 2021-12-31 |
| val | 29,404 | 2022-01-03 to 2022-12-30 |
| test | 46,279 | 2023-01-03 to 2023-12-27 |

### 5.2 Rationale generation

| Artifact | Mô tả |
|---|---|
| `data/rationales/raw/train_qwen3_4b_stage1_bulk.jsonl` | Bulk train-only raw rationale output |
| `data/rationales/raw/train_qwen3_4b_stage1_conflict_3cand.jsonl` | Conflict/hard train-only raw outputs |
| `data/rationales/parsed/train_candidates_stage1_strict.parquet` | Combined strict parsed candidates |
| `outputs/metrics/stage1_rationale_generation_summary.json` | Stage1 rationale summary |

Metrics:

| Metric | Value |
|---|---:|
| Parsed rows | 12,500 |
| Unique samples | 7,336 |
| Parse-ok rate | 0.99576 |
| Schema-ok rate | 0.98352 |
| Avg output tokens estimate | 219.47 |
| Explicit leak pattern count | 0 |

### 5.3 Judge and grounding

| Artifact | Mô tả |
|---|---|
| `data/judges/inferability_multi_judge_stage1.parquet` | Multi-judge inferability outputs |
| `outputs/metrics/inferability_judge_stability_stage1.json` | Judge stability metrics |
| `data/judges/claim_grounding_scores_stage1.parquet` | Claim-level grounding |
| `outputs/metrics/claim_grounding_summary_stage1.json` | Grounding summary |

Inferability:

| Metric | Value |
|---|---:|
| Rows | 100,000 |
| Parse-ok rate | 0.99576 |
| Label order consistency | 1.0 |
| Entropy mean | 1.3474 |
| Mean probability true label | 0.2139 |

Grounding:

| Claim/status | Count |
|---|---:|
| Total claims | 85,084 |
| technical_claim | 24,439 |
| news_claim | 23,304 |
| regime_claim | 12,447 |
| forecast_claim | 12,447 |
| risk_claim | 12,447 |
| supported | 19,615 |
| unverified | 28,128 |
| not_applicable | 37,341 |

### 5.4 Flow reward v2

| Artifact | Mô tả |
|---|---|
| `data/reward/flow_v2_train_dataset_stage1.pt` | Flow reward training dataset |
| `checkpoints/flow_reward_v2_stage1/model.pt` | Flow reward v2 checkpoint |
| `outputs/metrics/flow_reward_v2_train_metrics_stage1.json` | Train metrics |
| `outputs/metrics/flow_vs_proxy_eval_stage1.json` | Real evaluation vs proxy |
| `outputs/tables/flow_vs_proxy_eval_stage1.csv` | Evaluation table |

Train:

| Metric | Value |
|---|---:|
| Rows | 12,500 |
| Target dim | 7 |
| Loss epoch 1 | 2.3424 |
| Loss epoch 2 | 1.5847 |
| Loss epoch 3 | 1.0103 |

Flow-vs-proxy evaluation:

| Method | Eval rows | Rank corr | Calibration error | Top-k quality | Pair accuracy | Variance by regime |
|---|---:|---:|---:|---:|---:|---:|
| proxy_average_reward | 2,480 | 0.5731 | 0.2294 | 0.3500 | 0.8594 | 0.000095 |
| single_best_judge_reward | 2,480 | 1.0000 | 0.0000 | 0.4490 | 1.0000 | 0.000023 |
| flow_reward_v1 | 2,480 | 0.7859 | 0.0697 | 0.3859 | 0.9375 | 0.000109 |
| flow_reward_v2 | 2,480 | 0.0058 | 0.1180 | 0.2087 | 0.5313 | 0.000062 |

Conclusion: `claim_improvement=false`. Flow Reward v2 chưa thắng proxy average, nên không claim improvement.

### 5.5 Alignment

| Artifact | Mô tả |
|---|---|
| `data/alignment/rwsft_train_v2.jsonl` | RWSFT train-only data |
| `data/alignment/dpo_pairs_train_v2.jsonl` | DPO train-only chosen/rejected pairs |
| `checkpoints/aligned/qwen3_4b/rwsft_v2/adapter_model.safetensors` | RWSFT adapter |
| `checkpoints/aligned/qwen3_4b/dpo_v2/adapter_model.safetensors` | DPO adapter |

Metrics:

| Metric | Value |
|---|---:|
| RWSFT examples | 7,336 |
| DPO pairs | 3,362 |
| DPO unique samples | 1,565 |
| Reward gap min | 0.02 |
| RWSFT smoke pass | true |
| DPO smoke pass | true |
| Max steps | 10 |
| RWSFT first loss | 2.8327 |
| RWSFT last loss | 2.5759 |
| RWSFT max memory allocated | about 6.03 GB |
| DPO loss first/last | 0.6931 / 0.6931 |

## 6. Finance Evaluation Results

### 6.1 Prediction medium

Nguồn: `outputs/metrics/test_predictions_qwen3_dpo_flash_medium.json`

| Metric | Value |
|---|---:|
| Rows | 5,000 |
| Split | test |
| Selected trading days | 33 |
| Start date | 2023-01-03 |
| End date | 2023-02-17 |
| Parse-ok rate | 0.9866 |
| Schema-ok rate | 0.9798 |
| Parse-ok rows | 4,933 |
| Schema-ok rows | 4,899 |
| Fallback used | false |
| Attention backend | `flash_attention_2` |

| Action | Count | Share |
|---|---:|---:|
| hold | 2,442 | 0.4884 |
| long | 1,901 | 0.3802 |
| short | 556 | 0.1112 |
| invalid | 101 | 0.0202 |

### 6.2 Daily portfolio backtest

Nguồn: `outputs/metrics/backtest_daily_portfolio_v2.json`

| Metric | Value |
|---|---:|
| Trading days | 33 |
| Annualized daily Sharpe | -10.9598 |
| Mean daily return | -0.002809 |
| Nonzero daily return rate | 1.0 |
| Coverage | 1.0 |
| Average positions per day | 7.6120 |
| Average turnover per day | 6.8829 |
| Total turnover | 227.1351 |
| Cost | 5 bps |
| Slippage | 0 bps |
| Short borrow | 0 bps |

Interpretation: daily simulator đã hợp lệ về mechanics, nhưng trading performance đang âm. Không claim alpha.

### 6.3 Counterfactual directional eval

Nguồn: `outputs/metrics/counterfactual_directional_v2.json`

| Metric | Value |
|---|---:|
| Tasks | 500 |
| Generations | 1,000 |
| Parse-ok rate | 0.999 |
| Schema-ok rate | 0.969 |
| Directional pass rate | 0.160 |
| Wrong-direction rate | 0.144 |
| No-change rate | 0.696 |
| Mean delta | -0.03614 |

Interpretation: model có phản ứng đúng hướng ở một phần case, nhưng no-change còn rất cao.

## 7. Baselines and Statistics

### 7.1 Baseline aggregate B1-B4

Nguồn: `outputs/tables/baseline_suite_aggregate.csv`

| Baseline | Seeds | Accuracy | Macro-F1 | Macro-F1 std | MCC | MCC std | Brier | Train rows min | Test rows min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1_FinBERT_LR | 3 | 0.2346 | 0.1332 | 0.0000 | 0.0141 | 0.0000 | 0.7991 | 10,300 | 5,929 |
| B2_Technical_LightGBM | 3 | 0.3889 | 0.2635 | 0.0141 | 0.0827 | 0.0167 | 0.7138 | 30,000 | 10,000 |
| B3_News_Technical_Late_Fusion | 3 | 0.3895 | 0.2534 | 0.0120 | 0.0814 | 0.0121 | 0.7082 | 10,300 | 5,929 |
| B4_DLinear | 3 | 0.4357 | 0.1963 | 0.0109 | 0.0570 | 0.0098 | 0.7247 | 30,000 | 10,000 |

Ghi chú:

- B1-B4 đã chạy 3 seeds `[11,22,33]`.
- B5-B12 vẫn `NOT_RUN`, `multi_seed_missing=true`.
- Prediction-level baseline artifact: `outputs/predictions/baseline_suite_predictions.parquet`, 95,574 rows.

### 7.2 Prediction paired bootstrap summary

Nguồn: `outputs/tables/prediction_bootstrap_comparisons.csv`

| Comparator | Paired rows mean | Delta accuracy mean | Delta macro-F1 mean | Delta MCC mean |
|---|---:|---:|---:|---:|
| B1_FinBERT_LR | 635.0 | 0.0803 | 0.0368 | -0.0305 |
| B2_Technical_LightGBM | 1,077.7 | -0.0027 | -0.0950 | -0.0554 |
| B3_News_Technical_Late_Fusion | 635.0 | -0.0273 | -0.1000 | -0.0940 |
| B4_DLinear | 1,077.7 | -0.0985 | -0.0539 | -0.0891 |

Interpretation:

- FIRE-Fin/Qwen3 DPO tốt hơn B1 về accuracy và macro-F1 trên overlap nhỏ.
- FIRE-Fin thua hoặc không ổn định so với B2-B4.
- Không claim beat baselines.

### 7.3 Daily-return block bootstrap summary

Nguồn: `outputs/tables/backtest_block_bootstrap_comparisons.csv`

| Comparator | Overlap rows mean | Paired days | Delta mean daily return mean | Delta Sharpe mean |
|---|---:|---:|---:|---:|
| B1_FinBERT_LR | 635.0 | 33 | 0.000123 | 1.5260 |
| B2_Technical_LightGBM | 1,077.7 | 33 | 0.000173 | -0.6154 |
| B3_News_Technical_Late_Fusion | 635.0 | 33 | 0.001855 | 4.8804 |
| B4_DLinear | 1,077.7 | 33 | -0.000722 | -3.7359 |

Interpretation:

- Kết quả daily-return bootstrap lẫn lộn.
- FIRE-Fin tốt hơn B3 trên overlap này, nhưng kém B4 và chưa ổn định với B2.
- Do Sharpe tổng thể âm, đây chỉ là evidence mechanics/statistics, không phải alpha claim.

## 8. Paper Tables and Reproducibility Package

### 8.1 Final paper tables

Nguồn: `outputs/status/20_PAPER_TABLES_NO_DUMMY_GATES.status.json`

| Table | Rows | Evidence rows | Note |
|---|---:|---:|---|
| `table_1_prediction_main.csv` | 273 | 273 | Prediction, baselines, paired bootstrap |
| `table_2_explanation_quality.csv` | 32 | 32 | Inferability, grounding, flow eval |
| `table_3_daily_portfolio_backtest.csv` | 181 | 181 | Daily backtest and block bootstrap |
| `table_4_counterfactual_directional.csv` | 13 | 13 | Counterfactual metrics |
| `table_5_ablation.csv` | 9 | 0 | A0-A8 `NOT_RUN`, registry only |
| `table_6_scale_and_compute.csv` | 38 | 38 | Dataset scale and environment |

Manifest:

- `outputs/tables/final/table_manifest.json`
- `outputs/manifests/20_PAPER_TABLES_NO_DUMMY_GATES.manifest.json`

### 8.2 Final gate and package

Nguồn: `outputs/repro/aaai_gate_report.json`, `outputs/status/21_FINAL_REPRO_PACKAGE_AND_AAAI_GATE.status.json`

| Metric | Value |
|---|---:|
| Gate decision | GO |
| Blocking issues | 0 |
| Warning count | 1 |
| Warning | Ablation table contains NOT_RUN registry only; not used as evidence |
| Repro package | `outputs/repro/firefin_repro_package.zip` |
| Included files | 253 |
| Package size | 815,175 bytes |

## 9. Commands Đã Dùng Để Verify

Interpreter chuẩn:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe
```

Full tests:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest -q tests
```

Kết quả:

```text
25 passed
```

Verify 01-21:

```bash
for s in \
  01_REPO_AUDIT_AND_SAFE_BRANCH \
  02_CONFIG_PATHS_AND_REPRO_ENV \
  03_DATA_SCALE_AND_SPLIT_LOCK \
  04_LEAKAGE_GUARDS_AND_UNIT_TESTS \
  05_LABELS_ABNORMAL_RETURN_AND_BALANCE \
  06_TECHNICAL_FEATURES_V2 \
  07_TECH_EVENT_TOKENS_V2 \
  08_STRICT_RATIONALE_SCHEMA_NO_AUTOFIX \
  09_RATIONALE_GENERATION_SCALEUP_TRAIN_ONLY \
  10_PROXY_JUDGES_MULTI_MODEL_DEBIASED \
  11_CLAIM_LEVEL_GROUNDING_JUDGES \
  12_FLOW_REWARD_MULTITARGET_V2 \
  13_FLOW_REWARD_EVAL_VS_PROXY \
  14_RWSFT_DPO_DATASET_REBUILD \
  15_ALIGNMENT_TRAINING_REPRODUCIBLE \
  16_DAILY_PORTFOLIO_BACKTEST_V2 \
  17_COUNTERFACTUAL_DIRECTIONAL_EVAL_V2 \
  18_BASELINES_EXPANSION_AND_SEEDS \
  19_ABLATION_AND_STATISTICAL_TESTS \
  20_PAPER_TABLES_NO_DUMMY_GATES \
  21_FINAL_REPRO_PACKAGE_AND_AAAI_GATE
do
  /mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe src/utils/verify_status.py \
    --status outputs/status/${s}.status.json
done
```

Gate:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe src/repro/aaai_gate_check.py \
  --output outputs/repro/aaai_gate_report.json
```

Package:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe src/repro/build_repro_package.py \
  --output outputs/repro/firefin_repro_package.zip \
  --gate-report outputs/repro/aaai_gate_report.json
```

## 10. Claim Boundary Cho Paper

Có thể nói:

- Pipeline đã có locked split, leakage guard, strict parser, train-only rationale generation, judge/grounding, flow reward, RWSFT/DPO smoke, finance-valid evaluation, baselines, statistics, final tables và repro package.
- Final gate hiện tại GO về reproducibility/mechanics.
- Rationale generation có parse/schema high validity.
- Finance evaluation đã chuyển từ per-news sang daily portfolio simulator.
- Statistical tests đã có paired bootstrap và block bootstrap thực.

Không nên nói:

- Model có trading alpha.
- Model beat all baselines.
- Flow Reward v2 tốt hơn proxy average.
- Ablations đã support mọi design choice.
- Kết quả đã sẵn sàng làm final performance claim.

Lý do:

- Step16 Sharpe âm `-10.9598`.
- Flow Reward v2 `claim_improvement=false`.
- A0-A8 ablations là `NOT_RUN`.
- B5-B12 vẫn `NOT_RUN`.

## 11. Đề Xuất Phân Tích Tiếp

Nếu đưa file này vào ChatGPT để phân tích, nên hỏi:

1. Nguyên nhân nào có thể khiến Step16 Sharpe âm mặc dù schema/parse ok cao?
2. Prediction distribution có bị bias hold/long/short không?
3. Flow Reward v2 vì sao kém proxy average và flow v1?
4. Counterfactual no-change 0.696 cho thấy vấn đề ở prompt, adapter hay parser?
5. B2/B3/B4 baseline đang mạnh hơn FIRE-Fin ở macro-F1/MCC; cần cải thiện model hay objective?
6. Nên ưu tiên ablation nào trong A0-A8 để tạo evidence paper mạnh nhất?
7. Final table nào có thể đưa vào main paper, table nào nên đưa appendix?
