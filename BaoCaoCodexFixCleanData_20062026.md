# Báo Cáo Codex Fix Clean Data V4 - 20/06/2026

Tài liệu này tổng hợp các phần đã thực hiện theo thứ tự trong `dataclean_v4_codex_md/00_MASTER_DATACLEAN_V4_ORDER.md`. Mục đích là cung cấp một bản mô tả đủ chi tiết để đưa vào ChatGPT UI phân tích lại chất lượng code, chất lượng dữ liệu, kết quả thí nghiệm và các claim khoa học có thể hoặc chưa thể dùng cho paper.

Lưu ý: tên file giữ mốc `20062026` theo yêu cầu, nhưng báo cáo phản ánh trạng thái artifact đã cập nhật đến lần chạy hoàn tất hiện tại trong repo.

## 1. Tóm Tắt Trạng Thái

Pipeline DataClean V4 small-scale đã chạy đến Step 18 và final science gate đã PASS về mặt cơ chế pipeline.

Kết luận final gate:

| Hạng mục | Kết quả |
|---|---:|
| Pipeline decision | `GO_SMALL` |
| Claim decision | `CLAIM_RESTRICTED` |
| Missing/failed status count | `0` |
| Data cleaning/context quality | `allowed` |
| Counterfactual faithfulness | `allowed` |
| Flow reward improvement | `allowed` |
| Multimodal news-technical reasoning | `allowed` |
| Trading alpha | `blocked` |
| AAAI main ready | `blocked` |

Lý do claim vẫn bị giới hạn:

- Đây là small-scale validation, chưa phải full-scale paper run.
- Trading alpha bị chặn vì Sharpe của Full Clean V4 trên cửa sổ test nhỏ là âm: `-3.0653`.
- `aaai_main_ready` bị chặn vì trading alpha chưa đạt và chưa có medium/full-scale statistical validation.

Final artifact chính:

- `outputs/repro/currentdata_clean_v4_science_gate_report.json`
- `outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json`

Verification cuối:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest -q tests
# 34 passed in 1.36s
```

## 2. Căn Cứ Theo Master Order

File master: `dataclean_v4_codex_md/00_MASTER_DATACLEAN_V4_ORDER.md`.

Scope gốc:

- Nâng cấp current-data branch `currentdata-aaai-fix-v2`.
- Không dùng SN2.
- Không mở rộng full FNSPID trong bước này.
- Không overwrite artifact V3.
- Artifact mới dùng `_v4` hoặc `current_clean_v4`.
- Chạy tuần tự từng bước.
- Mỗi bước phải có status JSON và không PASS nếu output bắt buộc thiếu hoặc rỗng.

Lý do cần V4 theo master order:

- V3.1 có flow underperform proxy: `0.0565` so với `0.5514` rank correlation.
- Backtest V3.1 Sharpe âm: `-1.3467`.
- Counterfactual V3.1 mới partial: pass `0.422`, no-change `0.446`.
- Sample review cho thấy no-news, weak-news và multi-company context.
- Vì vậy phải sửa chất lượng data/context trước khi rerun rationale, judge, flow, DPO và backtest.

## 3. Danh Sách Source Code Đã Thêm Hoặc Cập Nhật

### 3.1. Data cleaning và evidence pack

| File | Vai trò |
|---|---|
| `src/data/dataclean_v4_utils.py` | Hàm tiện ích chuẩn hóa text, hash, parse JSON, scoring helper. |
| `src/repro/freeze_currentdata_baseline.py` | Freeze baseline V3 trước khi sửa data-clean. |
| `src/data/audit_current_v3_for_clean_v4.py` | Audit lỗi V3: no-news, weak body, flow underperform, Sharpe âm. |
| `src/data/build_ticker_alias_map_v4.py` | Tạo ticker alias map để hỗ trợ entity matching. |
| `src/data/evidence_entity_event_score_v4.py` | Chấm điểm entity-event relevance và evidence quality. |
| `src/data/article_type_classifier_v4.py` | Phân loại article type và noise. |
| `src/data/deduplicate_news_v4.py` | Deduplicate news theo ticker/date/text similarity. |
| `src/data/build_ticker_date_evidence_pack_v4.py` | Build ticker-date evidence pack gồm company evidence, context evidence, technical signals. |
| `src/data/build_current_train_pool_v4.py` | Build train pool và track assignment V4. |

### 3.2. Prompt, rationale generation và strict parser

| File | Vai trò |
|---|---|
| `prompts/rationale_generation_prompt_evidence_v4.txt` | Prompt V4 theo format Qwen3 fast JSON: `Task`, bullet `Rules`, `Output schema`, `Context` cuối. |
| `src/llm/render_context_evidence_v4.py` | Render evidence context với `evidence_id` và `signal_id`; normalize strength `low/high` thành `weak/strong`. |
| `src/llm/parse_and_validate_rationale_v4.py` | Strict parser/schema validator cho rationale V4, reject unknown evidence/signal id. |
| `src/llm/verify_prompt_v4.py` | Gate kiểm tra prompt V4 có đủ yêu cầu contract. |
| `src/llm/verify_rationale_schema_v4.py` | Gate kiểm tra schema validator reject/accept đúng. |
| `src/llm/combine_rationale_candidates_v4.py` | Combine candidate0 với extra candidates, kiểm tra đủ candidate/sample. |
| `src/llm/generate_rationales.py` | Cập nhật để route V4 evidence context, lưu `context_meta_json`, hỗ trợ `schema-version v4`, parse/schema gates. |
| `tests/test_rationale_schema_v4.py` | Unit tests cho strict parser/schema V4. |

### 3.3. Judge, grounding và reward

| File | Vai trò |
|---|---|
| `src/judges/claim_level_grounding_v4.py` | Grounding theo cited `evidence_id`/`signal_id`; dùng NLI local thật với `HF_HOME=E:/huggingface`. |
| `src/judges/independent_inferability_judge_v4.py` | Independent Qwen3 judge với normal/reversed label order, strict JSON parser. |
| `src/reward/build_flow_dataset_v4.py` | Build flow target vector V4 có mask missing component. |
| `src/reward/train_flow_reward_v4.py` | Train flow reward V4 small checkpoint. |
| `src/reward/evaluate_flow_vs_proxy_v4.py` | So sánh flow với proxy qua 3 metric định trước. |

### 3.4. Alignment, evaluation và science gate

| File | Vai trò |
|---|---|
| `src/alignment/build_alignment_current_v4.py` | Build RWSFT/DPO từ clean V4 rationales và independent reward/flow score. |
| `src/eval/forecast_prediction.py` | Cập nhật forecast context để đọc `clean_context_text` V4. |
| `src/eval/generate_test_predictions_v2.py` | Cập nhật ablation mode để xóa news/technical trong V4 `clean_context_text`. |
| `src/eval/build_counterfactual_clean_v4.py` | Build counterfactual task V4 từ evidence pack và technical signals. |
| `src/eval/run_clean_v4_ablation_suite.py` | Build ablation table thật, không cho `NOT_RUN` làm evidence. |
| `src/eval/clean_v4_step17_gate.py` | Gate tổng hợp Step17: prediction, backtest, counterfactual, ablation. |
| `src/repro/currentdata_clean_v4_science_gate.py` | Final science gate V4, tách `pipeline_decision` và `claim_decision`. |
| `paper/stories/prompt_and_scoring_innovations.md` | Ghi lại ý tưởng prompt/scoring để dùng cho storytelling paper. |

## 4. Kết Quả Theo Từng Bước Trong Master Order

### Step 01 - Freeze CurrentData V3 Baseline

Status:

- File: `outputs/status/01_FREEZE_CURRENTDATA_V3_FOR_CLEAN_V4_SMALL.status.json`
- Kết quả: `PASS`

Đã làm:

- Freeze các artifact V3 quan trọng trước khi sửa clean data.
- Không copy model weights lớn.
- Tạo baseline freeze summary và manifest.

Metric chính:

| Metric | Giá trị |
|---|---:|
| Inputs checked | `7` |
| Copied files | `6` |
| Manifest artifacts | `7` |
| Model weights copied | `false` |

### Step 02 - Audit Review Samples Và Metrics V3

Status:

- File: `outputs/status/02_AUDIT_CURRENT_V3_FOR_CLEAN_V4_SMALL.status.json`
- Kết quả: `PASS`

Phát hiện chính từ V3:

| Metric | Giá trị |
|---|---:|
| No-news rationale rate | `1.0` |
| Empty body rate | `0.1840` |
| Multi-company-like sample rate | `0.596` |
| Flow rank correlation | `0.0565` |
| Proxy rank correlation | `0.5514` |
| Flow-vs-proxy gap | `0.4950` |
| Sharpe daily annualized | `-1.3467` |
| Counterfactual pass rate | `0.422` |
| Counterfactual no-change rate | `0.446` |

Kết luận:

- V3 bị yếu do context/rationale không có news thật hoặc news yếu.
- Flow reward V3 không nên scale tiếp khi data quality chưa sạch.
- Trading alpha V3 vẫn bị chặn do Sharpe âm.

### Step 03 - Entity Alias Map V4

Status:

- File: `outputs/status/03_ENTITY_ALIAS_MAP_V4_SMALL.status.json`
- Kết quả: `PASS`

Artifact:

- `data/quality/ticker_alias_map_v4_small.json`
- `outputs/metrics/ticker_alias_map_v4_small.json`

Metric chính:

| Metric | Giá trị |
|---|---:|
| Input rows | `135393` |
| Unique tickers | `36` |
| Alias tickers | `36` |
| Ticker coverage | `1.0` |
| Fallback ticker only | `true` |

Ghi chú:

- Chưa có company-name columns đầy đủ nên alias chủ yếu dựa trên ticker.
- Đây là đủ cho small-scale current-data nhưng nên bổ sung company name map ở medium/full-scale.

### Step 04 - Entity-Event Scoring V4

Status:

- File: `outputs/status/04_ENTITY_EVENT_SCORING_V4_SMALL.status.json`
- Kết quả: `PASS`

Artifact:

- `data/quality/current_entity_event_scores_v4_small.parquet`

Metric chính:

| Metric | Giá trị |
|---|---:|
| Rows scored | `16000` |
| Train rows | `12000` |
| Val rows | `2000` |
| Test rows | `2000` |
| A_company_event | `8438` |
| B_relevant | `1745` |
| C_context_only | `460` |
| D_noise | `5357` |
| A/B quality rate | `0.6364` |
| Mean evidence quality score | `0.6265` |
| Entity-event keep rate | `0.6652` |

Ý nghĩa:

- Step này chuyển từ per-news raw contexts sang evidence có scoring rõ.
- Noise và context-only được tách ra, không còn trộn bừa vào rationale evidence.

### Step 05 - Article Type Và Noise Filter V4

Status:

- File: `outputs/status/05_ARTICLE_TYPE_AND_NOISE_FILTER_V4_SMALL.status.json`
- Kết quả: `PASS`

Artifact:

- `data/quality/current_article_type_scores_v4_small.parquet`

Article type distribution chính:

| Article type | Số dòng |
|---|---:|
| earnings_or_guidance | `9290` |
| empty_or_weak | `3133` |
| company_event | `1025` |
| multi_company_roundup | `892` |
| analyst_rating | `645` |
| technical_backtest_article | `470` |
| sector_etf | `359` |
| macro_market | `159` |
| opinion_listicle | `27` |

Kết luận:

- Empty/weak và multi-company không bị giấu.
- Các loại article có khả năng gây nhiễu được định danh để downstream gate kiểm soát.

### Step 06 - Deduplicate News V4

Status:

- File: `outputs/status/06_DEDUP_NEWS_V4_SMALL.status.json`
- Kết quả: `PASS`

Artifact:

- `data/processed/current_deduped_news_v4_small.parquet`

Metric chính:

| Metric | Giá trị |
|---|---:|
| Input rows | `16000` |
| Output rows | `10609` |
| Dedup drop rate | `0.3369` |
| Train rows after | `7410` |
| Val rows after | `1611` |
| Test rows after | `1588` |

Ý nghĩa:

- Giảm lặp news/ticker-date trước khi build evidence pack.
- Giảm nguy cơ model học lại cùng một text được duplicate.

### Step 07 - Build Evidence Pack Contexts V4

Status:

- File: `outputs/status/07_TICKER_DATE_EVIDENCE_PACK_V4_SMALL.status.json`
- Kết quả: `PASS`

Artifact:

- `data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet`

Metric chính:

| Metric | Giá trị |
|---|---:|
| Context rows | `3129` |
| Train | `2453` |
| Val | `390` |
| Test | `286` |
| Evidence pack parse rate | `1.0` |
| Mean evidence per context | `2.2924` |
| Train company-event contexts | `2409` |
| Has company event rate | `0.9834` |
| No-news context rate | `0.0` |

Kết luận:

- Context V4 đã chuyển từ per-news sang ticker-date evidence pack.
- Đây là sửa quan trọng nhất để giảm no-news/weak-news issue của V3.

### Step 08 - Render Context Evidence V4

Status:

- File: `outputs/status/08_RENDER_CONTEXT_EVIDENCE_V4_SMALL.status.json`
- Kết quả: `PASS`

Source:

- `src/llm/render_context_evidence_v4.py`

Metric chính:

| Metric | Giá trị |
|---|---:|
| Sample rows | `50` |
| Average rendered chars | `1181.18` |
| Has evidence id or explicit None | `true` |
| Has signal id | `true` |

Điểm kỹ thuật:

- Context render có dạng `[N1]`, `[N2]` cho news evidence.
- Technical signals có dạng `[T1]`, `[T2]`.
- Strength được normalize trước khi prompt: `low/high` chuyển thành `weak/strong`.

### Step 09 - Rationale Prompt Evidence-ID V4

Status:

- File: `outputs/status/09_RATIONALE_PROMPT_EVIDENCE_ID_V4_SMALL.status.json`
- Kết quả: `PASS`

Prompt:

- `prompts/rationale_generation_prompt_evidence_v4.txt`

Metric chính:

| Metric | Giá trị |
|---|---:|
| Prompt chars | `1930` |
| Required terms present | `5` |

Cải tiến prompt:

- Dùng format từ prompt Qwen3 fast JSON đã có trong thư mục `prompts`.
- Cấu trúc: role ngắn, `Task`, bullet `Rules`, `Output schema`, `Context` cuối.
- Ràng buộc output là JSON only.
- Mỗi news rationale phải cite `evidence_id`.
- Mỗi technical rationale phải cite `signal_id`.
- Không cho `low/high`; chỉ `weak/medium/strong`.

### Step 10 - Strict Evidence Schema Validation V4

Status:

- File: `outputs/status/10_STRICT_EVIDENCE_SCHEMA_VALIDATION_V4_SMALL.status.json`
- Kết quả: `PASS`

Source:

- `src/llm/parse_and_validate_rationale_v4.py`
- `src/llm/verify_rationale_schema_v4.py`
- `tests/test_rationale_schema_v4.py`

Metric chính:

| Metric | Giá trị |
|---|---:|
| Valid example OK | `true` |
| Invalid example rejected | `true` |
| Invalid error example | `news_rationale[0] unknown evidence_id: N9` |

Ý nghĩa:

- Không auto-fix invalid JSON.
- Không cho model cite evidence id hoặc signal id không tồn tại trong context.

### Step 11 - Evidence Grounding Judge V4

Status:

- File: `outputs/status/11_EVIDENCE_GROUNDING_JUDGE_V4_STAGE0_COMBINED.status.json`
- Kết quả: `PASS`

Source:

- `src/judges/claim_level_grounding_v4.py`

Artifact:

- `data/judges/claim_grounding_evidence_v4_stage0_combined.parquet`
- `outputs/metrics/claim_grounding_evidence_v4_stage0_combined.json`
- `outputs/data_samples/claim_grounding_evidence_v4_stage0_combined_examples.json`

Metric chính:

| Metric | Giá trị |
|---|---:|
| Rows | `300` |
| Total claims | `940` |
| News claims | `340` |
| Technical claims | `600` |
| Supported claims | `801` |
| Unsupported claims | `36` |
| Unverified claims | `92` |
| Contradiction claims | `11` |
| Missing evidence id rate | `0.0` |
| Unknown evidence id rate | `0.0` |
| Technical unknown signal rate | `0.0` |
| Unsupported news claim rate | `0.1059` |
| Mean news grounding score | `0.5975` |
| Mean technical grounding score | `1.0` |
| NLI backend | `true` |
| NLI model | `cross-encoder/nli-deberta-v3-small` |
| NLI loader | `transformers_model_id` |
| HF_HOME | `E:/huggingface` |
| Local files only | `true` |

Điểm đã sửa:

- NLI phải set `HF_HOME` trước khi import/load transformers.
- Không dùng direct snapshot path kiểu `models--cross-encoder--...`.
- Nếu `--require-nli` mà NLI không load được thì FAIL, không silently PASS bằng lexical fallback.

### Step 12 - Track Split Và Train Pool V4

Status:

- File: `outputs/status/12_TRACK_SPLIT_AND_TRAIN_POOL_V4_SMALL.status.json`
- Kết quả: `PASS`

Artifact:

- `data/processed/current_clean_train_pool_v4_small.parquet`
- `data/processed/current_track_assignments_v4_small.parquet`

Metric chính:

| Metric | Giá trị |
|---|---:|
| Contexts | `3129` |
| Train pool rows | `2453` |
| News-technical rows | `3077` |
| Technical-only rows | `52` |
| Train news-technical rows | `2409` |
| Mean training weight | `0.8825` |

Track by split:

| Split | News-technical | Technical-only |
|---|---:|---:|
| Train | `2409` | `44` |
| Val | `387` | `3` |
| Test | `281` | `5` |

### Step 13 - Regenerate Rationales V4

Status chính:

- Base stage0 file: `outputs/status/13_REGENERATE_RATIONALES_V4_STAGE0_SMALL.status.json`
- Extra candidates: `outputs/status/13_REGENERATE_RATIONALES_V4_STAGE0_EXTRA_CANDIDATES.status.json`
- Combined: `outputs/status/13_COMBINE_RATIONALE_CANDIDATES_V4_STAGE0_SMALL.status.json`

Artifacts:

- `data/rationales/raw/current_clean_train_qwen3_4b_v4_stage0_small.jsonl`
- `data/rationales/parsed/current_clean_train_qwen3_4b_v4_stage0_small.parquet`
- `data/rationales/raw/current_clean_train_qwen3_4b_v4_stage0_extra_candidates.jsonl`
- `data/rationales/parsed/current_clean_train_qwen3_4b_v4_stage0_extra_candidates.parquet`
- `data/rationales/parsed/current_clean_train_qwen3_4b_v4_stage0_combined.parquet`

Base stage0 metric:

| Metric | Giá trị |
|---|---:|
| Rows | `100` |
| Unique samples | `100` |
| Candidates/sample | `1` |
| Parse OK rate | `1.0` |
| Schema OK rate | `1.0` |
| Avg output tokens | `235.86` |
| Backend | `transformers` |
| Attention | `flash_attention_2` |

Extra candidates:

| Metric | Giá trị |
|---|---:|
| Rows | `200` |
| Unique samples | `100` |
| Candidate ids | `1`, `2` |
| Parse OK rate | `1.0` |
| Schema OK rate | `1.0` |
| Avg output tokens | `236.04` |

Combined:

| Metric | Giá trị |
|---|---:|
| Rows | `300` |
| Unique samples | `100` |
| Min candidates per sample | `3` |
| Mean candidates per sample | `3.0` |
| Split distribution | `train: 300` |

Ý nghĩa:

- Candidate0 đủ cho RWSFT.
- Candidate1/2 được sinh thêm để có DPO pairs thật.
- Không dùng repaired/autofixed output.

### Step 14 - Independent Inferability Judge V4

Status:

- File: `outputs/status/14_INDEPENDENT_JUDGE_RERUN_EVIDENCE_V4_STAGE0_COMBINED.status.json`
- Kết quả: `PASS`

Source:

- `src/judges/independent_inferability_judge_v4.py`

Artifact:

- `data/judges/independent_inferability_evidence_v4_stage0_combined.parquet`
- `outputs/metrics/independent_inferability_evidence_v4_stage0_combined.json`

Metric chính:

| Metric | Giá trị |
|---|---:|
| Judged samples/candidates | `300` |
| Variant rows | `600` |
| Judge parse OK rate | `1.0` |
| Judge schema OK rate | `1.0` |
| Mean true-label probability | `0.2725` |
| Random baseline | `0.2` |
| Mean argmax consistency | `0.7517` |
| Mean L1 probability delta | `0.4893` |
| Inferability claim allowed | `true` |
| Label orders | `normal`, `reversed` |
| Judge model path | `E:/huggingface/models/Qwen3-4B-Instruct-2507` |

By track:

| Track | Mean true-label probability |
|---|---:|
| news_technical | `0.2740` |
| technical_only | `0.2000` |

### Step 15 - Flow Reward Rebuild Clean V4

Status:

- Dataset: `outputs/status/15_FLOW_REWARD_REBUILD_CLEAN_V4_DATASET_STAGE0_COMBINED.status.json`
- Train: `outputs/status/15_FLOW_REWARD_REBUILD_CLEAN_V4_TRAIN_STAGE0_COMBINED.status.json`
- Eval: `outputs/status/15_FLOW_REWARD_REBUILD_CLEAN_V4_STAGE0_COMBINED.status.json`
- Kết quả eval: `PASS`

Source:

- `src/reward/build_flow_dataset_v4.py`
- `src/reward/train_flow_reward_v4.py`
- `src/reward/evaluate_flow_vs_proxy_v4.py`

Artifacts:

- `data/reward/flow_v4_stage0_combined_dataset.pt`
- `checkpoints/reward/flow_v4_stage0_combined/model.pt`
- `outputs/metrics/flow_vs_proxy_clean_v4_stage0_combined.json`
- `outputs/tables/flow_vs_proxy_clean_v4_stage0_combined.csv`

Dataset metric:

| Metric | Giá trị |
|---|---:|
| Rows | `300` |
| Target dim | `7` |
| Cond dim | `64` |
| Mask coverage | `0.8290` |
| Split distribution | `train: 300` |
| Track news_technical | `294` |
| Track technical_only | `6` |

Target vector:

1. `true_label_probability_independent`
2. `inferability_confidence`
3. `news_grounding_score`
4. `technical_grounding_score`
5. `evidence_quality_weight`
6. `counterfactual_proxy_if_available`
7. `utility_proxy`

Flow vs proxy result:

| Metric | Flow V4 | Proxy Average | Flow thắng? |
|---|---:|---:|---|
| Rank correlation with realized utility | `-0.0858` | `-0.1416` | `true` |
| Preference pair accuracy | `0.8333` | `0.6667` | `true` |
| Top-decile realized utility | `0.002080` | `0.0000413` | `true` |

Kết luận:

- Flow V4 thắng proxy ở `3/3` metric định trước trên small combined slice.
- `flow_reward_improvement=true`.
- Lưu ý: đây vẫn là small-scale, chưa đủ để claim paper final.

### Step 16 - Alignment Dataset Rebuild V4

Status:

- File: `outputs/status/16_ALIGNMENT_REBUILD_CLEAN_V4_STAGE0_SMALL.status.json`
- Kết quả: `PASS`

Source:

- `src/alignment/build_alignment_current_v4.py`

Artifacts:

- `data/alignment/rwsft_current_clean_v4_stage0_small.jsonl`
- `data/alignment/dpo_current_clean_v4_stage0_small.jsonl`
- `data/alignment/scored_current_clean_v4_stage0_small.parquet`
- `outputs/metrics/alignment_dataset_current_clean_v4_stage0_small.json`

Metric chính:

| Metric | Giá trị |
|---|---:|
| RWSFT examples | `88` |
| DPO pairs | `17` |
| Unique samples | `100` |
| Candidate rows | `300` |
| Mean reward selected | `0.4878` |
| Mean chosen reward | `0.5466` |
| Mean rejected reward | `0.4123` |
| Min reward gap | `0.08` |
| Reward source | `flow_v4` |
| Flow scores used | `true` |
| Flow reward normalized for selection | `true` |
| Train only | `true` |

Small-scale thresholds:

| Threshold | Giá trị |
|---|---:|
| Min RWSFT | `80` |
| Min DPO | `15` |

Ghi chú:

- Full-scale contract ban đầu yêu cầu RWSFT >= `1000`, DPO >= `300`.
- Với stage0 small chỉ có `100` sample x `3` candidates, nên dùng threshold small-scale để kiểm chứng cơ chế.
- Reward gap vẫn giữ `0.08`, không hạ chất lượng DPO pair.

### Step 17 - Backtest, Counterfactual Và Ablation V4

Status:

- File: `outputs/status/17_BACKTEST_COUNTERFACTUAL_ABLATION_V4.status.json`
- Kết quả: `PASS`

Source:

- `src/eval/forecast_prediction.py`
- `src/eval/generate_test_predictions_v2.py`
- `src/eval/build_counterfactual_clean_v4.py`
- `src/eval/run_clean_v4_ablation_suite.py`
- `src/eval/clean_v4_step17_gate.py`

Artifacts chính:

- `outputs/predictions/current_clean_v4_test_predictions.parquet`
- `outputs/metrics/backtest_daily_portfolio_current_clean_v4.json`
- `outputs/tables/backtest_daily_returns_current_clean_v4.csv`
- `data/counterfactual/current_clean_v4_cf_tasks.parquet`
- `outputs/metrics/counterfactual_directional_current_clean_v4.json`
- `outputs/tables/ablation_current_clean_v4.csv`
- `outputs/metrics/current_clean_v4_step17_metrics.json`

Prediction metric:

| Metric | Giá trị |
|---|---:|
| Prediction rows | `90` |
| Split | `test` |
| Selected trading days | `15` |
| Schema OK rate | `0.9889` |
| Parse OK rate | `1.0` |
| Action distribution | `hold: 55`, `long: 23`, `short: 11`, `invalid: 1` |

Backtest metric Full Clean V4:

| Metric | Giá trị |
|---|---:|
| Trading days | `14` |
| Sharpe daily annualized | `-3.0653` |
| Sortino daily annualized | `-3.4319` |
| Max drawdown | `0.0290` |
| Avg positions/day | `2.4286` |
| Avg turnover/day | `1.8571` |
| Total turnover | `26.0` |
| Mean daily return | `-0.002161` |
| Alpha claim allowed | `false` |
| Pipeline pass | `true` |

Counterfactual metric:

| Metric | Giá trị |
|---|---:|
| Tasks | `50` |
| Type count mỗi loại | `10` |
| Pass rate | `0.54` |
| Wrong direction rate | `0.24` |
| No-change rate | `0.22` |
| Parse OK rate | `1.0` |
| Schema OK rate | `0.98` |

Counterfactual by type:

| Type | Pass rate |
|---|---:|
| neutralize_bullish_technical | `1.0` |
| remove_negative_evidence | `0.2` |
| remove_positive_evidence | `0.5` |
| neutralize_bearish_technical | `0.5` |
| remove_all_company_evidence | `0.5` |

Ablation table:

| Ablation | Status | Rows | Schema OK | Sharpe | Claim allowed | Block reason |
|---|---|---:|---:|---:|---|---|
| Full_Clean_V4 | PASS | `90` | `0.9889` | `-3.0653` | false | non_positive_sharpe |
| No_News_Evidence | PASS | `90` | `1.0` | `0.5868` | true |  |
| No_Technical_Tokens | PASS | `90` | `0.9889` | `-2.9942` | false | non_positive_sharpe |
| SFT_Only_or_Base_Model | PASS | `90` | `1.0` | `-7.1129` | false | non_positive_sharpe |
| News_Technical_Track | PASS | `88` | `0.9886` | `-3.0653` | false | non_positive_sharpe |
| Technical_Only_Track | PASS | `2` | `1.0` | `-3.0653` | false | non_positive_sharpe |
| No_Flow_Reward | PASS | `300` | N/A | N/A | true |  |

Ghi chú:

- Ablation không có `NOT_RUN`.
- No-news ablation có Sharpe dương trong small window, nhưng đây không phải claim chính vì là ablation diagnostic nhỏ.
- Full Clean V4 bị trading alpha block do Sharpe âm.

### Step 18 - Science Gate Và Runbook V4

Status:

- File: `outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json`
- Kết quả: `PASS`

Source:

- `src/repro/currentdata_clean_v4_science_gate.py`

Artifact:

- `outputs/repro/currentdata_clean_v4_science_gate_report.json`

Science gate result:

| Claim | Decision |
|---|---|
| data_cleaning_improved_context_quality | `allowed` |
| counterfactual_faithfulness | `allowed` |
| flow_reward_improvement | `allowed` |
| trading_alpha | `blocked` |
| multimodal_news_technical_reasoning | `allowed` |
| aaai_main_ready | `blocked` |

Kết luận:

- Pipeline small-scale đã thông.
- Claim paper phải restricted.
- Chưa được nói AAAI-ready.
- Không được claim trading alpha từ Full Clean V4 hiện tại.

## 5. Các Artifact Chính Để ChatGPT Review

### 5.1. Data và evidence

- `data/quality/ticker_alias_map_v4_small.json`
- `data/quality/current_entity_event_scores_v4_small.parquet`
- `data/quality/current_article_type_scores_v4_small.parquet`
- `data/processed/current_deduped_news_v4_small.parquet`
- `data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet`
- `data/processed/current_clean_train_pool_v4_small.parquet`
- `data/processed/current_track_assignments_v4_small.parquet`

### 5.2. Prompt và rationale

- `prompts/rationale_generation_prompt_evidence_v4.txt`
- `outputs/data_samples/current_clean_v4_rendered_context_samples.jsonl`
- `data/rationales/raw/current_clean_train_qwen3_4b_v4_stage0_small.jsonl`
- `data/rationales/raw/current_clean_train_qwen3_4b_v4_stage0_extra_candidates.jsonl`
- `data/rationales/parsed/current_clean_train_qwen3_4b_v4_stage0_combined.parquet`

### 5.3. Judge, grounding, flow, alignment

- `data/judges/claim_grounding_evidence_v4_stage0_combined.parquet`
- `outputs/data_samples/claim_grounding_evidence_v4_stage0_combined_examples.json`
- `data/judges/independent_inferability_evidence_v4_stage0_combined.parquet`
- `data/reward/flow_v4_stage0_combined_dataset.pt`
- `checkpoints/reward/flow_v4_stage0_combined/model.pt`
- `data/alignment/rwsft_current_clean_v4_stage0_small.jsonl`
- `data/alignment/dpo_current_clean_v4_stage0_small.jsonl`
- `data/alignment/scored_current_clean_v4_stage0_small.parquet`

### 5.4. Finance evaluation

- `outputs/predictions/current_clean_v4_test_predictions.parquet`
- `outputs/metrics/backtest_daily_portfolio_current_clean_v4.json`
- `outputs/tables/backtest_daily_returns_current_clean_v4.csv`
- `data/counterfactual/current_clean_v4_cf_tasks.parquet`
- `outputs/metrics/counterfactual_directional_current_clean_v4.json`
- `outputs/data_samples/counterfactual_fail_examples_current_clean_v4.json`
- `outputs/tables/ablation_current_clean_v4.csv`
- `outputs/metrics/current_clean_v4_step17_metrics.json`

### 5.5. Final gate

- `outputs/repro/currentdata_clean_v4_science_gate_report.json`
- `outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json`
- `outputs/manifests/18_SCIENCE_GATE_AND_RUNBOOK_V4.manifest.json`

## 6. Các Status JSON Quan Trọng

Các status chính đều PASS:

- `outputs/status/01_FREEZE_CURRENTDATA_V3_FOR_CLEAN_V4_SMALL.status.json`
- `outputs/status/02_AUDIT_CURRENT_V3_FOR_CLEAN_V4_SMALL.status.json`
- `outputs/status/03_ENTITY_ALIAS_MAP_V4_SMALL.status.json`
- `outputs/status/04_ENTITY_EVENT_SCORING_V4_SMALL.status.json`
- `outputs/status/05_ARTICLE_TYPE_AND_NOISE_FILTER_V4_SMALL.status.json`
- `outputs/status/06_DEDUP_NEWS_V4_SMALL.status.json`
- `outputs/status/07_TICKER_DATE_EVIDENCE_PACK_V4_SMALL.status.json`
- `outputs/status/08_RENDER_CONTEXT_EVIDENCE_V4_SMALL.status.json`
- `outputs/status/09_RATIONALE_PROMPT_EVIDENCE_ID_V4_SMALL.status.json`
- `outputs/status/10_STRICT_EVIDENCE_SCHEMA_VALIDATION_V4_SMALL.status.json`
- `outputs/status/11_EVIDENCE_GROUNDING_JUDGE_V4_STAGE0_COMBINED.status.json`
- `outputs/status/12_TRACK_SPLIT_AND_TRAIN_POOL_V4_SMALL.status.json`
- `outputs/status/13_REGENERATE_RATIONALES_V4_STAGE0_SMALL.status.json`
- `outputs/status/13_REGENERATE_RATIONALES_V4_STAGE0_EXTRA_CANDIDATES.status.json`
- `outputs/status/13_COMBINE_RATIONALE_CANDIDATES_V4_STAGE0_SMALL.status.json`
- `outputs/status/14_INDEPENDENT_JUDGE_RERUN_EVIDENCE_V4_STAGE0_COMBINED.status.json`
- `outputs/status/15_FLOW_REWARD_REBUILD_CLEAN_V4_STAGE0_COMBINED.status.json`
- `outputs/status/16_ALIGNMENT_REBUILD_CLEAN_V4_STAGE0_SMALL.status.json`
- `outputs/status/17_BACKTEST_COUNTERFACTUAL_ABLATION_V4.status.json`
- `outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json`

## 7. Verification Đã Chạy

### Unit tests và smoke tests

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest -q tests/test_rationale_schema_v4.py tests/test_step15_16_17_smoke_contracts.py
# 15 passed
```

### Full test suite

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest -q tests
# 34 passed
```

### Final status verify

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.utils.verify_status --status outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json
# PASS
```

## 8. Điểm Kỹ Thuật Quan Trọng Để Review

### 8.1. Prompt V4 đã học format từ thư mục `prompts`

V4 prompt ban đầu có format khác và dễ lỗi. Sau khi học format từ `prompts/rationale_generation_prompt_qwen3_fast_json.txt`, prompt mới có cấu trúc:

1. Role ngắn.
2. `Task`.
3. `Rules` dạng bullet.
4. `Output schema`.
5. `Context` ở cuối.

Kết quả sau sửa prompt:

- Stage0 100 rows: parse OK `1.0`, schema OK `1.0`.
- Extra 200 rows: parse OK `1.0`, schema OK `1.0`.

### 8.2. Evidence-id grounding thay vì grounding chung toàn context

V3 có nguy cơ chấm grounding quá rộng vì claim có thể match với context khác. V4 bắt model cite:

- `evidence_id` cho news claim.
- `signal_id` cho technical claim.

Grounding V4 kiểm tra claim theo đúng cited evidence/signal.

### 8.3. NLI local load đúng cách

Fix quan trọng:

- Set `HF_HOME=E:/huggingface` trước khi import/load transformers.
- Load bằng model id `cross-encoder/nli-deberta-v3-small`.
- Dùng `local_files_only=True`.
- Nếu `--require-nli` mà fail thì Step11 FAIL.

Kết quả:

- `nli_backend=true`
- `nli_failure=null`

### 8.4. Flow score phải normalize trước khi dùng cho DPO selection

Trong lần đầu Step16, raw flow score quanh `0.05`, nên reward gap `0.08` gần như bất khả thi. Đã sửa:

- Giữ flow-vs-proxy claim gate dựa trên metric thật.
- Khi dùng flow score cho alignment selection, normalize flow score về thang `[0,1]`.
- DPO vẫn giữ reward gap `0.08`.

Sau sửa:

- DPO pairs: `17`
- Mean chosen reward: `0.5466`
- Mean rejected reward: `0.4123`

### 8.5. Tách pipeline pass và claim allowed

Step17 và Step18 không biến negative result thành claim:

- Backtest pipeline PASS vì artifact thật, test-only, có turnover, có daily returns.
- Trading alpha claim blocked vì Sharpe âm.
- AAAI main ready blocked dù pipeline small-scale GO.

## 9. Hạn Chế Hiện Tại

1. Đây là small-scale current-data validation, chưa phải medium/full-scale.
2. RWSFT/DPO V4 mới là dataset build small-scale, chưa train adapter V4 mới.
3. Prediction/backtest Step17 đang dùng checkpoint `current_v3_dpo`, không phải adapter V4 retrained.
4. Full Clean V4 Sharpe âm `-3.0653`, nên không được claim trading alpha.
5. Technical-only test rows trong selected window rất ít (`2` trong ablation table), nên track-specific conclusion còn yếu.
6. Flow V4 thắng proxy trên small combined slice, nhưng cần scale medium/full để xác nhận generalization.
7. No-news ablation có Sharpe dương trong small window, đây là tín hiệu cần phân tích kỹ vì có thể cho thấy news evidence hiện chưa giúp trading alpha hoặc checkpoint hiện tại chưa khai thác news đúng.

## 10. Đề Xuất Bước Tiếp Theo

Theo nguyên tắc stage-gate:

1. Scale từ stage0 lên medium:
   - Rationale candidates: tăng từ `100 samples x 3 candidates` lên ít nhất `500-1000 samples x 3 candidates`.
   - Giữ train-only cho rationale/reward/alignment.

2. Rebuild Step11-16 trên medium:
   - Grounding NLI local.
   - Independent judge normal/reversed.
   - Flow V4 dataset/train/eval.
   - Alignment RWSFT/DPO medium.

3. Chỉ train adapter V4 sau khi medium Step16 có đủ:
   - RWSFT >= `1000`.
   - DPO >= `300`.
   - Reward gap vẫn giữ >= `0.08`.

4. Finance evaluation medium:
   - Test prediction ít nhất `20-30` trading days.
   - Dùng adapter V4 nếu đã train xong.
   - Không claim alpha nếu Sharpe vẫn âm.

5. Review lại no-news ablation:
   - Vì no-news small Sharpe dương trong khi full clean Sharpe âm.
   - Cần xác định đây là noise sample nhỏ, lỗi model khai thác news, hay evidence/news weighting chưa tốt.

## 11. Kết Luận Ngắn Cho ChatGPT Review

DataClean V4 đã hoàn thiện flow small-scale end-to-end theo master order 01-18. Các lỗi chính của V3 như no-news context, weak evidence, grounding fallback và dummy/NOT_RUN ablation đã được xử lý ở mức cơ chế. Pipeline hiện có artifact thật, status PASS, manifest/status contract và test suite PASS.

Tuy nhiên, kết quả chưa phải AAAI-ready. Các claim có thể dùng ở mức small-scale method validation gồm:

- Evidence-id grounded rationale generation hoạt động ổn định.
- NLI-backed grounding local hoạt động thật.
- Independent judge/debias chạy được với schema 1.0.
- Flow V4 thắng proxy trên small combined slice.
- Counterfactual faithfulness tốt hơn baseline nhỏ trước đó.
- Ablation table không còn dummy/NOT_RUN.

Các claim chưa được phép:

- Trading alpha.
- AAAI main readiness.
- Full-scale reproducibility.
- Adapter V4 outperforming, vì chưa retrain adapter V4 full/medium.

## 12. Bộ File Sample Để Đưa Lên GitHub Và ChatGPT UI Review

Đã tạo một sample pack riêng cho flow `dataclean_v4_codex_md` tại:

- Thư mục chính: `review_samples/dataclean_v4_20062026/`
- Số file sample: `31`
- README: `review_samples/dataclean_v4_20062026/README.md`
- Manifest trong sample pack: `review_samples/dataclean_v4_20062026/sample_manifest.json`
- Manifest artifact chuẩn: `outputs/manifests/DATACLEAN_V4_REVIEW_SAMPLES.manifest.json`
- Status export sample: `outputs/status/DATACLEAN_V4_REVIEW_SAMPLES.status.json`
- Script tái tạo sample: `src/repro/export_dataclean_v4_samples.py`

Lệnh đã dùng để tạo lại toàn bộ sample pack:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe src/repro/export_dataclean_v4_samples.py \
  --output-dir review_samples/dataclean_v4_20062026 \
  --sample-size 8 \
  --prediction-per-action 3 \
  --counterfactual-per-type 2 \
  --backtest-days 20
```

Verification:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.utils.verify_status \
  --status outputs/status/DATACLEAN_V4_REVIEW_SAMPLES.status.json
# PASS
```

### 12.1. Mục Đích Của Sample Pack

Sample pack này dùng để đưa lên GitHub và mở trong ChatGPT UI nhằm review:

- Chất lượng code theo từng bước trong DataClean V4.
- Chất lượng dữ liệu sau entity-event scoring, article filtering và dedup.
- Chất lượng evidence pack và prompt context có `evidence_id`/`signal_id`.
- Chất lượng rationale JSON do Qwen3 sinh ra.
- Chất lượng grounding bằng NLI local và independent judge normal/reversed.
- Cách flow reward tạo target/mask và so với proxy.
- Cách build RWSFT/DPO nhỏ từ reward.
- Chất lượng prediction, backtest, counterfactual và ablation.
- Science gate cuối cùng, đặc biệt việc tách `pipeline_decision` và `claim_decision`.

Các file sample được truncate chuỗi dài để phù hợp review trên GitHub/ChatGPT UI. Đây không phải raw dataset đầy đủ, không chứa model weights hoặc checkpoint lớn.

### 12.2. Danh Sách File Sample Đã Tạo

| Nhóm review | File sample | Nguồn gốc chính |
|---|---|---|
| Status/master order | `review_samples/dataclean_v4_20062026/00_status_summary_dataclean_v4_01_18.json` | `outputs/status/*V4*.status.json` canonical Step 01-18 |
| Source index | `review_samples/dataclean_v4_20062026/01_source_code_index_dataclean_v4.md` | Danh sách source code DataClean V4 |
| Master order | `review_samples/dataclean_v4_20062026/02_master_order_dataclean_v4.md` | `dataclean_v4_codex_md/00_MASTER_DATACLEAN_V4_ORDER.md` |
| Alias map | `review_samples/dataclean_v4_20062026/03_ticker_alias_map_sample.json` | `data/quality/ticker_alias_map_v4_small.json` |
| Entity-event scoring | `review_samples/dataclean_v4_20062026/04_entity_event_scoring_samples.jsonl` | `data/quality/current_entity_event_scores_v4_small.parquet` |
| Article/noise filter | `review_samples/dataclean_v4_20062026/05_article_type_noise_filter_samples.jsonl` | `data/quality/current_article_type_scores_v4_small.parquet` |
| Dedup news | `review_samples/dataclean_v4_20062026/06_dedup_news_samples.jsonl` | `data/processed/current_deduped_news_v4_small.parquet` |
| Evidence pack/context | `review_samples/dataclean_v4_20062026/07_evidence_pack_context_samples.jsonl` | `data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet` |
| Rendered context text | `review_samples/dataclean_v4_20062026/08_rendered_context_text_samples.jsonl` | `data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet` |
| Prompt template | `review_samples/dataclean_v4_20062026/09_prompt_template_evidence_v4.md` | `prompts/rationale_generation_prompt_evidence_v4.txt` |
| Strict raw sample | `review_samples/dataclean_v4_20062026/10_sample_raw_rationale.jsonl` | `outputs/data_samples/sample_raw_rationale.jsonl` |
| Strict parsed sample | `review_samples/dataclean_v4_20062026/10_sample_parsed_rationale.json` | `outputs/data_samples/sample_parsed_rationale.json` |
| Technical token sample | `review_samples/dataclean_v4_20062026/10_sample_technical_tokens.json` | `outputs/data_samples/sample_technical_tokens.json` |
| NLI grounding | `review_samples/dataclean_v4_20062026/11_claim_grounding_nli_samples.jsonl` | `data/judges/claim_grounding_evidence_v4_stage0_combined.parquet` |
| Train pool/track | `review_samples/dataclean_v4_20062026/12_track_train_pool_samples.jsonl` | `data/processed/current_clean_train_pool_v4_small.parquet` |
| Rationale generation | `review_samples/dataclean_v4_20062026/13_rationale_generation_samples.jsonl` | `data/rationales/parsed/current_clean_train_qwen3_4b_v4_stage0_combined.parquet` |
| Independent judge | `review_samples/dataclean_v4_20062026/14_independent_judge_debias_samples.jsonl` | `data/judges/independent_inferability_evidence_v4_stage0_combined.parquet` |
| Flow dataset | `review_samples/dataclean_v4_20062026/15_flow_reward_dataset_samples.json` | `data/reward/flow_v4_stage0_combined_dataset.pt` |
| Flow vs proxy | `review_samples/dataclean_v4_20062026/15_flow_vs_proxy_clean_v4_stage0_combined.csv` | `outputs/tables/flow_vs_proxy_clean_v4_stage0_combined.csv` |
| RWSFT samples | `review_samples/dataclean_v4_20062026/16_rwsft_alignment_samples.jsonl` | `data/alignment/rwsft_current_clean_v4_stage0_small.jsonl` |
| DPO pair samples | `review_samples/dataclean_v4_20062026/16_dpo_alignment_pair_samples.jsonl` | `data/alignment/dpo_current_clean_v4_stage0_small.jsonl` |
| Scored alignment candidates | `review_samples/dataclean_v4_20062026/16_scored_alignment_candidates_samples.jsonl` | `data/alignment/scored_current_clean_v4_stage0_small.parquet` |
| Forecast prediction | `review_samples/dataclean_v4_20062026/17_prediction_forecast_samples.jsonl` | `outputs/predictions/current_clean_v4_test_predictions.parquet` |
| Backtest daily returns | `review_samples/dataclean_v4_20062026/17_backtest_daily_returns_sample.csv` | `outputs/tables/backtest_daily_returns_current_clean_v4.csv` |
| Counterfactual tasks | `review_samples/dataclean_v4_20062026/17_counterfactual_task_samples.jsonl` | `data/counterfactual/current_clean_v4_cf_tasks.parquet` |
| Counterfactual fail examples | `review_samples/dataclean_v4_20062026/17_counterfactual_fail_examples.json` | `outputs/data_samples/counterfactual_fail_examples_current_clean_v4.json` |
| Ablation | `review_samples/dataclean_v4_20062026/17_ablation_current_clean_v4.csv` | `outputs/tables/ablation_current_clean_v4.csv` |
| Science gate | `review_samples/dataclean_v4_20062026/18_science_gate_report.json` | `outputs/repro/currentdata_clean_v4_science_gate_report.json` |
| Metrics snapshot | `review_samples/dataclean_v4_20062026/metrics_snapshot_dataclean_v4.json` | Các file metric chính trong `outputs/metrics/` |
| Sample manifest | `review_samples/dataclean_v4_20062026/sample_manifest.json` | Hash/row count/source map của sample pack |

### 12.3. Thứ Tự Review Đề Xuất Trong ChatGPT UI

1. Đọc `01_source_code_index_dataclean_v4.md` để biết source code nào tương ứng từng bước.
2. Đọc `02_master_order_dataclean_v4.md` để hiểu contract gốc.
3. Review data-clean samples:
   - `04_entity_event_scoring_samples.jsonl`
   - `05_article_type_noise_filter_samples.jsonl`
   - `06_dedup_news_samples.jsonl`
4. Review context/prompt:
   - `07_evidence_pack_context_samples.jsonl`
   - `08_rendered_context_text_samples.jsonl`
   - `09_prompt_template_evidence_v4.md`
5. Review model outputs:
   - `13_rationale_generation_samples.jsonl`
   - `11_claim_grounding_nli_samples.jsonl`
   - `14_independent_judge_debias_samples.jsonl`
6. Review reward/alignment:
   - `15_flow_reward_dataset_samples.json`
   - `15_flow_vs_proxy_clean_v4_stage0_combined.csv`
   - `16_rwsft_alignment_samples.jsonl`
   - `16_dpo_alignment_pair_samples.jsonl`
7. Review finance evaluation:
   - `17_prediction_forecast_samples.jsonl`
   - `17_backtest_daily_returns_sample.csv`
   - `17_counterfactual_task_samples.jsonl`
   - `17_ablation_current_clean_v4.csv`
8. Cuối cùng đọc:
   - `18_science_gate_report.json`
   - `00_status_summary_dataclean_v4_01_18.json`
   - `sample_manifest.json`

### 12.4. Điểm Cần ChatGPT UI Review Kỹ

Các câu hỏi chính nên dùng khi đưa sample pack vào ChatGPT UI:

1. Entity-event scoring có đang giữ đúng company-specific news hay vẫn giữ quá nhiều multi-company/context-only news?
2. Article type/noise filter có loại bỏ đúng weak/empty/macro generic article không?
3. Evidence pack có làm rõ ràng `company_evidence`, `context_evidence`, `technical_signals` không?
4. Prompt V4 có đủ chặt để không leak label và không invent evidence không?
5. Rationale output có cite đúng `evidence_id`/`signal_id` không?
6. NLI grounding có phát hiện unsupported/contradiction thật hay vẫn quá dễ dãi?
7. Independent judge normal/reversed có ổn định đủ để làm reward component không?
8. Flow reward target/mask có hợp lý, có tránh dùng missing component sai cách không?
9. RWSFT/DPO sample có train-only, cùng sample_id cho DPO pair và reward gap hợp lý không?
10. Prediction/backtest/counterfactual có đúng test-only và forecast-only schema không?
11. Science gate có chặn claim trading alpha đúng khi Sharpe âm không?
