# MoTa: FIRE-Fin AAAI Upgrade

## 1. Cách hiểu ngắn gọn

Paper này giải quyết một bài toán rất khó trong tài chính: khi có tin tức và tín hiệu kỹ thuật của một cổ phiếu, làm sao để LLM không chỉ đưa ra dự báo, mà còn đưa ra lý do có thể kiểm chứng được.

Thay vì hỏi model trực tiếp "giá sẽ tăng hay giảm", pipeline tách bài toán thành các bước có gate:

1. Khóa dữ liệu theo thời gian để tránh leak từ tương lai.
2. Tạo label tài chính bằng abnormal return 5 lớp.
3. Chuyển chỉ báo kỹ thuật thành event tokens để LLM đọc như bằng chứng có cấu trúc.
4. Yêu cầu local LLM sinh rationale nhưng không được thấy label thật.
5. Dùng judge và grounding để chấm xem rationale có suy ra được label, có bám vào news/technical evidence, và có tránh mâu thuẫn hay không.
6. Học một Flow Reward v2 từ các điểm judge/grounding để thay thế việc gọi judge đắt tiền cho mỗi lần selection.
7. Tạo RWSFT và DPO dataset từ những rationale được chấm tốt.
8. Fine-tune student/generator và đánh giá trên validation/test bằng counterfactual metrics và daily portfolio simulator.

Nói ngắn gọn: đây là framework biến "giải thích tài chính" từ một output văn bản khó đo lường thành một đối tượng có thể audit, score, align và backtest.

## 2. Ví dụ minh họa để dễ review

Giả sử có một mẫu dữ liệu:

- Date: 2021-08-05
- Ticker: AAPL
- News: "Apple reports stronger iPhone demand and raises guidance."
- Technical tokens:
  - `MOMENTUM_POSITIVE`
  - `VOLUME_Z_HIGH`
  - `RSI_NEUTRAL`
- Label ẩn sau, model generator không được thấy: `Strong Positive`, tính từ abnormal return h1 sau tin.

Generator Qwen3-4B-Instruct-2507 chỉ thấy tin tức, technical tokens, horizon và yêu cầu JSON strict. Nó sinh rationale kiểu:

```json
{
  "forecast": "Positive",
  "probability": 0.64,
  "reasoning": "Guidance raise and strong demand imply stronger near-term revenue expectations. Positive momentum and high volume support the market reaction, while RSI is not yet overbought.",
  "risks": ["Guidance may already be priced in", "Market-wide selloff could dominate firm news"]
}
```

Sau đó judge không được thấy label gốc, chỉ đọc rationale và phải đoán lại phân phối label. Nếu judge có thể suy ra `Positive/Strong Positive`, rationale có inferability cao. Claim grounding tiếp tục kiểm tra:

- Claim về news có được entail bởi news không.
- Claim về technical có đúng với event tokens không.
- Claim về risk có hợp lý hay bị hallucinate không.

Flow Reward v2 học cách dự đoán điểm chất lượng rationale từ nhiều nguồn: inferability, grounding, schema quality, technical evidence, regime/volatility. Sau khi học được reward, pipeline có thể chọn rationale tốt nhất cho RWSFT và tạo cặp chosen/rejected cho DPO.

## 3. Contribution dự kiến

### Contribution 1: Train-only rationale alignment cho tài chính, có leakage guard

Pipeline không generate rationale trên validation/test cho alignment. Train split được dùng cho generation, reward và alignment; validation chỉ dùng để selection; test chỉ dùng final evaluation. Split là chronological:

- Train: đến hết 2021-12-31.
- Validation: năm 2022.
- Test: từ 2023 trở đi.

Điều này phòng thủ câu hỏi reviewer: model có học vẹt từ tương lai hoặc test label không?

### Contribution 2: Strict rationale schema, không auto-fix

Invalid JSON vẫn là invalid. Pipeline không sửa output rồi tính như hợp lệ. Parse-ok rate và schema-ok rate là metric thật. Điểm này quan trọng vì nếu parser tự sửa lỗi, paper sẽ khó bảo vệ tính reproducible.

### Contribution 3: Technical event tokens v2

Thay vì đưa raw indicator vào prompt, pipeline chuyển RSI, MACD, rolling return, Bollinger, volume z-score thành event tokens có field:

- `token`
- `value`
- `direction_prior`
- `strength`
- `rule`

Đây là cách nối domain finance với LLM bằng ngôn ngữ có cấu trúc. Reviewer có thể inspect được tại sao một token được tạo.

### Contribution 4: Multi-judge inferability và claim-level grounding

Đánh giá rationale không dựa vào một điểm cảm tính. Pipeline tách thành các câu hỏi:

- Từ rationale có suy ra được label không?
- Nếu đổi thứ tự label trong prompt judge, điểm có ổn định không?
- Claim nào dựa trên news?
- Claim nào dựa trên technical token?
- Claim nào là regime/forecast/risk?
- Claim nào không có bằng chứng?

Parse failure không default về neutral. Missing model phải ghi `missing`, không được PASS giả.

### Contribution 5: Flow Reward v2 thay vì proxy average thuần túy

Proxy average là baseline: lấy trung bình nhiều điểm judge/grounding. Flow Reward v2 là model học reward có điều kiện, có mask cho missing components và có input regime/volatility/model id.

Claim paper chỉ được phép nói Flow Reward v2 tốt hơn khi thắng proxy average trên ít nhất 2 metric định trước. Đây là cơ chế tự kiểm soát để tránh overclaim.

### Contribution 6: Finance-valid evaluation

Paper không dùng per-news backtest. Final evaluation phải dùng daily portfolio simulator:

- Một position trên mỗi ticker-date.
- Test split only.
- Có transaction cost.
- Có slippage/borrow optional.
- Sharpe tính trên daily returns.

Điều này giúp gắn kết NLP rationale với metric tài chính có ý nghĩa hơn.

## 4. Tại sao có khả năng competitive cho AAAI-style review

Đây không phải báo cáo engineering đơn thuần. Nếu happy path thành công, paper có 3 điểm có thể thuyết phục reviewer:

1. Bài toán có độ khó cao: tài chính là môi trường noisy, non-stationary, dễ leak, và khó đánh giá explanation.
2. Phương pháp có tính tổng quát: train-only rationale generation, strict schema, judge debiasing, claim grounding, flow reward và finance-valid evaluation có thể áp dụng cho domain noisy khác.
3. Evidence được gate hóa: mỗi bước có status JSON, manifest, hash, row count và failure rõ ràng. Paper không dựa vào dummy table hay cherry-pick artifact.

Cần nói cẩn trọng: không có gì đảm bảo accept. Khả năng accept chỉ mạnh nếu final result thực sự pass các gate sau:

- Flow Reward v2 thắng proxy average trên metric đã định trước.
- RWSFT/DPO dataset đủ số lượng và train-only.
- Aligned model tốt hơn baseline trên validation/test.
- Daily portfolio backtest tốt hơn baseline sau transaction cost.
- Baseline và ablation không dummy, có seed/statistics.
- Final repro gate không còn hard-coded path, contamination, per-news backtest hay missing status.

Nếu một trong các điều trên fail, paper vẫn có thể thành workshop/negative-result/engineering reproducibility story, nhưng chưa nên claim main AAAI-level contribution.

## 5. Mô hình và vai trò hiện tại

Quyết định model hiện tại:

- Generator chính: `Qwen3-4B-Instruct-2507`.
- Không dùng fallback student generator không có trong local cache.
- Không dùng `DeepSeek-R1-Distill-Qwen-1.5B` làm generator chính.
- Judge/teacher phụ: FinGPT forecaster, Llama-3-8B-Instruct nếu local load được.
- Grounding: `cross-encoder nli-deberta-v3-small`.

Lý do chọn Qwen3-4B-Instruct-2507:

- Có chat template và instruction-following tốt.
- Local safetensors có sẵn trong cache.
- Chạy được trên RTX 3090 24GB với batch có kiểm soát.
- Smoke stage-0 đã cho parse-ok tốt và output ngắn hơn.

## 6. Happy path thực nghiệm nếu thành công

### Step 01: Repo audit và safe branch

Mục tiêu:

- Tạo branch riêng cho upgrade.
- Không reset/revert thay đổi đang có.
- Audit source/config/prompts/outputs.
- Flag các risk: hard-coded path, dummy table, parser auto-fix, PASS thiếu output, neutral fallback, per-news backtest.

Output chính:

- `outputs/status/01_REPO_AUDIT_AND_SAFE_BRANCH.status.json`
- Manifest hash các file quan trọng.

Gate:

- Chỉ sang step sau nếu status PASS và `next_step_allowed=true`.

### Step 02: Config paths và reproducible environment

Mục tiêu:

- Tất cả path đi qua config/env, không hard-code path máy cá nhân.
- Dùng local cache `/mnt/e/huggingface` hoặc `E:/huggingface`.
- Không cài thư viện nếu môi trường sẵn có đủ.

Output chính:

- `configs/default_paths.yaml`
- `configs/local_paths.template.yaml`
- `src/utils/config.py`
- `outputs/status/02_CONFIG_PATHS_AND_REPRO_ENV.status.json`

Gate:

- Step02 PASS cho Step09-14 nếu các thư viện inference/data có đủ.
- Thiếu `peft/trl/bitsandbytes` chỉ chặn Step15 training, không chặn rationale generation.

### Step 03: Data scale và locked split

Mục tiêu:

- Build manifest FNSPID từ local cache.
- Dùng chronological split, không random split.

Split policy:

- Train: <= 2021-12-31.
- Val: 2022.
- Test: >= 2023.

Output chính:

- `data/processed/split_membership.parquet`
- `outputs/status/03_DATA_SCALE_AND_SPLIT_LOCK.status.json`

### Step 04: Leakage guards

Mục tiêu:

- Unit test bắt synthetic leak.
- Kiểm tra prompt, split, alignment outputs.
- Legacy artifact chỉ là audit risk nếu không thuộc current v2 gate.

Output chính:

- Leakage reports.
- `outputs/status/04_LEAKAGE_GUARDS_AND_UNIT_TESTS.status.json`

### Step 05: Abnormal-return labels

Mục tiêu:

- Tạo label 5 lớp từ abnormal return.
- Report class balance theo split/year/ticker/regime.

Ví dụ label:

- Strong Negative
- Negative
- Neutral
- Positive
- Strong Positive

Output chính:

- `data/labels/labels_h1_abnormal.parquet`
- Class balance reports.

### Step 06-07: Technical features và event tokens

Mục tiêu:

- Refactor technical indicators thành pure functions.
- Validate RSI, MACD, rolling return, Bollinger, volume z-score.
- Tạo technical event tokens có rule traceable.

Output chính:

- `data/indicators/technical_event_tokens_h1_v2.parquet`
- No-token rate report.

### Step 08: Strict rationale schema

Mục tiêu:

- Raw output lưu riêng.
- Parsed/error lưu riêng.
- Không overwrite probability theo action.
- Không dùng repaired output cho main metrics.

Output chính:

- Prompt strict JSON.
- Parser strict.
- Unit tests schema.

### Step 09: Train-only rationale generation

Mục tiêu:

- Generate rationale chỉ trên train split.
- Stage hóa theo scale:
  - `stage_0_sanity_check`
  - `stage_1_small_scale`
  - `stage_2_medium_scale`
  - `stage_3_full_scale`

Happy path stage-1:

- Bulk branch: 5,000 train samples x 1 candidate.
- Conflict branch: 2,500 hard train samples x 3 candidates.
- Combined output: 12,500 candidate rationales.

Required gates:

- All rows train-only.
- Raw/parsed row counts match.
- Parse-ok >= 0.95.
- Raw schema-ok >= 0.85.
- Avg output tokens <= 280.
- No label leak.

Output chính:

- `data/rationales/raw/train_qwen3_4b_stage1_bulk.jsonl`
- `data/rationales/raw/train_qwen3_4b_stage1_conflict_3cand.jsonl`
- `data/rationales/parsed/train_candidates_stage1_strict.parquet`
- `outputs/status/09_RATIONALE_GENERATION_SCALEUP_TRAIN_ONLY.status.json`

### Step 10: Multi-judge inferability

Mục tiêu:

- Judge đọc rationale và đoán label distribution.
- Prompt JSON-only deterministic.
- Có normal/reversed label order để giảm bias vị trí.
- Parse failure là `parse_ok=false`, không fallback neutral.

Output chính:

- `data/judges/inferability_multi_judge_stage1.parquet`
- `outputs/status/10_PROXY_JUDGES_MULTI_MODEL_DEBIASED.status.json`

### Step 11: Claim-level grounding

Mục tiêu:

- Tách claim theo news, technical, regime, forecast, risk.
- News claim dùng NLI entailment.
- Technical claim đối chiếu event tokens.
- No-token context là `not_applicable`, không tính là perfect grounding.

Output chính:

- `data/judges/claim_grounding_scores_stage1.parquet`
- `outputs/status/11_CLAIM_LEVEL_GROUNDING_JUDGES.status.json`

### Step 12: Flow Reward v2

Mục tiêu:

- Build dataset multi-target có mask missing components.
- Conditioning gồm embedding, technical vector, regime/volatility, model id.
- Train smoke trước, chỉ scale khi loss giảm.

Output chính:

- `data/reward/flow_v2_train_dataset_stage1.pt`
- `checkpoints/flow_reward_v2_stage1`
- `outputs/status/12_FLOW_REWARD_MULTITARGET_V2.status.json`

### Step 13: Flow vs proxy validation

Mục tiêu:

- So sánh Flow Reward v2 với proxy average, single-best judge, flow v1 nếu có.
- Không claim improvement nếu Flow Reward v2 không thắng proxy average trên ít nhất 2 metric đã định trước.

Output chính:

- `outputs/metrics/flow_vs_proxy_eval_stage1.json`
- `outputs/status/13_FLOW_REWARD_EVAL_VS_PROXY.status.json`

### Step 14: RWSFT/DPO alignment datasets

Mục tiêu:

- RWSFT chọn best candidate per sample bằng flow/proxy score.
- DPO tạo chosen/rejected cùng `sample_id`, chosen score > rejected score, có min reward gap.
- Train-only tuyệt đối.

MVP gate:

- RWSFT >= 5,000 examples.
- DPO >= 2,000 pairs.

Output chính:

- `data/alignment/rwsft_train_v2.jsonl`
- `data/alignment/dpo_pairs_train_v2.jsonl`
- `outputs/status/14_RWSFT_DPO_DATASET_REBUILD.status.json`

### Step 15: Alignment training

Mục tiêu:

- QLoRA 4-bit trên RTX 3090.
- Smoke `--max-steps 10` trước full run.
- Missing/OOM là FAIL, không deferred PASS.

Rủi ro/chẩn bị môi trường:

- `peft`, `trl`, `bitsandbytes` là các dependency bắt buộc cho QLoRA/DPO.
- `D:/LOBProj/LOBExp/.venv` có Torch CUDA/Transformers nhưng không có các package này.
- `C:/Python/Python311` có `peft`, `trl`, `bitsandbytes`; pipeline smoke hiện tại dùng venv LOBExp và append `C:/Python/Python311/Lib/site-packages` sau venv path để tái sử dụng package sẵn có, không cài mới.

### Step 16-17: Finance-valid evaluation

Mục tiêu:

- Daily portfolio simulator trên test split only.
- Một position mỗi ticker-date.
- Có transaction cost.
- Counterfactual eval deterministic, directional metrics.

Output chính:

- Daily returns.
- Sharpe/turnover/drawdown.
- Counterfactual directional metrics.

### Step 18-19: Baselines, ablations, statistics

Mục tiêu:

- Baseline B1-B4 với seeds `[11, 22, 33]`.
- GPU-heavy LLM baseline có thể một seed cho MVP, nhưng phải ghi `multi_seed_missing=true`.
- Ablation A0-A8 không được dummy.
- Prediction dùng paired bootstrap.
- Daily returns dùng block bootstrap.

### Step 20-21: Paper tables và final repro gate

Mục tiêu:

- Paper tables chỉ từ metric thật, không zero-fill.
- Table manifest map source file/run_id/seed/split/timestamp.
- Final package không include raw dataset/model weights lớn.

Final gate fail nếu còn:

- Dummy table.
- Hard-coded path.
- Test contamination.
- Per-news backtest.
- Thiếu Flow Reward v2/counterfactual eval.
- Baseline < 3.
- Status missing/failed.

## 7. Hiện trạng thực nghiệm hiện tại

Tại thời điểm cập nhật file này:

- Step02 config/repro env: PASS.
- Step04 leakage guards: PASS.
- Stage-0 Qwen3 fast pipeline: đã pass trước đó.
- Step09 stage-1 bulk: PASS.
  - Row count: 5,000.
  - Unique sample count: 5,000.
  - Parse-ok rate: 0.9958.
  - Schema-ok rate: 0.979.
  - Avg output tokens estimate: 215.1806.
  - Run id: `stage_1_small_scale_a70d8930`.
- Stage-1 train conflict subset: PASS.
  - 2,500 train-only hard/conflict samples đã được tạo.
- Stage-1 conflict generation: PASS.
  - Target: 2,500 samples x 3 candidates = 7,500 raw rows.
  - Row count: 7,500.
  - Unique sample count: 2,500.
  - Parse-ok rate: 0.9957.
  - Schema-ok rate: 0.9865.
  - Avg output tokens estimate: 222.3337.
  - Run id: `stage_1_small_scale_5bb2222f`.
- Step09 stage-1 combined: PASS.
  - Combined row count: 12,500.
  - Unique sample count: 7,336.
  - All rows are `train`.
  - Parse-ok rate: 0.99576.
  - Schema-ok rate: 0.98352.
  - Duplicate key count: 0.
  - Explicit leak pattern count: 0.

Trạng thái downstream:

- Step10 multi-judge inferability trên stage-1: PASS.
  - Judge rows: 100,000.
  - Parse-ok rate: 0.99576.
  - Label-order consistency: 1.0.
- Step11 claim-level grounding trên stage-1: PASS.
  - Claim rows: 85,084.
  - Claim types gồm technical, news, regime, forecast, risk.
- Step12 Flow Reward v2 smoke train: PASS.
  - Dataset rows: 12,500.
  - Target dim: 7.
  - Cond dim: 93.
  - Train loss giảm: 2.3424 -> 1.5847 -> 1.0103.
- Step13 flow-vs-proxy validation smoke: PASS.
  - `claim_improvement=false` vì metrics thật còn pending, đúng contract không overclaim.
- Step14 RWSFT/DPO rebuild: PASS.
  - RWSFT examples: 7,336.
  - DPO pairs: 3,362.
  - DPO unique samples: 1,565.
  - Alignment datasets đều train-only.
- Step15 RWSFT + DPO QLoRA smoke: PASS.
  - Dùng `D:/LOBProj/LOBExp/.venv/Scripts/python.exe`.
  - Append `C:/Python/Python311/Lib/site-packages` để tái sử dụng `peft`, `trl`, `bitsandbytes`.
  - Model: `E:/huggingface/models/Qwen3-4B-Instruct-2507`.
  - Max steps: 10.
  - Batch size: 1.
  - Max seq length: 768.
  - RWSFT adapter: `checkpoints/aligned/qwen3_4b/rwsft_v2/adapter_model.safetensors`.
  - DPO adapter: `checkpoints/aligned/qwen3_4b/dpo_v2/adapter_model.safetensors`.
  - RWSFT max memory allocated: about 6.0GB.
  - DPO max memory allocated: about 6.3GB.
- Prediction smoke for Step16: PASS.
  - Prompt/interface đã đổi sang forecast-only JSON, không dùng rationale schema.
  - Rows: 200 test-only.
  - Parse-ok rate: 1.0.
  - Schema-ok rate: 0.98.
  - Action distribution: hold 107, short 63, long 26, invalid 4.
  - Checkpoint: `checkpoints/aligned/qwen3_4b/dpo_v2`, không dùng fallback.
  - Attention backend: `flash_attention_2`.
- Step16 daily portfolio backtest smoke: PASS.
  - Input: `outputs/predictions/test_predictions_qwen3_dpo_flash_smoke.parquet`.
  - Test split only, schema-ok prediction rows only.
  - Trading days in smoke subset: 1.
  - Total turnover: 7.5158.
  - Mean daily return after transaction cost: -0.00768.
  - Đây là smoke finance-valid, chưa phải performance claim.
- Step17 counterfactual directional eval smoke: PASS.
  - Tasks: 200 test-only counterfactual tasks.
  - LLM forecast generations: 400.
  - Parse-ok rate: 1.0.
  - Schema-ok rate: 0.9675.
  - Directional pass rate: 0.165.
  - No-change rate: 0.67.
  - Evaluator: DPO adapter + forecast-only parser, không phải heuristic-only.

Ý nghĩa hiện trạng:

- Stage-1 data/reward/alignment MVP đã được unblock đến Step17 smoke.
- Không nên claim Flow Reward v2 improvement hoặc finance performance cho đến khi Step16-19 full-scale, baselines, ablations và statistics hoàn tất.
- Bước kế tiếp hợp lý là mở Step18 baselines/statistics ở quy mô smoke trước, hoặc tăng Step16 prediction/backtest lên nhiều trading days trước khi claim tài chính.

## 8. Điều cần review bằng domain knowledge

Nhờ domain knowledge tài chính, cần review các điểm sau:

1. Label abnormal return 5 lớp có hợp lý với horizon h1 không.
2. Threshold strong/weak positive/negative có quá nhạy với volatility regime không.
3. Technical event tokens có phản ánh đúng logic trading không.
4. Conflict subset rule có tạo hard case thật không hay chỉ tạo noise.
5. Prompt rationale có ép model nói đúng evidence không, hay model vẫn có thể viết chung chung.
6. Judge inferability có chấm đúng explanation useful hay chỉ chấm wording hợp lý.
7. Flow Reward v2 có đang học quality signal hay học artifact của prompt/schema.
8. Backtest daily portfolio có dùng assumption thực tế về execution, cost và holding horizon không.
9. Baseline nào bắt buộc phải có để reviewer không nói thiếu đối chiếu.
10. Ablation nào có giá trị paper cao nhất nếu GPU/time có hạn.

## 9. Storyline để viết paper

Một storyline mạnh có thể là:

> Financial LLMs often produce plausible explanations, but plausible text is not enough for noisy markets. We introduce a gated, leakage-safe rationale alignment pipeline where explanations are generated without seeing labels, audited through inferability and claim grounding, distilled into a flow-based reward model, and finally evaluated by finance-valid counterfactual and portfolio metrics.

Tiếng Việt:

> LLM tài chính không nên được đánh giá bằng việc văn bản nghe có vẻ hợp lý. Paper này đề xuất một pipeline biến explanation thành artifact có thể kiểm chứng: không leak label, có strict schema, có judge suy ngược label, có grounding theo claim, có flow reward học từ nhiều nguồn điểm, và có backtest đúng theo ngày trên test split.

## 10. Ranh giới claim

Claim được phép nếu evidence PASS:

- "Pipeline generates train-only rationales with high parse/schema validity."
- "Claim-level grounding reduces unverified rationale evidence."
- "Flow Reward v2 can approximate or improve over proxy aggregation under validation metrics."
- "Alignment dataset construction is leakage-safe and score-based."
- "Aligned model improves finance-valid evaluation on test split."

Claim chưa được phép nếu chưa có evidence:

- "Model beats all baselines."
- "Flow Reward v2 universally better than proxy average."
- "Rationales improve real trading performance."
- "Framework is AAAI-ready."
- "Ablations support every design choice."

Những claim này chỉ nên đưa vào paper sau khi Step14-21 PASS và final tables không dummy.

## 11. Cập nhật hiện trạng sau medium finance evaluation

Sau vòng scale mới, pipeline không còn dừng ở smoke 1 trading day:

- Prediction medium: PASS.
  - Artifact: `outputs/predictions/test_predictions_qwen3_dpo_flash_medium.parquet`.
  - Rows: 5,000 test-only rows.
  - Trading days: 33, từ 2023-01-03 đến 2023-02-17.
  - Parse-ok rate: 0.9866.
  - Schema-ok rate: 0.9798.
  - Action distribution: hold 2442, long 1901, short 556, invalid 101.
  - Checkpoint: `checkpoints/aligned/qwen3_4b/dpo_v2`, không dùng fallback.

- Step16 medium daily portfolio: PASS về mặt cơ chế.
  - Trading days: 33.
  - Nonzero daily return rate: 1.0.
  - Total turnover: 227.1351.
  - Mean daily return: -0.002809.
  - Annualized daily Sharpe: -10.9598.
  - Kết quả này xác nhận simulator và prediction artifact đủ ngày để thống kê, nhưng hiệu năng âm nên chưa được dùng làm performance claim.

- Step17 medium counterfactual: PASS.
  - Tasks: 500 test-only counterfactual tasks.
  - Generations: 1,000.
  - Parse-ok rate: 0.999.
  - Schema-ok rate: 0.969.
  - Directional pass rate: 0.16.
  - Wrong-direction rate: 0.144.
  - No-change rate: 0.696.
  - Ý nghĩa: mô hình phản ứng có hướng trong một phần case, nhưng no-change còn chiếm đa số nên cần phân tích thêm trước khi claim robustness.

- Step18 baseline minimum: PASS.
  - B1-B4 đã chạy đủ seeds `[11,22,33]`.
  - Baseline prediction-level artifact: `outputs/predictions/baseline_suite_predictions.parquet`.
  - Prediction rows: 95,574.
  - B5-B12 vẫn `NOT_RUN`, `multi_seed_missing=true`; không được dùng làm bằng chứng paper.

- Step19 statistics: PASS.
  - Prediction paired bootstrap comparisons: 12.
  - Backtest block bootstrap comparisons: 12.
  - Bootstrap samples: 500.
  - Block length: 5 trading days.
  - A0-A8 ablations vẫn `NOT_RUN` và được ghi rõ `not_used_as_evidence=true`.

Ranh giới claim mới:

- Có thể nói pipeline đã có finance-valid medium-scale evaluation mechanics, baseline comparator theo sample-id, và statistical test artifacts thật.
- Chưa nên nói model beat baselines hoặc có trading alpha, vì Step16 medium Sharpe đang âm và nhiều bootstrap CI chưa ủng hộ cải thiện nhất quán.
- Bước tiếp theo hợp lý là phân tích nguyên nhân chất lượng dự báo âm hoặc mở rộng baseline/backtest đầy đủ trước Step20 paper tables.

## 12. Bảng kết quả chi tiết đã verify

Các bảng dưới đây lấy trực tiếp từ artifact đã PASS ở `outputs/metrics`, `outputs/tables`, `outputs/status`, và `outputs/repro`. Đây là trạng thái reproducibility gate hiện tại, không phải claim cuối cùng rằng mô hình đã có alpha giao dịch.

### 12.1 Prediction medium artifact

Nguồn: `outputs/metrics/test_predictions_qwen3_dpo_flash_medium.json`

| Metric | Value |
|---|---:|
| Rows | 5,000 |
| Split | test |
| Checkpoint | `checkpoints/aligned/qwen3_4b/dpo_v2` |
| Checkpoint source | primary |
| Fallback used | false |
| Parse-ok rate | 0.9866 |
| Schema-ok rate | 0.9798 |
| Parse-ok rows | 4,933 |
| Schema-ok rows | 4,899 |
| Selected trading days | 33 |
| Start date | 2023-01-03 |
| End date | 2023-02-17 |
| Decoding | temperature 0.0, `do_sample=false` |
| Attention backend | `flash_attention_2` |

| Action | Count | Share |
|---|---:|---:|
| hold | 2,442 | 0.4884 |
| long | 1,901 | 0.3802 |
| short | 556 | 0.1112 |
| invalid | 101 | 0.0202 |

### 12.2 Daily portfolio backtest medium

Nguồn: `outputs/metrics/backtest_daily_portfolio_v2.json`

| Metric | Value |
|---|---:|
| Trading days | 33 |
| Annualized daily Sharpe | -10.9598 |
| Mean daily return | -0.002809 |
| Nonzero daily return rate | 1.0000 |
| Coverage | 1.0000 |
| Average positions per day | 7.6120 |
| Average turnover per day | 6.8829 |
| Total turnover | 227.1351 |
| Transaction cost | 5 bps |
| Slippage | 0 bps |
| Short borrow | 0 bps |
| Minimum schema-ok rate required | 0.80 |
| Minimum trading days required | 20 |

Ý nghĩa: simulator đã đúng dạng daily portfolio và đủ số ngày để thống kê, nhưng Sharpe âm nên không dùng làm claim trading performance.

### 12.3 Counterfactual directional evaluation

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
| Evaluator | `llm_forecast_counterfactual` |
| Checkpoint | `checkpoints/aligned/qwen3_4b/dpo_v2` |
| Attention backend | `flash_attention_2` |

Ý nghĩa: mô hình có phản ứng đúng hướng ở một phần case, nhưng no-change còn rất cao. Đây là điểm cần phân tích thêm trước khi claim robustness.

### 12.4 Baseline aggregate B1-B4, 3 seeds

Nguồn: `outputs/tables/baseline_suite_aggregate.csv`

| Baseline | Seeds | Accuracy mean | Macro-F1 mean | Macro-F1 std | MCC mean | MCC std | Brier mean | Train rows min | Test rows min | Feature dim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1_FinBERT_LR | 3 | 0.2346 | 0.1332 | 0.0000 | 0.0141 | 0.0000 | 0.7991 | 10,300 | 5,929 | 3 |
| B2_Technical_LightGBM | 3 | 0.3889 | 0.2635 | 0.0141 | 0.0827 | 0.0167 | 0.7138 | 30,000 | 10,000 | 28 |
| B3_News_Technical_Late_Fusion | 3 | 0.3895 | 0.2534 | 0.0120 | 0.0814 | 0.0121 | 0.7082 | 10,300 | 5,929 | 31 |
| B4_DLinear | 3 | 0.4357 | 0.1963 | 0.0109 | 0.0570 | 0.0098 | 0.7247 | 30,000 | 10,000 | 28 |

Ghi chú: B5-B12 vẫn `NOT_RUN` và không được dùng làm evidence. Artifact prediction-level cho paired test là `outputs/predictions/baseline_suite_predictions.parquet` với 95,574 rows.

### 12.5 Flow reward v2 versus proxy, validation holdout

Nguồn: `outputs/metrics/flow_vs_proxy_eval_stage1.json`

| Method | Eval rows | Rank corr with realized utility | Calibration error | Top-k rationale quality | Preference pair accuracy | Variance by regime | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| proxy_average_reward | 2,480 | 0.5731 | 0.2294 | 0.3500 | 0.8594 | 0.000095 | RUN |
| single_best_judge_reward | 2,480 | 1.0000 | 0.0000 | 0.4490 | 1.0000 | 0.000023 | RUN |
| flow_reward_v1 | 2,480 | 0.7859 | 0.0697 | 0.3859 | 0.9375 | 0.000109 | RUN |
| flow_reward_v2 | 2,480 | 0.0058 | 0.1180 | 0.2087 | 0.5313 | 0.000062 | RUN |

| Gate item | Value |
|---|---:|
| Dataset rows | 12,500 |
| Eval rows | 2,480 |
| Holdout fraction | 0.20 |
| Integration steps | 12 |
| Claim improvement | false |

Ý nghĩa: Step13 hiện có metric thật, không còn `PENDING_REAL_EVAL`. Tuy nhiên Flow Reward v2 chưa thắng proxy average, nên không claim improvement.

### 12.6 Prediction paired bootstrap against baselines

Nguồn: `outputs/tables/prediction_bootstrap_comparisons.csv`

| Comparator | Paired rows mean | Paired rows range | Delta accuracy mean | Delta accuracy range | Delta macro-F1 mean | Delta macro-F1 range | Delta MCC mean | Delta MCC range |
|---|---:|---|---:|---|---:|---|---:|---|
| B1_FinBERT_LR | 635.0 | 635-635 | 0.0803 | 0.0803 to 0.0803 | 0.0368 | 0.0368 to 0.0368 | -0.0305 | -0.0305 to -0.0305 |
| B2_Technical_LightGBM | 1,077.7 | 1,048-1,101 | -0.0027 | -0.0203 to 0.0245 | -0.0950 | -0.1291 to -0.0745 | -0.0554 | -0.1013 to -0.0107 |
| B3_News_Technical_Late_Fusion | 635.0 | 635-635 | -0.0273 | -0.0551 to -0.0126 | -0.1000 | -0.1255 to -0.0791 | -0.0940 | -0.1306 to -0.0661 |
| B4_DLinear | 1,077.7 | 1,048-1,101 | -0.0985 | -0.1307 to -0.0812 | -0.0539 | -0.0662 to -0.0475 | -0.0891 | -0.1025 to -0.0722 |

Ý nghĩa: FIRE-Fin/Qwen3 DPO thắng B1 về accuracy và macro-F1 trong overlap nhỏ, nhưng thua hoặc không ổn định trước B2-B4. Không được claim “beat baselines”.

### 12.7 Daily-return block bootstrap against baselines

Nguồn: `outputs/tables/backtest_block_bootstrap_comparisons.csv`

| Comparator | Overlap rows mean | Paired days | Delta mean daily return mean | Delta mean daily return range | Delta Sharpe mean | Delta Sharpe range |
|---|---:|---:|---:|---|---:|---|
| B1_FinBERT_LR | 635.0 | 33 | 0.000123 | 0.000123 to 0.000123 | 1.5260 | 1.5260 to 1.5260 |
| B2_Technical_LightGBM | 1,077.7 | 33 | 0.000173 | -0.000481 to 0.000826 | -0.6154 | -3.7614 to 1.2851 |
| B3_News_Technical_Late_Fusion | 635.0 | 33 | 0.001855 | 0.001347 to 0.002384 | 4.8804 | 3.4922 to 5.7345 |
| B4_DLinear | 1,077.7 | 33 | -0.000722 | -0.002280 to 0.000423 | -3.7359 | -7.7072 to -0.6770 |

Ý nghĩa: daily-return bootstrap có kết quả lẫn lộn. FIRE-Fin tốt hơn B3 trên overlap này, nhưng kém B4 và chưa ổn định với B2. Do Step16 Sharpe tổng thể vẫn âm, bảng này chỉ nên dùng để debug và định hướng cải thiện.

### 12.8 Paper tables và reproducibility gate

Nguồn: `outputs/status/20_PAPER_TABLES_NO_DUMMY_GATES.status.json`, `outputs/repro/aaai_gate_report.json`, `outputs/status/21_FINAL_REPRO_PACKAGE_AND_AAAI_GATE.status.json`

| Artifact | Status | Rows or metric | Evidence rows | Note |
|---|---|---:|---:|---|
| `table_1_prediction_main.csv` | PASS | 273 rows | 273 | Prediction, baselines, paired bootstrap |
| `table_2_explanation_quality.csv` | PASS | 32 rows | 32 | Inferability, grounding, flow eval |
| `table_3_daily_portfolio_backtest.csv` | PASS | 181 rows | 181 | Daily backtest and block bootstrap |
| `table_4_counterfactual_directional.csv` | PASS | 13 rows | 13 | Counterfactual metrics |
| `table_5_ablation.csv` | PASS | 9 rows | 0 | A0-A8 `NOT_RUN`, registry only |
| `table_6_scale_and_compute.csv` | PASS | 38 rows | 38 | Dataset scale and environment |
| AAAI gate report | GO | 0 blockers | n/a | Warning: ablation registry only |
| Repro package | PASS | 253 files | n/a | Size 815,175 bytes |

Final status:

| Check | Result |
|---|---|
| Status 01-21 verification | PASS |
| Unit tests | 25 passed |
| Final gate decision | GO |
| Known warning | Ablations are `NOT_RUN` and not evidence |
| Main quality caveat | Trading Sharpe is negative; Flow v2 does not beat proxy |
