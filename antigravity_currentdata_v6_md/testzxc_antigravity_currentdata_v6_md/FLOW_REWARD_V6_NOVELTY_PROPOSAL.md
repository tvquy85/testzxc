# Flow Reward V6 Novelty Proposal — Decision-Faithful Distributional Reward

> Scope: current-data V6 only; no SN2; no full FNSPID expansion. This note is a research-backed design proposal for `10_FLOW_REWARD_V6_DECISION_TARGETS.md` and `11_FLOW_EVAL_RAW_UTILITY_AND_PROXY.md`.

## 1. Bối cảnh hiện tại

Clean V4 Medium đã chạy được end-to-end nhưng chưa đủ làm claim khoa học:

- Flow V5 chưa thắng proxy average trên đủ metric.
- Judge signal chỉ vượt random rất sát, nên không thể coi scalar proxy là ground truth mạnh.
- Backtest sau chi phí có Sharpe âm, nên không được claim alpha.
- Counterfactual faithfulness chưa đạt ngưỡng strict.
- Technical_Rule vẫn là baseline macro-F1 tốt nhất.

Vì vậy novelty của V6 không nên chỉ là "thêm flow reward". Điểm bán mạnh hơn là biến Flow Reward thành bộ lọc phân phối cho reward nhiễu, có kiểm soát bias judge, bám evidence, và chỉ được claim khi nó cải thiện quyết định tài chính thật.

## 2. Định vị novelty

Tên đề xuất:

```text
Decision-Faithful Distributional Flow Reward for Explainable Financial Forecasting
```

Tên ngắn trong repo:

```text
DFD-FlowReward-V6
```

Một câu bán hàng:

```text
We align financial rationales using a debiased distributional flow reward that denoises LLM judge distributions while preserving evidence faithfulness and selecting explanations by downstream after-cost decision utility.
```

Ý nghĩa dễ hiểu:

- SEP tự phản tỉnh để sinh giải thích và dùng PPO, nhưng reward chủ yếu gắn với dự báo đúng/sai.
- PEN học alignment text-price nhưng explanation thiên về chọn text quan trọng, chưa phải free-form rationale có kiểm chứng faithfulness.
- Policy paper dùng rectified flow để học reward distribution từ proxy LLM cho explanation.
- V6 chuyển ý tưởng đó sang tài chính bằng cách điều kiện hóa reward trên hard-event evidence, technical tokens, judge uncertainty, grounding, và utility sau chi phí. Flow không được tự nhận thắng nếu không vượt proxy trên held-out validation và không cải thiện các metric quyết định.

## 3. Các contribution có khả năng mạnh nhất

### Contribution A — Debiased Judge Distribution as Noisy Reward Target

Step 08/09 không chỉ tạo một điểm proxy. Nó tạo phân phối 5 lớp đã kiểm tra label-order normal/reversed/stable-random:

```text
p_judge = [p_strong_down, p_mild_down, p_neutral, p_mild_up, p_strong_up]
```

Flow học phân phối này với reliability weight:

```text
w_reliability =
  schema_ok
  * argmax_consistency_ensemble
  * (1 - normalized_label_order_kl)
  * (1 - judge_disagreement_entropy_norm)
  * evidence_quality_weight
```

Nếu judge bị bias hoặc bất ổn, row vẫn được lưu nhưng weight giảm. Đây là cách biến hạn chế của LLM-as-a-judge thành biến quan sát được, thay vì âm thầm average.

### Contribution B — Decision-Utility-Aware Reward, không chỉ Plausibility Reward

Reward chính vẫn là distribution target 5 lớp để giữ đúng tinh thần Flow Matching Generated Rewards. Tuy nhiên Step 10 phải lưu auxiliary utility để Step 11 kiểm tra Flow có hữu ích cho quyết định tài chính thật hay không:

```text
expected_direction_score = dot(p_judge, [-2, -1, 0, 1, 2])
action = short|hold|long derived from forecast_distribution
raw_realized_utility = position(action) * abnormal_return_h1 - transaction_cost_proxy
technical_rule_delta = raw_realized_utility(candidate) - raw_realized_utility(technical_rule)
```

Điểm mới nằm ở claim gate: Flow chỉ được coi là contribution nếu prediction-quality reward cũng cải thiện utility/ranking trên validation, không phải chỉ khớp judge.

### Contribution C — Evidence-Faithful Reward via Grounding and Counterfactual Readiness

Flow V6 không thưởng cho rationale nghe hợp lý nhưng không bám evidence. Step 10 lưu riêng:

```text
news_grounding_score
technical_grounding_score
unsupported_news_claim_rate
negative_news_claim_supported_rate
evidence_usage_track
```

Step 11 đánh giá thêm `faithfulness_lift`: trong nhóm candidate cùng sample, candidate được Flow chọn phải có grounding tốt hơn proxy/top-prob baseline. Counterfactual Step 16 là downstream audit chính, nhưng Step 10/11 phải chuẩn bị các cột để truy vết vì sao một candidate được chọn.

### Contribution D — Regime/Track-Conditioned Distributional Reward

Tài chính không phải môi trường IID. V6 nên condition Flow trên:

```text
hard_event_track
volatility_regime
technical_token_direction_mix
news_evidence_direction
market_day_context
```

Ý tưởng: cùng một rationale có thể đáng tin hơn trong `hard_event_news` nhưng yếu trong `weak_or_context_only`. Flow được đánh giá by-track để tránh thắng trung bình nhưng fail đúng phần cần claim.

### Contribution E — Negative-Result-Aware Scientific Gate

Strong selling point cho AAAI không phải là ép mọi metric thắng. V6 phải có cơ chế chặn claim:

- Nếu Flow không thắng proxy trên >=2/3 core utility metrics: không claim Flow superiority.
- Nếu Technical_Rule vẫn thắng macro-F1/MCC: không claim aligned LLM tốt hơn technical baseline.
- Nếu Sharpe sau chi phí âm hoặc ngày giao dịch quá ít: không claim alpha.
- Nếu counterfactual no-change cao: không claim faithful reasoning.

Điểm này làm paper đáng tin hơn vì hệ thống phân biệt pipeline pass và scientific claim.

## 4. Cập nhật đề xuất cho Step 10

Step 10 nên giữ primary target dimension = 5:

```text
target_distribution = calibrated_debiased_judge_distribution
```

Nhưng dataset phải chứa thêm các field:

```text
target_names:
  - p_strong_down
  - p_mild_down
  - p_neutral
  - p_mild_up
  - p_strong_up

auxiliary:
  - true_label_probability_ensemble
  - judge_reliability_weight
  - label_order_kl_mean
  - judge_disagreement_entropy
  - news_grounding_score
  - technical_grounding_score
  - unsupported_news_claim_rate
  - evidence_quality_weight
  - abnormal_return_h1
  - raw_realized_utility
  - technical_rule_delta
  - hard_event_track
  - volatility_regime
  - source_split
```

Không dùng test rows cho training/reward/alignment. Utility chỉ được tính từ train/val theo locked split; test chỉ xuất hiện ở final evaluation.

## 5. Cập nhật đề xuất cho Step 11

Step 11 phải đánh giá Flow theo 4 nhóm metric:

### 5.1 Distributional fidelity

- `kl_to_calibrated_judge_distribution`
- `js_to_calibrated_judge_distribution`
- `brier_true_label_probability`
- `ece_true_label_probability`

### 5.2 Decision utility

- `rank_correlation_with_raw_realized_utility`
- `preference_pair_accuracy_by_raw_utility`
- `top_decile_raw_realized_utility`
- `technical_rule_delta_top_decile`

### 5.3 Faithfulness and evidence use

- `top_decile_news_grounding_score`
- `top_decile_technical_grounding_score`
- `unsupported_claim_rate_top_decile`
- `faithfulness_lift_vs_proxy`

### 5.4 Robustness by track/regime

- `metric_wins_by_track`
- `metric_wins_by_volatility_regime`
- `hard_event_news_win_rate`
- `weak_context_failure_rate`

Core claim gate:

```text
flow_claim_allowed =
  wins_at_least_2_of_3_core_utility_metrics
  and no_critical_track_regression
  and distributional_fidelity_not_worse_than_proxy
  and top_decile_unsupported_claim_rate_not_worse
```

Core utility metrics:

```text
1. rank_correlation_with_raw_realized_utility
2. preference_pair_accuracy_by_raw_utility
3. top_decile_raw_realized_utility
```

## 6. Tại sao hướng này có novelty hơn bản hiện tại

| Thành phần | Bản hiện tại | V6 đề xuất | Selling point |
|---|---|---|---|
| Reward target | Multi-target/utility phụ, Flow chưa thắng proxy | Primary 5-class debiased distribution + reliability weight | Bám Policy paper nhưng xử lý bias/noise judge |
| Utility | Có raw utility nhưng chưa đủ thắng | Utility là eval gate bắt buộc, có delta vs Technical_Rule | Decision-focused, không tối ưu metric rỗng |
| Faithfulness | Grounding và counterfactual là bước sau | Flow chọn candidate phải có grounding lift; traceable fields | XAI faithfulness được đưa vào reward audit |
| Finance validity | Backtest sau cùng | Raw utility và rule delta vào Step 11 validation | Tránh reward đẹp nhưng trading vô nghĩa |
| Claim logic | CLAIM_RESTRICTED sau medium | Claim gate ngay ở Step 11 | Negative-result-aware, review-friendly |

## 7. Ablation bắt buộc để chứng minh novelty

Step 18 nên có ít nhất các ablation sau để bảo vệ contribution Step 10/11:

```text
A0 Full DFD-FlowReward-V6
A1 No judge reliability weight
A2 No grounding auxiliary fields
A3 No raw utility evaluation gate
A4 No track/regime conditioning
A5 Proxy average reward only
A6 Single best judge only
A7 Technical rule score only
A8 Flow distribution target without utility/faithfulness audit
```

Flow claim chỉ mạnh nếu A0 thắng A5/A6 trên utility metrics và không làm grounding/counterfactual tệ hơn.

## 8. Rủi ro và cách chặn

- Nếu judge true-label probability vẫn gần 0.20, Flow có thể học nhiễu. Chặn bằng `judge_reliability_weight` và không train alignment nếu Step 09 FAIL.
- Nếu raw utility quá noisy, chỉ dùng làm evaluation/gate, không trộn vào primary distribution target.
- Nếu Technical_Rule vẫn thắng, chuyển claim thành "explanation faithfulness under hard-event evidence" thay vì "better forecaster".
- Nếu counterfactual no-change cao, không claim causal/faithful reasoning; chỉ claim pipeline mechanics.

## 9. Nguồn nghiên cứu đã dùng

- Flow Matching Generated Rewards for LLM Explanations, ICLR 2026: https://openreview.net/forum?id=zmZsWCGzUV
- Direct Preference Optimization, arXiv 2305.18290: https://arxiv.org/abs/2305.18290
- RewardBench, arXiv 2403.13787: https://arxiv.org/abs/2403.13787
- Decision-Focused Learning survey, JAIR/arXiv 2307.13565: https://arxiv.org/abs/2307.13565
- Faithful Model Explanation in NLP survey, arXiv 2209.11326: https://arxiv.org/html/2209.11326v4
- LLM-as-a-judge position bias study, arXiv 2406.07791: https://arxiv.org/html/2406.07791v7
- SEP, WWW 2024: https://arxiv.org/abs/2402.03659
- PEN, AAAI 2023: https://ojs.aaai.org/index.php/AAAI/article/view/25648
- FNSPID, arXiv 2402.06698: https://arxiv.org/abs/2402.06698
- BenchStock, OpenReview ICLR 2025 submission: https://openreview.net/forum?id=bsXxNkhvm6
- Lo, The Statistics of Sharpe Ratios, Financial Analysts Journal: https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios
