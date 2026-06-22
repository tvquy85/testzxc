# Story Note — Flow Reward V6 as Decision-Faithful Distributional Alignment

Ngày 21/06/2026, khi review Clean V4 Medium và thiết kế Antigravity Current-Data V6, điểm nghẽn khoa học rõ nhất là Flow Reward chưa thắng proxy average, trong khi Technical_Rule vẫn mạnh và backtest Sharpe âm. Vì vậy hướng paper không nên kể rằng "Flow tự động tốt hơn"; hướng mạnh hơn là một hệ thống claim-gated, decision-faithful reward.

Ý tưởng chính:

```text
FIRE-Fin V6 treats LLM judge feedback as a noisy distribution, not a scalar truth.
Rectified Flow learns to denoise this distribution under market context, evidence grounding, and judge reliability.
The reward is only scientifically useful if it improves held-out after-cost decision utility and does not degrade evidence faithfulness.
```

Selling point cho phần method:

- Ẩn future label khỏi generator, nhưng dùng judge distribution để đo inferability của rationale.
- Dùng label-order permutation và judge disagreement để estimate độ nhiễu của reward.
- Dùng Flow như bộ denoise phân phối reward thay vì scalar ranker.
- Không trộn raw return vào primary target để tránh overfit tài chính; chỉ dùng raw utility làm validation gate.
- Dùng Technical_Rule delta để buộc model phải vượt baseline domain knowledge, không chỉ vượt LLM yếu.
- Dùng grounding/counterfactual audit để chặn rationale nghe hợp lý nhưng không phản ứng với evidence.

Claim wording nếu thành công:

```text
We introduce a decision-faithful distributional reward for financial rationales, where a rectified-flow reward model denoises calibrated LLM judge distributions and is validated by after-cost portfolio utility and evidence-level faithfulness rather than explanation plausibility alone.
```

Claim wording nếu metric không thắng:

```text
The current-data study shows that distributional reward modeling can be audited under finance-valid gates; however, Flow superiority and trading alpha are restricted when the learned reward does not beat proxy averaging or technical baselines on held-out decision utility.
```

Điểm đáng nhớ cho paper:

- Đây là một paper về trustworthy alignment under noisy financial evidence, không phải chỉ một stock forecasting paper.
- Điểm mạnh AAAI là negative-result-aware methodology: pipeline pass không tự động biến thành claim.
- Nếu Flow V6 thắng, novelty là sự kết hợp hiếm giữa distributional reward modeling, decision-focused evaluation, and faithful financial explanation.
