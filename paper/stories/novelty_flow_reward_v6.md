# Storytelling: Decision-Faithful Distributional Flow Reward (DFD-FlowReward-V6)

## Ý tưởng cốt lõi (Core Insight)
Trong quá trình phát triển V6 để đáp ứng tiêu chuẩn Strong-Accept của AAAI 2027, chúng tôi nhận thấy rằng việc sử dụng LLM như một judge sinh ra điểm số (scalar proxy) chứa nhiều rủi ro về bias (vị trí label) và thiếu độ tin cậy thực tiễn (financial utility). 

Thay vì "nhắm mắt" học theo proxy một chiều như các phương pháp trước đó (SEP, PEN, hoặc Policy paper cơ bản), chúng tôi định hình lại bài toán reward:
1. **Denoising Noisy Distributions**: Reward không phải là một điểm số hoàn hảo, mà là một phân phối 5 lớp đầy nhiễu. Chúng tôi ép Flow model học cách khử nhiễu bằng việc gán trọng số độ tin cậy (\judge_reliability_weight\) dựa trên disagreement và label-order KL.
2. **Decision-Utility Gate**: Flow model chỉ được phép công nhận là thành công (claim allowed) nếu nó vượt qua được bài kiểm tra khốc liệt về *utility thực tế* (Lợi nhuận ròng sau chi phí giao dịch - \aw_realized_utility\), chứ không chỉ là khớp với proxy.
3. **Evidence Faithfulness under Hard Events**: Lời giải thích (rationale) được chọn không chỉ cần nghe bùi tai (plausible) mà phải bám rễ sâu vào các bằng chứng \hard_events\.

## Đóng góp khoa học (Selling Points for AAAI)
- Lần đầu tiên đưa khái niệm "Decision-Faithful Distributional Reward" vào tài chính, kết nối mảng Explainable Finance với Decision-Focused Learning.
- Công khai việc biến các sai số (bias/noise) của LLM judge thành dữ liệu quan sát được và xử lý chúng, thay vì giấu giếm qua việc trung bình hóa (averaging).
- Gate đánh giá chặt chẽ, từ chối việc overfitting vào metric rỗng để bảo vệ tính toàn vẹn khoa học.
