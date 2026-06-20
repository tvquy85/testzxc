# AGENTS.md
1. Lập kế hoạch

Mọi plan phải dựa trên:

- Context trước đó trong trao đổi.
- Phân tích và kết quả đã có ở bước trước.

Khi đề xuất bước tiếp theo, phải dựa trên phân tích trước đó, không làm rời rạc hoặc cảm tính.

Luôn ưu tiên giải pháp tổng quát, có thể tái sử dụng và mở rộng. Không xử lý theo kiểu ad hoc nếu chưa có lý do kỹ thuật rõ ràng.

**QUAN TRỌNG:** Luôn ghi nhớ phải kiểm chứng (verify) chất lượng từng bước nhỏ hoàn chỉnh, chắc chắn thành công và đạt chuẩn thì mới qua bước mới. Không nhảy bước khi chưa xác nhận kết quả bước hiện tại.

2. Điều tra và tìm giải pháp

Khi gặp lỗi, kết quả yếu hoặc hiện tượng bất thường, phải xử lý với vai trò **chuyên gia ML hàng đầu**:

- Investigate kỹ nguyên nhân.
- Tìm và đối chiếu với nguồn uy tín.
- Ưu tiên paper, benchmark, tài liệu kỹ thuật chính thống hoặc tài liệu từ nguồn đáng tin cậy.
- Không kết luận vội nếu chưa đủ bằng chứng.

Giải pháp đưa ra phải có cơ sở kỹ thuật, có khả năng tổng quát hóa và phù hợp với mục tiêu paper.

3. Góc nhìn đánh giá

Luôn đánh giá từ vai trò:

- **Principal ML Scientist**: tập trung vào mô hình, dữ liệu, phương pháp, độ tin cậy và khả năng tổng quát.

4. Chạy thí nghiệm

Tận dụng GPU **RTX 3090**, nhưng phải kiểm soát thời gian và tài nguyên.

Với thí nghiệm mới hoặc có khả năng chạy lâu:

- Bắt đầu từ scale nhỏ.
- Mở rộng dần lên scale trung bình/lớn.
- Mỗi stage có `run_id` riêng.
- Mỗi stage có audit riêng.
- Chỉ sang stage tiếp theo khi pass gate đã định nghĩa.

Stage gợi ý:

```text
stage_0_sanity_check
stage_1_small_scale
stage_2_medium_scale
stage_3_full_scale
```

5. Thư viện và môi trường

Khi thiếu thư viện, kiểm tra theo thứ tự:

- môi trường hiện tại hoặc thư mục `C:\Python\Python311\`.
- `d:\LOBProj\LOBExp\`.
- Download từ Internet khi hai nguồn trên không có.

Không cài đặt tùy tiện nếu có thể tái sử dụng môi trường, thư viện hoặc cache sẵn có.

## 6. Local LLM và cache

Local LLM phải được:
Đường dẫn Local LLM: 'e:\huggingface\'
- Download một lần.
- Load từ cache local ở các lần sau.
- Tránh download hoặc load lại không cần thiết.

Trước khi tải model từ internet, luôn kiểm tra cache local/model path.

Bộ dữ liệu fnspid đã có trong 'e:\huggingface\datasets\'

## . Ghi nhận các ý tưởng để viết vào paper

Luôn ghi lại những ý tưởng đáng giá trong quá trình làm việc vào thư mục paper\stories để reference về sau viết vào paper như storytelling. Đặc biệt là những cải tiến về cách prompt và cách tính toán điểm. Cả những ý tưởng chưa được áp dụng. 