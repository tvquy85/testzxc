# Báo Cáo Phân Tích Kỹ Thuật: Nền Tảng AI-FIT LLM Services

Dựa trên tài liệu `LLMServices.md`, kiến trúc hệ thống của Open WebUI, và quá trình probing trực tiếp vào các API endpoints của hệ thống (ai-fit.hcmus.edu.vn), dưới đây là báo cáo phân tích chuyên sâu về các tính năng, giới hạn kỹ thuật và cách khai thác tối đa hệ thống này.

---

## 1. Kết Quả Probing Cấp Độ Hệ Thống (API Level)

Trong quá trình điều tra, tôi đã tiến hành gửi trực tiếp các HTTP requests (mô phỏng chuẩn OpenAI API) để xác minh mức độ mở của hệ thống. 

**Kết quả thí nghiệm:**
- `GET /models`: **✅ Thành công.** Hệ thống public hai phiên bản: `Qwen3.6-27B` (cho context lớn, tối đa 256K) và `Qwen3.6-27B-CTX-64K` (bản tối ưu cho truy vấn ngắn gọn hơn).
- `POST /chat/completions`: **✅ Thành công.** Hỗ trợ tinh chỉnh `temperature`, `max_tokens` (giới hạn 10000), `system prompt`.
- `POST /embeddings`: ❌ Bị chặn (`403 Forbidden - Direct API passthrough is disabled`).
- `GET /files` & `POST /files`: ❌ Bị chặn.
- `POST /images/generations`: ❌ Bị chặn.
- `Vision Chat (Multimodal)`: ❌ Trả về lỗi 500 do backend (LiteLLM) không thể truy xuất hình ảnh qua URL hoặc model không được cấu hình cho Multimodal qua chuẩn API.

> [!IMPORTANT]
> **Đánh giá Kỹ thuật:** Hệ thống backend chặn tính năng `ENABLE_OPENAI_API_PASSTHROUGH` đối với các API nâng cao (files, embeddings, images). Điều này có nghĩa là **các tính năng RAG, Text-to-Image, và Voice chỉ hoạt động thông qua WebUI Front-end**, và không được cấp phép để tự động hóa (programmatic access) qua API chuẩn của OpenAI (dùng thư viện `openai-python`).

---

## 2. Chiến Lược Khai Thác Triệt Để Các Tính Năng (UI vs. API)

Với vai trò là **Principal ML Scientist**, để xây dựng một pipeline bền vững với nguồn tài nguyên này, chúng ta cần phân chia chiến lược sử dụng rõ ràng:

### A. Khai thác qua API (Dành cho Lập trình & Tự động hóa)
Vì hệ thống chỉ mở cổng `/chat/completions`, bạn có thể dùng nó làm **Generator (LLM Inference Engine)** siêu mạnh.
- **Tối ưu hóa Context (256K):** Tận dụng tối đa `Qwen-3.6-27B` cho các tác vụ cần nhồi nhiều context (ví dụ: summarize nguyên một cuốn sách, hay pass toàn bộ codebase/log files).
- **Kiến trúc Custom RAG:** Vì không gọi được API `/embeddings` của server, nếu bạn muốn tự code RAG bằng Python:
  1. Sử dụng một model Embedding cục bộ miễn phí (VD: `sentence-transformers/all-MiniLM-L6-v2` trên HuggingFace).
  2. Dùng Vector Database nội bộ (như ChromaDB, FAISS).
  3. Chỉ gửi prompt cuối cùng (Context + Câu hỏi) đến AI-FIT `/chat/completions`.
- **Tham số Inference:** Đặt `temperature = 0.0` đến `0.2` cho các bài toán code, toán học, hoặc trích xuất thông tin. Đặt `0.7` cho sáng tạo nội dung.

### B. Khai thác qua WebUI (Dành cho Tương tác & Nghiên cứu trực tiếp)
Giao diện người dùng tại trang web cung cấp sức mạnh toàn diện do Open WebUI hỗ trợ:

- **RAG Nội Bộ (Công cụ cực mạnh nhưng có giới hạn):**
  - *Giới hạn:* 20 files, max 20MB.
  - *Chiến lược:* Đừng tải lên file PDF chứa nhiều hình ảnh vô nghĩa. Hãy dùng thư viện (như `pymupdf` hoặc `marker`) để chuyển đổi PDF thành `.txt` hoặc `.md` trước khi tải lên UI. Bạn sẽ nhét được gấp 10 lần kiến thức vào giới hạn 20MB này.
  - *Sử dụng `#`:* Đây là tính năng lập chỉ mục tag của Open WebUI. Rất hữu ích để phân tách các knowledge base khác nhau (ví dụ: `#math_paper`, `#code_docs`) mà không sợ LLM bị "ảo giác" (hallucination) chéo.

- **Web Search Integration:**
  - Kích hoạt tính năng này khi cần kiểm chứng sự kiện (fact-checking) đối với các thông tin sau thời điểm training của model. Tính năng này kết nối trực tiếp với công cụ tìm kiếm bên dưới, rất tốt cho quá trình tổng hợp báo cáo.

- **Đa Mô Hình & Voice/Image:**
  - Giúp bạn brainstorm đa chiều. Dùng STT (Speech-to-Text) để ra lệnh nhanh thay vì gõ văn bản dài, và sinh ảnh minh họa ngay trong flow trò chuyện mà không cần bật Midjourney/DALL-E.

---

## 3. Khuyến nghị Tiếp theo (Next Steps)
Nếu mục tiêu của bạn là sử dụng model này cho nghiên cứu (Experiments):
1. **Thí nghiệm chuẩn bị (Sanity Check):** Bạn có muốn tôi thiết lập một đoạn mã boilerplate (ví dụ: dùng LangChain hoặc LlamaIndex) kết nối vào model này để bạn tự code logic tự động hóa không?
2. **Khắc phục Multimodal:** Nếu bạn cần xử lý hình ảnh tự động, chúng ta có thể chuyển hướng dùng model khác trên hệ thống cục bộ (nếu bạn có GPU 3090 như đề cập trong file AGENTS.md) và chỉ dùng AI-FIT cho text generation.
