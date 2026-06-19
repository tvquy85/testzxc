**LLM Services**  
[**https://ai-fit.hcmus.edu.vn**](https://ai-fit.hcmus.edu.vn)  
Email hỗ trợ: webmaster@fit.hcmus.edu.vn

**Giới thiệu**  
Đăng nhập thông qua tài khoản Google FIT  
Hệ thống được setup dựa trên Open WebUI (https://docs.openwebui.com)  
Cung cấp model LLM mở, cập nhật mới và phổ biến:  
Model name: **Qwen-3.6-27B**  
Context: 256k (32k batched)  
Max output tokens: 10000

**API Keys**  
Base\_url**:** https://ai-fit.hcmus.edu.vn/openai  
Sử dụng API Key của tài khoản cá nhân:  
Click Username tại góc dưới-trái \> Settings \> Account \> **Show API Key**

**Ví dụ:**

curl \-sX GET "https://ai-fit.hcmus.edu.vn/openai/models" \\  
  \-H "Authorization: Bearer **sk-xxx**" | jq '.data\[\].id'

curl \-sX POST "https://ai-fit.hcmus.edu.vn/openai/chat/completions" \\  
  \-H "Content-Type: application/json" \\  
  \-H "Authorization: Bearer **sk-xxx**" \\  
  \-d '{  
    "model": "Qwen3.6-27B",  
    "messages": \[  
      {"role": "system", "content": "You are a helpful assistant."},  
      {"role": "user", "content": "Giới thiệu về Trường ĐH Khoa học tự nhiên TP.HCM"}  
    \],  
    "max\_tokens": 500  
  }'

**Chat & Models**

* Đa mô hình: Chuyển đổi linh hoạt giữa các model cục bộ trong cùng một luồng chat.  
* Voice & Image: Hỗ trợ nhập liệu bằng giọng nói (Speech-to-Text) và tạo hình ảnh minh họa cho bài giảng trực tiếp.  
* Controls: Cho phép tùy chỉnh các tham số kỹ thuật như Temperature, System Prompt… ngay tại khung chat.

**RAG:** Công cụ hỗ trợ đưa tài liệu chuyên môn vào AI  
*Tối đa 20 file với tổng dung lượng 20MB.*

* Tải lên tài liệu: Hỗ trợ các loại file văn bản, hình ảnh… Hệ thống sẽ tự động phân tách và lập chỉ mục (chunking & embedding).  
* Triệu hồi dữ liệu: Sử dụng ký tự \# trong khung chat để gọi một tập hồ sơ cụ thể. AI sẽ chỉ trả lời dựa trên nội dung tài liệu đó.  
* Web Search: Tích hợp công cụ tìm kiếm để cập nhật các nghiên cứu mới nhất mà model gốc chưa kịp học.

