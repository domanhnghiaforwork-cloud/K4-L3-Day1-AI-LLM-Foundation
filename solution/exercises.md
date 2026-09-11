# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
Khi temperature tăng từ 0.0 lên 1.5, câu trả lời trở nên đa dạng, sáng tạo và ít lặp lại hơn, nhưng cũng dễ xuất hiện thông tin sai lệch (hallucination) hoặc văn phong rời rạc hơn. Ngược lại, ở mức 0.0, mô hình luôn đưa ra một kết quả duy nhất, mang tính xác định cao và bám sát ý nghĩa phổ biến nhất.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
Tôi sẽ đặt temperature ở mức thấp (khoảng từ 0.2 đến 0.3) cho chatbot hỗ trợ khách hàng. Mức này đảm bảo câu trả lời ổn định, chính xác, bám sát tài liệu hướng dẫn và tránh việc chatbot tự "sáng tạo" ra các chính sách sai lệch gây phiền toái cho người dùng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
GPT-4o thường đắt hơn GPT-4o-mini khoảng 15 đến 20 lần tùy theo biểu giá token của OpenAI. GPT-4o xứng đáng khi cần xử lý các tác vụ suy luận logic phức tạp, lập trình nâng cao hoặc sáng tạo nội dung nghệ thuật tinh tế. Trường hợp nên dùng mini là cho các tác vụ đơn giản, lặp đi lặp lại như phân loại văn bản cơ bản, dịch thuật thông thường hoặc chatbot hỗ trợ FAQ khối lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
Phản hồi của giáo viên tiểu học rất ngắn gọn, dùng từ ngữ gần gũi, ẩn dụ đơn giản, trong khi bản của chuyên gia tài chính dùng nhiều thuật ngữ kỹ thuật phức tạp và chuyên sâu. System prompt định hình trực tiếp góc nhìn, giọng điệu, mức độ chuyên sâu và cấu trúc trình bày của mô hình, biến nó thành một chuyên gia chuyên biệt thay vì một trợ lý chung chung.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
Số token thực tế từ tiktoken thường cao hơn khoảng 30% đến 50% so với ước lượng đơn giản số từ / 0.75 (tỷ lệ token/từ tiếng Việt rơi vào khoảng 1.3 - 1.5 token/từ). Tiếng Việt tốn nhiều token hơn vì bộ mã hóa tokenizer (được huấn luyện chủ yếu trên tiếng Anh) không có nhiều từ vựng tiếng Việt nguyên khối trong từ điển, dẫn đến việc các dấu thanh hoặc âm tiết phải bị cắt nhỏ thành nhiều byte hoặc subword tokens.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
Streaming đóng vai trò cực kỳ quan trọng trong các giao diện trò chuyện thời gian thực để giảm thời gian chờ đợi cảm nhận của người dùng khi văn bản dài đang được sinh ra từng từ một. Ngược lại, non-streaming lại phù hợp hơn cho các tác vụ phía sau như tự động phân loại email, trích xuất dữ liệu JSON, hoặc gọi API nội bộ nơi ứng dụng cần nhận toàn bộ kết quả hoàn chỉnh một lần để xử lý logic tiếp theo mà không cần hiển thị trực quan cho người dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
Exponential backoff giúp giảm dần tần suất gửi yêu cầu, tạo khoảng thời gian nghỉ hợp lý để hệ thống máy chủ hạ tầng kịp phục hồi sau quá tải. Nếu hàng nghìn client cùng retry với delay cố định, hiện tượng yêu cầu đồng loạt sẽ xảy ra, khiến máy chủ tiếp tục sập ngay lập tức do lượng traffic dồn dập vào cùng một thời điểm.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
Tôi chọn persona là một trợ lý lập trình Python súc tích và thực tế.
System prompt: "Bạn là chuyên gia lập trình Python cao cấp. Hãy trả lời ngắn gọn, tập trung vào đoạn code tối ưu và giải thích súc tích bằng tiếng Việt. Không dùng lời chào dài dòng."
Lựa chọn "ngắn gọn" giúp tiết kiệm token và thời gian đọc của developer, còn "tiếng Việt" đảm bảo kết quả trả về đúng ngôn ngữ giao tiếp của người dùng mà không cần dịch thủ công.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
Hạn chế lớn nhất hiện tại là thiếu bộ nhớ dài hạn. Cải thiện cụ thể là tích hợp cơ chế Retrieval-Augmented Generation (RAG) kết hợp bộ nhớ vector (như ChromaDB hoặc FAISS) để lưu trữ và truy xuất lại lịch sử hội thoại hoặc tài liệu liên quan khi cần thiết, triển khai bằng cách nhúng các đoạn chat cũ thành vector và tìm kiếm độ tương đồng cosine trước mỗi lần gọi API.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
