# Phương án hướng tới tạp chí Q1 ngành gia công (bản nháp để thảo luận với thầy)

Lưu ý: xếp hạng/IF của tạp chí và mức độ nhận bài tôi KHÔNG biết; tiêu chuẩn dưới đây là hiểu biết chung + khảo sát tiêu đề các bài gần đây (chưa đọc toàn văn). Số lượng thí nghiệm là đề xuất, cần phép tính cỡ mẫu từ một thí điểm.

## 1. Khảo sát nơi đăng (tiêu đề bài 2024–2026 khớp chủ đề; `paper/lit/venue_survey.json`)
| Tạp chí | Các bài gần với đề tài (tiêu đề) | Nhận xét |
|---|---|---|
| J. Manuf. Systems | Physics-informed multi-task heterogeneous transfer learning for milling tool wear/RUL (2026); digital-twin anomaly detection for tool condition monitoring (2024); tool digital twin (2025) | Chấp nhận hướng vật lý + chuyển giao + digital twin; cạnh tranh cao |
| Mech. Syst. Signal Process. | Cross-machine fault diagnosis với ít mẫu (continual learning, 2025); semi-supervised kết hợp quy luật mòn dao (2025); few-shot meta-learning | Thích phương pháp học mới + đánh giá chuyển giao máy |
| J. Manuf. Processes | Meta-transfer learning dự đoán ổn định phay theo vị trí khi đổi chiều dài nhô (2025); hybrid mô phỏng vật lý + ML cho lực phay (2024) | Rất gần ý tưởng "đổi tương tác máy–dao"; phải đọc kỹ để định vị |
| Robotics & Comput.-Integr. Manuf. | Mô hình mòn dao kết hợp cơ chế + dữ liệu + chuyển giao (2026); digital twin cấp máy cho phay vòng kín (2026/27) | Thích tích hợp, vòng kín |
| J. Intell. Manuf. | Few-shot meta-transfer chẩn đoán lỗi (2024–2025); chẩn đoán lỗi phức hợp trên máy phay (2026) | Nhiều bài ML chẩn đoán; cần khác biệt rõ |
| Int. J. Mach. Tools Manuf. / CIRP JMST | Receptance coupling dự đoán FRF đầu dao (2025); ổn định chatter; Bayesian có vật lý cho chatter (2024) | Thiên vật lý và thực nghiệm; ngưỡng cao về đo đạc |
**Ứng viên đề xuất:** J. Manuf. Systems hoặc Mech. Syst. Signal Process. (nếu nhấn phương pháp và chuyển giao), J. Manuf. Processes (nếu nhấn tương tác máy–dao và vật lý). Cần thầy xác nhận.

## 2. Định vị đóng góp (cần kiểm lại sau khi đọc toàn văn các bài gần nhất)
Không chọn tính mới ở "học trước bằng mô phỏng" (đã có: Alberts 2024, Ebrahimi Araghizad 2023). Đề xuất:
1. Chẩn đoán **nhiều nguyên nhân gốc** (kẹp lỏng/độ cứng, nhô dao, mòn, độ sâu/tốc độ ngoài ổn định, mất cân bằng, sứt rãnh) thay vì một loại lỗi.
2. **Khởi động lạnh (cold-start) trên máy mới**: mô phỏng neo vào FRF đo thật + đặc trưng tương đối theo nền khỏe, không cần nhãn lỗi thật; chứng minh bằng đường cong k = 0..32 mẫu.
3. **Đo khoảng cách mô phỏng–thật theo thành phần** (nền nhiễu, chữ ký lỗi) và nói khi nào mô phỏng giúp, khi nào gây hại (kết quả âm tính đã có).
4. **Bộ dữ liệu** có nhãn theo thiết kế + FRF + hiệu chuẩn lực (nếu thu được) — tài sản riêng, có thể thành bài dữ liệu.
Tiêu đề tạm: "Cold-start root cause diagnosis in milling across machines: physics-grounded simulation anchored on measured FRF".

## 3. Chương trình thực nghiệm (cần dữ liệu của bạn)
### 3.1 Phạm vi tối thiểu / mục tiêu
| | Tối thiểu khả thi | Mục tiêu |
|---|---|---|
| Máy / cấu hình | 2 | 3 |
| Dao | 2 (đường kính hoặc chiều dài nhô khác nhau) | 3 |
| Vật liệu phôi | 1 | 2 |
| Nguyên nhân | khỏe + 5 (kẹp lỏng, nhô dao, mòn dao, độ sâu/tốc độ quá ổn định, mất cân bằng) | + sứt rãnh |
| Mức độ mỗi nguyên nhân | 2 | 3 |
| Số lần cắt mỗi máy | ~100 (khỏe 24; 5 nguyên nhân x 2 mức x 2 điều kiện x 2 lặp = 40; +dự phòng) | ~200 |
### 3.2 Đo đạc bắt buộc
- **Gõ búa FRF đầu dao (x, y)** mỗi khi đổi cấu hình (nhô dao, kẹp). Lưu dữ liệu thô FRF và tham số modal đã khớp.
- **Cắt hiệu chuẩn hệ số lực** (rãnh ở vài lượng ăn) cho từng dao/vật liệu.
- Cảm biến: gia tốc 3 trục (thân trục chính), mic, công suất/dòng trục chính từ bộ điều khiển; lực kế nếu có. Lấy mẫu >= 25 kHz; ghi mọi tham số (rpm, ăn, ap, ae, ID dao, nhô dao, lực xiết kẹp, khối lượng thêm, VB đo bằng kính hiển vi).
- Nhãn có từ thiết kế (đại lượng đặt) và được kiểm tra (cờ lê lực, VB, khối lượng).
### 3.3 Chia dữ liệu và đánh giá (đăng ký trước)
Bỏ-một-máy-ra và bỏ-một-cấu-hình-ra; tập kiểm không dùng để chọn siêu tham số; >= 10 lần chia/hạt giống, báo trung bình ± độ lệch chuẩn và khoảng tin cậy bootstrap; kiểm định cặp.
Chỉ số: Top-1/Top-3, macro-F1, recall từng nguyên nhân, ECE (hiệu chỉnh), đường cong phạm vi–độ chính xác khi cho phép "không chắc".

## 4. Phương pháp cần cài đặt và so sánh
1. Mô phỏng DDE có **FRF đo thật** (nạp FRF -> khớp mode -> mô hình), thêm cơ chế sứt rãnh/không đối xứng răng và cản quá trình; kiểm chứng lobe mô phỏng so với các lần cắt thật.
2. Đặc trưng **tương đối theo nền khỏe** của máy/dao, bất biến dấu trục lực (đã có bằng chứng sơ bộ trên LUH).
3. Mô-đun theo nguyên nhân + hợp nhất xác suất đã hiệu chỉnh + lựa chọn "không chắc".
4. Chế độ huấn luyện: (a) chỉ mô phỏng (k=0), (b) mô phỏng rồi tinh chỉnh k mẫu thật, (c) chỉ k mẫu thật, (d) ngẫu nhiên hóa tham số có ràng buộc vật lý neo quanh FRF đo.
5. Baseline: bộ phân loại chuẩn; DANN (Wang 2020); meta-DG (Li 2023); baseline công nghiệp (ngưỡng RMS, quy tắc phổ, kiểm tra lobe bằng FRF đo); lai vật lý–dữ liệu (Yang 2022; Li 2025).
6. Ablation: cảm biến (chỉ gia tốc / + mic / + công suất), có/không FRF đo, có/không chuẩn hóa nền khỏe, kích thước tập mô phỏng.
## 5. Kiểm chứng hành động (bằng chứng mạnh cho RCA)
Với nguyên nhân sửa được (kẹp, nhô dao, mất cân bằng, tốc độ/độ sâu): chẩn đoán -> thực hiện biện pháp -> chứng minh tín hiệu và bề mặt trở về bình thường.
## 6. Cổng quyết định và phương án dự phòng
- **Cổng 1 (thí điểm trên 1 máy, vài tuần):** mô phỏng với FRF đo tái tạo ranh giới ổn định thực nghiệm (tiêu chí đề xuất: sai lệch độ sâu giới hạn trong khoảng đã thống nhất với thầy) và có mẫu dấu hiệu lỗi cùng chiều. Không đạt: sửa mô hình trước khi thu thêm.
- **Cổng 2 (sau thu đủ):** chứng minh được lợi ích khi k nhỏ (so với học từ đầu) trên >= 1 nhiệm vụ nhiều nguyên nhân. Không đạt: định vị lại là **khởi động lạnh (k=0) + bộ dữ liệu + phân tích khi nào mô phỏng giúp**.
- Dự phòng đăng: bài dữ liệu (ví dụ Data in Brief / Scientific Data) cho bộ dữ liệu; bài phương pháp riêng.
## 7. Việc tôi làm được ngay không cần dữ liệu mới
1. Đọc toàn văn các bài gần nhất (Alberts 2024; Ebrahimi Araghizad 2023; meta-transfer đổi nhô dao JMP 2025; JMS 2026 multi-task chuyển giao; MSSP continual learning chéo máy) để chốt tính mới.
2. Cài đặt: nạp FRF (CSV/UFF/MAT) -> khớp mode; cơ chế sứt rãnh/không đối xứng răng; cản quá trình.
3. Đóng gói giao thức đánh giá (đường cong k, ECE, phạm vi–độ chính xác, kiểm định) để chạy ngay khi có dữ liệu.
4. Viết lại khung bản thảo theo định vị trên, theo mẫu tạp chí đã chọn.
## 8. Rủi ro
Thu dữ liệu chậm hoặc ít; mô phỏng không tái tạo dấu hiệu lỗi; lợi ích tinh chỉnh không xuất hiện; tính mới bị đối thủ gần nhất (JMP 2025, JMS 2026) bao phủ. Mỗi rủi ro có cổng/dự phòng ở mục 6.
