# Việc còn mở: quyết định của bạn, việc bạn cần làm, và điểm chưa chắc (cập nhật 13/10/2026)

## A. Cần bạn quyết
1. **Kiến trúc KG.** Mình tự chọn: 5 nguyên nhân, mỗi nguyên nhân một cơ chế (một bước), chatter chỉ là hai chỉ số phi đồng bộ chứ chưa là nút riêng, dấu theo lập luận vật lý (tier ASSUMED), cơ chế KHÔNG là biến ẩn trong mạng Bayes (chỉ để truy vết). Cần chốt: tập nguyên nhân; có cần chuỗi cơ chế nhiều bước; có đưa chatter thành nút riêng; có dùng cơ chế làm biến ẩn không.
2. **Cách dựng KG.** Bạn muốn KG sinh tự động từ tài liệu. Hiện `kg/milling_kg.json` là bản viết tay; bộ dựng tự động `kg_build.py` mới có chế độ quy tắc trên kho tài liệu demo và khung chế độ LLM (xem `LITERATURE.md`, `documents/`). Cần xác nhận ontology (loại nút: cause, mechanism, feature, action; loại quan hệ: leads_to, manifests_as, verified_by), coi trọng số ban đầu do LLM đưa ra chỉ là tiên nghiệm ASSUMED độ bất định cao, và đồng ý chạy bước trích xuất bằng LLM thật (cần khóa, mỗi lần chạy bật cờ cho phép).
3. **Luồng ứng dụng web** (đã dựng): thiết lập điều kiện và chế độ giải thích -> Start -> máy chạy, học nền khỏe trong 10 cửa sổ đầu -> tick một lỗi và Apply (lỗi tăng dần trong vài cửa sổ) -> bộ phát hiện chỉ dùng cảm biến báo đèn đỏ và âm thanh -> thu 10 cửa sổ -> xếp hạng nguyên nhân kèm đồ thị nhân quả -> lời giải thích. Cần bạn xác nhận: (a) chỉ một lỗi mỗi lần (mô hình đơn nguyên nhân); (b) lỗi tăng dần rồi mới vượt ngưỡng; (c) phát hiện chỉ dựa trên cảm biến.
4. **Độ chính xác**: chấp nhận như một demo trung thực hay cải thiện trước? Số liệu hiện tại ở `README.md` mục Web app (xếp hạng top-1 khoảng 0.6 với 5 nguyên nhân; cản thấp hay bị nhầm; mất cân bằng và độ đảo gần như không được nhận ra là lỗi lạ).
5. **Đưa kết quả prototype vào paper hay không.** Paper v4 chưa chứa phần học liên tục, ứng dụng web hay KG tự động; trộn vào sẽ làm bài lệch trọng tâm đo lường.
6. **Giấy phép và thông tin nộp:** giấy phép MIT (thầy hoặc trường xác nhận), email thật của ba tác giả (hiện đang đoán, màu đỏ), DOI Zenodo, tạp chí nhắm tới.
7. **Commit và push.** Chưa commit gì. Nếu commit prototype và ứng dụng: không đưa `.pyc`, `.npz` dữ liệu, `.env`, `app/server.log`.

## B. Cần bạn làm
- **Tài liệu**: tải các file ưu tiên trong `documents/DOWNLOAD_REQUEST.md` (C1, C2, C5, C10, A1; nếu kịp B2, B3), đặt đúng đường dẫn rồi nhắn "xong".
- **Khóa OpenRouter**: ĐÃ có `.env` và đã thử gọi thành công một lần (mô hình mặc định `google/gemma-3-4b-it`). Khóa đặt ở `prototype_kg_bn_llm/.env`, dòng `OPENROUTER_API_KEY=...`; đổi mô hình bằng dòng `OPENROUTER_MODEL=...`. Không dán khóa vào chat.
- **Thu hồi token Kaggle** đã dán trước đó (nếu chưa).

## C. Mình chưa chắc / chưa làm
- Giao diện web **chưa được nhìn thấy trên trình duyệt** (tiện ích trình duyệt ngắt kết nối giữa chừng). Đã kiểm tra bằng gọi API, kiểm tra cú pháp JavaScript (esprima) và tồn tại mọi phần tử HTML. Bạn mở thử và báo lỗi nếu có.
- Âm thanh cảnh báo cần bạn bấm Start (trình duyệt chỉ cho phát âm thanh sau thao tác của người dùng); chưa nghe thử.
- Kết quả ứng dụng chỉ nằm trong bộ sinh mô phỏng, không nói lên khả năng chẩn đoán trên máy thật; bài Q1 vẫn cần dữ liệu thật đa nguyên nhân kèm FRF.
- Tài liệu FMEA, MTConnect, OPC UA, Haas, Kennametal mới thấy qua kết quả tìm kiếm; chỉ Sandvik được đọc trực tiếp.
- Truy hồi kiểu LightRAG là bản mô phỏng ý tưởng (không dùng thư viện `lightrag`), kho demo chỉ 11 đoạn ngắn.
- Chất lượng lời giải thích của LLM chưa được đánh giá bởi người; bộ kiểm tra chỉ đảm bảo số, tên nguyên nhân và trích dẫn có trong bằng chứng.

## Việc hiệu chuẩn bộ phát hiện (hoãn, ghi ngày 2026-10-13)
- Hiện dùng bộ phát hiện theo rung động tương đối dụng cụ–phôi (CUSUM của log RMS, k_v=0.5, h_v=2.0, sàn 0.02) theo yêu cầu; 0/40 báo động giả (tuning), 0/20 (hold-out); phát hiện trung bình 0.69.
- Yếu: RPM_MISMATCH (~60%), IMBALANCE (25–42%), RUNOUT (0–17%). Phương án thay thế đã đo: spectral 0.905 (báo động giả hold-out 10%), union 0.917 (10%).
- Cần làm sau: (1) chọn lại chế độ (vibration / union) và tăng số phiên healthy (>=200) để chặn tỉ lệ báo động giả; (2) hiệu chuẩn lại trên máy/điều kiện khác; (3) kiểm tra ngưỡng với dữ liệu thật; (4) cập nhật `app/report/detector.json` bằng `python -m app.calibrate_detector`.
