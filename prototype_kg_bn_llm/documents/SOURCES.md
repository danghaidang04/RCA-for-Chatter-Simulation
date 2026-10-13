# Tài liệu đầu vào để dựng KG nền cho phay/gia công (tìm ngày 13/10/2026)

Mức xác minh: **ĐÃ ĐỌC** = mình đã mở và đọc nội dung; **CHỈ KẾT QUẢ TÌM KIẾM** = chỉ thấy mô tả trong kết quả tìm kiếm (trang trả 403 hoặc không có thân bài khi truy cập tự động); chưa tải file nào về.
Bản quyền: tài liệu hãng thường có bản quyền. Đề xuất: chỉ lưu bộ ba có cấu trúc (nguyên nhân -> triệu chứng -> biện pháp) kèm trích dẫn ngắn và URL nguồn, KHÔNG sao chép và phân phối lại nguyên văn; kiểm tra điều khoản sử dụng của từng hãng trước khi chạy trích xuất hàng loạt bằng LLM.

## 1. Bảng FMEA (hiện tượng lỗi -> nguyên nhân -> tác động)

| Nguồn | Nội dung | Mức xác minh |
|---|---|---|
| McLeay, Turner, *Journal of Machine Engineering* 2011 (yadda.icm.edu.pl/baztech/... "9_MCLEAY.pdf") | FMEA cấp quá trình cho công đoạn "Milling Operation" (Bảng 1) gắn lỗi, nguyên nhân, tác động với giám sát bằng cảm biến — gần nhất với mục tiêu | CHỈ KẾT QUẢ TÌM KIẾM (PDF trả 403; chưa xác nhận các hàng chatter/mòn dao trong Bảng 1) |
| Nghiên cứu FMEA máy phay CNC, PT. IJA Surabaya (archive.umsida.ac.id, preprint 8248) | Điểm S/O/D 1-10 -> RPN; trục chính là chi tiết tới hạn nhất: mòn/kẹt do thiếu bôi trơn, quá tải (RPN 933) | CHỈ KẾT QUẢ TÌM KIẾM |
| FTA + FMEA trên máy phay CNC, Univ. Riau (jom.unri.ac.id/index.php/JOMFTEKNIK/article/view/20170) | Cây lỗi tìm nguyên nhân sự kiện cơ bản, FMEA xếp ưu tiên bằng RPN | CHỈ KẾT QUẢ TÌM KIẾM |
| FTA + FMEA máy phay CNC HOLZHER EcoMaster 7120, *AFTMT* X(1) 2023 (aftmt.uni-ruse.bg/wp-content/uploads/2024/10/AFTMT-X-1-2023-4.pdf) | Xếp hạng lỗi theo mức nghiêm trọng, tần suất, khả năng phát hiện (máy phay gỗ-CNC, không phải kim loại) | CHỈ KẾT QUẢ TÌM KIẾM |
| FMEA quá trình phay đầu (end milling), IJMET 2013 (bản SlideShare) | RPN, nguyên nhân, tác động, biện pháp phòng ngừa; chưa rõ truy cập mở | CHỈ KẾT QUẢ TÌM KIẾM |
| Sabancı University: chatter làm giảm tuổi thọ dao khoảng 30% trong thử nghiệm phay (research.sabanciuniv.edu/id/eprint/12769/) | Dữ liệu để chấm điểm mức nghiêm trọng "chatter -> mòn dao" | CHỈ KẾT QUẢ TÌM KIẾM |
Khoảng trống: chưa thấy FMEA công khai riêng cho vít me bi/bàn máy ở phay; cần tìm mẫu FMEA (SAE J1739/AIAG) hoặc FMEA từ nhà sản xuất.

## 2. Cây phân rã chức năng, sơ đồ máy (ontology / asset hierarchy)

| Nguồn | Nội dung | Mức xác minh |
|---|---|---|
| MTConnect, Device Information Model (model.mtconnect.org, bản 2.x; docs.mtconnect.org Part 2) | Thiết bị -> Component -> DataItem; Axis là kiểu con của Component (Linear X/Y/Z, Rotary); Spindle là kiểu con của Axis nhưng bị đánh dấu lỗi thời từ v1.1 và thay bằng RotaryMode; quy tắc đặt tên trục (X, Y, Z; X2, X3) | CHỈ KẾT QUẢ TÌM KIẾM (thân trang là ứng dụng động; chưa đọc được bảng thành phần) |
| OPC UA OPC 40501-1 "Machine Tools - Monitoring and Job Management" (reference.opcfoundation.org/MachineTool), phiên bản 1.02 (2024-11-01); thực thi mẫu: umati | Mô hình thông tin máy công cụ trung lập về công nghệ (MachineToolType...) để trao đổi với MES/SCADA | CHỈ KẾT QUẢ TÌM KIẾM (chưa thấy chi tiết trục chính/trục tiến; có thể nằm ở phần khác của họ tiêu chuẩn) |
Khoảng trống: cây phân rã tới mức ổ trục, trục vít me bi, ray dẫn, động cơ servo cần tài liệu bảo trì của hãng hoặc ontology học thuật; cần tìm thêm.

## 3. Sổ tay xử lý sự cố của hãng ("Nếu... thì...")

| Nguồn | Nội dung | Mức xác minh |
|---|---|---|
| Sandvik Coromant, "Troubleshooting milling" (www.sandvik.coromant.com/en-us/knowledge/milling/troubleshooting-milling) | Hai bảng cột Cause / Solution: Milling (Vibration, chip clogging, re-cutting, poor surface finish, burr, machine power) và Insert wear (flank, crater, plastic deformation, chipping, notch, thermal crack, built-up edge). Nhóm Vibration có 8 hàng: weak fixture, axially weak workpiece, excessive tool overhang, square shoulder milling with weak spindle, irregular table feed (siết cơ cấu tiến, thay vít me bi), cutting data, poor stability, vibration in corners | ĐÃ ĐỌC |
| Kennametal, tech tips về giảm chatter khi phay đầu (kennametal.com/.../reducing-chatter-and-vibration-in-end-milling.html) | Biện pháp: kiểm tra tốc độ/lượng ăn, tăng độ cứng vững, giảm khoảng cách chi tiết-trục chính, nhiều rãnh hơn, kiểm tra độ đảo, hình học rãnh/góc xoắn, phay thuận | CHỈ KẾT QUẢ TÌM KIẾM |
| Seco Tools, "Reduce chatter for better machining" (secotools.com/article/125930) | Nguyên nhân chatter từ dụng cụ hoặc phôi, hướng giảm | CHỈ KẾT QUẢ TÌM KIẾM |
| Haas, tài liệu tham khảo "Mill - spindle vibration - tolerances" (RD0071 theo kết quả tìm kiếm) và sổ tay dịch vụ VF (haascnc.com) | Dung sai dịch chuyển rung trục chính theo cỡ và tốc độ; quy trình cô lập nguồn rung: chạy từng trục không quay trục chính, quay trục chính không chạy trục | CHỈ KẾT QUẢ TÌM KIẾM (trang trả 403) |

## Gợi ý bước tiếp theo
1. Chọn một tập nhỏ ưu tiên (Sandvik bảng rung + 1 FMEA phay + MTConnect Axis/Spindle) làm "tài liệu nền", thiết kế lược đồ trích: `(cause | symptom/feature | remedy | source URL | quote)`.
2. Chạy trích xuất bằng LLM (OpenRouter, khi bạn cho phép) và so với bộ KG viết tay trong `kg/milling_kg.json`; mọi cạnh tạo ra gắn tier ASSUMED và độ bất định cao.
3. Với các nguồn bị 403, bạn tải thủ công rồi đặt vào `documents/raw/` (đã nằm ngoài git nếu bạn thêm vào .gitignore) — mình chưa tải để tránh sao chép tài liệu có bản quyền.
