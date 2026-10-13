# Giá trị giả / chưa xác nhận trong bản thảo (in màu ĐỎ trong PDF, macro \fake{...})
Tìm trong LaTeX bằng: `grep -n "fake{" paper/main.tex`. Không có giá trị giả nào nằm trong abstract, kết luận hay các bảng/hình kết quả thật.

| Vị trí | Nội dung | Phải làm trước khi nộp |
|---|---|---|
| Khối tác giả | email của 3 tác giả (`hai-dang.dang@`, `dinh.quang@`, `simon.park@` + `vinuni.edu.vn`) — ĐOÁN theo mẫu, không tìm được nguồn công khai | thay bằng email thật |
| Mục Availability | "Licence: MIT (to be confirmed...)" | thầy/trường xác nhận giấy phép và chủ sở hữu bản quyền |
| Mục Availability | "Archival DOI: to be assigned (Zenodo)" | tạo bản phát hành Zenodo, điền DOI (và vào `CITATION.cff`) |
| Mục 5.3 (Future work) | Bảng `tab:planned` — TOÀN BỘ số liệu (0.46, 0.78, ... ) là GIẢ minh họa bố cục | thay bằng kết quả thật khi có dữ liệu đa nguyên nhân; nếu không có, xóa bảng và hình |
| Mục 5.3 (Future work) | Hình `f_planned_kcurve.png` (dữ liệu giả, có watermark đỏ) | như trên |
| Nơi nộp | bản thảo KHÔNG còn ghi nơi nộp (theo yêu cầu); đây là bài journal, mẫu `article` hiện tại | khi chọn tạp chí: chuyển sang mẫu tạp chí đó; (đã khảo sát IEEE CASE 2027, hạn 1/3/2027, nhưng bài này nhắm journal) |

Ngoài ra còn các dấu hiệu xanh (`\verify`): bài Ebrahimi Araghizad 2023 chỉ đọc tóm tắt (trong bảng related work có dấu ‡).

Ghi chú: khối mô tả đầu bài về chữ đỏ đã được bỏ khỏi PDF theo yêu cầu; chữ đỏ vẫn còn ở email, giấy phép, DOI, bảng/hình planned (mục Future work).
