# archive/ — script thế hệ cũ (không còn được module nào import, không dùng cho bản v2 của paper)
Gồm các bản hiệu chỉnh (calibrate_v2_B/v3/v4, sens_v4), chẩn đoán (diag*), RCA đời đầu (exp4/exp5) và vẽ loss.
Lý do giữ: lịch sử kết quả âm tính (hiệu chỉnh trực tiếp DDE không hội tụ, xem paper mục "Negative result").
Muốn chạy lại: `cd cutting_sim && PYTHONPATH=. python archive/<file>.py` (chúng import các module ở thư mục cha và có thể cần dữ liệu `data/`).
