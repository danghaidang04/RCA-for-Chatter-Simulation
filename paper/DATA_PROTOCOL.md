# Protocol thu dữ liệu thật và đánh giá (bản nháp để thảo luận với thầy)

Mục tiêu: trả lời bằng thực nghiệm hai câu hỏi mà dữ liệu công khai không trả lời được.
- **H1**: mô phỏng neo vào FRF đo có lợi khi nguyên nhân KHÔNG có nhãn thật trên máy nào.
- **H2**: mô phỏng có lợi khi máy–dao đích có động lực học NGOÀI dải của các cấu hình huấn luyện.
Nếu cả hai đều không được ủng hộ, bài định vị lại thành "khoảng cách mô phỏng–thật + bộ dữ liệu".

## 1. Thiết kế thí nghiệm
| Yếu tố | Tối thiểu | Mục tiêu |
|---|---|---|
| Máy (hoặc cấu hình kẹp khác hẳn) | 2 | 3 |
| Dao/nhô dao (động lực học đầu dao khác nhau, có FRF riêng) | 3 nhô dao khác nhau mỗi máy | 4 |
| Vật liệu | 1 | 2 |
| Nguyên nhân | khỏe + kẹp lỏng + nhô dao + mòn dao + độ sâu/tốc độ vượt ổn định + mất cân bằng | + sứt rãnh |
| Mức độ mỗi nguyên nhân | 2 | 3 |
| Lặp | 2 | 3 |
Thứ tự cắt NGẪU NHIÊN hóa theo khối (chống trôi nhiệt/mòn lẫn nhãn); ghi cắt khỏe xen kẽ để kiểm tra trôi.

## 2. Đo đạc bắt buộc (cho mỗi cấu hình)
1. **FRF đầu dao** (búa, trục x và y, ≥3 điểm/lần đo, ghi dữ liệu thô + tham số modal khớp bằng `cutting_sim/frf_import.py`); lặp lại FRF sau mỗi lần đổi nhô dao/kẹp và sau chuỗi cắt.
2. **Cắt hiệu chuẩn lực** (rãnh ở ≥4 lượng ăn/răng) cho từng dao–vật liệu → Kt, Kr, Kte, Kre (bù cho dữ liệu hiện chỉ là giả định).
3. **Cảm biến**: gia tốc 3 trục trên thân trục chính, mic, công suất/dòng trục chính từ bộ điều khiển; lực kế nếu có. Lấy mẫu ≥25 kHz. Lưu CÙNG định dạng cho mọi máy.
4. **Thông số cắt thật** từ bộ điều khiển (rpm, ăn, ap, ae, hướng phay), ID dao, nhô dao đo bằng thước, lực xiết kẹp (cờ lê lực), khối lượng thêm (mất cân bằng), VB đo kính hiển vi trước/sau mỗi cắt.
5. **Nhãn từ thiết kế** (đại lượng đặt) và được kiểm tra bằng các đo đạc trên; không dùng nhãn chuyên gia chủ quan.

## 3. Kiểm tra cổng 1 (thí điểm 1 máy, vài tuần)
- Mô phỏng với FRF đo + Kt/Kr hiệu chuẩn có tái tạo được ranh giới ổn định thực nghiệm (cắt rãnh tăng dần độ sâu tại ≥5 tốc độ) trong sai lệch độ sâu giới hạn đã thống nhất (đề xuất ±25%) không?
- Mỗi nguyên nhân có dấu hiệu cùng chiều với mô phỏng (cosine > 0.5 sau khi loại thành phần "to hơn" chung) không?
- Không đạt: sửa mô hình (đặc biệt cơ chế dao hỏng, cản quá trình) trước khi thu thêm.

## 4. Chia dữ liệu và chế độ huấn luyện (đăng ký trước)
- **Chia**: leave-one-machine-out; leave-one-configuration-out (nhô dao/dao); leave-one-cause-out (giả lập "nguyên nhân chưa có nhãn thật" để kiểm H1).
- **Chế độ**: (a) chỉ mô phỏng (k=0); (b) mô phỏng rồi tinh chỉnh k mẫu thật, k∈{2,4,8,16,32}; (c) chỉ k mẫu thật; (d) chỉ dữ liệu thật của các cấu hình khác (không mô phỏng); (e) (d) + mô phỏng neo FRF đích.
- **Mô phỏng**: ngẫu nhiên hóa tham số có ràng buộc, neo quanh FRF đo của cấu hình ĐÍCH (đây là điều kiện biết trước, không dùng tín hiệu đích).
- Tập kiểm không dùng để chọn siêu tham số; ≥10 hạt giống/chia; trung bình ± sd; bootstrap CI theo cắt (cụm), không theo cửa sổ.

## 5. Chỉ số và so sánh
- Top-1/Top-3, macro-F1, recall từng nguyên nhân, ECE, đường cong phạm vi–độ chính xác khi cho phép "không chắc".
- Baseline: bộ phân loại chuẩn (GB, LR), DANN (Wang 2020), meta-DG (Li 2023), quy tắc ngưỡng RMS/phổ, kiểm tra lobe bằng FRF đo, và (d) ở trên (dữ liệu thật không mô phỏng).
- Kiểm định: so cặp theo chia (Wilcoxon), hiệu chỉnh đa so sánh.
- **Tiêu chí thành công (đăng ký trước)**: ở ít nhất một chia (leave-one-cause-out hoặc leave-one-configuration-out), chế độ (a) hoặc (b)/(e) vượt (c)/(d) ở k nhỏ (≤8) với hiệu số có CI không chứa 0.

## 6. Kiểm chứng hành động (bằng chứng mạnh cho RCA)
Với nguyên nhân sửa được (kẹp, nhô dao, mất cân bằng, tốc độ/độ sâu): chẩn đoán → thực hiện biện pháp → xác nhận tín hiệu và bề mặt trở lại bình thường.

## 7. Cỡ mẫu và ngân sách tham khảo
Mỗi máy–cấu hình ~100 cắt (khỏe 24; 5 nguyên nhân × 2 mức × 2 điều kiện × 2 lặp = 40; dự phòng). 2 máy × 3 nhô dao ≈ 600 cắt. Cần phép tính cỡ mẫu từ thí điểm (cổng 1) trước khi chốt.

## 8. Rủi ro
Thu dữ liệu chậm/ít; mô phỏng không tái tạo dấu hiệu lỗi (cổng 1 bắt sớm); lợi ích không xuất hiện (khi đó định vị lại như đã nêu); đa so sánh (đã đăng ký trước tiêu chí).
