# Trạng thái công việc — mô phỏng cắt phay + RCA nhân quả (dừng thực nghiệm theo yêu cầu)

Môi trường chạy: `/Users/danghaidang04/miniforge3/bin/python3` (numpy, scipy, sklearn, numba, cma, networkx). `.venv` của repo bị hỏng, không dùng.
Mã nằm trong `cutting_sim/`, kết quả/hình trong `cutting_sim/output/`. Dữ liệu thô nằm ngoài git: `data/msm/` (MSM, Kim et al., Sci. Data 2026, CC BY 4.0).

## 1. Đã làm xong
| Hạng mục | File chính | Kết quả |
|---|---|---|
| Mô hình phay tái sinh DDE 2-DOF, nhiều răng, lõi numba, rãnh xoắn, lực cạnh | `milling_model.py`, `model_v4.py` | numba khớp bản Python tới 7e-16; tắt xoắn/lực cạnh = mô hình cũ |
| Stability lobe ZOA (Altintas) | `milling_model.py` | khớp code cũ của repo (sai số ~1e-16); khớp mô phỏng miền thời gian ~79% (nửa immersion) |
| 4 loại cảm biến (rung tương đối, gia tốc, mic, lực) + hình | `sensors.py`, `exp1_baseline.py`, `exp2_faults.py` | `output/exp1_*.png`, `exp2_*.png` |
| Tải và đọc dữ liệu thật MSM (dataset 5, 6; dao A, AA6061, imi_vm20i) | `real_features.py`, `real_windows.py` | 73 đường cắt, 219 cửa sổ 0.3 s |
| Bảng nguồn gốc tham số (FIXED / LEARNED / ASSUMED) kèm ngữ cảnh + mức xác minh | `model_v4.py::parameter_registry`, `output/parameter_registry.md` | góc xoắn 30° mức "trung bình"; Kt, FRF, gain... không có nguồn → học/giả định |
| Phân tích độ nhạy | `sens_v4.py`, `output/sensitivity_v4.csv` | runout gần như không ảnh hưởng → cố định 5 µm (ASSUMED) |
| Nền nhiễu từ phổ thật (đường bao phổ + hài, hồi quy phi tuyến, biến thiên phân cấp) | `noise_model.py` | C2ST (balanced acc) ≈ 0.82–0.85 trên đường cắt giữ lại → VẪN phân biệt được thật/giả |
| Sinh dataset RCA có nhãn (nền thật + lỗi DDE), chia theo đường cắt | `rca_data.py`, `rca_gain_effect.json` | hệ số khuếch đại hiệu chuẩn theo độ lớn thay đổi: g_acc=7.7e-6, g_mic=0.0116 |
| RCA phân loại (7 thuật toán) + đường cong độ khó | `rca_run.py`, `rca_sweep.py` | GB 0.886, RF 0.870, QDA 0.787, NB 0.579 (top-1, đoán bừa 0.167) — CHỈ trong mô phỏng |
| RCA NHÂN QUẢ (BRCD, RCD, RCG, SmoothTraversal, BARO, SimpleRCA của repo) | `rca_causal.py` | xem mục 2 |

## 2. Kết quả RCA nhân quả (m=10, Top-1 trung bình trên 5 nguyên nhân; đoán bừa 0.07)
| Thuật toán | Cùng bộ sinh | Tham số DDE LỆCH |
|---|---|---|
| SmoothTraversal | 0.98 | 0.71 |
| BRCD | 0.90 | 0.58 |
| RCG | 0.90 | 0.59 |
| BARO | 0.94 | 0.35 |
| RCD | 0.80 | 0.39 |
| SimpleRCA | 0.52 | 0.44 |
Độ tin cậy: THẤP. Một cách chia, một hạt giống, không có khoảng tin cậy. Cột "cùng bộ sinh" bị vòng tròn (xem mục 3).

## 3. Vấn đề đã biết (cần nói thẳng trong paper)
1. **Vòng tròn / inverse crime:** nút rpm = nhân tố ẩn thật + nhiễu 0.5%; 4 nút còn lại là cảm biến mềm huấn luyện có giám sát trên nhãn thật của chính mô phỏng; đồ thị nhân quả do tác giả dựng khớp với cách mô phỏng tạo lỗi. Các số ~1.00 vì vậy KHÔNG phải bằng chứng.
2. **Không có dữ liệu thật có nhãn nguyên nhân gốc** được dùng. Ca thật có nhãn (dao hỏng nhãn 2: dataset 2, 3, 4, 26; ghi chú kẹp lỏng/độ sâu: dataset 7, 8, 9, 17, 18, 22, 33) đều KHÔNG phải dao A, mỗi ca 1–7 đường cắt (trừ dao hỏng). Tải dở dataset 3 (D*) và 14 (D khỏe) để kiểm chứng; chưa chạy.
3. **Nền nhiễu chưa đạt cổng C2ST** (≈0.82–0.85, cần ≈0.5). Không đo được mức đáy vì không có điều kiện cắt lặp lại; `ae` chỉ có 3 giá trị.
4. **Hiệu chỉnh DDE trực tiếp không hội tụ** (v2, v3, v4; loss kẹt ~3 độ lệch chuẩn). Đã bỏ hướng này; DDE chỉ còn dùng để tạo PHẦN CHÊNH LỆCH do lỗi, cộng lên nền học từ dữ liệu thật.
5. **Nhiều giả định tự đặt:** khoảng độ nặng lỗi, hệ số khuếch đại (hiệu chuẩn theo độ lớn thay đổi gia tốc +0.375, mic +4.9 dB), tham số DDE danh định, đồ thị nhân quả.
6. **Nhãn "bất thường" của MSM trong dataset 5/6 chủ yếu = phay nghịch (UP)**, không phải chatter hay nguyên nhân gốc; mô hình DDE không tái tạo được khác biệt thuận/nghịch (thiếu cơ chế cọ xát).
7. **Các con số C2ST báo trước khi sửa là sai mốc** (tỉ lệ giả:thật 5:1, mốc đoán bừa 0.83); đã sửa sang balanced accuracy trong `noise_model.py`, `c2st.py`. Mọi kết quả `*_v1`, `calibration_v2_*` không còn dùng.
8. **Kết quả cũ trong repo (benchmark 14 nút ~0.9–1.0) dùng cùng một SCM để huấn luyện và kiểm tra**; không so sánh được với phần này.
9. **Tài liệu tham khảo:** xác minh được RCD (Ikram 2022), BARO (Pham 2024), Orchard et al. 2025, Kennedy & O'Hagan 2001, Shrivastava 2017, Altintas 2020 (theo README). Người dùng xác nhận các nguồn BRCD/RCG/SimpleRCA là chuẩn; tôi chưa tự kiểm được.

## 4. Việc cần làm (ưu tiên)
1. Kiểm chứng THẬT dao hỏng: tải xong dataset 3 và 14, chạy `real_windows.py real_windows_defect.npz 3 14` rồi `real_defect_check.py` (so mẫu lệch thật với mẫu lệch mô phỏng từng nguyên nhân).
2. Chạy `rca_ablation.py` (3 hạt giống × gain ×0.3/1/3; nút "vật lý không nhãn" vs cảm biến mềm; đồ thị physics/shuffled/none) để gỡ vòng tròn và có khoảng tin cậy. Lần chạy bị dừng giữa chừng, chưa có kết quả.
3. Cải thiện nền nhiễu cho C2ST: thêm đường cắt (dataset 28, 30; file gia tốc 1.5 GB mỗi cái), biến điều kiện còn thiếu (hài trùng mode — giả thuyết CHƯA được chứng minh).
4. Thêm cơ chế cọ xát (ploughing) và/hoặc FRF đầu dao tham số hóa tốt hơn; kiểm tra độ nhạy với góc xoắn.
5. Kiểm nguồn BRCD/RCG/SimpleRCA và viết lại phần related work.
6. Dọn mã: `exp2_faults.py`/`exp4_rca.py` là bản cũ (mức tay đặt); `calibrate_cma.py`, `calibrate_v2/v3/v4.py` là các lần thử khớp DDE không hội tụ (giữ làm bằng chứng phủ định).
7. Thu hồi token Kaggle đã dán trong cuộc trò chuyện (Kaggle → Settings → API).

## 5. Tái lập nhanh
```
cd cutting_sim && export OPENBLAS_NUM_THREADS=1
python real_features.py 5 6 ; python real_windows.py 5 6
python rca_gain2.py   # hệ số khuếch đại theo độ lớn
python rca_causal.py  # RCA nhân quả (cần output/real_windows.npz)
```
