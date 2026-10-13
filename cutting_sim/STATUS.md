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

## 6. Nhật ký tiến độ khi tạm dừng (16:57, 6/10/2026)
- Bản thảo paper: `paper/main.pdf` (10 trang, biên dịch sạch; nhánh `cutting-sim-experiments`). Số liệu trong bảng sinh tự động từ JSON (`paper/make_tables.py`).
- Đang chạy nền lúc ghi (có thể bị dừng nếu máy/phiên đóng; kết quả sẽ nằm trong `cutting_sim/output/` nếu chạy xong):
  1. `rca_ablation.py` (3 hạt giống x gain 0.3/1/3; nút vật lý không nhãn vs cảm biến mềm; đồ thị physics/shuffled/none) -> `output/rca_ablation.json`, log `output/rca_ablation.log`. Dự kiến xong ~17:10.
  2. Tải dataset 14 (dao D khỏe) -> sau đó chuỗi `chain2.sh` tự chạy `real_windows.py real_windows_defect.npz 3 14` và `real_defect_check.py` -> `output/real_defect_check.json`. Dự kiến ~17:15-17:20.
- Nếu các tiến trình bị dừng: chạy lại thủ công theo mục 4 (việc cần làm) và mục 5 (tái lập). Dataset 3 (dao hỏng D*) đã tải đủ; dataset 14 cần tải lại file `*_acc.csv` (1.17 GB) nếu bị dở.
- Sau khi có kết quả: cập nhật mục Experiments/Limitations trong `paper/main.tex`, chạy `python paper/make_tables.py` rồi `pdflatex+bibtex` trong `paper/`.
- Chưa làm (không được nói đã làm trong paper): kiểm chứng nguyên nhân gốc thật, ablation (đang chạy), nguồn BRCD/RCG/SimpleRCA (cần thêm bản ghi thư mục), mẫu tạp chí đích.
- Việc bên ngoài: thu hồi token Kaggle đã dán trong cuộc trò chuyện.

## 7. Cập nhật sau khi chạy xong (7/10/2026)
- `rca_ablation.py` XONG: đồ thị physics/shuffled/none cho kết quả như nhau (đồ thị không có vai trò); nút vật lý không nhãn gần mức đoán bừa; gain x3 làm tệ hơn khi tham số DDE lệch. Xem `paper/main.tex` mục Ablations.
- `real_defect_check.py` XONG (dao D* hỏng vs D khỏe, thép 4140): mô phỏng KHÔNG tái tạo được dịch chuyển băng thấp (+3.9/+4.0 sd) và chỉ số phi đồng bộ (-3.7/-1.2 sd) của dao hỏng thật. Giả thuyết (chưa kiểm): thiếu cơ chế sứt rãnh (lực 1 lần/vòng).
- Việc còn lại chính: (a) mô hình hóa cơ chế dao sứt/không đối xứng rãnh và kiểm lại với D*; (b) thiết kế lại benchmark để nguyên nhân gốc không phải gốc đồ thị được ước lượng có giám sát (hiện benchmark không kiểm tra suy luận nhân quả); (c) nền nhiễu C2ST; (d) bản ghi thư mục BRCD/RCG/SimpleRCA; (e) chọn mẫu tạp chí.

## 8. Văn liệu và thí nghiệm khung A (8/10/2026)
- Văn liệu: `paper/lit/literature_map.md`, `paper/lit/verify_refs.py`, 12 nguồn mới đã xác minh trong `paper/refs.bib` (21 mục). Phát hiện: mô phỏng -> thật đã có tiền lệ cho chatter (Alberts 2024) và lỗi dao (Ebrahimi Araghizad 2023) => tính mới KHÔNG nằm ở "học trước bằng mô phỏng". Chưa xác định bài "tiền huấn luyện lực mô phỏng + tinh chỉnh ít mẫu cho mòn dao" (xem mục D của bản đồ).
- Thí nghiệm `sim_to_real_wear.py` (LUH, mòn dao VB>=100, 3 máy đích chưa thấy, đặc trưng lực x,y tương đối theo nền khỏe của từng dao):
  - Lần 1 (dùng lực trung bình có dấu): zero-shot dưới ngẫu nhiên (AUROC 0.55/0.26/0.10) do quy ước dấu trục lực; tinh chỉnh tệ hơn học từ đầu.
  - Lần 2 (dùng |lực trung bình|, thay đổi SAU khi thấy kết quả lần 1, có cơ sở bất biến dấu): zero-shot (không nhãn thật) AUROC M1 0.887, M2 0.807, M3 0.571; học từ k mẫu thật: 0.85-0.99; tinh chỉnh từ mô phỏng KHÔNG tốt hơn học từ đầu ở mọi k (k=2: 0.83-0.89 vs 0.85-0.89; k=32: 0.97-0.99 vs 0.99).
  - Kết luận sơ bộ: mô phỏng có giá trị trong chế độ không nhãn thật; chưa chứng minh lợi ích tinh chỉnh; nhiệm vụ này dễ (đặc trưng tương đối + vài nhãn đã đủ).
- File: `cutting_sim/output/sim_to_real_wear*.json`, `luh_transfer.json`, `quick_*.json`.

## 9. Kiểm tra cơ chế dao sứt rãnh (runout hiệu dụng) — KẾT QUẢ ÂM TÍNH (2026-10-07)
- Thêm lớp TOOL_DEFECT (runout 10–80 µm, hệ số khuếch đại g x1/x10/x100) vào `rca_data.dde_signals`; script `defect_mechanism_check.py`, kết quả `output/defect_mechanism_check.json`.
- Cosine với mẫu lệch thật D* chỉ ≈0.48–0.54, và phần lớn do thành phần chung (log10_ac, mic_dB đều tăng). Các nút phân biệt KHÔNG khớp:
  thật: chatter_acc −3.7, chatter_mic −1.2, acc_low +3.9, mic_low +4.0 ; mô phỏng: chatter ≈ +0.1..+0.9 (sai dấu), mic_low ≈ 0–0.9, acc_high tăng rất mạnh (+1..+38) mà thật chỉ +1.9.
- Kết luận: runout 1x KHÔNG tái tạo chữ ký dao hỏng thật (tăng năng lượng băng thấp, tín hiệu đồng bộ hơn). Không nên báo là đã mô hình hóa được dao sứt. Giả thuyết còn lại: tương tác lực cạnh/ma sát phi tuyến, hoặc dao hỏng đổi tải theo răng (Kt từng răng khác nhau) — chưa thử.

## 10. Nạp FRF đo (frf_import.py, test_frf_import.py) — mới có kiểm thử trên FRF tổng hợp
- Khớp mode bằng cách thêm dần mode tại đỉnh phần dư + variable projection cho (σ, ν); dừng khi sai số giảm <3%.
- Kiểm thử khép kín (FRF trục x của Altintas + nhiễu, dải ≤3500 Hz): sai số hình dạng FRF so với bản sạch 0.8% (không nhiễu), 1.7% (nhiễu 1% đỉnh), 16% (nhiễu 3%; mất mode 504 Hz). Mode chính 262/504/668/887/2201/2799 Hz được tìm đúng khi nhiễu ≤1%.
- Hạn chế đã biết: (i) không nhiễu thì thêm mode giả ngoài dải (≥3465 Hz) để hấp thụ đuôi; (ii) mới thử trục x, chưa thử y; (iii) là inverse crime, FRF búa thật (nhiễu, phi tuyến, đo gia tốc) chưa thử — chỉ nhận receptance hoặc accelerance (chia (jω)²), chưa có định dạng UFF/MAT.

## 11. Bước 1 kế hoạch: LUH leave-one-machine-out với baseline (luh_baselines.py -> output/luh_baselines.json)
- 10 hạt giống, bootstrap theo lần chạy (đánh giá thấp bất định). Baseline: ngưỡng RMS lực 1 đặc trưng, GB, LR, CORAL+LR, DANN (2 phương pháp cuối dùng đặc trưng máy đích không nhãn).
- Kết quả (đặc trưng thô, AUROC M1/M2/M3): RMS 0.996/0.993/0.462; GB 0.995/0.999/0.976; LR 0.996/0.995/0.943; CORAL 0.992/0.995/0.886; DANN 0.996/0.997/0.974.
- Hệ quả: nhiệm vụ mòn LUH quá dễ để thấy lợi ích của mô phỏng; zero-shot mô phỏng (0.887/0.807/0.571) thấp hơn quy tắc 1 đặc trưng ở M1, M2. DANN/CORAL không hơn GB. Đã sửa abstract/đóng góp/kết luận của paper theo đó.
- Khuyết điểm: CI bootstrap ở mức lần chạy; DANN ở dao-tương đối không ổn định (sd 0.13); chưa kiểm cột RMS bằng cách tách riêng.

## 12. Bước 1b: ngưỡng mòn sớm (sim_to_real_wear.py, THR=100/60/40 đặt trước khi chạy; output/sim_to_real_wear*_thr*.json)
- Label-free, AUROC (M1/M2/M3, TB): ngưỡng 100: sim 0.887/0.807/0.571 (0.755) vs quy tắc RMS(tool-rel) 0.853/0.936/0.433 (0.741); 60: sim 0.925/0.809/0.759 (0.831) vs 0.927/0.870/0.415 (0.737); 40: sim 0.827/0.893/0.782 (0.834) vs 0.994/0.941/0.398 (0.778).
- Lợi ích của sim chỉ đến từ máy 3 (quy tắc < ngẫu nhiên, chưa giải thích dấu); máy 1,2 quy tắc ≥ sim. Tinh chỉnh KHÔNG hơn học từ đầu ở mọi k, mọi ngưỡng. Kết luận: bằng chứng lợi ích mô phỏng yếu, chưa "chứng minh".

## 13. Thử hướng: chuyển giao thật có suy giảm khi động lực học đích xa không? (luh_dynamics_shift.py -> output/luh_dynamics_shift.json)
- Proxy động lực học = phổ lực 21 băng của 3 lần chạy mòn ít nhất mỗi dao (KHÔNG phải FRF). 9 dao đích, huấn luyện trên dao của máy khác (tất cả 6 / 3 gần nhất / 3 xa nhất).
- AUROC GB (raw): VB>=100 all 0.993, near3 0.984, far3 0.976; Spearman(khoảng cách, AUROC) = -0.16 (p=0.67). VB>=60: Spearman -0.34 (p=0.36). Chỉ LR tool-relative ở VB>=60 có xu hướng (rho=-0.62, p=0.08; AUROC thấp nhất 0.248).
- Kết luận: trên LUH KHÔNG thấy suy giảm có ý nghĩa; dữ liệu thật đủ cho nhiệm vụ này. Với n=9, proxy yếu và dải động lực học không biết, đây là bằng chứng KHÔNG-phát-hiện, không bác bỏ giả thuyết. Cần dữ liệu có FRF và nhiều cấu hình để kiểm chứng.

## 14. Hoàn thiện (7/10/2026)
- C2ST chạy lại 3 hạt giống, lưu JSON `output/noise_model_gate_nonlin{0,1}_seed{0,1,2}.json` (noise_model.py ghi file theo NONLIN/SEED); `paper/make_tables.py` đọc trực tiếp từ JSON. Biến thể cuối 0.820±0.036 tái tạo đúng; hai dòng tuyến tính cũ không tái tạo được (0.986/0.876 → 0.946/0.853).
- Hình pipeline `paper/figures/pipeline.png`; protocol thu dữ liệu `paper/DATA_PROTOCOL.md`; paper v2 16 trang biên dịch sạch; `paper/self_review.md` cập nhật.
- Chưa làm: bảng causal chính 3–5 hạt giống (ablation đã có 3 hạt giống cho cấu hình đồ thị vật lý), meta-DG, đọc 4 bài PDF ưu tiên (thư mục paper/lit/pdfs chưa tồn tại), thiết kế lại benchmark nhân quả, thông tin thư mục BRCD/RCG/SimpleRCA.

## 15. Bản v3 của paper, phát hiện chế độ vận hành, release (7/10/2026)
- PHÁT HIỆN: ở tham số danh định mô phỏng nằm trên giới hạn ổn định (ZOA a_lim 18–31 mm < ap 38.1 mm; x rms 0.4–0.8 mm; h_min=0) ở rpm 9000, ae 0.0635/0.3175 mm. Độ cứng x10: a_lim ~180–195 mm, x rms 5–8 µm. Lưu `output/regime_check.json` (`regime_check.py`). Hệ số gain 7.7e-6 bù biên độ lệch.
- Robustness: `KSCALE=10 TAG=_k10 rca_gain2.py/rca_causal.py` -> `rca_causal_k10.json`, `rca_gain_effect_k10.json` (gain 3.5e-4 / 0.43; lưới gain phải mở rộng). Top-1 m=10: cùng bộ sinh 0.53–0.98; lệch 0.32–0.62; BARO không còn sụp đổ (0.50).
- Thuật toán RCA trong repo là bản đơn giản hóa (BRCD: mô hình Gauss tuyến tính trên DAG cho sẵn + hằng số 0.12/0.40; RCD: tương quan riêng; RCG: dịch chuyển phần dư chuẩn hóa; Smooth Traversal: hằng số 0.70/0.05). BARO (bước chấm điểm) và SimpleRCA khớp mô tả của Lee et al. 2026 phụ lục C. Đã ghi vào paper (Bảng RCA algorithms).
- Thư mục gốc `references/30078_Root_Cause_Analysis_of_F.pdf` chính là bài BRCD (Lee, Zhou, Kocaoglu, ICML 2026; PMLR 306). Đã thêm refs.bib: lee2026brcd, ikram2025rcg (chép từ danh mục tham khảo của bài BRCD), fang2025simplerca (arXiv 2510.04711, đã kiểm API), lopezpaz2017c2st (arXiv 1610.06545, đã kiểm API). Ebrahimi 2023: đọc tóm tắt chính thức (94% mô phỏng, 93% đo thực), chưa đọc toàn văn.
- Paper v3: tác giả Hai-Dang Dang, Dinh Thieu Quang, Simon Park (VinUniversity; email ĐOÁN, đỏ); khối thuật toán, phương trình mô phỏng, bảng so sánh bài khác, ~25 hình/bảng sinh từ JSON (`paper/make_figures.py`, `make_tables.py`); số giả duy nhất: bảng/hình planned (đỏ). Hạ tầng release: LICENSE (MIT, đề xuất), requirements.txt, CITATION.cff, REPRODUCE.md, Makefile, skill dự án.
- Nơi nộp đề xuất: IEEE CASE 2027 (hạn 1/3/2027 theo trang IEEE RAS); định dạng IEEE chưa chuyển theo yêu cầu giữ mẫu hiện tại; quy định số trang năm 2027 chưa xác minh.

## 16. Bản v4 của paper (7/10/2026): viết lại theo cấu trúc yêu cầu và skill
- Bỏ ghi chú chữ đỏ và nơi nộp ở đầu bài; Abstract ngắn, ít số liệu, câu cuối là URL mã nguồn.
- Related Work đầy đủ (5 mục) + Bảng 1 tổng hợp khoảng trống C1–C6; Method mở đầu bằng sơ đồ kiến trúc lớn (trang nằm ngang, `paper/figures/f_architecture.png`, `make_figures.py::fig_architecture`), rồi Input / Simulation / RCA; Experiments: Setup (config, datasets, metrics, baselines), Main, Analysis, Ablation; cuối cùng Conclusion, Limitations, Future work.
- Bản trước giữ ở `paper/main_v3_article.tex`, `main_v2_article.tex`, `main_v1.tex`.

## 17. Prototype KG + BN + LLM (13/10/2026): `prototype_kg_bn_llm/`
- KG (5 nguyên nhân, 5 cơ chế, 9 đặc trưng, hành động kiểm tra; tier ASSUMED) -> 19 cạnh cause->feature; BN Gauss có điều kiện, hiệp phương sai chung, trọng số cạnh + CI bootstrap theo cut; LLM chỉ giải thích (kiểm tra bằng mã).
- Kết quả mô phỏng (5 nguyên nhân, chance 0.2, m=10): KG gốc top-1 0.38 (lệch 0.24); KG sau kiểm chứng 0.84 (0.44); đầy đủ 0.88 (0.45); xáo trộn 0.22. Chỉ dùng trọng số (matched filter) 0.38-0.44; PageRank ≈ ngẫu nhiên. Dao hỏng thật D*: BN tự tin STIFFNESS 1.00 nhưng kiểm tra ngoài mô hình d²=3472 (ngưỡng 44.3).
- Chưa gọi LLM thật (không có khóa; khóa OpenRouter sẽ do người dùng đặt trong .env, không dán vào chat). Chưa đánh giá chất lượng giải thích của LLM.
- Văn liệu: không tìm thấy KG riêng cho chatter; gần nhất Wehner et al. 2024 (arXiv:2402.00043, KG ràng buộc BN nhân quả, không LLM) và các bài LLM+KG cho CNC (xem LITERATURE.md).

## 18. Học liên tục qua 12 máy, truy hồi kiểu LightRAG, danh mục tài liệu (13/10/2026)
- `prototype_kg_bn_llm/continual.py`: quên tăng theo số máy học (giữ lại 0.55 → 0.44 → 0.40 với ghi đè; nền chung + lớp phủ 0.53–0.62); máy mới không có ca xác nhận không cơ chế nào giúp; từ k=5–10 nền+lớp phủ tốt nhất ở L=6 (0.446±0.018 vs 0.381 vs 0.357); nhãn giả trôi (0.26–0.27). Chi tiết trong README của prototype.
- `retrieval.py` + `documents/corpus_demo.json`: truy hồi kiểu LightRAG (không LLM) chỉ để lập luận; `documents/SOURCES.md` và `documents/DOWNLOAD_REQUEST.md` liệt kê tài liệu cần người dùng tải.

## 19. Ứng dụng web, bộ phát hiện, KG tự động (13/10/2026)
- `prototype_kg_bn_llm/app/` (FastAPI + HTML + Plotly): luồng thiết lập -> Start (10 cửa sổ học nền) -> Apply lỗi tăng dần -> CUSUM báo đèn đỏ + âm thanh -> xếp hạng + đồ thị nhân quả -> giải thích. Mô hình xếp hạng: Bayes Gauss có điều kiện trên 24 đặc trưng (`cond24.py`): top-1 0.795 trong bộ sinh (hiệu chuẩn tốt ECE 0.01), 0.47 khi lệch; 9 nút theo KG chỉ 0.48–0.58, nên KG không ràng buộc mô hình xếp hạng. Ma trận kiểm tra qua đúng đường chạy của ứng dụng: top-1 0.60; cản thấp khó nhất; mất cân bằng/độ đảo không bị gắn cờ ngoài mô hình.
- Phát hiện lỗi thiết kế: thanh severity của độ cứng/cản bị đảo nghĩa (đã sửa); phát hiện chuẩn hóa chung theo điều kiện làm sai lệch khi rút cùng điều kiện (đã sửa bằng mô hình điều kiện); độ chính xác 0.80/0.93 trước đó là do rút cửa sổ lẫn lộn điều kiện.
- `kg_build.py`: KG tự động từ 11 đoạn tài liệu (quy tắc), so với KG viết tay và với cạnh được dữ liệu ủng hộ. Chế độ LLM chưa chạy.
- OpenRouter: đã gọi thử thành công một lần (gemma-3-4b-it), đầu ra qua kiểm tra. Xem `OPEN_ITEMS.md`.
