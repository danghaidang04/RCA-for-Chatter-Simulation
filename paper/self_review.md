# Tự phản biện bản v2 (theo `research-paper-writing`, paper-review.md)

## Claim–evidence map
| Claim | Evidence | Status |
|---|---|---|
| Top-1 của 6 thuật toán RCA giảm 0.80–0.98 → 0.35–0.71 khi lệch tham số | Bảng causal, `output/rca_causal*.json` | supported (một lần chia/hạt giống; chỉ ablation có 3 hạt giống) |
| Đồ thị nhân quả không có vai trò trong benchmark | Bảng ablation, 3 hạt giống | supported |
| Nút không nhãn → gần ngẫu nhiên | Bảng ablation | supported |
| Nền khỏe mô phỏng vẫn phân biệt được (0.82±0.04) | Bảng C2ST | supported; đã chạy lại 7/10 với 3 hạt giống và lưu JSON (noise_model_gate_nonlin*_seed*.json), tái tạo 0.820±0.036; hai dòng tuyến tính cũ (0.986/0.876) không tái tạo được nên đã thay bằng 0.946/0.853 |
| Không lỗi mô phỏng nào (kể cả runout 1x) tái tạo dấu hiệu dao hỏng thật | real_defect_check.json, defect_mechanism_check.json | supported cho 1 dao hỏng khác loại dao mô phỏng |
| Mô phỏng-only zero-shot trên LUH: AUROC 0.89/0.81/0.57 | sim_to_real_wear_absmean.json | supported nhưng THĂM DÒ (đổi đặc trưng sau khi thấy kết quả lần 1) |
| Tinh chỉnh không tốt hơn học từ đầu | sim_to_real_wear*.json | supported trên nhiệm vụ dễ này |
| Chuẩn hóa theo nền khỏe không nâng AUROC chéo máy | luh_transfer.json | supported |
| FRF import khôi phục mode | test_frf_import.py | chỉ trên FRF tổng hợp (inverse crime) |
| Mô phỏng "giúp khi không có nhãn thật" | LUH zero-shot, 3 ngưỡng mòn | yếu: trung bình hơn quy tắc RMS ở cả 3 ngưỡng nhưng chỉ nhờ máy 3; máy 1, 2 quy tắc ≥ mô phỏng; tinh chỉnh không hơn học từ đầu |
| Chuyển giao thật không suy giảm khi động lực học đích xa | luh_dynamics_shift.json (9 đích, proxy phổ) | KHÔNG phát hiện suy giảm (rho −0.16, p 0.67); không bác bỏ H2 vì proxy yếu, n=9 |
| DANN/CORAL không hơn GB trên LUH | luh_baselines.json, 10 hạt giống | supported; CI theo lần chạy, đánh giá thấp bất định |

## Năm chiều (pass / needs revision / needs new experiment)
1. **Đóng góp**: tri thức mới = đo độ vòng tròn của benchmark + khoảng cách thành phần + bằng chứng âm tính. Rủi ro: bị coi là "toàn kết quả âm tính", thiếu phương pháp mới → *needs new experiment* (dữ liệu thật đa nguyên nhân là then chốt).
2. **Rõ ràng**: Method theo bộ ba motivation/design/advantage — *pass*; hình pipeline chưa vẽ (khung chữ) — *needs revision*.
3. **Mạnh thực nghiệm**: không có kết quả dương trên nguyên nhân gốc thật — *needs new experiment*; nhiều bảng chỉ 1 hạt giống — *needs revision* (chạy lại 3–5 hạt giống).
4. **Đầy đủ đánh giá**: thiếu baseline DANN/meta-DG trên LUH (có dữ liệu, có thể làm ngay) — *needs new experiment*; thiếu CI bootstrap cho LUH.
5. **Thiết kế phương pháp**: tham số "assumed" (mức độ lỗi, gain); dao hỏng thật khác loại dao mô phỏng; benchmark nhân quả chưa kiểm tra suy luận nhân quả — *needs revision* (thiết kế lại để nguyên nhân gốc không phải gốc đồ thị).

## Việc có thể làm ngay không cần dữ liệu mới
- Đã làm (7/10): baseline + DANN/CORAL trên LUH 10 hạt giống; C2ST lưu JSON 3 hạt giống; hình pipeline; DATA_PROTOCOL.md; thử dịch chuyển động lực học.
- Chưa làm: meta-DG (cần cài đặt riêng); chạy lại bảng causal chính nhiều hạt giống (số 3 hạt giống hiện chỉ có trong ablation); sửa tên 'chatter' của các chỉ số; benchmark nhân quả thiết kế lại để nguyên nhân không là gốc đồ thị.

## Cập nhật 7/10/2026 (bản v3)
| Claim | Evidence | Status |
|---|---|---|
| Mô phỏng danh định nằm trên giới hạn ổn định ở mọi điều kiện cắt thật | regime_check.json | supported (một tổ hợp rpm 9000, hai ae; không phải toàn dải) |
| Gain 7.7e-6 bù lệch biên độ; chế độ ổn định cần gain lớn hơn 45x/38x | rca_gain_effect_k10.json | supported |
| Sụt giảm khi lệch tham số vẫn còn ở chế độ ổn định (0.32–0.62 ở m=10) | rca_causal_k10.json | supported; graph ablation và so sánh dao hỏng chưa lặp lại ở chế độ này |
| Thuật toán RCA là bản đơn giản hóa | run_full_paper_benchmark.py so với phụ lục C của Lee et al. 2026 | supported (đối chiếu thủ công); CHƯA cài bản trung thành để so |
| Đồ thị không có vai trò | ablation 3 hạt giống | supported cho bản cài đặt đơn giản hóa; không suy ra cho thuật toán gốc |
| Bảng planned (tab:planned) và hình k-curve | KHÔNG có | giả, in đỏ, xem PLACEHOLDERS.md |
Việc còn mở: cài BRCD/RCD/RCG trung thành (mã tác giả có tại github.com/kenneth-lee-ch/brcd) để kiểm tra kết luận "đồ thị không có vai trò"; lặp lại ablation và dao hỏng ở chế độ ổn định; đọc toàn văn Ebrahimi 2023 và 4 bài ưu tiên.

## Tự phản biện bản v4 (theo references/paper-review.md của skill)
Cấu trúc mới: Abstract (ít số liệu, câu cuối là mã nguồn) → Introduction (bài toán, đã có gì, thách thức, giải pháp, đóng góp) → Related Work (5 mục + bảng tổng hợp khoảng trống C1–C6) → Method (sơ đồ kiến trúc lớn, Input / Simulation / RCA, mỗi module có motivation–design–advantage) → Experiments (Setup: configuration, datasets, metrics, baselines; Main; Analysis "mô phỏng đã đúng chưa"; Ablation) → Conclusion, Limitations, Future work.
| Chiều | Câu hỏi nguy cơ | Trạng thái |
|---|---|---|
| Đóng góp | Chỉ có kết quả âm tính, thiếu phương pháp mới? | needs new experiment: cần dữ liệu thật đa nguyên nhân (đã nêu rõ trong Future work) |
| Rõ ràng | Tái lập được không? | pass: phương trình, bảng cấu hình, REPRODUCE.md; chưa: mã nguồn thuật toán RCA gốc |
| Thực nghiệm | Có baseline mạnh/hiện đại? | needs revision: thiếu meta-DG, thiếu BRCD/RCD/RCG trung thành |
| Đánh giá | Ablation đủ? Dữ liệu đủ khó? | partial: ablation đồ thị/nút/gain/chuẩn hóa/ngưỡng/fine-tune đã có; LUH quá dễ; ablation chưa lặp ở chế độ ổn định |
| Thiết kế | Hợp lý? | needs revision: mô phỏng danh định trên giới hạn ổn định (đã báo cáo, đã kiểm tra độ bền ở độ cứng x10) |
Claim–evidence cập nhật: mọi con số trong Abstract (0.80–0.98 → 0.35–0.71) truy được tới `rca_causal.json`; "đồ thị không có vai trò" → `rca_ablation.json` (3 hạt giống) và chỉ đúng cho bản cài đặt đơn giản hóa; "không lỗi mô phỏng nào tái tạo dấu hiệu dao hỏng" → `real_defect_check.json`, `defect_mechanism_check.json` (một dao hỏng khác loại dao mô phỏng).
