# Bản đồ văn liệu (Phase 1) — chỉ gồm nguồn đã xác minh bằng chương trình

Phương pháp: tìm bằng WebSearch → khớp tiêu đề qua Crossref/arXiv (`verify_refs.py`, độ giống ≥0.97 trừ khi ghi chú) → lấy BibTeX qua DOI (`refs.bib`).
Ghi chú "nội dung" chỉ dựa trên tiêu đề, tóm tắt Crossref (nếu có) hoặc tóm tắt của công cụ tìm kiếm; CHƯA đọc toàn văn. Cần đọc toàn văn trước khi trích số liệu.

## A. Mô phỏng -> thực (huấn luyện trên dữ liệu mô phỏng, áp dụng cho dữ liệu thật)
| Khóa | Nguồn | Nội dung (mức đã xác nhận) | Liên hệ với bài của ta |
|---|---|---|---|
| `alberts2024sim2real` | Alberts, St. John, Odie, Khojandi, 2024, preprint (Preprints.org) | Tóm tắt Crossref: mô hình ML huấn luyện trên dữ liệu mô phỏng cho thấy triển vọng phát hiện chatter nhưng khả năng áp dụng thật chưa chắc do khác biệt mô phỏng-thật; kiểm chứng bằng cả mô phỏng và dữ liệu thật. Theo tóm tắt công cụ tìm kiếm (chưa kiểm): Random Forest, hơn 140 000 bộ mô phỏng, OMA/RCSA, học chuyển giao, 1 600 bộ thật, độ chính xác 86.1%. | Gần nhất với khung (A) cho chatter; ta mở rộng sang nhiều nguyên nhân và đo khoảng cách. [VERIFY số liệu] |
| `ebrahimi2023smartfaults` | Ebrahimi Araghizad, Tehranizadeh, Kilic, Budak, J. Machine Eng. 23(4), 2023 | Giám sát lỗi liên quan đến dao bằng ML huấn luyện trên mô phỏng quá trình (Random Forest); theo tóm tắt chính thức trên trang tạp chí (jmacheng.not.pl, đọc 7/10/2026): Random Forest huấn luyện trên dữ liệu mô phỏng giải tích; 94% độ chính xác dự đoán lỗi liên quan đến dao, 93% trên dữ liệu đo thực. CHƯA đọc toàn văn (loại lỗi cụ thể chưa rõ). | Tiền lệ trực tiếp cho "mô phỏng -> ML -> máy thật" với lỗi dao. |
| `yesilli2022tlchatter` | Yesilli et al., J. Manuf. Process. 2022 | Học chuyển giao cho phát hiện chatter tự động (tiêu đề). | Chuyển giao giữa điều kiện/dữ liệu cho chatter. |
| `han2025dtmech` | Han, Liang, Luo, Tian, J. Manuf. Process. 2025 | Digital twin có lõi mô phỏng cơ học động cho dự đoán mòn dao phay (tiêu đề). | Nền cho hướng mô phỏng + mòn dao. |
| `li2025physguided` | Li, Li, Zhu, Mech. Syst. Signal Process. 2025 | Mô hình học sâu có hướng dẫn vật lý cho giám sát mòn dao phay tốc độ cao (tiêu đề). | Cách đưa vật lý vào mô hình học. |
| `yang2022hybridwear` | Yang, Pattipati, Awasthi, Bollas, J. Manuf. Syst. 2022 | Phát hiện mòn dao trực tuyến lai dữ liệu + mô hình (tiêu đề). | Hướng lai. |
| `ma2025dtpdm` | Ma 2025, Computing in Civil Engineering 2024 | Digital twin trong bảo trì dự đoán, khả năng chuyển giao sim-to-real / real-to-sim (tiêu đề). | Khung chung sim-to-real. |
| `braun2026pinnphm` | Braun 2026, J. Intell. Manuf. | Tổng quan có hệ thống: học máy có hiểu biết vật lý trong PHM (tiêu đề). | Related work tổng quan. |

## B. Chuyển giao / tổng quát hóa giữa máy, giữa điều kiện (mòn dao)
| Khóa | Nguồn | Nội dung | Liên hệ |
|---|---|---|---|
| `wang2020datwear` | Wang et al., PHM Society 2020 | Học chuyển giao đối kháng miền cho dự đoán mòn dao tổng quát (tiêu đề). | Baseline thích nghi miền. |
| `li2023metadg` | Li, Fu, Zhuo, Liu, J. Manuf. Syst. 2023 | Tổng quát hóa miền meta bán giám sát đa nguồn cho trạng thái mòn dao khi đổi điều kiện cắt (tiêu đề; độ giống 0.81 do tiêu đề dài, DOI khớp). | Baseline meta-learning. |
| `hirsch2025singlesensor` | Hirsch et al., 2025, SSRN preprint (arXiv 2412.19950) | Dự đoán mòn dao phay bằng một cảm biến tích hợp quá trình (tiêu đề). | Có thể dùng chung bộ dữ liệu; cần đọc. |

## C. Bộ dữ liệu
| Khóa | Nguồn | Ghi chú |
|---|---|---|
| `kim2026msm` | Kim et al., Sci. Data 2026 | MSM (đã dùng). |
| `denkena2023luh` | Denkena, Klemme, Stiehl, Data in Brief 2023 | Bộ LUH: nhiều máy, nhiều dao, mòn VB (đã dùng một tập con). |

## D. Cái chưa xác định / chưa tìm đủ (không được trích dẫn như đã xác nhận)
- **Chưa xác định** bài mà tóm tắt công cụ tìm kiếm mô tả: hiệu chỉnh mô hình lực + mòn cơ học trên ít phép đo, sinh dữ liệu tổng hợp, tiền huấn luyện (ResNet + attention ngưỡng mềm + BiLSTM) trên lực mô phỏng rồi tinh chỉnh bằng ít mẫu thật cho mòn dao. Nếu bài này tồn tại đúng như mô tả thì nó là tiền lệ gần nhất cho khung (A) trên mòn dao. CẦN tìm và đọc.
- RCA nhân quả trong sản xuất/gia công: một vài kết quả tìm kiếm (hội nghị IFAC, Fraunhofer), chưa xác minh nên chưa đưa vào bib.
- BRCD (ICML 2026), RCG (UAI 2025), SimpleRCA (Fang 2025): nguồn do tác giả xác nhận, tôi chưa tự kiểm được.
- Tổng quan chẩn đoán lỗi/gốc nguyên nhân gia công bằng ML; ngẫu nhiên hóa miền trong gia công; phương pháp thích nghi miền tổng quát (CORAL, DANN…) — chưa tìm.

## E. Hệ quả cho định vị đóng góp (khung A)
1. "Huấn luyện trên mô phỏng rồi dùng cho thật" **đã có tiền lệ** cho chatter (`alberts2024sim2real`) và lỗi dao (`ebrahimi2023smartfaults`), và có vẻ cho mòn dao. Vì vậy **không thể** đặt tính mới ở việc "dùng mô phỏng để học trước".
2. Khoảng trống khả dĩ (chưa kiểm chứng rằng chưa ai làm; cần tìm kỹ hơn): (i) nhiều nguyên nhân gốc cùng lúc thay vì một loại lỗi; (ii) **độ bền khi tương tác máy–dao thay đổi**, với tham số có nguồn gốc rõ ràng; (iii) **đo khoảng cách mô phỏng–thật theo từng thành phần** (nền nhiễu, chữ ký lỗi) bằng dữ liệu công khai, kèm kết quả âm tính.
3. Baseline cần có: DANN (`wang2020datwear`), meta-DG (`li2023metadg`), mô hình chỉ học từ ít mẫu thật.
