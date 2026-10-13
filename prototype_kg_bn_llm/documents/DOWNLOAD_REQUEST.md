# Danh sách tài liệu mình KHÔNG tải/đọc được — nhờ bạn tải hộ (cập nhật 13/10/2026)

Cách làm: tải file, đặt đúng tên vào đúng thư mục ở cột "Lưu thành", rồi nhắn mình "xong". Mình sẽ dùng `pdftotext` đọc và cập nhật paper/prototype.
Thư mục `paper/lit/pdfs/` và `prototype_kg_bn_llm/documents/raw/` đã được thêm vào `.gitignore` (tài liệu có bản quyền, không đưa lên GitHub).
Cột "Truy cập": OA = truy cập mở (tải trực tiếp); TRẢ PHÍ = cần tài khoản thư viện VinUni (hoặc nhờ thầy); TRANG = lưu trang web thành PDF ("In -> Lưu PDF") hoặc "Lưu trang dạng HTML".
DOI đã được xác minh qua Crossref (tiêu đề và tác giả khớp) trừ khi ghi chú khác.

## A. Cần để chốt tính mới của bài nghiên cứu chính (ưu tiên cao nhất)

| ID | Công trình | Liên kết | Vì sao chưa đọc được | Truy cập | Lưu thành | Mình sẽ trích gì |
|---|---|---|---|---|---|---|
| A1 | Ebrahimi Araghizad, Tehranizadeh, Kilic, Budak — *Smart Tool-Related Faults Monitoring System Using Process Simulation-Based Machine Learning Algorithms*, J. Machine Engineering 23(4):18-32, 2023 | DOI 10.36897/jme/174018; trang: http://jmacheng.not.pl/Smart-Tool-Related-Faults-Monitoring-System-Using-Process-Simulation-Based-Machine,174018,0,2.html (có liên kết "Article (PDF)") | mới đọc được tóm tắt (HTTPS của máy chủ lỗi chứng chỉ) | OA | `paper/lit/pdfs/A1_ebrahimi2023.pdf` | loại lỗi, cách tạo dữ liệu mô phỏng, thiết lập thử nghiệm thật, số liệu 94%/93% |
| A2 | Deng, Wu, Miao et al. — *Meta-transfer learning for predicting position-dependent milling stability across varying overhang lengths ...*, J. Manuf. Processes, 2025 | DOI 10.1016/j.jmapro.2025.05.037 | trả phí/chặn | TRẢ PHÍ | `paper/lit/pdfs/A2_deng2025_jmp.pdf` | họ chuyển giao qua thay đổi nhô dao thế nào; có dùng mô phỏng không |
| A3 | Liu, Wang, Zhou et al. — *Physics-informed multi-task heterogeneous transfer learning network for robust milling tool wear and RUL prediction ...*, J. Manuf. Systems, 2026 | DOI 10.1016/j.jmsy.2025.12.011 | trả phí/chặn | TRẢ PHÍ | `paper/lit/pdfs/A3_liu2026_jms.pdf` | cách đưa vật lý vào, dữ liệu, chuyển giao giữa điều kiện/máy |
| A4 | Li, Yue, Wu et al. — *KDN: A class-added continual learning framework for cross-machine fault diagnosis with limited samples*, Mech. Syst. Signal Process., 2025 | DOI 10.1016/j.ymssp.2025.112379 | trả phí/chặn | TRẢ PHÍ | `paper/lit/pdfs/A4_li2025_kdn.pdf` | cơ chế học liên tục chéo máy (liên quan chống quên) |

## B. Liên quan KG + LLM + mạng Bayes (cho prototype và phần related work)

| ID | Công trình | Liên kết | Vì sao chưa đọc được | Truy cập | Lưu thành | Mình sẽ trích gì |
|---|---|---|---|---|---|---|
| B1 | Liu, Zhou, Liu et al. — *Intelligent Fault Diagnosis for CNC Through the Integration of Large Language Models and Domain Knowledge Graphs*, Engineering, 2025 | DOI 10.1016/j.eng.2025.04.003; https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2025.04.003 | trang lỗi khi truy cập tự động | TRẢ PHÍ hoặc OA (tạp chí Engineering thường mở; thử tải PDF từ trang) | `paper/lit/pdfs/B1_liu2025_engineering_cnc_kg_llm.pdf` | cách dựng KG, RAG, dữ liệu, độ chính xác |
| B2 | Wu, Mou, Gong et al. — *An automatic machine fault identification method using the knowledge graph-embedded large language model*, IJAMT, 2025 | DOI 10.1007/s00170-025-15555-2 (giấy phép CC BY) | chưa mở | OA | `paper/lit/pdfs/B2_wu2025_ijamt_kg_llm.pdf` | KG nhúng vào LLM; nguồn tạo cạnh; kết quả 97.50% (theo kết quả tìm kiếm) |
| B3 | Chen, Zhou, Zhang et al. — *Knowledge-Graph-Driven Fault Diagnosis Methods for Intelligent Production Lines*, Sensors, 2025 | DOI 10.3390/s25133912 (CC BY 4.0); https://www.mdpi.com/1424-8220/25/13/3912 (PMC chặn bot) | PMC trả trang xác minh trình duyệt | OA | `paper/lit/pdfs/B3_chen2025_sensors_kg.pdf` | ontology CNC, cách sinh cạnh lỗi-nguyên nhân-biện pháp |
| B4 | Qiu et al. — *A novel method for machine tool structure condition monitoring based on knowledge graph*, IJAMT 120:563-582, 2022 | DOI 10.1007/s00170-022-08757-5 | chỉ biết qua tổng quan | TRẢ PHÍ | `paper/lit/pdfs/B4_qiu2022_ijamt.pdf` | KG nhiều tầng cho vít me bi; cách so với nền khỏe |
| B5 | Bài Fraunhofer: mạng Bayes dựa FMEA + RAG trên KG hồ sơ 8D + LLM nhỏ | https://publica.fraunhofer.de/handle/publica/497663 | bị chặn (Anubis); chưa biết tiêu đề/tác giả | OA (thường) | `paper/lit/pdfs/B5_fraunhofer_rag_bn.pdf` | xác nhận tiêu đề/tác giả; kết hợp BN + KG + LLM |
| B6 | Cai et al. — KG chẩn đoán đa cấp kết hợp suy luận Bayes cho máy quay | chưa có tiêu đề (chỉ biết qua Hoang et al., ref [21]); bạn tra giúp từ danh mục tham khảo của arXiv:2506.13026 | chưa xác định được | ? | `paper/lit/pdfs/B6_cai_kg_bayes.pdf` | độ chính xác 91.1% khi thiếu cảm biến |
| B7 | OGPAL — *Fault Cause Identification across Manufacturing Lines through Ontology-Guided and Process-Aware FMEA Graph Learning with LLMs* | arXiv:2510.15428 (PDF: https://arxiv.org/pdf/2510.15428) | mình đọc bản HTML, chưa có PDF | OA | `paper/lit/pdfs/B7_ogpal_2510.15428.pdf` | ontology, prompt trích xuất, đánh giá |
| B8 | LightRAG (arXiv:2410.05779) và GraphRAG (arXiv:2404.16130) | https://arxiv.org/pdf/2410.05779 ; https://arxiv.org/pdf/2404.16130 | mình mới đọc HTML/tóm tắt | OA | `paper/lit/pdfs/B8_lightrag.pdf`, `B8_graphrag.pdf` | prompt trích xuất, cách hợp nhất cập nhật |

## C. Tài liệu nền để dựng KG (FMEA, cây máy, sổ tay hãng) — xem `SOURCES.md`

| ID | Tài liệu | Liên kết | Vì sao chưa đọc được | Truy cập | Lưu thành |
|---|---|---|---|---|---|
| C1 | McLeay & Turner, *Journal of Machine Engineering* 2011 — FMEA công đoạn "Milling Operation" (Bảng 1) | https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-f3b97e6a-b84e-45b0-8a49-ad3006ef0e47/c/9_MCLEAY.pdf (DOI chưa tra được trên Crossref) | trả 403 khi truy cập tự động | OA | `prototype_kg_bn_llm/documents/raw/C1_mcleay2011_fmea_milling.pdf` |
| C2 | FMEA máy phay CNC (PT. IJA Surabaya), preprint | DOI 10.21070/ups.8248; https://archive.umsida.ac.id/index.php/archive/preprint/download/8248/59291/65785 | chưa mở | OA | `.../raw/C2_umsida_fmea_cnc.pdf` |
| C3 | FTA + FMEA máy phay CNC HOLZHER, AFTMT X(1) 2023 | https://aftmt.uni-ruse.bg/wp-content/uploads/2024/10/AFTMT-X-1-2023-4.pdf | chưa mở | OA | `.../raw/C3_aftmt2023_fmea.pdf` |
| C4 | FTA + FMEA máy phay CNC, Univ. Riau | https://jom.unri.ac.id/index.php/JOMFTEKNIK/article/view/20170 | chưa mở | OA | `.../raw/C4_riau_fta_fmea.pdf` |
| C5 | MTConnect Part 2 Devices Information Model (bản mới nhất) và trang mô hình Component/Axis/Spindle | https://docs.mtconnect.org/MTConnect_Part_2-0_Devices_Information_Model_1-8-0.pdf ; https://model.mtconnect.org/Version2.6/DeviceInformationModel/Components/ComponentTypes/Axis | PDF chỉ ra mục lục; trang model là ứng dụng động | OA | `.../raw/C5_mtconnect_part2.pdf` |
| C6 | OPC UA 40501-1 Machine Tools (toàn bộ đặc tả) | https://reference.opcfoundation.org/specs/OPC-40501-1/full | chỉ thấy mô tả phạm vi | OA | `.../raw/C6_opcua_40501_1.html` (lưu trang) |
| C7 | Haas: tài liệu "Mill - spindle vibration - tolerances" (RD0071) và sổ tay dịch vụ VF | https://www.haascnc.com/content/haascnc/en/service/troubleshooting-and-how-to/reference-documents/mill---spindle-vibration---tolerances.html ; https://www.haascnc.com/content/dam/haascnc/en/service/manual/service/english---vf-series-service-manual---1997.pdf | 403 | OA (đăng nhập có thể cần) | `.../raw/C7_haas_spindle_vibration.pdf`, `C7_haas_vf_service.pdf` |
| C8 | Kennametal: Tech Tip giảm chatter khi phay đầu | https://www.kennametal.com/us/en/resources/blog/metal-cutting/reducing-chatter-and-vibration-in-end-milling.html | mới thấy trích dẫn trong kết quả tìm kiếm | TRANG | `.../raw/C8_kennametal_chatter.pdf` |
| C9 | Seco Tools: Reduce chatter for better machining | https://secotools.com/article/125930 | mới thấy mô tả | TRANG | `.../raw/C9_seco_chatter.pdf` |
| C10 | Sandvik Coromant: Troubleshooting milling (đã ĐỌC nhưng nên lưu bản gốc để trích đủ bảng Insert wear) | https://www.sandvik.coromant.com/en-us/knowledge/milling/troubleshooting-milling | đã đọc; cần bản lưu | TRANG | `.../raw/C10_sandvik_troubleshooting_milling.pdf` |

## D. Ngoài danh sách tài liệu (chỉ khi bạn muốn)
* Mã nguồn BRCD của tác giả (github.com/kenneth-lee-ch/brcd) để cài bản trung thành của BRCD/RCD/RCG — bạn tải (hoặc cho mình chạy `git clone` khi bạn đồng ý).
* Dữ liệu thật đa nguyên nhân kèm FRF: xem `paper/DATA_PROTOCOL.md`.

## Ghi chú pháp lý
Nhiều tài liệu hãng và tạp chí có bản quyền. Mình chỉ dùng để trích sự kiện/bộ ba có cấu trúc kèm trích dẫn ngắn và URL nguồn, không phân phối lại nguyên văn.
