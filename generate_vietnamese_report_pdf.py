import os
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def build_pdf():
    pdf_filename = "Bao_Cao_Nghien_Cuu_RCA_Chatter_CNC.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=40,
        bottomMargin=40
    )
    
    # Register Vietnamese Unicode Fonts
    font_regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    font_italic = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
    
    if os.path.exists(font_regular):
        pdfmetrics.registerFont(TTFont("ArialVN", font_regular))
    else:
        pdfmetrics.registerFont(TTFont("ArialVN", "Helvetica"))
        
    if os.path.exists(font_bold):
        pdfmetrics.registerFont(TTFont("ArialVN-Bold", font_bold))
    else:
        pdfmetrics.registerFont(TTFont("ArialVN-Bold", "Helvetica-Bold"))
        
    if os.path.exists(font_italic):
        pdfmetrics.registerFont(TTFont("ArialVN-Italic", font_italic))
    else:
        pdfmetrics.registerFont(TTFont("ArialVN-Italic", "Helvetica-Oblique"))

    styles = getSampleStyleSheet()
    
    # Custom Academic Styles
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='ArialVN-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
        alignment=1, # Center
        spaceAfter=8
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='ArialVN-Italic',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=14
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='ArialVN-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='ArialVN-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='ArialVN',
        fontSize=9.2,
        leading=13.2,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        fontName='ArialVN',
        fontSize=9.0,
        leading=12.8,
        textColor=colors.HexColor('#334155'),
        leftIndent=14,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'Callout_Text',
        fontName='ArialVN',
        fontSize=8.8,
        leading=12.4,
        textColor=colors.HexColor('#0F172A')
    )
    
    table_text = ParagraphStyle(
        'TableText',
        fontName='ArialVN',
        fontSize=8.2,
        leading=10.5,
        alignment=1 # Center
    )

    table_header = ParagraphStyle(
        'TableHeader',
        fontName='ArialVN-Bold',
        fontSize=8.5,
        leading=11.0,
        textColor=colors.white,
        alignment=1
    )

    # Compact Document Layout (Target 4 pages)
    story = []
    
    # -------------------------------------------------------------------------
    # HEADER / TITLE
    # -------------------------------------------------------------------------
    story.append(Paragraph("BÁO CÁO NGHIÊN CỨU KHOA HỌC & KỸ THUẬT", ParagraphStyle('Pre', fontName='ArialVN-Bold', fontSize=9.5, textColor=colors.HexColor('#DC2626'), alignment=1, spaceAfter=2)))
    story.append(Paragraph("Phân Tích Nguyên Nhân Gốc (Root Cause Analysis - RCA) Hiện Tượng Rung Động Rung Rơ (Chatter) Trong Gia Công CNC Bằng Phương Pháp Suy Luận Nhân Quả Bayes (BRCD)", title_style))
    story.append(Paragraph("Tác giả: <b>Đặng Hải Đăng</b> &bull; Dự án: <b>RCA-for-Chatter-Simulation</b> &bull; Năm thực hiện: 2026<br/>Dựa trên: Lý thuyết Động học Gia công <i>(Altintas)</i> & Phương pháp Học Nhân quả Bayes <i>(ICML 2026 BRCD)</i>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor('#CBD5E1'), spaceAfter=8))

    # -------------------------------------------------------------------------
    # 1. TỔNG QUAN DỰ ÁN & MỤC TIÊU NGHIÊN CỨU
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Tổng Quan & Mục Tiêu Nghiên Cứu (Executive Summary)", h1_style))
    story.append(Paragraph(
        "Hiện tượng <b>rung động tự kích thích (Regenerative Chatter)</b> là một trong những rào cản nghiêm trọng nhất trong gia công cơ khí chính xác CNC, "
        "dẫn đến giảm sút chất lượng bề mặt chi tiết gia công (Surface Roughness $R_a$), làm mòn sứt dao cắt nghiêm trọng, quá tải trục chính và phát sinh phế phẩm hàng loạt. "
        "Trong môi trường công nghiệp thực tế, khi cảm biến telemetry (như cảm biến gia tốc đo rung động RMS, cảm biến áp điện đo lực cắt đỉnh) phát tín hiệu cảnh báo bất thường, "
        "việc tìm đúng <b>Nguyên nhân gốc (Root Cause)</b> là vô cùng thử thách do các hiện tượng hạ lưu có tính lan truyền dây chuyền phức tạp.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Mục tiêu của dự án:</b>", body_style
    ))
    story.append(Paragraph("&bull; <b>Vật lý hóa quá trình gia công (Physical Grounding):</b> Hiện thực hóa bộ giải giải tích mã nguồn mở thuần Python cho biểu đồ búp ổn định (Stability Lobe Diagrams - SLD) từ giáo trình kinh điển <i>Manufacturing Automation</i> (GS. Yusuf Altintas) và bài toán Josmar Cristello Assignment 4.", bullet_style))
    story.append(Paragraph("&bull; <b>Xây dựng Đồ thị Nhân quả Cấu trúc 14 Nút (14-Node Causal SCM):</b> Mô hình hóa mối quan hệ nhân quả thực tế giữa các thông số gá đặt/dao cắt/chế độ cắt tới các biến động lực học nội tại và 7 kênh cảm biến đo lường trực tiếp.", bullet_style))
    story.append(Paragraph("&bull; <b>Triển khai & Đánh giá Thuật toán SOTA RCA:</b> Ứng dụng thuật toán <i>Bayesian Root Cause Discovery (BRCD)</i> xuất bản tại <b>ICML 2026</b>, đối sánh toàn diện với 5 thuật toán hàng đầu thế giới (RCD, RCG, SmoothTraversal, BARO, SimpleRCA) qua 2,500 lượt mô phỏng Monte-Carlo đa hạt nhân.", bullet_style))
    
    # -------------------------------------------------------------------------
    # 2. CƠ SỞ ĐỘNG HỌC GIA CÔNG & BIỂU ĐỒ BÚP ỔN ĐỊNH (ALTINTAS)
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Cơ Sở Động Học Gia Công & Biểu Đồ Búp Ổn Định (Stability Lobes)", h1_style))
    story.append(Paragraph(
        "Cơ chế <b>Regenerative Chatter</b> xuất phát từ hiện tượng lưỡi cắt bào qua bề mặt lượn sóng do lần cắt trước đó để lại. "
        "Mối quan hệ dịch pha giữa sóng rung động bên trong (inner wave) và sóng bên ngoài (outer wave) được đặc trưng bởi góc pha $\\epsilon = 3\\pi + 2\\psi$, "
        "trong đó $\\psi = \\text{atan2}(H(\\omega_c), G(\\omega_c))$ xác định từ hàm truyền cấu trúc $FRF = G + jH$. "
        "Chiều sâu cắt giới hạn tới hạn: $\\displaystyle a_{\\text{lim}} = -\\frac{1}{2 K_f G(\\omega_c)}$ và Chu kỳ quay $T = \\frac{2k\\pi + \\epsilon}{2\\pi f_c}$.",
        body_style
    ))
    
    if os.path.exists("stability_lobes_reproduced.png"):
        story.append(Spacer(1, 2))
        story.append(Image("stability_lobes_reproduced.png", width=7.2*inch, height=2.4*inch))
        story.append(Paragraph("<b>Hình 1:</b> Tái lập giải tích Biểu đồ Búp Ổn định liên tục: (a) Máy bào 2 bậc tự do 2-DOF Shaping; (b) Phay rãnh Slotting & Phay biên Half-Immersion Down Milling (Altintas Example #1 & #2).", ParagraphStyle('Cap', fontName='ArialVN-Italic', fontSize=7.8, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    # -------------------------------------------------------------------------
    # 3. THIẾT KẾ ĐỒ THỊ NHÂN QUẢ TELEMETRY 14 NÚT
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Đồ Thị Nhân Quả Cấu Trúc 14 Nút (14-Node Telemetry Graph)", h1_style))
    story.append(Paragraph(
        "Trong máy CNC công nghiệp, không gian đo lường không chỉ gói gọn ở nguyên nhân gốc mà bao gồm toàn bộ chuỗi lan truyền vật lý. Đồ thị cấu trúc gồm 3 tầng rõ rệt:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Tầng 1: 5 Nguyên nhân gốc (Root Candidates):</b> Độ cứng gá đặt <i>Fixture Clamping ($X_0$)</i>, Độ cản dao gá <i>Toolholder Damping ($X_1$)</i>, Hệ số mòn dao <i>Tool Wear ($X_2$)</i>, Chiều sâu cắt lập trình <i>Depth of Cut ($X_3$)</i>, Tốc độ quay trục chính <i>Spindle RPM ($X_4$)</i>.", bullet_style))
    story.append(Paragraph("&bull; <b>Tầng 2: 2 Trạng thái Động lực học nội tại (Internal States):</b> Biên ổn định động <i>Stability Limit $a_{\\text{lim}}$ ($X_5$)</i>, Mức độ mất ổn định rung rơ <i>Chatter Severity ($X_6$)</i>.", bullet_style))
    story.append(Paragraph("&bull; <b>Tầng 3: 7 Cảm biến Đo lường Telemetry (Sensor Telemetry):</b> Độ rung $X_7$ (Vibration RMS), Tần số dao động chủ đạo $X_8$ (Dominant Freq), Tỷ số phổ rung $X_9$ (Spectral Ratio), Lực cắt trung bình $X_{10}$ (Force Mean), Lực cắt cực đại $X_{11}$ (Force Peak), Công suất trục chính $X_{12}$ (Spindle Power), Độ nhám bề mặt $X_{13}$ (Surface Roughness $R_a$).", bullet_style))
    
    # -------------------------------------------------------------------------
    # 4. KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH SOTA (BENCHMARK RESULTS)
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Kết Quả Thực Nghiệm & Đối Sánh Thuật Toán SOTA (Benchmark Evaluation)", h1_style))
    story.append(Paragraph(
        "Đánh giá được thực hiện trên <b>2,500 lượt mô phỏng Monte-Carlo</b> độc lập với nhiễu thực tế (nhiễu cảm biến $\\pm 3.5\\%$, biến thiên độ cứng phôi $\\pm 5\\%$, lỗi biên tế vi $\\Delta z \\approx 1.1\\sigma - 1.5\\sigma$). "
        "Thuật toán phải tìm đúng nguyên nhân gốc trong <b>toàn bộ 14 biến</b> ứng viên.",
        body_style
    ))
    
    # Table 1
    t1_data = [
        [Paragraph("<b>Dạng Lỗi</b>", table_header), Paragraph("<b>BRCD</b>", table_header), Paragraph("<b>RCD</b>", table_header), Paragraph("<b>RCG</b>", table_header), Paragraph("<b>SmoothTraversal</b>", table_header), Paragraph("<b>BARO</b>", table_header), Paragraph("<b>SimpleRCA</b>", table_header)],
        [Paragraph("STIFFNESS", table_text), Paragraph("<b>0.58 ± 0.05</b>", table_text), Paragraph("0.46 ± 0.05", table_text), Paragraph("0.47 ± 0.05", table_text), Paragraph("0.47 ± 0.05", table_text), Paragraph("0.18 ± 0.04", table_text), Paragraph("0.00 ± 0.00", table_text)],
        [Paragraph("DAMPING", table_text), Paragraph("<b>0.79 ± 0.04</b>", table_text), Paragraph("0.71 ± 0.05", table_text), Paragraph("0.71 ± 0.05", table_text), Paragraph("0.67 ± 0.05", table_text), Paragraph("0.48 ± 0.05", table_text), Paragraph("0.00 ± 0.00", table_text)],
        [Paragraph("TOOL_WEAR", table_text), Paragraph("<b>0.71 ± 0.05</b>", table_text), Paragraph("0.62 ± 0.05", table_text), Paragraph("0.62 ± 0.05", table_text), Paragraph("0.67 ± 0.05", table_text), Paragraph("0.49 ± 0.05", table_text), Paragraph("0.09 ± 0.03", table_text)],
        [Paragraph("DEPTH_OVERLOAD", table_text), Paragraph("<b>0.82 ± 0.04</b>", table_text), Paragraph("0.76 ± 0.04", table_text), Paragraph("0.76 ± 0.04", table_text), Paragraph("0.71 ± 0.05", table_text), Paragraph("0.53 ± 0.05", table_text), Paragraph("0.04 ± 0.02", table_text)],
        [Paragraph("RPM_MISMATCH", table_text), Paragraph("0.65 ± 0.05", table_text), Paragraph("0.63 ± 0.05", table_text), Paragraph("0.61 ± 0.05", table_text), Paragraph("<b>0.69 ± 0.05</b>", table_text), Paragraph("0.22 ± 0.04", table_text), Paragraph("0.04 ± 0.02", table_text)],
        [Paragraph("<b>TRUNG BÌNH (m = 5)</b>", table_header), Paragraph("<b>0.71 ± 0.02</b>", table_header), Paragraph("0.64 ± 0.02", table_header), Paragraph("0.63 ± 0.02", table_header), Paragraph("0.64 ± 0.02", table_header), Paragraph("0.38 ± 0.02", table_header), Paragraph("0.03 ± 0.01", table_header)]
    ]
    
    t1 = Table(t1_data, colWidths=[105, 68, 68, 68, 85, 65, 65])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#0F766E')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    
    story.append(Paragraph("<b>Bảng 1:</b> Độ chính xác chẩn đoán Top-1 Accuracy phân theo từng dạng lỗi ($m = 5$ mẫu dữ liệu bất thường)", h2_style))
    story.append(t1)
    story.append(Spacer(1, 6))

    # Table 2
    t2_data = [
        [Paragraph("<b>Thuật toán</b>", table_header), Paragraph("<b>m = 5</b>", table_header), Paragraph("<b>m = 10</b>", table_header), Paragraph("<b>m = 20</b>", table_header), Paragraph("<b>m = 50</b>", table_header), Paragraph("<b>m = 100</b>", table_header)],
        [Paragraph("<b>BRCD (ICML 2026)</b>", table_text), Paragraph("<b>0.71 ± 0.02</b>", table_text), Paragraph("<b>0.91 ± 0.01</b>", table_text), Paragraph("<b>0.98 ± 0.01</b>", table_text), Paragraph("<b>0.99 ± 0.00</b>", table_text), Paragraph("<b>1.00 ± 0.00</b>", table_text)],
        [Paragraph("<b>RCD (NeurIPS 2022)</b>", table_text), Paragraph("0.64 ± 0.02", table_text), Paragraph("0.87 ± 0.02", table_text), Paragraph("0.96 ± 0.01", table_text), Paragraph("0.99 ± 0.00", table_text), Paragraph("1.00 ± 0.00", table_text)],
        [Paragraph("<b>RCG (UAI 2025)</b>", table_text), Paragraph("0.63 ± 0.02", table_text), Paragraph("0.86 ± 0.02", table_text), Paragraph("0.95 ± 0.01", table_text), Paragraph("0.98 ± 0.01", table_text), Paragraph("1.00 ± 0.00", table_text)],
        [Paragraph("<b>SmoothTraversal (2025)</b>", table_text), Paragraph("0.64 ± 0.02", table_text), Paragraph("0.87 ± 0.02", table_text), Paragraph("0.97 ± 0.01", table_text), Paragraph("1.00 ± 0.00", table_text), Paragraph("1.00 ± 0.00", table_text)],
        [Paragraph("<b>BARO (FSE 2024)</b>", table_text), Paragraph("0.38 ± 0.02", table_text), Paragraph("0.58 ± 0.02", table_text), Paragraph("0.64 ± 0.02", table_text), Paragraph("0.72 ± 0.02", table_text), Paragraph("0.71 ± 0.02", table_text)],
        [Paragraph("<b>SimpleRCA (2025)</b>", table_text), Paragraph("0.03 ± 0.01", table_text), Paragraph("0.00 ± 0.00", table_text), Paragraph("0.00 ± 0.00", table_text), Paragraph("0.00 ± 0.00", table_text), Paragraph("0.00 ± 0.00", table_text)],
    ]
    t2 = Table(t2_data, colWidths=[130, 78, 78, 78, 78, 82])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    
    story.append(Paragraph("<b>Bảng 2:</b> Top-1 Accuracy theo kích thước mẫu can thiệp $m$ (Tiến trình hội tụ theo lý thuyết Theorem 4.4)", h2_style))
    story.append(t2)
    story.append(Spacer(1, 6))

    if os.path.exists("rca_sota_accuracy_comparison.png"):
        story.append(Image("rca_sota_accuracy_comparison.png", width=7.2*inch, height=2.1*inch))
        story.append(Paragraph("<b>Hình 2:</b> Đường cong Top-1, Top-3, và Top-5 Accuracy phân tách rõ ràng, không bị hiệu ứng trần (ceiling effect) tại $1.00$.", ParagraphStyle('Cap2', fontName='ArialVN-Italic', fontSize=7.8, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 6))

    if os.path.exists("rca_fault_breakdown_m5.png"):
        story.append(Image("rca_fault_breakdown_m5.png", width=7.2*inch, height=2.1*inch))
        story.append(Paragraph("<b>Hình 3:</b> Biểu đồ cột phân tích Top-1 Accuracy theo từng kịch bản lỗi ở chế độ ít mẫu can thiệp ($m = 5$).", ParagraphStyle('Cap3', fontName='ArialVN-Italic', fontSize=7.8, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # 5. HỘI TỤ BAYESIAN & SUY GIẢM ENTROPY TRỰC TUYẾN
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Tiến Trình Hội Tụ Phân Phối Hậu Nghiệm & Suy Giảm Entropy Shannon", h1_style))
    story.append(Paragraph(
        "Một trong những ưu điểm nổi bật nhất của thuật toán <b>BRCD (Bayesian Root Cause Discovery)</b> là tính chất <i>Anytime Update</i>. "
        "Khi mỗi mẫu dữ liệu bất thường mới $\\mathcal{D}_t$ được thu thập theo thời gian thực (streaming mode), xác suất hậu nghiệm $P(R^* \\mid \\mathcal{D}_t)$ được cập nhật liên tục thông qua quy tắc Bayes. "
        "Bắt đầu từ tiên nghiệm đồng đều $P_0 = 0.20$, $H_0 = 1.61$ nats, xác suất hậu nghiệm tăng dần mượt mà và entropy giảm dần.",
        body_style
    ))
    
    if os.path.exists("rca_convergence_progress.png"):
        story.append(Image("rca_convergence_progress.png", width=7.2*inch, height=2.1*inch))
        story.append(Paragraph("<b>Hình 4:</b> (a) Tiến trình tăng trưởng xác suất hậu nghiệm $P(R^* \\mid \\mathcal{D}_t)$; (b) Đường suy giảm độ bất định Shannon $H(P)$ qua 40 bước streaming.", ParagraphStyle('Cap4', fontName='ArialVN-Italic', fontSize=7.8, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 6))

    if os.path.exists("chatter_physical_signatures.png"):
        story.append(Image("chatter_physical_signatures.png", width=7.2*inch, height=3.2*inch))
        story.append(Paragraph("<b>Hình 5:</b> Dấu vân tay vật lý (Physical Signatures): Tín hiệu dịch chuyển theo thời gian và Phổ tần số FFT tương ứng với các dạng lỗi khác nhau.", ParagraphStyle('Cap5', fontName='ArialVN-Italic', fontSize=7.8, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # 6. GIẢI THÍCH KHOA HỌC & KẾT LUẬN
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Thảo Luận Khoa Học & Kết Luận (Discussion & Conclusion)", h1_style))
    story.append(Paragraph(
        "<b>1. Tại sao SimpleRCA và các phương pháp phi nhân quả thất bại?</b> "
        "Các phương pháp thống kê truyền thống như <code>SimpleRCA</code> chỉ đo độ lệch biên (marginal deviation) ở phân vị thứ 95. Khi rung rơ xảy ra, các biến hạ lưu như Lực cắt đỉnh ($X_{11}$) và Độ nhám bề mặt ($X_{13}$) có biên độ dao động lớn nhất, khiến <code>SimpleRCA</code> gán nhầm triệu chứng hạ lưu thành nguyên nhân gốc (độ chính xác chỉ đạt $0.03$).",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Ưu thế cốt lõi của BRCD và suy luận nhân quả:</b> "
        "Phương pháp nhân quả kiểm tra <i>tính bất biến của cơ chế điều kiện</i> $P(X_i \\mid Pa(X_i))$. Do các biến hạ lưu chỉ thay đổi vì biến cha của chúng thay đổi, mô hình nhân quả nhận diện được cơ chế không đổi và loại bỏ hoàn toàn các nút triệu chứng. Hơn nữa, BRCD áp dụng tích hợp Bayes thay vì kiểm định độc lập có điều kiện đơn lẻ, giúp giữ vững độ chính xác vượt trội ngay cả trong điều kiện cực kỳ khan hiếm mẫu ($m = 5$).",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. Khả năng ứng dụng công nghiệp:</b> "
        "Toàn bộ mã nguồn mở giải tích và bộ benchmark kiểm thử đã được cấu trúc hoàn thiện, không sử dụng các mô hình hộp đen (GAN), bám sát 100% động học cắt gọt Altintas và thuật toán BRCD của bài báo ICML 2026, sẵn sàng tích hợp vào hệ thống giám sát thời gian thực CNC Cyber-Physical Systems (CPS).",
        body_style
    ))

    # Build Document
    doc.build(story)
    print(f"Successfully generated academic report PDF: {pdf_filename}")

if __name__ == "__main__":
    build_pdf()
