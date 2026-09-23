import os
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

# -------------------------------------------------------------------------
# Helper function: Render Crisp LaTeX Equation PNG using Matplotlib
# -------------------------------------------------------------------------
def render_latex_equation(latex_str, filepath, fontsize=13, color="#1E3A8A"):
    plt.figure(figsize=(0.01, 0.01))
    plt.text(0.5, 0.5, f"${latex_str}$", fontsize=fontsize, color=color,
             horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=350, bbox_inches='tight', pad_inches=0.04, transparent=True)
    plt.close()
    
    # Return width and height in points for ReportLab
    with PILImage.open(filepath) as img:
        w_px, h_px = img.size
        # at 350 DPI, 1 pt = 350 / 72 pixels
        scale = 72.0 / 350.0
        return w_px * scale, h_px * scale

def build_pdf():
    pdf_filename = "Bao_Cao_Nghien_Cuu_RCA_Chatter_CNC.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
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
        fontSize=17,
        leading=21,
        textColor=colors.HexColor('#0F172A'),
        alignment=1, # Center
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='ArialVN-Italic',
        fontSize=10,
        leading=13.5,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='ArialVN-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='ArialVN-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='ArialVN',
        fontSize=9.0,
        leading=12.6,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        fontName='ArialVN',
        fontSize=8.8,
        leading=12.2,
        textColor=colors.HexColor('#334155'),
        leftIndent=12,
        spaceAfter=2.5
    )

    table_text = ParagraphStyle(
        'TableText',
        fontName='ArialVN',
        fontSize=8.0,
        leading=10.0,
        alignment=1 # Center
    )

    table_header = ParagraphStyle(
        'TableHeader',
        fontName='ArialVN-Bold',
        fontSize=8.2,
        leading=10.5,
        textColor=colors.white,
        alignment=1
    )

    # Pre-render LaTeX Equation Images
    eq_dir = "math_formulas"
    w1, h1 = render_latex_equation(r"a_{\mathrm{lim}} = -\frac{1}{2 K_f G(\omega_c)}, \quad T = \frac{2k\pi + \epsilon}{2\pi f_c}, \quad N = \frac{60}{N_{\mathrm{teeth}} T}, \quad \epsilon = 3\pi + 2\psi", f"{eq_dir}/eq_chatter.png", fontsize=11.5)
    w2, h2 = render_latex_equation(r"\Phi_y(s) = \frac{\omega_{n1}^2 \cos^2(\theta_1)}{k_1 (s^2 + 2\zeta_1 \omega_{n1} s + \omega_{n1}^2)} + \frac{\omega_{n2}^2 \cos^2(\theta_2)}{k_2 (s^2 + 2\zeta_2 \omega_{n2} s + \omega_{n2}^2)}", f"{eq_dir}/eq_tf.png", fontsize=11.0)
    w3, h3 = render_latex_equation(r"p(R \vert \mathcal{D}) = \frac{p(\mathcal{D} \vert R) p(R)}{\sum_{R^{\prime}} p(\mathcal{D} \vert R^{\prime}) p(R^{\prime})}, \quad p(\mathcal{D} \vert R) = \sum_{G \in [\mathcal{G}^*]} p(\mathcal{D} \vert G, R) p(G \vert R)", f"{eq_dir}/eq_brcd.png", fontsize=11.5)
    w4, h4 = render_latex_equation(r"H(P) = -\sum_{i=1}^n p(R_i \vert \mathcal{D}_t) \ln p(R_i \vert \mathcal{D}_t)", f"{eq_dir}/eq_entropy.png", fontsize=11.0)
    w5, h5 = render_latex_equation(r"p(G^*, R^* \vert \mathcal{D}) \geq 1 - M \mathrm{exp}\left\{-n \left(\Delta_{\mathrm{min}}^{\mathrm{eff}}(n) - t_n\right)\right\} \max_{(G,R) \neq (G^*,R^*)} \frac{p(G,R)}{p(G^*,R^*)}", f"{eq_dir}/eq_bound.png", fontsize=11.0)

    story = []
    
    # -------------------------------------------------------------------------
    # HEADER / TITLE
    # -------------------------------------------------------------------------
    story.append(Paragraph("BÁO CÁO NGHIÊN CỨU KHOA HỌC & KỸ THUẬT", ParagraphStyle('Pre', fontName='ArialVN-Bold', fontSize=9.0, textColor=colors.HexColor('#DC2626'), alignment=1, spaceAfter=2)))
    story.append(Paragraph("Phân Tích Nguyên Nhân Gốc (Root Cause Analysis - RCA) Hiện Tượng Rung Động Rung Rơ (Chatter) Trong Gia Công CNC Bằng Phương Pháp Suy Luận Nhân Quả Bayes (BRCD)", title_style))
    story.append(Paragraph("Tác giả: <b>Đặng Hải Đăng</b> &bull; Dự án: <b>RCA-for-Chatter-Simulation</b> &bull; Năm thực hiện: 2026<br/>Dựa trên: Lý thuyết Động học Gia công <i>(Altintas)</i> & Phương pháp Học Nhân quả Bayes <i>(ICML 2026 BRCD)</i>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor('#CBD5E1'), spaceAfter=6))

    # -------------------------------------------------------------------------
    # 1. TỔNG QUAN DỰ ÁN & MỤC TIÊU NGHIÊN CỨU
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Tổng Quan & Mục Tiêu Nghiên Cứu (Executive Summary)", h1_style))
    story.append(Paragraph(
        "Hiện tượng <b>rung động tự kích thích (Regenerative Chatter)</b> là một trong những rào cản nghiêm trọng nhất trong gia công cơ khí chính xác CNC, "
        "dẫn đến giảm sút chất lượng bề mặt chi tiết gia công (Surface Roughness <i>R<sub>a</sub></i>), làm mòn sứt dao cắt nghiêm trọng, quá tải trục chính và phát sinh phế phẩm hàng loạt. "
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
        "Mối quan hệ dịch pha giữa sóng rung động bên trong (inner wave) và sóng bên ngoài (outer wave) được đặc trưng bởi góc pha <i>&epsilon;</i> = 3&pi; + 2<i>&psi;</i>, "
        "trong đó <i>&psi;</i> = atan2(<i>H</i>(<i>&omega;<sub>c</sub></i>), <i>G</i>(<i>&omega;<sub>c</sub></i>)) xác định từ hàm truyền cấu trúc <i>FRF</i> = <i>G</i> + <i>jH</i>.",
        body_style
    ))
    
    # Equation 1 & 2
    story.append(Spacer(1, 2))
    story.append(Table([[Image(f"{eq_dir}/eq_tf.png", width=w2*0.9, height=h2*0.9)]], colWidths=[520], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 2))
    story.append(Table([[Image(f"{eq_dir}/eq_chatter.png", width=w1*0.85, height=h1*0.85)]], colWidths=[520], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 2))

    if os.path.exists("stability_lobes_reproduced.png"):
        story.append(Image("stability_lobes_reproduced.png", width=7.2*inch, height=2.3*inch))
        story.append(Paragraph("<b>Hình 1:</b> Tái lập giải tích Biểu đồ Búp Ổn định liên tục: (a) Máy bào 2 bậc tự do 2-DOF Shaping; (b) Phay rãnh Slotting & Phay biên Half-Immersion Down Milling (Altintas Example #1 & #2).", ParagraphStyle('Cap', fontName='ArialVN-Italic', fontSize=7.6, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    # -------------------------------------------------------------------------
    # 3. THIẾT KẾ ĐỒ THỊ NHÂN QUẢ TELEMETRY 14 NÚT
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Đồ Thị Nhân Quả Cấu Trúc 14 Nút (14-Node Telemetry Graph)", h1_style))
    story.append(Paragraph(
        "Trong máy CNC công nghiệp, không gian đo lường không chỉ gói gọn ở nguyên nhân gốc mà bao gồm toàn bộ chuỗi lan truyền vật lý. Đồ thị cấu trúc gồm 3 tầng rõ rệt:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Tầng 1: 5 Nguyên nhân gốc (Root Candidates):</b> Độ cứng gá đặt <i>Fixture Clamping (X<sub>0</sub>)</i>, Độ cản dao gá <i>Toolholder Damping (X<sub>1</sub>)</i>, Hệ số mòn dao <i>Tool Wear (X<sub>2</sub>)</i>, Chiều sâu cắt lập trình <i>Depth of Cut (X<sub>3</sub>)</i>, Tốc độ quay trục chính <i>Spindle RPM (X<sub>4</sub>)</i>.", bullet_style))
    story.append(Paragraph("&bull; <b>Tầng 2: 2 Trạng thái Động lực học nội tại (Internal States):</b> Biên ổn định động <i>Stability Limit a<sub>lim</sub> (X<sub>5</sub>)</i>, Mức độ mất ổn định rung rơ <i>Chatter Severity (X<sub>6</sub>)</i>.", bullet_style))
    story.append(Paragraph("&bull; <b>Tầng 3: 7 Cảm biến Đo lường Telemetry (Sensor Telemetry):</b> Độ rung <i>X<sub>7</sub></i> (Vibration RMS), Tần số dao động chủ đạo <i>X<sub>8</sub></i> (Dominant Freq), Tỷ số phổ rung <i>X<sub>9</sub></i> (Spectral Ratio), Lực cắt trung bình <i>X<sub>10</sub></i> (Force Mean), Lực cắt cực đại <i>X<sub>11</sub></i> (Force Peak), Công suất trục chính <i>X<sub>12</sub></i> (Spindle Power), Độ nhám bề mặt <i>X<sub>13</sub></i> (Surface Roughness <i>R<sub>a</sub></i>).", bullet_style))
    
    # -------------------------------------------------------------------------
    # 4. KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH SOTA (BENCHMARK RESULTS)
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Kết Quả Thực Nghiệm & Đối Sánh Thuật Toán SOTA (Benchmark Evaluation)", h1_style))
    story.append(Paragraph(
        "Đánh giá được thực hiện trên <b>2,500 lượt mô phỏng Monte-Carlo</b> độc lập với nhiễu thực tế (nhiễu cảm biến &plusmn; 3.5%, biến thiên độ cứng phôi &plusmn; 5%, lỗi biên tế vi &Delta;<i>z</i> &approx; 1.1&sigma; - 1.5&sigma;). "
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
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    
    story.append(Paragraph("<b>Bảng 1:</b> Độ chính xác chẩn đoán Top-1 Accuracy phân theo từng dạng lỗi (<i>m</i> = 5 mẫu dữ liệu bất thường)", h2_style))
    story.append(t1)
    story.append(Spacer(1, 4))

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
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    
    story.append(Paragraph("<b>Bảng 2:</b> Top-1 Accuracy theo kích thước mẫu can thiệp <i>m</i> (Tiến trình hội tụ theo lý thuyết Theorem 4.4)", h2_style))
    story.append(t2)
    story.append(Spacer(1, 4))

    # Bound equation
    story.append(Table([[Image(f"{eq_dir}/eq_bound.png", width=w5*0.82, height=h5*0.82)]], colWidths=[520], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 4))

    if os.path.exists("rca_sota_accuracy_comparison.png"):
        story.append(Image("rca_sota_accuracy_comparison.png", width=7.2*inch, height=2.0*inch))
        story.append(Paragraph("<b>Hình 2:</b> Đường cong Top-1, Top-3, và Top-5 Accuracy phân tách rõ ràng, không bị hiệu ứng trần (ceiling effect) tại 1.00.", ParagraphStyle('Cap2', fontName='ArialVN-Italic', fontSize=7.6, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    if os.path.exists("rca_fault_breakdown_m5.png"):
        story.append(Image("rca_fault_breakdown_m5.png", width=7.2*inch, height=2.0*inch))
        story.append(Paragraph("<b>Hình 3:</b> Biểu đồ cột phân tích Top-1 Accuracy theo từng kịch bản lỗi ở chế độ ít mẫu can thiệp (<i>m</i> = 5).", ParagraphStyle('Cap3', fontName='ArialVN-Italic', fontSize=7.6, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    # -------------------------------------------------------------------------
    # 5. HỘI TỤ BAYESIAN & SUY GIẢM ENTROPY TRỰC TUYẾN
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Tiến Trình Hội Tụ Phân Phối Hậu Nghiệm & Suy Giảm Entropy Shannon", h1_style))
    story.append(Paragraph(
        "Một trong những ưu điểm nổi bật nhất của thuật toán <b>BRCD (Bayesian Root Cause Discovery)</b> là tính chất <i>Anytime Update</i>. "
        "Khi mỗi mẫu dữ liệu bất thường mới <b>D</b><sub><i>t</i></sub> được thu thập theo thời gian thực (streaming mode), xác suất hậu nghiệm <i>P</i>(<i>R</i><sup>*</sup> | <b>D</b><sub><i>t</i></sub>) được cập nhật liên tục thông qua quy tắc Bayes.",
        body_style
    ))
    
    # BRCD & Entropy Equation
    story.append(Spacer(1, 2))
    story.append(Table([[Image(f"{eq_dir}/eq_brcd.png", width=w3*0.82, height=h3*0.82)]], colWidths=[520], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 2))
    story.append(Table([[Image(f"{eq_dir}/eq_entropy.png", width=w4*0.82, height=h4*0.82)]], colWidths=[520], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 2))
    
    if os.path.exists("rca_convergence_progress.png"):
        story.append(Image("rca_convergence_progress.png", width=7.2*inch, height=2.0*inch))
        story.append(Paragraph("<b>Hình 4:</b> (a) Tiến trình tăng trưởng xác suất hậu nghiệm <i>P</i>(<i>R</i><sup>*</sup> | <b>D</b><sub><i>t</i></sub>); (b) Đường suy giảm độ bất định Shannon <i>H</i>(<i>P</i>) qua 40 bước streaming.", ParagraphStyle('Cap4', fontName='ArialVN-Italic', fontSize=7.6, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    if os.path.exists("chatter_physical_signatures.png"):
        story.append(Image("chatter_physical_signatures.png", width=7.2*inch, height=3.0*inch))
        story.append(Paragraph("<b>Hình 5:</b> Dấu vân tay vật lý (Physical Signatures): Tín hiệu dịch chuyển theo thời gian và Phổ tần số FFT tương ứng với các dạng lỗi khác nhau.", ParagraphStyle('Cap5', fontName='ArialVN-Italic', fontSize=7.6, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    # -------------------------------------------------------------------------
    # 6. GIẢI THÍCH KHOA HỌC & KẾT LUẬN
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Thảo Luận Khoa Học & Kết Luận (Discussion & Conclusion)", h1_style))
    story.append(Paragraph(
        "<b>1. Tại sao SimpleRCA và các phương pháp phi nhân quả thất bại?</b> "
        "Các phương pháp thống kê truyền thống như <code>SimpleRCA</code> chỉ đo độ lệch biên (marginal deviation) ở phân vị thứ 95. Khi rung rơ xảy ra, các biến hạ lưu như Lực cắt đỉnh (<i>X<sub>11</sub></i>) và Độ nhám bề mặt (<i>X<sub>13</sub></i>) có biên độ dao động lớn nhất, khiến <code>SimpleRCA</code> gán nhầm triệu chứng hạ lưu thành nguyên nhân gốc (độ chính xác chỉ đạt 0.03).",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Ưu thế cốt lõi của BRCD và suy luận nhân quả:</b> "
        "Phương pháp nhân quả kiểm tra <i>tính bất biến của cơ chế điều kiện</i> <i>P</i>(<i>X<sub>i</sub></i> | <i>Pa</i>(<i>X<sub>i</sub></i>)). Do các biến hạ lưu chỉ thay đổi vì biến cha của chúng thay đổi, mô hình nhân quả nhận diện được cơ chế không đổi và loại bỏ hoàn toàn các nút triệu chứng. Hơn nữa, BRCD áp dụng tích hợp Bayes thay vì kiểm định độc lập có điều kiện đơn lẻ, giúp giữ vững độ chính xác vượt trội ngay cả trong điều kiện cực kỳ khan hiếm mẫu (<i>m</i> = 5).",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. Khả năng ứng dụng công nghiệp:</b> "
        "Toàn bộ mã nguồn mở giải tích và bộ benchmark kiểm thử đã được cấu trúc hoàn thiện, không sử dụng các mô hình hộp đen (GAN), bám sát 100% động học cắt gọt Altintas và thuật toán BRCD của bài báo ICML 2026, sẵn sàng tích hợp vào hệ thống giám sát thời gian thực CNC Cyber-Physical Systems (CPS).",
        body_style
    ))

    # Build Document
    doc.build(story)
    print(f"Successfully generated academic report PDF with rendered LaTeX: {pdf_filename}")

if __name__ == "__main__":
    build_pdf()
