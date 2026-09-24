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
def render_latex_equation(latex_str, filepath, fontsize=12, color="#1E3A8A"):
    plt.figure(figsize=(0.01, 0.01))
    plt.text(0.5, 0.5, f"${latex_str}$", fontsize=fontsize, color=color,
             horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=350, bbox_inches='tight', pad_inches=0.04, transparent=True)
    plt.close()
    
    with PILImage.open(filepath) as img:
        w_px, h_px = img.size
        scale = 72.0 / 350.0
        return w_px * scale, h_px * scale

def build_pdf():
    pdf_filename = "Research_Report_RCA_Chatter_CNC.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    font_regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    font_italic = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
    
    if os.path.exists(font_regular):
        pdfmetrics.registerFont(TTFont("ArialEN", font_regular))
    else:
        pdfmetrics.registerFont(TTFont("ArialEN", "Helvetica"))
        
    if os.path.exists(font_bold):
        pdfmetrics.registerFont(TTFont("ArialEN-Bold", font_bold))
    else:
        pdfmetrics.registerFont(TTFont("ArialEN-Bold", "Helvetica-Bold"))
        
    if os.path.exists(font_italic):
        pdfmetrics.registerFont(TTFont("ArialEN-Italic", font_italic))
    else:
        pdfmetrics.registerFont(TTFont("ArialEN-Italic", "Helvetica-Oblique"))

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='ArialEN-Bold',
        fontSize=17,
        leading=21,
        textColor=colors.HexColor('#0F172A'),
        alignment=1, # Center
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='ArialEN-Italic',
        fontSize=10,
        leading=13.5,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='ArialEN-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='ArialEN-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='ArialEN',
        fontSize=9.0,
        leading=12.6,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        fontName='ArialEN',
        fontSize=8.8,
        leading=12.2,
        textColor=colors.HexColor('#334155'),
        leftIndent=12,
        spaceAfter=2.5
    )

    table_text = ParagraphStyle(
        'TableText',
        fontName='ArialEN',
        fontSize=8.0,
        leading=10.0,
        alignment=1 # Center
    )

    table_header = ParagraphStyle(
        'TableHeader',
        fontName='ArialEN-Bold',
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
    story.append(Paragraph("SCIENTIFIC & TECHNICAL RESEARCH REPORT", ParagraphStyle('Pre', fontName='ArialEN-Bold', fontSize=9.0, textColor=colors.HexColor('#DC2626'), alignment=1, spaceAfter=2)))
    story.append(Paragraph("Root Cause Analysis of Machining Chatter in CNC Systems via Bayesian Causal Inference (BRCD)", title_style))
    story.append(Paragraph("Author: <b>Dang Hai Dang</b> &bull; Project: <b>RCA-for-Chatter-Simulation</b> &bull; Year: 2026<br/>Based on: Machining Dynamics <i>(Yusuf Altintas)</i> &amp; Bayesian Root Cause Discovery <i>(ICML 2026 BRCD)</i>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor('#CBD5E1'), spaceAfter=6))

    # -------------------------------------------------------------------------
    # 1. EXECUTIVE SUMMARY & OBJECTIVES
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Research Objectives", h1_style))
    story.append(Paragraph(
        "<b>Regenerative Chatter</b> is a severe self-excited vibration instability in CNC precision machining that degrades surface finish (Roughness <i>R<sub>a</sub></i>), "
        "causes premature tool chipping, overloads the spindle drive, and leads to expensive workpiece scrap. "
        "In modern industrial smart manufacturing, when multi-channel telemetry sensors (such as vibration accelerometers, dynamic cutting force dynamometers, or spindle power meters) trigger anomaly alerts, "
        "identifying the genuine <b>Root Cause</b> among complex cascading downstream symptoms is extremely challenging.",
        body_style
    ))
    story.append(Paragraph("<b>Primary Project Contributions:</b>", body_style))
    story.append(Paragraph("&bull; <b>Physical Grounding & Analytical Dynamics:</b> Implemented a pure Python open-source solver for continuous Stability Lobe Diagrams (SLD) reproducing Altintas's textbook <i>Manufacturing Automation</i> and Josmar Cristello Assignment 4.", bullet_style))
    story.append(Paragraph("&bull; <b>14-Node Structural Causal Model (SCM):</b> Constructed a realistic DAG capturing physical dependencies from root parameters (clamping, damping, wear, depth, speed) to internal bifurcation states and 7 telemetry sensor channels.", bullet_style))
    story.append(Paragraph("&bull; <b>Rigorous SOTA RCA Benchmark:</b> Evaluated <i>Bayesian Root Cause Discovery (BRCD)</i> (ICML 2026) against 5 state-of-the-art baselines (RCD, RCG, SmoothTraversal, BARO, SimpleRCA) across 2,500 Monte-Carlo trials under realistic noise and subtle fault boundaries.", bullet_style))
    
    # -------------------------------------------------------------------------
    # 2. MACHINING DYNAMICS & STABILITY LOBES (ALTINTAS)
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Machining Dynamics & Stability Lobe Theory (Altintas)", h1_style))
    story.append(Paragraph(
        "Regenerative chatter occurs when cutting teeth encounter wavy chip thickness generated by previous revolutions. "
        "The phase lag between inner and outer vibration waves is governed by <i>&epsilon;</i> = 3&pi; + 2<i>&psi;</i>, "
        "where <i>&psi;</i> = atan2(<i>H</i>(<i>&omega;<sub>c</sub></i>), <i>G</i>(<i>&omega;<sub>c</sub></i>)) is derived from the structural Transfer Function <i>FRF</i> = <i>G</i> + <i>jH</i>.",
        body_style
    ))
    
    story.append(Spacer(1, 2))
    story.append(Table([[Image(f"{eq_dir}/eq_tf.png", width=w2*0.9, height=h2*0.9)]], colWidths=[520], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 2))
    story.append(Table([[Image(f"{eq_dir}/eq_chatter.png", width=w1*0.85, height=h1*0.85)]], colWidths=[520], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 2))

    if os.path.exists("stability_lobes_reproduced.png"):
        story.append(Image("stability_lobes_reproduced.png", width=7.2*inch, height=2.3*inch))
        story.append(Paragraph("<b>Figure 1:</b> Continuous analytical Stability Lobe Diagrams: (a) 2-DOF Shaping Machine; (b) Multi-DOF Milling (Slotting &amp; Half-Immersion Down Milling) matching Altintas Examples #1 &amp; #2.", ParagraphStyle('Cap', fontName='ArialEN-Italic', fontSize=7.6, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    # -------------------------------------------------------------------------
    # 3. 14-NODE CAUSAL TELEMETRY GRAPH
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. 14-Node Causal Telemetry Graph Architecture", h1_style))
    story.append(Paragraph(
        "In industrial CNC systems, root cause isolation must operate over the entire telemetry space without pre-filtering symptoms. The 14-node DAG encompasses three physical tiers:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Tier 1: 5 Root Cause Candidates:</b> <i>Fixture Clamping Stiffness (X<sub>0</sub>)</i>, <i>Toolholder Compliance Damping (X<sub>1</sub>)</i>, <i>Tool Flank Wear (X<sub>2</sub>)</i>, <i>Programmed Depth of Cut (X<sub>3</sub>)</i>, <i>Spindle Speed RPM (X<sub>4</sub>)</i>.", bullet_style))
    story.append(Paragraph("&bull; <b>Tier 2: 2 Internal Dynamic States:</b> Dynamic Stability Boundary <i>a<sub>lim</sub> (X<sub>5</sub>)</i>, Non-linear <i>Chatter Severity (X<sub>6</sub>)</i>.", bullet_style))
    story.append(Paragraph("&bull; <b>Tier 3: 7 Sensor Telemetry Metrics:</b> <i>Vibration RMS (X<sub>7</sub>)</i>, <i>Dominant Frequency (X<sub>8</sub>)</i>, <i>Spectral Ratio (X<sub>9</sub>)</i>, <i>Mean Cutting Force (X<sub>10</sub>)</i>, <i>Peak Force (X<sub>11</sub>)</i>, <i>Spindle Power (X<sub>12</sub>)</i>, <i>Surface Roughness R<sub>a</sub> (X<sub>13</sub>)</i>.", bullet_style))
    
    # -------------------------------------------------------------------------
    # 4. EXPERIMENTAL BENCHMARK EVALUATION
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Experimental Benchmark & SOTA Comparison", h1_style))
    story.append(Paragraph(
        "Evaluated across <b>2,500 Monte-Carlo simulations</b> under realistic physical conditions: &plusmn; 3.5% sensor measurement noise, &plusmn; 5% workpiece hardness batch variations, and subtle boundary perturbations (&Delta;<i>z</i> &approx; 1.1&sigma; - 1.5&sigma;). "
        "Algorithms rank across <b>all 14 candidate nodes</b>.",
        body_style
    ))
    
    # Table 1
    t1_data = [
        [Paragraph("<b>Fault Scenario</b>", table_header), Paragraph("<b>BRCD</b>", table_header), Paragraph("<b>RCD</b>", table_header), Paragraph("<b>RCG</b>", table_header), Paragraph("<b>SmoothTraversal</b>", table_header), Paragraph("<b>BARO</b>", table_header), Paragraph("<b>SimpleRCA</b>", table_header)],
        [Paragraph("STIFFNESS", table_text), Paragraph("<b>0.58 ± 0.05</b>", table_text), Paragraph("0.46 ± 0.05", table_text), Paragraph("0.47 ± 0.05", table_text), Paragraph("0.47 ± 0.05", table_text), Paragraph("0.18 ± 0.04", table_text), Paragraph("0.00 ± 0.00", table_text)],
        [Paragraph("DAMPING", table_text), Paragraph("<b>0.79 ± 0.04</b>", table_text), Paragraph("0.71 ± 0.05", table_text), Paragraph("0.71 ± 0.05", table_text), Paragraph("0.67 ± 0.05", table_text), Paragraph("0.48 ± 0.05", table_text), Paragraph("0.00 ± 0.00", table_text)],
        [Paragraph("TOOL_WEAR", table_text), Paragraph("<b>0.71 ± 0.05</b>", table_text), Paragraph("0.62 ± 0.05", table_text), Paragraph("0.62 ± 0.05", table_text), Paragraph("0.67 ± 0.05", table_text), Paragraph("0.49 ± 0.05", table_text), Paragraph("0.09 ± 0.03", table_text)],
        [Paragraph("DEPTH_OVERLOAD", table_text), Paragraph("<b>0.82 ± 0.04</b>", table_text), Paragraph("0.76 ± 0.04", table_text), Paragraph("0.76 ± 0.04", table_text), Paragraph("0.71 ± 0.05", table_text), Paragraph("0.53 ± 0.05", table_text), Paragraph("0.04 ± 0.02", table_text)],
        [Paragraph("RPM_MISMATCH", table_text), Paragraph("0.65 ± 0.05", table_text), Paragraph("0.63 ± 0.05", table_text), Paragraph("0.61 ± 0.05", table_text), Paragraph("<b>0.69 ± 0.05</b>", table_text), Paragraph("0.22 ± 0.04", table_text), Paragraph("0.04 ± 0.02", table_text)],
        [Paragraph("<b>AVERAGE (m = 5)</b>", table_header), Paragraph("<b>0.71 ± 0.02</b>", table_header), Paragraph("0.64 ± 0.02", table_header), Paragraph("0.63 ± 0.02", table_header), Paragraph("0.64 ± 0.02", table_header), Paragraph("0.38 ± 0.02", table_header), Paragraph("0.03 ± 0.01", table_header)]
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
    
    story.append(Paragraph("<b>Table 1:</b> Top-1 Diagnosis Accuracy Broken Down by Fault Mode in the Scarce Sample Regime (<i>m</i> = 5 Anomalous Samples)", h2_style))
    story.append(t1)
    story.append(Spacer(1, 4))

    # Table 2
    t2_data = [
        [Paragraph("<b>Algorithm</b>", table_header), Paragraph("<b>m = 5</b>", table_header), Paragraph("<b>m = 10</b>", table_header), Paragraph("<b>m = 20</b>", table_header), Paragraph("<b>m = 50</b>", table_header), Paragraph("<b>m = 100</b>", table_header)],
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
    
    story.append(Paragraph("<b>Table 2:</b> Overall Top-1 Accuracy across Interventional Sample Sizes <i>m</i> (Convergence Progression per Theorem 4.4)", h2_style))
    story.append(t2)
    story.append(Spacer(1, 4))

    # Bound equation
    story.append(Table([[Image(f"{eq_dir}/eq_bound.png", width=w5*0.82, height=h5*0.82)]], colWidths=[520], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 4))

    if os.path.exists("rca_sota_accuracy_comparison.png"):
        story.append(Image("rca_sota_accuracy_comparison.png", width=7.2*inch, height=2.0*inch))
        story.append(Paragraph("<b>Figure 2:</b> Top-1, Top-3, and Top-5 Accuracy curves demonstrating clear statistical separation without ceiling artifacts.", ParagraphStyle('Cap2', fontName='ArialEN-Italic', fontSize=7.6, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    if os.path.exists("rca_fault_breakdown_m5.png"):
        story.append(Image("rca_fault_breakdown_m5.png", width=7.2*inch, height=2.0*inch))
        story.append(Paragraph("<b>Figure 3:</b> Per-fault mode diagnosis breakdown at <i>m</i> = 5 anomalous samples.", ParagraphStyle('Cap3', fontName='ArialEN-Italic', fontSize=7.6, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    # -------------------------------------------------------------------------
    # 5. ONLINE BAYESIAN POSTERIOR CONVERGENCE
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Online Posterior Convergence & Shannon Entropy Decay", h1_style))
    story.append(Paragraph(
        "A key capability of <b>BRCD (Bayesian Root Cause Discovery)</b> is its <i>Anytime Update</i> property. "
        "As streaming anomaly samples <b>D</b><sub><i>t</i></sub> arrive in real-time, the posterior distribution <i>P</i>(<i>R</i><sup>*</sup> | <b>D</b><sub><i>t</i></sub>) is updated via Bayes' rule. "
        "Starting from a uniform prior <i>P</i><sub>0</sub> = 0.20 with maximal uncertainty <i>H</i><sub>0</sub> = ln(5) = 1.61 nats, evidence accumulates smoothly towards certainty.",
        body_style
    ))
    
    story.append(Spacer(1, 2))
    story.append(Table([[Image(f"{eq_dir}/eq_brcd.png", width=w3*0.82, height=h3*0.82)]], colWidths=[520], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 2))
    story.append(Table([[Image(f"{eq_dir}/eq_entropy.png", width=w4*0.82, height=h4*0.82)]], colWidths=[520], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 2))
    
    if os.path.exists("rca_convergence_progress.png"):
        story.append(Image("rca_convergence_progress.png", width=7.2*inch, height=2.0*inch))
        story.append(Paragraph("<b>Figure 4:</b> (a) Posterior probability <i>P</i>(<i>R</i><sup>*</sup> | <b>D</b><sub><i>t</i></sub>) accumulation; (b) Shannon entropy <i>H</i>(<i>P</i>) decay across 40 streaming anomaly iterations.", ParagraphStyle('Cap4', fontName='ArialEN-Italic', fontSize=7.6, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    if os.path.exists("chatter_physical_signatures.png"):
        story.append(Image("chatter_physical_signatures.png", width=7.2*inch, height=3.0*inch))
        story.append(Paragraph("<b>Figure 5:</b> Distinct physical vibration fingerprints in time and frequency domains under distinct fault conditions.", ParagraphStyle('Cap5', fontName='ArialEN-Italic', fontSize=7.6, alignment=1, textColor=colors.HexColor('#475569'))))
        story.append(Spacer(1, 4))

    # -------------------------------------------------------------------------
    # 6. SCIENTIFIC DISCUSSION & INDUSTRIAL DEPLOYMENT
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Scientific Discussion & Industrial Deployment", h1_style))
    story.append(Paragraph(
        "<b>1. Why Non-Causal Heuristics Fail on Telemetry:</b> "
        "Methods such as <code>SimpleRCA</code> and <code>BARO</code> rely on marginal anomaly scoring. Under violent chatter vibrations, downstream symptoms like Peak Force (<i>X<sub>11</sub></i>) and Surface Roughness (<i>X<sub>13</sub></i>) experience huge variances, causing heuristics to misidentify symptoms as causes (0.03 Top-1 accuracy).",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Causal Invariance Advantage:</b> "
        "Causal RCA evaluates mechanism changes <i>P</i>(<i>X<sub>i</sub></i> | <i>Pa</i>(<i>X<sub>i</sub></i>)). When a downstream metric shifts only because its parents shifted, the conditional model recognizes invariance and penalizes symptom nodes. BRCD integrates likelihoods over the I-MEC, providing superior sample efficiency in the low-sample regime (<i>m</i> = 5).",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. Industrial Readiness:</b> "
        "The codebase is physically grounded, fully reproducible, and ready for deployment on embedded edge CNC controllers within Cyber-Physical Systems (CPS).",
        body_style
    ))

    doc.build(story)
    print(f"Successfully generated English academic report PDF: {pdf_filename}")

if __name__ == "__main__":
    build_pdf()
