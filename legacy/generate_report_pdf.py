"""Sinh báo cáo PDF song ngữ (Việt + Anh) về RCA cho hiện tượng chatter.

  python make_report_data.py        # thu số liệu thật (≈1 phút)
  python generate_report_pdf.py     # tạo cả 2 PDF

Mọi con số/hình đều lấy từ report_results.json, physics_results.json, physics_waveforms.npz.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from chatter_simulation import example_1_shaping_stability, example_2_milling_stability

OUT = {"vi": "Bao_Cao_Nghien_Cuu_RCA_Chatter_CNC.pdf", "en": "Research_Report_RCA_Chatter_CNC.pdf"}
FIG_DIR = "figures"
FONT_DIR = "/System/Library/Fonts/Supplemental"
INK, MUTED = "#0F172A", "#475569"
ALGOS = ["BRCD", "RCD", "RCG", "SmoothTraversal", "BARO", "SimpleRCA"]
COL = {"BRCD": "#DC2626", "RCD": "#2563EB", "RCG": "#0D9488",
       "SmoothTraversal": "#64748B", "BARO": "#D97706", "SimpleRCA": "#7C3AED"}
FAULTS = ["STIFFNESS", "DAMPING", "TOOL_WEAR", "DEPTH_OVERLOAD", "RPM_MISMATCH"]
FAULT_NAME = {
    "vi": {"STIFFNESS": "Độ cứng\ngiảm", "DAMPING": "Độ cản\ngiảm", "TOOL_WEAR": "Dao\nmòn",
           "DEPTH_OVERLOAD": "Cắt\nquá sâu", "RPM_MISMATCH": "Tốc độ\nkhông hợp"},
    "en": {"STIFFNESS": "Stiffness\ndrop", "DAMPING": "Damping\nloss", "TOOL_WEAR": "Tool\nwear",
           "DEPTH_OVERLOAD": "Depth\noverload", "RPM_MISMATCH": "RPM\nmismatch"},
}


def setup_fonts():
    for name in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"):
        fm.fontManager.addfont(os.path.join(FONT_DIR, name))
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    for n, f in (("ArialVN", "Arial.ttf"), ("ArialVN-Bold", "Arial Bold.ttf"), ("ArialVN-Italic", "Arial Italic.ttf")):
        pdfmetrics.registerFont(TTFont(n, os.path.join(FONT_DIR, f)))
    pdfmetrics.registerFontFamily("ArialVN", normal="ArialVN", bold="ArialVN-Bold",
                                  italic="ArialVN-Italic", boldItalic="ArialVN-Bold")


# ---------------------------------------------------------------------------
# Hình vẽ (nhãn theo ngôn ngữ)
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, face, edge, fs=10, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                linewidth=1.5, edgecolor=edge, facecolor=face, linestyle=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0, linestyle=ls))


def fig_system(path, lang):
    """Sơ đồ khối hệ thống: input -> quá trình cắt (DDE) -> đại lượng vật lý -> cảm biến -> RCA."""
    T = lambda vi, en: vi if lang == "vi" else en
    fig, ax = plt.subplots(figsize=(13, 6.4), dpi=200)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.4); ax.axis("off")
    # Input
    _box(ax, 0.1, 3.9, 3.2, 1.9, "", "#F0FDF4", "#15803D")
    ax.text(1.7, 5.55, T("THAM SỐ VẬT LIỆU–DAO–MÁY", "MATERIAL–TOOL–MACHINE"), ha="center", fontsize=8.6, fontweight="bold", color="#15803D")
    ax.text(1.7, 4.75, T("K_t, K_r, C_p, k, ζ, ω_n, số răng\n(cố định tương đối,\ntrôi chậm: mòn dao, phôi mỏng…)", "K_t, K_r, C_p, k, ζ, ω_n, tooth count\n(roughly fixed, slow drift:\ntool wear, thin-wall part…)"),
            ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.45)
    _box(ax, 0.1, 1.6, 3.2, 1.9, "", "#EFF6FF", "#1D4ED8")
    ax.text(1.7, 3.25, T("THAM SỐ VẬN HÀNH", "OPERATING PARAMETERS"), ha="center", fontsize=8.6, fontweight="bold", color="#1D4ED8")
    ax.text(1.7, 2.45, T("tốc độ n, chiều sâu a,\nđộ ăn dao, lượng chạy dao\n(điều khiển được, lệnh từ CNC)", "speed n, depth a,\nradial immersion, feed\n(controllable, set by the CNC program)"),
            ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.45)
    _box(ax, 0.1, 0.1, 3.2, 1.2, "", "#FFFBEB", "#B45309")
    ax.text(1.7, 1.05, T("BẤT ĐỊNH", "UNCERTAINTY"), ha="center", fontsize=8.6, fontweight="bold", color="#B45309")
    ax.text(1.7, 0.55, T("độ đảo dao, độ cứng vật liệu dao\nđộng, nhiễu cảm biến", "tool runout, material hardness\nvariation, sensor noise"),
            ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.45)
    # Process
    _box(ax, 4.2, 1.6, 3.4, 4.2, "", "#FEF3C7", "#B45309")
    ax.text(5.9, 5.45, T("QUÁ TRÌNH CẮT", "CUTTING PROCESS"), ha="center", fontsize=9.6, fontweight="bold", color="#B45309")
    ax.text(5.9, 4.55, T("Phương trình dao động có trễ (DDE)", "Delay differential equation (DDE)"), ha="center", fontsize=8.8, color=INK)
    ax.text(5.9, 3.85, "m·r″ + c·r′ + k·r = K_r·K_t·a·h(t)\nh(t) = h0 − [ r(t) − r(t−T) ]", ha="center", va="center", fontsize=9.2, color="#7C2D12", linespacing=1.6)
    ax.text(5.9, 2.85, T("T = 60 / n  (chu kỳ tái sinh)\nvòng hồi tiếp tái sinh nằm\nsẵn trong quá trình cắt", "T = 60 / n  (regeneration delay)\nthe regenerative feedback loop\nis built into the cutting process"),
            ha="center", va="center", fontsize=8.6, color=MUTED, linespacing=1.45, style="italic")
    ax.text(5.9, 1.95, T("ổn định khi a < a_lim(n, k, ζ, K_t…)", "stable when a < a_lim(n, k, ζ, K_t…)"), ha="center", fontsize=9, fontweight="bold", color="#991B1B")
    # Physical quantities
    _box(ax, 8.4, 3.9, 2.0, 1.9, "", "#F8FAFC", "#64748B", ls="--")
    ax.text(9.4, 5.45, T("ĐẠI LƯỢNG VẬT LÝ", "PHYSICAL QUANTITIES"), ha="center", fontsize=8.2, fontweight="bold", color="#475569")
    ax.text(9.4, 4.6, T("dao động tương đối\ndao–phôi r(t),\nlực cắt F(t)\n(khó đo trực tiếp)", "relative tool–work\nvibration r(t),\ncutting force F(t)\n(hard to measure directly)"),
            ha="center", va="center", fontsize=8.6, color=INK, linespacing=1.4)
    # Sensors
    _box(ax, 8.4, 0.1, 2.0, 3.5, "", "#FEF2F2", "#B91C1C")
    ax.text(9.4, 3.3, T("CẢM BIẾN THỰC TẾ", "REAL SENSORS"), ha="center", fontsize=8.4, fontweight="bold", color="#B91C1C")
    ax.text(9.4, 1.85, T("• accelerometer, micro\n  (rẻ, phổ biến)\n• dòng điện / công suất\n  trục chính, tín hiệu CNC\n• dynamometer lực\n  (đắt, phòng thí nghiệm)\n• độ nhám: đo sau", "• accelerometer, microphone\n  (cheap, common)\n• spindle current / power,\n  CNC signals\n• force dynamometer\n  (costly, lab use)\n• roughness: offline"),
            ha="center", va="center", fontsize=7.8, color=INK, linespacing=1.4)
    # Features + RCA
    _box(ax, 10.9, 1.6, 2.0, 2.2, "", "#FAF5FF", "#7E22CE")
    ax.text(11.9, 3.5, T("ĐẶC TRƯNG", "FEATURES"), ha="center", fontsize=8.6, fontweight="bold", color="#7E22CE")
    ax.text(11.9, 2.6, T("RMS rung, tần số\nđỉnh, tỷ số phổ,\nlực, công suất…", "vibration RMS, peak\nfrequency, spectral\nratio, force, power…"), ha="center", va="center", fontsize=8.6, color=INK, linespacing=1.4)
    _box(ax, 10.9, 4.2, 2.0, 1.6, "", "#F0FDF4", "#15803D", bold=True)
    ax.text(11.9, 5.45, "RCA", ha="center", fontsize=10.5, fontweight="bold", color="#15803D")
    ax.text(11.9, 4.75, T("suy ra nguyên nhân\ngốc (ẩn) từ đặc trưng", "infer the hidden root\ncause from features"), ha="center", va="center", fontsize=8.6, color=INK, linespacing=1.4)
    # Arrows
    _arrow(ax, 3.3, 4.85, 4.2, 4.85); _arrow(ax, 3.3, 2.55, 4.2, 3.0); _arrow(ax, 3.3, 0.9, 4.2, 1.9, color="#B45309")
    _arrow(ax, 7.6, 4.85, 8.4, 4.85)
    _arrow(ax, 9.4, 3.9, 9.4, 3.6, color="#64748B")
    _arrow(ax, 10.4, 1.85, 10.9, 2.3)
    _arrow(ax, 11.9, 3.8, 11.9, 4.2)
    # Optional closed-loop controller (not in project): ô riêng, không mũi tên để tránh chồng chéo
    _box(ax, 4.2, 0.1, 3.7, 1.2, "", "#F8FAFC", "#94A3B8", ls="--")
    ax.text(6.05, 0.7, T("(tùy chọn – KHÔNG có trong dự án)\nĐiều khiển vòng kín: cảm biến → bộ điều khiển\n→ chỉnh n, a (SSV, giảm chấn chủ động)",
                         "(optional – NOT in this project)\nClosed-loop control: sensors → controller\n→ adjusts n, a (SSV, active damping)"),
            ha="center", va="center", fontsize=8.2, color="#64748B", style="italic", linespacing=1.45)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_rootgraph(path, lang):
    """Đồ thị: nguyên nhân gốc -> tham số máy/dao/vật liệu và tham số vận hành -> a_lim -> chatter.
    Nét: liền = đã đọc trong nguồn; đứt = nguồn thứ cấp/chỉ tóm tắt; chấm = nguồn yếu (blog).
    Màu: đỏ = làm giảm ổn định; xanh = tăng ổn định; xám = biến thiên/không một chiều."""
    T = lambda vi, en: vi if lang == "vi" else en
    fig, ax = plt.subplots(figsize=(13, 9.2), dpi=200)
    ax.set_xlim(0, 13); ax.set_ylim(-1.3, 9.0); ax.axis("off")
    RED, GRN, GRY = "#DC2626", "#16A34A", "#64748B"
    ax.text(1.75, 8.75, T("NGUYÊN NHÂN GỐC (sự kiện lỗi / điều kiện)", "ROOT CAUSES (fault events / conditions)"), ha="center", fontsize=9.6, fontweight="bold", color="#334155")
    ax.text(6.9, 8.75, T("THAM SỐ MÁY – DAO – VẬT LIỆU và VẬN HÀNH", "MACHINE–TOOL–MATERIAL and OPERATING PARAMETERS"), ha="center", fontsize=9.6, fontweight="bold", color="#334155")
    ax.text(11.6, 8.75, T("ỔN ĐỊNH", "STABILITY"), ha="center", fontsize=9.6, fontweight="bold", color="#334155")

    roots = [
        ("R1", T("Dao nhô dài", "Long tool overhang"), "#B45309", "#FFFBEB"),
        ("R2", T("Phôi thành mỏng (mất vật liệu)", "Thin-wall part (material removal)"), "#0F766E", "#F0FDFA"),
        ("R3", T("Gá phôi lỏng / đỡ kém", "Loose or poor workholding"), "#0F766E", "#F0FDFA"),
        ("R4", T("Ổ trục chính mòn, mối nối lỏng", "Worn spindle bearings, loose joints"), "#7C3AED", "#FAF5FF"),
        ("R5", T("Vị trí trục máy (máy lớn)", "Machine axis position (large machines)"), "#7C3AED", "#FAF5FF"),
        ("R6", T("Vật liệu phôi khác danh định", "Workpiece material differs from nominal"), "#0F766E", "#F0FDFA"),
        ("R7", T("Mòn mặt sau dao", "Tool flank wear"), "#B45309", "#FFFBEB"),
        ("R8", T("Lập trình chiều sâu cắt quá lớn", "Over-deep programmed depth of cut"), "#1D4ED8", "#EFF6FF"),
        ("R9", T("Chọn tốc độ ở vùng trũng", "Speed chosen in a lobe valley"), "#1D4ED8", "#EFF6FF"),
    ]
    ry = {}
    for i, (rid, name, edge_c, face) in enumerate(roots):
        y = 7.95 - i * 0.92
        ry[rid] = y + 0.31
        _box(ax, 0.1, y, 3.3, 0.62, "", face, edge_c)
        ax.text(0.22, y + 0.31, rid, ha="left", va="center", fontsize=9, fontweight="bold", color=edge_c)
        ax.text(0.72, y + 0.31, name, ha="left", va="center", fontsize=8.6, color=INK)

    nodes = {  # id: (x, y, w, h, text, face, edge)
        "TOOL": (5.5, 7.85, 2.8, 0.62, T("Đặc trưng dao: đường kính,\nvật liệu, số răng", "Tool features: diameter,\nmaterial, tooth count"), "#FFFBEB", "#B45309"),
        "FRF": (5.5, 4.55, 2.8, 2.9, T("Động lực học cấu trúc (FRF)\nk, ζ, ω_n", "Structural dynamics (FRF)\nk, ζ, ω_n"), "#FFFBEB", "#B45309"),
        "KT": (5.5, 3.15, 2.8, 0.8, T("Hệ số lực cắt K_t, K_r", "Cutting coefficients K_t, K_r"), "#FFFBEB", "#B45309"),
        "CP": (5.5, 2.0, 2.8, 0.7, T("Cản quá trình C_p", "Process damping C_p"), "#FFFBEB", "#B45309"),
        "A": (5.5, 1.05, 2.8, 0.7, T("a: chiều sâu cắt (vận hành)", "a: depth of cut (operating)"), "#EFF6FF", "#1D4ED8"),
        "N": (5.5, 0.1, 2.8, 0.7, T("n: tốc độ trục chính (vận hành)", "n: spindle speed (operating)"), "#EFF6FF", "#1D4ED8"),
        "ALIM": (10.2, 3.6, 2.7, 1.2, T("Giới hạn ổn định\na_lim(n)", "Stability limit\na_lim(n)"), "#FEF3C7", "#B45309"),
        "CH": (10.2, 1.0, 2.7, 1.0, T("Chatter\n(khi a > a_lim)", "Chatter\n(when a > a_lim)"), "#FEE2E2", "#B91C1C"),
    }
    for k, (x, y, w, h, t, face, edge_c) in nodes.items():
        _box(ax, x, y, w, h, t, face, edge_c, fs=8.8, bold=k in ("ALIM", "CH"))
    cy = {k: v[1] + v[3] / 2 for k, v in nodes.items()}
    R = lambda k, dy=0.0: (nodes[k][0] + nodes[k][2], cy[k] + dy)
    Lf = lambda k, dy=0.0: (nodes[k][0], cy[k] + dy)

    def edge(p, q, label, color, style, frac=0.5, dy=0.0, fs=7.4):
        ls = {"solid": "-", "dashed": "--", "dotted": ":"}[style]
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, linewidth=1.7 if style == "solid" else 1.5,
                                     color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))
        x = p[0] + (q[0] - p[0]) * frac; y = p[1] + (q[1] - p[1]) * frac + dy
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=3,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92))

    rr = lambda r: (3.4, ry[r])
    # R -> tham số (mỗi cạnh vào một điểm khác nhau của nút FRF để khỏi chồng)
    edge(rr("R1"), (5.5, 7.1), "k↓ ω_n↓ [14][15]", RED, "solid", 0.5)
    edge((6.9, 7.85), (6.9, 7.45), "[14]", GRY, "solid", 0.5, dy=0.0)
    edge(rr("R2"), (5.5, 6.55), T("FRF biến thiên [1][20]", "time-varying FRF [1][20]"), GRY, "solid", 0.5)
    edge(rr("R3"), (5.5, 6.0), "k↓ [17]", RED, "dotted", 0.5)
    edge(rr("R4"), (5.5, 5.45), "k↓ [17]", RED, "dotted", 0.5)
    edge(rr("R5"), (5.5, 4.9), T("mode tần thấp dịch [1]", "low modes shift [1]"), GRY, "solid", 0.5)
    edge(rr("R6"), (5.5, 3.7), T("K_t, K_r đổi [14]", "K_t, K_r change [14]"), GRY, "solid", 0.5)
    edge(rr("R7"), (5.5, 3.35), "K_t↑ [16]", RED, "dashed", 0.55)
    edge(rr("R7"), (5.5, 2.35), "C_p↑ [1][16]", GRN, "solid", 0.55)
    edge(rr("R8"), (5.5, 1.5), "[1]", RED, "solid", 0.5)
    edge(rr("R9"), (5.5, 0.55), "[1]", GRY, "solid", 0.5)
    # tham số -> ổn định
    edge(R("FRF", 0.7), (10.2, 4.55), "[1] Eq.19–20", GRY, "solid", 0.5)
    edge(R("KT"), (10.2, 4.15), "[1] Eq.19–20", RED, "solid", 0.5)
    edge(R("CP"), (10.2, 3.8), "[1] Eq.18", GRN, "solid", 0.5)
    edge(R("N"), (10.2, 3.65), T("T = 60/n; búp [1]", "T = 60/n; lobes [1]"), GRY, "solid", 0.62)
    edge(R("A"), (10.2, 1.5), "a > a_lim [1]", RED, "solid", 0.8)
    edge((11.55, 3.6), (11.55, 2.0), "", RED, "solid")

    # chú giải
    ly = -0.35
    ax.plot([0.2, 0.9], [ly, ly], color="#334155", lw=1.8)
    ax.text(1.0, ly, T("nét liền: đã đọc trong nguồn", "solid: read in the source"), va="center", fontsize=8.2, color=INK)
    ax.plot([4.2, 4.9], [ly, ly], color="#334155", lw=1.5, ls="--")
    ax.text(5.0, ly, T("nét đứt: nguồn thứ cấp / chỉ tóm tắt", "dashed: secondary source / summary only"), va="center", fontsize=8.2, color=INK)
    ax.plot([8.9, 9.6], [ly, ly], color="#334155", lw=1.5, ls=":")
    ax.text(9.7, ly, T("nét chấm: nguồn yếu (blog)", "dotted: weak source (blog)"), va="center", fontsize=8.2, color=INK)
    ly2 = -0.85
    for x, c, t in ((0.2, RED, T("đỏ: làm giảm ổn định", "red: lowers stability")), (4.2, GRN, T("xanh: tăng ổn định (tốc độ thấp)", "green: raises stability (low speed)")), (8.9, GRY, T("xám: biến thiên / không một chiều", "grey: varies / not one-directional"))):
        ax.add_patch(FancyBboxPatch((x, ly2 - 0.1), 0.6, 0.2, boxstyle="round,pad=0,rounding_size=0.05", fc=c, ec="none"))
        ax.text(x + 0.72, ly2, t, va="center", fontsize=8.2, color=INK)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_cause_effect(path, lang):
    T = lambda vi, en: vi if lang == "vi" else en
    fig, ax = plt.subplots(figsize=(13, 6.0), dpi=200)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.0); ax.axis("off")
    ax.text(1.55, 5.75, T("NGUYÊN NHÂN GỐC (5)", "ROOT CAUSES (5)"), ha="center", fontsize=10.5, fontweight="bold", color="#15803D")
    ax.text(5.75, 5.75, T("TRẠNG THÁI BÊN TRONG (2)", "INTERNAL STATES (2)"), ha="center", fontsize=10.5, fontweight="bold", color="#B45309")
    ax.text(10.6, 5.75, T("CẢM BIẾN (7)", "SENSORS (7)"), ha="center", fontsize=10.5, fontweight="bold", color="#B91C1C")
    causes = [(T("Độ cứng gá đặt k", "Fixture stiffness k"), True), (T("Độ cản ζ", "Damping ratio ζ"), True),
              (T("Mòn dao Kt", "Tool wear Kt"), True), (T("Chiều sâu cắt a", "Depth of cut a"), False),
              (T("Tốc độ trục chính rpm", "Spindle speed rpm"), False)]
    ys = [4.6, 3.7, 2.8, 1.9, 1.0]
    for (t, hid), y in zip(causes, ys):
        _box(ax, 0.1, y, 2.9, 0.7, t, "#F0FDF4", "#15803D", ls="--" if hid else "-")
    _box(ax, 4.5, 3.5, 2.5, 1.0, T("Giới hạn ổn định\na_lim", "Stability limit\na_lim"), "#FFFBEB", "#B45309", 10.5, True, ls="--")
    _box(ax, 4.5, 1.5, 2.5, 1.0, T("Mức rung chatter\n(Chatter Severity)", "Chatter\nSeverity"), "#FFFBEB", "#B45309", 10.5, True, ls="--")
    sensors = [T("Rung RMS", "Vibration RMS"), T("Tần số chủ đạo", "Dominant frequency"),
               T("Tỷ số phổ", "Spectral ratio"), T("Lực cắt trung bình", "Mean cutting force"),
               T("Lực cắt đỉnh", "Peak cutting force"), T("Công suất trục chính", "Spindle power"),
               T("Độ nhám bề mặt Ra (đo sau)", "Surface roughness Ra (offline)")]
    sy = [4.95, 4.25, 3.55, 2.85, 2.15, 1.45, 0.75]
    for t, y in zip(sensors, sy):
        _box(ax, 8.9, y, 3.5, 0.5, t, "#FEF2F2", "#B91C1C", 9.5)
    for y in ys[:3] + [ys[4]]:
        _arrow(ax, 3.0, y + 0.35, 4.5, 4.0, color="#94A3B8", lw=1.3)
    _arrow(ax, 3.0, ys[3] + 0.35, 4.5, 2.0, color="#94A3B8", lw=1.3)
    _arrow(ax, 5.75, 3.5, 5.75, 2.5)
    for y in sy:
        _arrow(ax, 7.0, 2.0, 8.9, y + 0.25, color="#CBD5E1", lw=1.0)
    ax.text(6.5, 0.2, T("Nét liền: quan sát được trên máy thật (lệnh CNC a, rpm hoặc cảm biến).   Nét đứt: KHÔNG đo trực tiếp được.\n"
                        "Lưu ý: benchmark Thí nghiệm 2 vẫn đưa cả các cột nét đứt vào dữ liệu quan sát (xem mục 3.3 và 6.2).",
                        "Solid outline: observable on a real machine (CNC setpoints a, rpm, or sensors).   Dashed outline: NOT directly measurable.\n"
                        "Note: the Experiment 2 benchmark still feeds the dashed variables to the algorithms as observed data (see Sections 3.3 and 6.2)."),
            ha="center", va="center", fontsize=9, color="#991B1B", style="italic")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_architecture(path, lang):
    T = lambda vi, en: vi if lang == "vi" else en
    fig, ax = plt.subplots(figsize=(13, 5.4), dpi=200)
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis("off")
    _box(ax, 0.1, 3.7, 3.4, 1.5, T("Knowledge Graph (KG)\nbộ phận, tham số, dạng lỗi (FMEA),\ncảm biến, quan hệ vật lý", "Knowledge Graph (KG)\ncomponents, parameters, failure modes\n(FMEA), sensors, physical relations"), "#EFF6FF", "#1D4ED8", 9.4)
    _box(ax, 0.1, 1.9, 3.4, 1.4, T("Mô phỏng vật lý (DDE, búp ổn định)\nsinh dữ liệu có nhãn", "Physics simulation (DDE, lobes)\ngenerates labelled data"), "#FFFBEB", "#B45309", 9.4)
    _box(ax, 0.1, 0.1, 3.4, 1.4, T("Dữ liệu cảm biến thực\nRMS, tần số đỉnh, tỷ số phổ,\ncông suất…", "Real sensor data\nvibration RMS, peak frequency,\nspectral ratio, power…"), "#FEF2F2", "#B91C1C", 9.4)
    _box(ax, 4.6, 1.3, 3.6, 2.7, T("Mạng Bayes nhân quả\n\nnguyên nhân (biến ẩn)\n→ trạng thái → triệu chứng\n\nxác suất của từng nguyên nhân", "Causal Bayesian network\n\ncauses (hidden)\n→ states → symptoms\n\nprobability of each cause"), "#F0FDF4", "#15803D", 9.8, True)
    _box(ax, 4.6, 4.3, 3.6, 0.9, T("RAG: truy xuất sổ tay,\nnhật ký bảo trì, tài liệu", "RAG: retrieve manuals,\nmaintenance logs, papers"), "#FAF5FF", "#7E22CE", 9.4)
    _box(ax, 9.2, 2.3, 3.7, 1.7, T("LLM – lớp giải thích\n(không phải bên quyết định cuối)", "LLM – explanation layer\n(not the final decision maker)"), "#FAF5FF", "#7E22CE", 9.4)
    _box(ax, 9.2, 0.2, 3.7, 1.5, T("Kết quả: xếp hạng nguyên nhân\n+ giải thích + đề xuất kiểm tra", "Output: ranked causes\n+ explanation + suggested checks"), "#F8FAFC", "#475569", 9.4, True)
    _arrow(ax, 3.5, 4.2, 4.6, 3.6)                 # KG -> BN
    _arrow(ax, 3.5, 2.6, 4.6, 2.6)                 # Sim -> BN
    _arrow(ax, 3.5, 0.9, 4.6, 1.8)                 # data -> BN
    _arrow(ax, 3.5, 4.9, 4.6, 4.8)                 # KG -> RAG
    _arrow(ax, 8.2, 2.9, 9.2, 3.0)                 # BN -> LLM
    _arrow(ax, 8.2, 4.7, 10.5, 4.0)                # RAG -> LLM
    _arrow(ax, 11.05, 2.3, 11.05, 1.7)             # LLM -> output
    ax.text(4.05, 4.0, T("cấu trúc", "structure"), fontsize=8.2, color=MUTED, style="italic")
    ax.text(4.0, 2.75, T("xác suất", "probabilities"), fontsize=8.2, color=MUTED, style="italic")
    ax.text(3.9, 1.1, T("bằng chứng", "evidence"), fontsize=8.2, color=MUTED, style="italic")
    ax.text(6.5, 0.45, T("ĐỀ XUẤT – chưa cài đặt trong dự án", "PROPOSAL – not implemented in this project"), ha="center", fontsize=9.4, color="#991B1B", fontweight="bold")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_pipeline(path, lang):
    T = lambda vi, en: vi if lang == "vi" else en
    fig, ax = plt.subplots(figsize=(13, 3.2), dpi=200)
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.2); ax.axis("off")
    steps = [
        (T("1  Mô phỏng bình thường", "1  Simulate normal runs"), T("5.000 mẫu  →  D_obs\n(dữ liệu chuẩn để so sánh)", "5,000 samples  →  D_obs\n(reference data)"), "#EFF6FF", "#1D4ED8"),
        (T("2  Cố ý gây 1 lỗi", "2  Inject one fault"), T("VD: giảm độ cứng k\nthu m mẫu  →  D_int", "e.g. lower stiffness k\ncollect m samples  →  D_int"), "#FEF2F2", "#B91C1C"),
        (T("3  RCA chấm điểm", "3  RCA scores"), T("so D_int với D_obs\nchấm điểm cả 14 biến", "compare D_int to D_obs\nscore all 14 variables"), "#FAF5FF", "#7E22CE"),
        (T("4  Xếp hạng & kiểm tra", "4  Rank & check"), T("Nguyên nhân thật có đứng\nđầu danh sách không?", "Is the true cause\nat the top of the list?"), "#F0FDF4", "#15803D"),
    ]
    w, gap = 2.85, 0.45
    for i, (t, b, face, edge) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        _box(ax, x, 0.35, w, 2.3, "", face, edge)
        ax.text(x + w / 2, 2.2, t, ha="center", va="center", fontsize=9.6, fontweight="bold", color=edge)
        ax.text(x + w / 2, 1.2, b, ha="center", va="center", fontsize=9.6, color=INK, linespacing=1.5)
        if i < 3:
            _arrow(ax, x + w + 0.03, 1.5, x + w + gap - 0.03, 1.5)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_lobes(path, lang):
    """Biểu đồ búp ổn định (Altintas Ví dụ 1 và 2) tính từ chatter_simulation.py."""
    T = lambda vi, en: vi if lang == "vi" else en
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=200)
    res1 = example_1_shaping_stability()
    for mode, ls, ks, lab in (("mode1", "-", range(8), "ω_n1 = 250 Hz"), ("mode2", "--", range(7), "ω_n2 = 150 Hz")):
        m = res1[mode]
        labelled = False
        for k in ks:
            Tk = (2 * k * np.pi + (3 * np.pi + 2 * m["psi"])) / (2 * np.pi * m["fc"])
            N = 60.0 / Tk
            o = np.argsort(N)
            n_s, a_s = N[o], m["a_lim"][o]
            v = (n_s >= 1000) & (n_s <= 15000) & (a_s <= 60) & (a_s >= 0)
            if v.any():
                ax1.plot(n_s[v], a_s[v], "k" + ls, lw=1.4, label=None if labelled else lab)
                labelled = True
    ax1.set_xlim(1000, 15000); ax1.set_ylim(0, 60)
    ax1.set_title(T("(a) Máy bào 2 bậc tự do (2 dạng dao động)", "(a) 2-DOF shaping machine (two vibration modes)"), fontsize=10.5, fontweight="bold")
    ax1.set_xlabel(T("Tốc độ trục chính [vòng/phút]", "Spindle speed [rev/min]"))
    ax1.set_ylabel(T("Chiều sâu cắt tối đa ổn định a_lim [mm]", "Max stable depth of cut a_lim [mm]"))
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, ls=":", alpha=0.6)

    def envelope(sp, al, n_bins=140):
        edges = np.linspace(500, 20000, n_bins + 1)
        xs, ys = [], []
        for i in range(n_bins):
            mk = (sp >= edges[i]) & (sp < edges[i + 1])
            if mk.any():
                xs.append(0.5 * (edges[i] + edges[i + 1])); ys.append(al[mk].min())
        return np.array(xs), np.array(ys)
    s1, a1 = example_2_milling_stability(immersion="slotting")
    s2, a2 = example_2_milling_stability(immersion="down_milling")
    x1, y1 = envelope(s1, a1); x2, y2 = envelope(s2, a2)
    ax2.plot(x1, y1, "k-", lw=1.6, label=T("Phay rãnh (cắt toàn phần)", "Slotting (full immersion)"))
    ax2.plot(x2, y2, color="#D90429", ls="--", lw=1.6, label=T("Phay biên, ăn dao một nửa", "Half-immersion down-milling"))
    ax2.set_xlim(0, 20000); ax2.set_ylim(0, 30)
    ax2.set_title(T("(b) Phay nhiều bậc tự do", "(b) Multi-DOF end milling"), fontsize=10.5, fontweight="bold")
    ax2.set_xlabel(T("Tốc độ trục chính [vòng/phút]", "Spindle speed [rev/min]"))
    ax2.set_ylabel(T("Chiều sâu cắt tối đa ổn định a_lim [mm]", "Max stable depth of cut a_lim [mm]"))
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, ls=":", alpha=0.6)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_signatures(path, lang, wave, phys):
    T = lambda vi, en: vi if lang == "vi" else en
    rows = [("Normal", T("Chạy bình thường (ổn định)", "Normal operation (stable)"), "#0D9488"),
            ("Stiffness", T("Độ cứng giảm 50% (cộng hưởng dịch xuống ≈ 177 Hz)", "Stiffness −50% (resonance shifts to ≈ 177 Hz)"), "#E76F51"),
            ("Depth", T("Cắt quá sâu a = 28 mm", "Excessive depth a = 28 mm"), "#D90429")]
    info = {s["key"]: s for s in phys["scenarios"]}
    fig, axes = plt.subplots(3, 2, figsize=(13, 8.2), dpi=200)
    for i, (key, title, col) in enumerate(rows):
        a = axes[i, 0]
        a.plot(wave[f"{key}_t"], wave[f"{key}_y"], color=col, lw=1.4)
        a.set_title(title, fontsize=10, fontweight="bold", loc="left")
        a.set_xlabel(T("Thời gian [ms]", "Time [ms]")); a.set_ylabel(T("Dịch chuyển [µm]", "Displacement [µm]"))
        a.grid(True, ls=":", alpha=0.6)
        b = axes[i, 1]
        xf, mag = wave[f"{key}_xf"], wave[f"{key}_mag"]
        b.plot(xf, mag / (mag.max() + 1e-12), color=col, lw=1.4)
        b.axvline(250, color="gray", ls="--", alpha=0.7, label=T("Tần số riêng danh định 250 Hz", "Nominal natural frequency 250 Hz"))
        b.set_xlim(0, 500)
        b.set_title(T(f"Phổ tần số – đỉnh tại {info[key]['dom_freq']:.0f} Hz; rung RMS = {info[key]['rms']:.2f} µm",
                      f"Spectrum – peak at {info[key]['dom_freq']:.0f} Hz; vibration RMS = {info[key]['rms']:.2f} µm"),
                    fontsize=10, loc="left")
        b.set_xlabel(T("Tần số [Hz]", "Frequency [Hz]")); b.set_ylabel(T("Biên độ chuẩn hóa", "Normalized magnitude"))
        b.legend(loc="upper right", fontsize=8.5); b.grid(True, ls=":", alpha=0.6)
    for a in axes.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_accuracy(path, lang, res):
    T = lambda vi, en: vi if lang == "vi" else en
    ms = [5, 10, 20, 50, 100]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), dpi=200, sharey=True)
    for ax, k, ttl in zip(axes, ("top1", "top3", "top5"),
                          (T("(a) Top-1: đúng ngay lần đầu", "(a) Top-1: right on first guess"),
                           T("(b) Top-3: nằm trong 3 hạng đầu", "(b) Top-3: within the top 3"),
                           T("(c) Top-5: nằm trong 5 hạng đầu", "(c) Top-5: within the top 5"))):
        for a in ALGOS:
            y = [res[a][str(m)][k][0] for m in ms]; e = [res[a][str(m)][k][1] for m in ms]
            hl = a == "BRCD"
            ax.errorbar(range(5), y, yerr=e, color=COL[a], lw=3 if hl else 1.6, marker="o",
                        ms=6 if hl else 4, capsize=2, label=a, zorder=3 if hl else 2)
        ax.set_xticks(range(5)); ax.set_xticklabels(ms)
        ax.set_xlabel(T("Số mẫu bất thường m", "Number of anomalous samples m"))
        ax.set_title(ttl, fontsize=10.5, fontweight="bold")
        ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(T("Tỷ lệ đoán đúng", "Accuracy"))
    axes[0].legend(loc="center right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_by_fault(path, lang, res):
    T = lambda vi, en: vi if lang == "vi" else en
    algos = ["BRCD", "RCD", "SmoothTraversal", "BARO", "SimpleRCA"]
    fig, ax = plt.subplots(figsize=(9, 4.0), dpi=200)
    wd = 0.16
    for i, a in enumerate(algos):
        ax.bar(np.arange(5) + (i - 2) * wd, [res[a]["5"]["by_fault"][f][0] for f in FAULTS], wd, color=COL[a], label=a)
    ax.set_xticks(range(5)); ax.set_xticklabels([FAULT_NAME[lang][f] for f in FAULTS], fontsize=9.5)
    ax.set_ylabel(T("Top-1 (m = 5)", "Top-1 (m = 5)")); ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=5, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_convergence(path, lang, conv):
    T = lambda vi, en: vi if lang == "vi" else en
    names = {f: FAULT_NAME[lang][f].replace("\n", " ") for f in FAULTS}
    pal = ["#DC2626", "#2563EB", "#0D9488", "#D97706", "#7C3AED"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), dpi=200)
    t = np.arange(1, 41)
    for f, c in zip(FAULTS, pal):
        d = conv["faults"][f]
        a.plot(t, d["p_mean"], color=c, lw=2, label=names[f]); a.fill_between(t, d["p_lo"], d["p_hi"], color=c, alpha=0.12)
        b.plot(t, d["h_mean"], color=c, lw=2); b.fill_between(t, d["h_lo"], d["h_hi"], color=c, alpha=0.12)
    a.axhline(conv["prior"], color="gray", ls=":"); a.text(41, conv["prior"], T(" đoán mò = 0,2", " guessing = 0.2"), va="center", fontsize=8.5, color="gray")
    b.axhline(conv["max_entropy"], color="gray", ls=":"); b.text(41, conv["max_entropy"], T(" tối đa = 1,61", " maximum = 1.61"), va="center", fontsize=8.5, color="gray")
    a.set_ylim(0, 1); b.set_ylim(0, 1.8)
    a.set_title(T("(a) Xác suất BRCD gán cho nguyên nhân thật", "(a) Probability BRCD gives the true cause"), fontsize=10.5, fontweight="bold")
    b.set_title(T("(b) Độ bất định (entropy Shannon, nats)", "(b) Uncertainty (Shannon entropy, nats)"), fontsize=10.5, fontweight="bold")
    for ax in (a, b):
        ax.set_xlabel(T("Số mẫu bất thường đã thu t", "Anomalous samples collected t")); ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False); ax.set_xlim(1, 40)
    a.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def build(lang):
    T = lambda vi, en: vi if lang == "vi" else en
    res_all = json.load(open("report_results.json"))
    res, conv = res_all["benchmark"], res_all["convergence"]
    phys = json.load(open("physics_results.json"))
    wave = np.load("physics_waveforms.npz")

    keys = ("system", "rootgraph", "lobes", "signatures", "cause_effect", "pipeline", "accuracy", "by_fault", "convergence", "architecture")
    FN = {k: i + 1 for i, k in enumerate(keys)}                       # số hình theo thứ tự xuất hiện
    TN = {"inputs": 1, "observ": 2, "rootcause": 3, "bn": 4, "top1": 5, "crit": 6}   # số bảng
    fg = lambda k: T(f"Hình {FN[k]}", f"Figure {FN[k]}")
    tb = lambda k: T(f"Bảng {TN[k]}", f"Table {TN[k]}")

    figs = {k: f"{FIG_DIR}/{k}_{lang}.png" for k in keys}
    fig_system(figs["system"], lang)
    fig_rootgraph(figs["rootgraph"], lang)
    fig_lobes(figs["lobes"], lang)
    fig_signatures(figs["signatures"], lang, wave, phys)
    fig_cause_effect(figs["cause_effect"], lang)
    fig_pipeline(figs["pipeline"], lang)
    fig_accuracy(figs["accuracy"], lang, res)
    fig_by_fault(figs["by_fault"], lang, res)
    fig_convergence(figs["convergence"], lang, conv)
    fig_architecture(figs["architecture"], lang)

    S = lambda name, **kw: ParagraphStyle(name, **kw)
    title = S("t", fontName="ArialVN-Bold", fontSize=18, leading=23, alignment=1, textColor=colors.HexColor(INK), spaceAfter=4)
    sub = S("s", fontName="ArialVN-Italic", fontSize=9.5, leading=13, alignment=1, textColor=colors.HexColor(MUTED), spaceAfter=8)
    h1 = S("h1", fontName="ArialVN-Bold", fontSize=13, leading=17, textColor=colors.HexColor("#1E3A8A"), spaceBefore=12, spaceAfter=5, keepWithNext=True)
    h2 = S("h2", fontName="ArialVN-Bold", fontSize=10.5, leading=14, textColor=colors.HexColor("#0F766E"), spaceBefore=7, spaceAfter=3, keepWithNext=True)
    body = S("b", fontName="ArialVN", fontSize=9.7, leading=13.6, textColor=colors.HexColor("#1E293B"), spaceAfter=5)
    bullet = S("bl", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=3)
    cap = S("c", fontName="ArialVN-Italic", fontSize=8.8, leading=11.5, alignment=1, textColor=colors.HexColor(MUTED), spaceBefore=2, spaceAfter=8)
    cell = S("cell", fontName="ArialVN", fontSize=8.8, leading=11.5)
    cellc = S("cellc", parent=cell, alignment=1)
    head = S("head", fontName="ArialVN-Bold", fontSize=8.8, leading=11.5, textColor=colors.white, alignment=1)
    boxs = S("box", parent=body, spaceAfter=0)
    eqs = S("eq", fontName="ArialVN", fontSize=10, leading=15, alignment=1, textColor=colors.HexColor("#1E3A8A"))

    P = lambda t, st=body: Paragraph(t, st)
    bl = lambda t: Paragraph(t, bullet, bulletText="•")
    f = lambda t: Paragraph(t, cell)
    c_ = lambda t: Paragraph(t, cellc)
    hd = lambda t: Paragraph(t, head)

    def fig(key, width_in, caption):
        img = Image(figs[key]); r = img.imageHeight / img.imageWidth
        img.drawWidth, img.drawHeight = width_in * inch, width_in * inch * r
        return KeepTogether([img, P(f"<b>{fg(key)}.</b> " + caption, cap)])

    def callout(items, bg="#EFF6FF", edge="#1E3A8A"):
        t = Table([[items]], colWidths=[7.0 * inch])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                               ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(edge)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                               ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        return t

    def table(rows, widths):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])]))
        return t

    def captioned(key, caption, tbl):
        return KeepTogether([P(f"<b>{tb(key)}.</b> " + caption, cap), tbl])

    def eq_box(text):
        t = Table([[Paragraph(text, eqs)]], colWidths=[7.0 * inch])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        return t

    def pct(a, m, k="top1"):
        v, e = res[a][str(m)][k]
        return f"{v:.2f} ± {e:.2f}"

    dec = lambda x, n=2: (f"{x:.{n}f}").replace(".", ",") if lang == "vi" else f"{x:.{n}f}"
    r5 = lambda a, k="top1": res[a]["5"][k][0]
    sc = {s["key"]: s for s in phys["scenarios"]}
    mc = phys["mc"]
    cv = lambda fk, key: conv["faults"][fk][key][-1]
    p40 = np.mean([cv(x, "p_mean") for x in FAULTS])
    h40 = np.mean([cv(x, "h_mean") for x in FAULTS])

    s = []
    s += [P(T("Tìm nguyên nhân gốc của hiện tượng rung chatter<br/>trong gia công CNC bằng mô phỏng và suy luận nhân quả",
              "Root Cause Analysis of Machining Chatter in CNC Milling<br/>via Physics Simulation and Causal Inference"), title),
          P(T("Đặng Hải Đăng · Dự án RCA-for-Chatter-Simulation · 2026", "Dang Hai Dang · RCA-for-Chatter-Simulation project · 2026"), sub),
          HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=8)]

    # 1 ----------------------------------------------------------------------
    s.append(P(T("1. Tóm tắt trong một phút", "1. One-minute summary"), h1))
    s.append(callout([
        P(T("<b>Bài toán.</b> Máy CNC đang cắt thì bắt đầu rung mạnh (gọi là <i>chatter</i>), bề mặt chi tiết xấu đi. Cảm biến chỉ cho biết “đang rung”, "
            "không cho biết <b>vì sao</b>: do gá lỏng, dao mòn, cắt quá sâu hay chạy sai tốc độ? Tìm ra nguyên nhân thật gọi là <b>RCA</b> (Root Cause Analysis – phân tích nguyên nhân gốc).",
            "<b>The problem.</b> A CNC machine suddenly starts to vibrate violently while cutting (<i>chatter</i>) and the part surface is ruined. Sensors only say “it is vibrating”, "
            "not <b>why</b>: a loose fixture, a worn tool, too deep a cut, or a bad spindle speed? Finding the real reason is called <b>RCA</b> (Root Cause Analysis)."), boxs),
        Spacer(1, 4),
        P(T("<b>Hệ thống.</b> Quá trình cắt là một <b>phương trình dao động có trễ (DDE)</b>. Đầu vào gồm tham số vật liệu–dao–máy (trôi chậm), tham số vận hành (điều khiển được) và các bất định; "
            "đầu ra là dao động và lực cắt, nhưng trên máy thật ta chỉ nhìn thấy chúng <b>gián tiếp</b> qua cảm biến như accelerometer, micro, dòng điện trục chính.",
            "<b>The system.</b> The cutting process is a <b>delay differential equation (DDE)</b>. Inputs are material–tool–machine parameters (slowly drifting), operating parameters (controllable) and uncertainties; "
            "outputs are vibration and cutting force, but on a real machine they are seen only <b>indirectly</b> through sensors such as accelerometers, microphones and spindle current."), boxs),
        Spacer(1, 4),
        P(T("<b>Dự án gồm hai thí nghiệm.</b> <b>(1) Mô phỏng vật lý:</b> búp ổn định, sóng rung theo thời gian, chẩn đoán bằng mạng Bayes nhỏ. "
            "<b>(2) Benchmark RCA:</b> 14 đại lượng sinh từ mô hình đại số, cố ý gây lỗi, 6 thuật toán đoán nguyên nhân (2.500 lần thử mỗi thuật toán).",
            "<b>The project has two experiments.</b> <b>(1) Physics simulation:</b> stability lobes, time-domain vibration, a small Bayesian-network diagnosis. "
            "<b>(2) RCA benchmark:</b> 14 variables from an algebraic model, injected faults, 6 algorithms guessing the cause (2,500 trials per algorithm)."), boxs),
        Spacer(1, 4),
        P(T(f"<b>Kết quả chính.</b> (1) Mạng Bayes đúng ngay lần đầu <b>{mc['top1']*100:.0f}%</b>, đúng trong 3 hạng đầu <b>{mc['top3']*100:.0f}%</b> (120 lần chạy). "
            f"(2) Với 5 mẫu bất thường, BRCD đúng ngay lần đầu <b>{r5('BRCD')*100:.0f}%</b>, các thuật toán nhân quả khác ≈ 63–64%, BARO {r5('BARO')*100:.0f}%, SimpleRCA {r5('SimpleRCA')*100:.0f}%; "
            "từ 50 mẫu trở lên các thuật toán nhân quả đạt ≈ 99–100%.",
            f"<b>Main results.</b> (1) The Bayesian network is right on the first guess <b>{mc['top1']*100:.0f}%</b> of the time and within the top 3 <b>{mc['top3']*100:.0f}%</b> (120 runs). "
            f"(2) With 5 anomalous samples, BRCD is right first time <b>{r5('BRCD')*100:.0f}%</b> of the time, the other causal methods ≈ 63–64%, BARO {r5('BARO')*100:.0f}%, SimpleRCA {r5('SimpleRCA')*100:.0f}%; "
            "from 50 samples on, the causal methods reach ≈ 99–100%."), boxs),
        Spacer(1, 4),
        P(T("<b>Lưu ý quan trọng.</b> Kết quả đều trên <b>dữ liệu mô phỏng</b>. Đặc biệt, benchmark Thí nghiệm 2 đưa cả những đại lượng <b>không đo được trên máy thật</b> (độ cứng k, độ cản ζ, hệ số lực cắt K<sub>t</sub>…) vào dữ liệu quan sát, "
            "nên bài toán dễ hơn thực tế và con số gần 100% không thể suy ra cho máy thật (xem mục 3.3 và 6.2).",
            "<b>Important.</b> All results are on <b>simulated data</b>. In particular, the Experiment 2 benchmark feeds <b>quantities that cannot be measured on a real machine</b> (stiffness k, damping ζ, cutting coefficient K<sub>t</sub>…) to the algorithms as observed data, "
            "so the problem is easier than reality and near-100% accuracy cannot be extrapolated to a real machine (see Sections 3.3 and 6.2)."), boxs),
    ]))

    # 2 ----------------------------------------------------------------------
    s.append(P(T("2. Chatter là gì và vì sao khó tìm nguyên nhân?", "2. What is chatter and why is its cause hard to find?"), h1))
    s.append(P(T("Khi dao cắt qua phôi, dao hơi rung và để lại bề mặt gợn sóng. Nhát cắt tiếp theo gặp bề mặt gợn sóng đó nên rung mạnh thêm, rồi lại để lại sóng lớn hơn… "
                 "Vòng lặp tự khuếch đại này là <b>chatter tái sinh</b>. Nó xảy ra khi chiều sâu cắt <i>a</i> vượt một <b>giới hạn ổn định</b> <i>a</i><sub>lim</sub>; "
                 "giới hạn này phụ thuộc độ cứng, độ cản của hệ máy–dao–phôi, hệ số lực cắt và tốc độ trục chính [1].",
                 "As the tool cuts, it vibrates slightly and leaves a wavy surface. The next pass meets that wavy surface and vibrates even more, leaving bigger waves… "
                 "This self-amplifying loop is <b>regenerative chatter</b>. It occurs when the depth of cut <i>a</i> exceeds a <b>stability limit</b> <i>a</i><sub>lim</sub>, "
                 "which depends on the stiffness and damping of the machine–tool–workpiece system, the cutting-force coefficient and the spindle speed [1].")))
    s.append(P(T("Khó khăn là <b>chuỗi lan truyền</b>: nguyên nhân làm <i>a</i><sub>lim</sub> giảm → chatter → rung, lực cắt, công suất, độ nhám đều tăng. "
                 "Nhìn vào cảm biến thì <i>mọi thứ</i> đều bất thường, và những đại lượng thay đổi nhiều nhất (lực đỉnh, độ nhám) lại là <b>triệu chứng</b>, không phải nguyên nhân.",
                 "The difficulty is the <b>propagation chain</b>: a cause lowers <i>a</i><sub>lim</sub> → chatter → vibration, forces, power and roughness all rise. "
                 "On the sensors <i>everything</i> looks abnormal, and the quantities that change most (peak force, roughness) are <b>symptoms</b>, not causes.")))

    # 3 (mô hình hệ thống) ----------------------------------------------------
    s.append(P(T("3. Mô hình hệ thống của bài toán", "3. System model of the problem"), h1))
    s.append(P(T(f"{fg('system')} tóm tắt cả hệ thống: đầu vào đi qua quá trình cắt (một DDE), sinh ra các đại lượng vật lý, ta chỉ quan sát chúng qua cảm biến, rồi thuật toán RCA phải suy ngược về nguyên nhân. "
                 "Các mục dưới đây trả lời lần lượt: đầu vào là gì, hệ thống là gì, quan sát được gì, có điều khiển vòng kín không, khi nào xảy ra bất thường và “nguyên nhân gốc” nằm ở đâu.",
                 f"{fg('system')} summarises the whole system: inputs pass through the cutting process (a DDE) and produce physical quantities; we observe them only through sensors, and the RCA algorithm must reason backwards to the cause. "
                 "The subsections below answer in turn: what the inputs are, what the system is, what is observable, whether there is closed-loop control, when anomalies occur and where the “root cause” lies.")))
    s.append(fig("system", 7.0, T("Sơ đồ khối của hệ thống. Nét đứt: không đo trực tiếp được. Vòng điều khiển kín (đường đứt xám phía dưới) có trong thực tế nhưng <b>không</b> được mô hình hóa trong dự án.",
                                  "Block diagram of the system. Dashed outline: not directly measurable. The closed control loop (grey dashed line at the bottom) exists in practice but is <b>not</b> modelled in this project.")))
    s.append(P(T("3.1 Đầu vào: ba nhóm", "3.1 Inputs: three groups"), h2))
    s.append(P(T("Chia đầu vào thành “tĩnh” và “động” là đúng hướng nhưng chưa đủ, vì nhóm “tĩnh” thực ra <b>trôi theo thời gian</b>. Nên chia thành ba nhóm:",
                 "Splitting inputs into “static” and “dynamic” is the right direction but not enough, because the “static” group actually <b>drifts over time</b>. Three groups are more accurate:")))
    rows = [[hd(T("Nhóm", "Group")), hd(T("Ví dụ", "Examples")), hd(T("Thay đổi thế nào", "How it changes")), hd(T("Vai trò trong RCA", "Role in RCA"))],
            [f(T("<b>Tham số vật liệu–dao–máy</b>", "<b>Material–tool–machine parameters</b>")), f(T("hệ số lực cắt K<sub>t</sub>, K<sub>r</sub>; hệ số cản quá trình C<sub>p</sub>; độ cứng k, cản ζ, tần số riêng ω<sub>n</sub>; số răng", "cutting coefficients K<sub>t</sub>, K<sub>r</sub>; process-damping coefficient C<sub>p</sub>; stiffness k, damping ζ, natural frequency ω<sub>n</sub>; tooth count")),
             f(T("cố định tương đối, <b>trôi chậm</b>: mòn dao, phôi thành mỏng mất vật liệu, vị trí trục máy [1]", "roughly fixed but <b>drifting slowly</b>: tool wear, thin-wall part losing material, machine axis position [1]")), f(T("ứng viên nguyên nhân gốc", "root-cause candidates"))],
            [f(T("<b>Tham số vận hành</b>", "<b>Operating parameters</b>")), f(T("tốc độ trục chính n, chiều sâu cắt a, độ ăn dao hướng kính, lượng chạy dao", "spindle speed n, axial depth a, radial immersion, feed")),
             f(T("điều khiển được, đặt trong chương trình CNC", "controllable, set in the CNC program")), f(T("ứng viên nguyên nhân (lập trình sai, override)", "candidates (bad programming, overrides)"))],
            [f(T("<b>Bất định</b>", "<b>Uncertainty</b>")), f(T("độ đảo dao, độ cứng vật liệu dao động, nhiễu cảm biến, điều kiện đầu", "tool runout, material hardness variation, sensor noise, initial conditions")),
             f(T("ngẫu nhiên", "random")), f(T("gây nhiễu; có thể kích hoạt chatter", "adds noise; can trigger chatter"))]]
    s.append(captioned("inputs", T("Ba nhóm đầu vào của hệ thống cắt.", "Three groups of inputs to the cutting system."), table(rows, [1.35 * inch, 2.25 * inch, 2.1 * inch, 1.3 * inch])))
    s.append(P(T("3.2 Hệ thống: một phương trình dao động có trễ", "3.2 The system: a delay differential equation"), h2))
    s.append(P(T("Đúng như bạn nghĩ, hệ thống là một DDE. Với cắt trực giao [1, Eq. 15–16], dao có độ cứng <i>k</i>, cản <i>ζ</i>, tần số riêng <i>ω</i><sub>n</sub> chịu lực cắt phụ thuộc bề dày phoi "
                 "<i>h</i>(<i>t</i>) = <i>h</i><sub>0</sub> − [<i>r</i>(<i>t</i>) − <i>r</i>(<i>t</i> − <i>T</i>)], trong đó <i>T</i> = 60/<i>n</i> là độ trễ tái sinh (một vòng quay) và <i>C</i><sub>p</sub> là hệ số cản quá trình:",
                 "As you suspected, the system is a DDE. For orthogonal cutting [1, Eq. 15–16], a tool with stiffness <i>k</i>, damping <i>ζ</i> and natural frequency <i>ω</i><sub>n</sub> is driven by a force that depends on chip thickness "
                 "<i>h</i>(<i>t</i>) = <i>h</i><sub>0</sub> − [<i>r</i>(<i>t</i>) − <i>r</i>(<i>t</i> − <i>T</i>)], where <i>T</i> = 60/<i>n</i> is the regeneration delay (one revolution) and <i>C</i><sub>p</sub> is the process-damping coefficient:")))
    s.append(eq_box("<i>r</i>″ + 2<i>ζω</i><sub>n</sub><i>r</i>′ + <i>ω</i><sub>n</sub><sup>2</sup><i>r</i> = (<i>ω</i><sub>n</sub><sup>2</sup>/<i>k</i>)·[ <i>K</i><sub>r</sub><i>K</i><sub>t</sub><i>a</i>·<i>h</i>(<i>t</i>) − <i>a</i><i>C</i><sub>p</sub><i>r</i>′/<i>v</i><sub>c</sub> ]"))
    s.append(Spacer(1, 4))
    s.append(P(T("Bài báo cũng đưa ra điều kiện ổn định ở miền tần số, dùng để vẽ biểu đồ búp ổn định (Mục 4.1): <i>a</i><sub>lim</sub> = −1/(2<i>K</i><sub>r</sub><i>K</i><sub>t</sub><i>G</i>(<i>ω</i><sub>c</sub>)) [1, Eq. 19], "
                 "và giới hạn thấp nhất của Tlusty <i>a</i><sub>min</sub> = 2<i>kζ</i>/(<i>K</i><sub>r</sub><i>K</i><sub>t</sub>) [1, Eq. 20]. Hai công thức này cho thấy: <b>cứng hơn, cản nhiều hơn → cắt sâu được hơn; hệ số lực cắt lớn hơn (dao mòn, vật liệu cứng) → phải cắt nông hơn</b>.",
                 "The paper also gives the frequency-domain stability condition used to draw stability-lobe diagrams (Section 4.1): <i>a</i><sub>lim</sub> = −1/(2<i>K</i><sub>r</sub><i>K</i><sub>t</sub><i>G</i>(<i>ω</i><sub>c</sub>)) [1, Eq. 19], "
                 "and Tlusty’s lower bound <i>a</i><sub>min</sub> = 2<i>kζ</i>/(<i>K</i><sub>r</sub><i>K</i><sub>t</sub>) [1, Eq. 20]. Together they say: <b>stiffer and more damped → deeper cuts are possible; a larger cutting coefficient (worn tool, harder material) → shallower cuts are needed</b>.")))
    s.append(P(T("<b>Đối chiếu với dự án:</b> công thức surrogate của Thí nghiệm 2 có <i>a</i><sub>lim</sub> tỷ lệ với <i>k</i>·<i>ζ</i>/<i>K</i><sub>t</sub>, <b>cùng dạng phụ thuộc</b> với Eq. 20, nên hợp lý về xu hướng. "
                 "Tuy nhiên hằng số 0,85 và hệ số lượn sóng 1 − 0,35·sin(4π·rpm) là do dự án tự đặt, không đến từ bài báo; mô hình cũng bỏ qua cản quá trình và <i>K</i><sub>r</sub>. "
                 "Bộ mô phỏng miền thời gian của Thí nghiệm 1 là DDE một dạng dao động, có lực cắt bằng 0 khi dao “nhảy ra khỏi” phôi, không có cản quá trình.",
                 "<b>Comparison with the project:</b> the Experiment 2 surrogate has <i>a</i><sub>lim</sub> proportional to <i>k</i>·<i>ζ</i>/<i>K</i><sub>t</sub>, the <b>same dependence</b> as Eq. 20, so it is plausible in trend. "
                 "However the constant 0.85 and the lobing factor 1 − 0.35·sin(4π·rpm) are the project’s own choices, not from the paper; the model also omits process damping and <i>K</i><sub>r</sub>. "
                 "The Experiment 1 time-domain simulator is a single-mode DDE whose force drops to zero when the tool “jumps out” of the cut, with no process damping.")))
    s.append(P(T("3.3 Quan sát được gì trên máy thật?", "3.3 What can be observed on a real machine?"), h2))
    s.append(P(T("Bạn đúng: đại lượng quyết định “có chatter hay không” là dao động tương đối giữa dao và phôi, nhưng đo trực tiếp ở mũi dao rất khó hoặc đắt. "
                 "Các nghiên cứu phát hiện chatter chủ yếu dùng accelerometer, micro và dynamometer [2]; dòng điện trục chính hoặc tín hiệu nội bộ của CNC cũng truyền được tần số chatter mà không cần cảm biến ngoài [3].",
                 "You are right: the quantity that decides “chatter or not” is the relative vibration between tool and workpiece, but measuring it directly at the tool tip is hard or expensive. "
                 "Chatter-detection research mainly uses accelerometers, microphones and dynamometers [2]; spindle current or internal CNC signals can also carry the chatter frequency without extra sensors [3].")))
    rows = [[hd(T("Đại lượng", "Quantity")), hd(T("Trong mô phỏng", "In simulation")), hd(T("Trên máy thật", "On a real machine"))],
            [f(T("Dao động tương đối dao–phôi r(t)", "Relative tool–workpiece vibration r(t)")), f(T("có", "available")), f(T("khó đo trực tiếp; đo gián tiếp qua gia tốc, âm thanh", "hard to measure directly; inferred via acceleration, sound"))],
            [f(T("Gia tốc / âm thanh (rung RMS, tần số đỉnh, tỷ số phổ)", "Acceleration / sound (vibration RMS, peak frequency, spectral ratio)")), f(T("có", "available")), f(T("<b>quan sát được</b>, rẻ, phổ biến [2]", "<b>observable</b>, cheap, common [2]"))],
            [f(T("Dòng điện / công suất trục chính, tín hiệu CNC", "Spindle current / power, CNC signals")), f(T("có", "available")), f(T("<b>quan sát được</b>, không cần cảm biến ngoài [3]", "<b>observable</b>, no external sensor needed [3]"))],
            [f(T("Lực cắt (dynamometer)", "Cutting force (dynamometer)")), f(T("có", "available")), f(T("đo chính xác nhưng đắt, chủ yếu ở phòng thí nghiệm [2]", "accurate but costly, mostly in laboratories [2]"))],
            [f(T("Độ nhám bề mặt Ra", "Surface roughness Ra")), f(T("có", "available")), f(T("thường đo sau gia công, không phải tín hiệu thời gian thực", "usually measured after cutting, not a real-time signal"))],
            [f(T("Lệnh a, n (giá trị lập trình)", "Commanded a, n (programmed values)")), f(T("có", "available")), f(T("<b>biết</b> từ chương trình CNC (giá trị thực có thể khác do override)", "<b>known</b> from the CNC program (actual values may differ due to overrides)"))],
            [f(T("k, ζ, K<sub>t</sub>, C<sub>p</sub>, a<sub>lim</sub>, mức chatter", "k, ζ, K<sub>t</sub>, C<sub>p</sub>, a<sub>lim</sub>, chatter severity")), f(T("có (biết chính xác)", "available (known exactly)")), f(T("<b>không đo trực tiếp</b>; chỉ suy ra hoặc đo ngoại tuyến (vd thử va đập lấy FRF [1])", "<b>not directly measurable</b>; only inferred or measured offline (e.g. impact test for the FRF [1])"))]]
    s.append(captioned("observ", T("Đại lượng nào quan sát được trong mô phỏng so với trên máy thật.", "Which quantities are observable in simulation versus on a real machine."), table(rows, [2.5 * inch, 1.3 * inch, 3.2 * inch])))
    s.append(Spacer(1, 4))
    s.append(P(T("<b>Hệ quả cho dự án.</b> Thí nghiệm 1 làm đúng nguyên tắc: nguyên nhân là biến ẩn, bằng chứng chỉ là triệu chứng nhị phân rút từ đặc trưng cảm biến. "
                 f"Thí nghiệm 2 thì <b>không</b>: bảng 14 cột ({fg('cause_effect')}) coi k, ζ, K<sub>t</sub>, a<sub>lim</sub> và mức chatter là cột quan sát được (cộng nhiễu cảm biến). "
                 "Khi một nguyên nhân bị gây lỗi, chính cột của nó lệch ngay, nên thuật toán chỉ cần tìm cột lệch (có hiệu chỉnh theo đồ thị) mà không phải suy luận ẩn. Đây là lý do chính khiến độ chính xác tiến tới 100% và là hạn chế lớn nhất của benchmark (mục 6.2).",
                 "<b>Consequence for the project.</b> Experiment 1 follows the right principle: causes are hidden variables and the evidence is binary symptoms derived from sensor features. "
                 f"Experiment 2 does <b>not</b>: its 14-column table ({fg('cause_effect')}) treats k, ζ, K<sub>t</sub>, a<sub>lim</sub> and chatter severity as observed columns (plus sensor noise). "
                 "When a cause is faulted, its own column shifts immediately, so an algorithm only has to find the shifted column (adjusted for the graph) instead of doing hidden-variable inference. This is the main reason accuracy approaches 100% and the biggest limitation of the benchmark (Section 6.2).")))
    s.append(P(T("3.4 Vòng hở hay vòng kín?", "3.4 Open loop or closed loop?"), h2))
    s.append(P(T("Trong dự án, mọi đầu vào vận hành đều <b>cố định</b> (mỗi kịch bản một cặp n, a không đổi): đó là <b>vòng hở</b>. Cần phân biệt hai loại “vòng kín”:",
                 "In the project every operating input is <b>fixed</b> (each scenario has one constant pair n, a): that is <b>open loop</b>. Two kinds of “closed loop” must be distinguished:")))
    s.append(bl(T("<b>Vòng hồi tiếp tái sinh</b> nằm sẵn trong bản thân quá trình cắt: bài báo [1] nêu rằng Merritt đã biểu diễn nó thành hệ vòng kín có hồi tiếp trễ. Đây là tính chất vật lý, không phải bộ điều khiển, và DDE ở mục 3.2 đã chứa nó.",
                  "The <b>regenerative feedback loop</b> is built into the cutting process itself: the paper [1] notes that Merritt expressed it as a closed-loop system with delayed feedback. It is a physical property, not a controller, and the DDE in Section 3.2 already contains it.")))
    s.append(bl(T("<b>Điều khiển chatter vòng kín</b> có tồn tại trong thực tế: biến thiên tốc độ trục chính (SSV) thích nghi dùng tín hiệu gia tốc làm hồi tiếp [4], giảm chấn chủ động bằng cơ cấu áp điện [5]. "
                  "Các tổng quan ghi nhận SSV hiệu quả với chatter nhân đôi chu kỳ nhưng không hiệu quả với chatter bán tuần hoàn phía trên búp Hopf. Dự án <b>không</b> mô hình hóa các bộ điều khiển này.",
                  "<b>Closed-loop chatter control</b> does exist in practice: adaptive spindle speed variation (SSV) using acceleration as feedback [4], and active damping with piezoelectric actuators [5]. "
                  "Reviews report that SSV works against period-doubling chatter but not against quasi-periodic chatter above the Hopf lobe. The project does <b>not</b> model these controllers.")))
    s.append(P(T("Theo suy luận của tôi (chưa có nguồn trực tiếp): nếu máy có điều khiển thích nghi, bộ điều khiển sẽ bù và có thể che nguyên nhân gốc, đồng thời tạo vòng lặp trong đồ thị nhân quả; khi đó RCA cần xử lý khác với trường hợp vòng hở.",
                 "In my own reasoning (no direct source yet): if the machine has adaptive control, the controller compensates and may mask the root cause, and it creates cycles in the causal graph; RCA would then need different handling than in the open-loop case.")))
    s.append(P(T("3.5 Khi nào xảy ra bất thường?", "3.5 When do anomalies occur?"), h2))
    s.append(P(T("Hệ chạy bình thường khi điểm vận hành (n, a) nằm trong vùng ổn định và tham số đúng danh định. Nhưng “không nhiễu” <b>không</b> đồng nghĩa “luôn normal”. Các nguyên nhân bất thường chính:",
                 "The system runs normally when the operating point (n, a) lies in the stable region and the parameters are nominal. But “no noise” does <b>not</b> mean “always normal”. The main causes of anomalies:")))
    s.append(bl(T("<b>Điểm vận hành sai ngay từ đầu:</b> nếu <i>a</i> &gt; <i>a</i><sub>lim</sub> với tham số danh định (lập trình quá sâu, chọn tốc độ ở vùng trũng) thì chatter là tất định; nhiễu chỉ đóng vai kích hoạt.",
                  "<b>Wrong operating point from the start:</b> if <i>a</i> &gt; <i>a</i><sub>lim</sub> with nominal parameters (programmed too deep, speed in a valley), chatter is deterministic; noise only acts as a trigger.")))
    s.append(bl(T("<b>Tham số của hệ trôi đi</b> (đây là nguyên nhân thường gặp hơn việc “input vận hành bị đổi”): hệ số lực cắt và cản quá trình thay đổi theo mòn dao, động lực học phôi thành mỏng đổi liên tục khi mất vật liệu, mode tần thấp của máy lớn đổi theo vị trí [1, mục 6]. "
                  "Bài báo kết luận độ chính xác dự đoán còn bị giới hạn bởi bất định đo lường, phi tuyến và động lực học biến thiên theo thời gian.",
                  "<b>Drifting system parameters</b> (more common than “operating inputs got changed”): cutting and process-damping coefficients change with tool wear, thin-wall workpiece dynamics change as material is removed, low-frequency modes of large machines shift with position [1, Sec. 6]. "
                  "The paper concludes that prediction accuracy is still limited by measurement uncertainty, nonlinearities and time-varying dynamics.")))
    s.append(bl(T("<b>Vùng lưỡng ổn định:</b> dưới biên ổn định vẫn có “vùng không an toàn” nơi cắt ổn định và chatter cùng tồn tại; nhiễu nhỏ thì ổn định, nhiễu đủ lớn thì chuyển sang chatter [6]. Bài báo [1, mục 3.3] cũng mô tả các dạng bifurcation khác (Hopf thứ cấp, period-2, period-3). Dự án chưa mô hình hóa hiện tượng này.",
                  "<b>Bistable zones:</b> below the stability boundary there are “unsafe zones” where stable cutting and chatter coexist; small disturbances decay, large enough ones switch the process to chatter [6]. The paper [1, Sec. 3.3] also describes other bifurcations (secondary Hopf, period-2, period-3). The project does not model this.")))
    s.append(bl(T("<b>Rung cưỡng bức:</b> Arnold phân biệt rung cưỡng bức với tự kích [1]. Khi phay, “normal” vẫn có rung ở tần số răng cắt, không phải rung bằng 0.",
                  "<b>Forced vibration:</b> Arnold distinguishes forced from self-induced vibration [1]. In milling, “normal” still includes vibration at the tooth-passing frequency; it is not zero vibration.")))
    s.append(P(T("3.6 “Nguyên nhân gốc” nằm ở đâu?", "3.6 Where does the “root cause” lie?"), h2))
    s.append(P(T("Các tham số trong mô hình chính là <b>nguyên nhân cơ học trực tiếp</b>. Nguyên nhân gốc đúng nghĩa của RCA thường nằm sâu hơn một bậc. Chuỗi nhân quả đầy đủ là:",
                 "The parameters in the model are the <b>direct mechanical causes</b>. The root cause in the RCA sense usually lies one level deeper. The full causal chain is:")))
    rows = [[hd(T("Tầng", "Level")), hd(T("Ví dụ", "Example")), hd(T("Quan sát được?", "Observable?"))],
            [f(T("1. Sự kiện lỗi (nguyên nhân gốc thực sự)", "1. Fault event (the real root cause)")), f(T("gá lỏng, dao nhô dài, mòn mặt sau dao, lập trình chiều sâu quá lớn, chọn sai tốc độ", "loose clamp, long overhang, flank wear, over-deep program, wrong speed")), f(T("không", "no"))],
            [f(T("2. Tham số bị đổi (nguyên nhân cơ học)", "2. Changed parameter (mechanical cause)")), f("k↓, ζ↓, K<sub>t</sub>↑, C<sub>p</sub>, a↑, n lệch" if lang == "vi" else "k↓, ζ↓, K<sub>t</sub>↑, C<sub>p</sub>, a↑, n off"), f(T("không (trừ a, n)", "no (except a, n)"))],
            [f(T("3. Biên ổn định a<sub>lim</sub> giảm", "3. Stability limit a<sub>lim</sub> drops")), f(T("a &gt; a<sub>lim</sub>", "a &gt; a<sub>lim</sub>")), f(T("không", "no"))],
            [f(T("4. Chatter", "4. Chatter")), f(T("rung tái sinh tăng dần", "growing regenerative vibration")), f(T("gián tiếp", "indirectly"))],
            [f(T("5. Triệu chứng", "5. Symptoms")), f(T("rung RMS, tần số đỉnh, lực, công suất, độ nhám", "vibration RMS, peak frequency, force, power, roughness")), f(T("có (qua cảm biến)", "yes (through sensors)"))]]
    s.append(table(rows, [2.5 * inch, 3.2 * inch, 1.3 * inch]))
    s.append(Spacer(1, 4))
    s.append(P(T("Quan hệ giữa tầng 1 và 2 <b>không phải 1–1</b>: dao nhô dài làm đổi cả k lẫn ζ, mòn dao làm đổi cả K<sub>t</sub> lẫn C<sub>p</sub> [1]. Vì vậy hai lỗi khác nhau có thể cho triệu chứng giống nhau, và RCA có thể chỉ phân biệt được đến mức tham số chứ chưa chắc đến mức sự kiện lỗi. "
                 "<b>Dự án hiện gộp tầng 1 và 2 làm một:</b> “5 nguyên nhân gốc” thực chất là 5 tham số.",
                 "The mapping between levels 1 and 2 is <b>not one-to-one</b>: a long overhang changes both k and ζ, tool wear changes both K<sub>t</sub> and C<sub>p</sub> [1]. Two different faults can therefore produce similar symptoms, and RCA may only be able to separate them down to the parameter level rather than the fault-event level. "
                 "<b>The project currently merges levels 1 and 2:</b> its “5 root causes” are really 5 parameters.")))

    s.append(P(T("3.7 Nghiên cứu đã làm được đến đâu về nguyên nhân gốc?", "3.7 What has research achieved on root causes so far?"), h2))
    s.append(P(T("Tôi đã tra các nguồn ở phần tài liệu tham khảo. Kết luận ngắn: quan hệ <b>tham số → ổn định</b> được biết rất rõ; quan hệ <b>điều kiện hoặc lỗi → tham số</b> đã được đo lẻ tẻ cho từng yếu tố; "
                 "còn một bản đồ thống nhất <b>lỗi → tham số → ổn định → triệu chứng</b>, kèm dấu của từng tác động, thì tôi <b>chưa tìm thấy</b> trong các nguồn đã tra.",
                 "I searched the sources listed in the references. In short: the relation <b>parameter → stability</b> is very well known; the relation <b>condition or fault → parameter</b> has been measured piecemeal for individual factors; "
                 "but a unified map <b>fault → parameter → stability → symptoms</b> with the sign of each effect I <b>did not find</b> in the sources I checked.")))
    s.append(bl(T("<b>Đã biết chắc:</b> luật ổn định của Tlusty và Tobias từ thập niên 1950; giới hạn thấp nhất <i>a</i><sub>min</sub> = 2<i>kζ</i>/(<i>K</i><sub>r</sub><i>K</i><sub>t</sub>); phụ thuộc tốc độ qua độ trễ <i>T</i> = 60/<i>n</i>; cản quá trình làm ổn định hơn ở tốc độ thấp [1].",
                  "<b>Firmly known:</b> the Tlusty and Tobias stability laws from the 1950s; the lower bound <i>a</i><sub>min</sub> = 2<i>kζ</i>/(<i>K</i><sub>r</sub><i>K</i><sub>t</sub>); the speed dependence through the delay <i>T</i> = 60/<i>n</i>; process damping adding stability at low speed [1].")))
    s.append(bl(T("<b>Đã đo cho từng yếu tố:</b> dao nhô dài từ 35 lên 65 mm làm tần số riêng bậc 1 rơi từ 565 Hz xuống 325 Hz [15], và nhô từ 35 lên 70 mm làm <i>a</i><sub>lim</sub> giảm trên cả ba vật liệu thử [14]; "
                  "hệ số lực cắt thay đổi mạnh theo vật liệu [14]; động lực học phôi thành mỏng và máy lớn biến thiên theo vị trí [1][20].",
                  "<b>Measured factor by factor:</b> a longer overhang from 35 to 65 mm drops the first natural frequency from 565 Hz to 325 Hz [15], and from 35 to 70 mm lowers <i>a</i><sub>lim</sub> for all three tested materials [14]; "
                  "cutting coefficients vary strongly with material [14]; the dynamics of thin-wall parts and large machines vary with position [1][20].")))
    s.append(bl(T("<b>Còn nhiều bất định:</b> <i>K</i><sub>t</sub> và <i>C</i><sub>p</sub> có bất định lớn; mòn dao đổi tiếp xúc mặt sau nên đổi cản quá trình [1, tr. 17]. Dao mòn làm lực cắt tăng nhưng đồng thời làm tăng cản quá trình [16][21], nên <b>tác động ròng lên chatter không đơn điệu</b>.",
                  "<b>Much remains uncertain:</b> <i>K</i><sub>t</sub> and <i>C</i><sub>p</sub> carry large uncertainty; wear changes the flank contact and hence process damping [1, p. 17]. Wear raises cutting force but also raises process damping [16][21], so the <b>net effect on chatter is not monotonic</b>.")))
    s.append(bl(T("<b>Danh mục lỗi:</b> FMEA/FMECA của máy CNC liệt kê dạng lỗi theo bộ phận (trục chính mòn hoặc kẹt, dao mẻ hoặc mòn, lỗi kẹp…) [18]; một công trình knowledge graph cho chẩn đoán lỗi CNC dùng các loại thực thể: thiết bị, mô-đun, tham số, cảnh báo, hiện tượng, nguyên nhân, giải pháp [19]; "
                  "Wehner và cộng sự kết hợp mạng Bayes nhân quả với KG cho RCA trong sản xuất [7]. Các công trình này mô tả lỗi ở mức bộ phận hoặc hiện tượng, <b>không</b> định lượng tác động lên <i>a</i><sub>lim</sub>.",
                  "<b>Fault catalogues:</b> FMEA/FMECA of CNC machines lists failure modes by component (spindle wear or seizure, tool chipping or wear, clamping faults…) [18]; a knowledge-graph work for CNC fault diagnosis uses the entity types equipment, module, parameter, alarm, phenomenon, cause, solution [19]; "
                  "Wehner et al. combine a causal Bayesian network with a KG for manufacturing RCA [7]. These describe faults at component or phenomenon level and do <b>not</b> quantify the effect on <i>a</i><sub>lim</sub>.")))
    s.append(bl(T("<b>Khoảng trống:</b> chưa có bảng hay đồ thị thống nhất nối nguyên nhân với tham số máy–dao–vận hành kèm dấu tác động. Nguồn cho “gá phôi lỏng” và “ổ trục mòn” tôi mới thấy ở blog kỹ thuật [17], độ tin cậy thấp.",
                  "<b>Gap:</b> there is no unified table or graph linking causes to machine–tool–operating parameters with signs of effect. For “loose workholding” and “worn spindle bearings” I only found engineering blogs [17], a low-reliability source.")))
    s.append(P(T("3.8 Danh sách nguyên nhân gốc và quan hệ với tham số", "3.8 Root causes and their relations to the parameters"), h2))
    s.append(P(T(f"{fg('rootgraph')} là đồ thị tôi <b>tổng hợp từ các nguồn trên</b>, không phải kết quả thực nghiệm của dự án. Cột trái là 9 nguyên nhân gốc (chia theo nhóm: dao–bộ gá, phôi, máy, vận hành); cột giữa là tham số máy–dao–vật liệu (màu vàng) và tham số vận hành (màu xanh dương); cột phải là ổn định. "
                 "Mỗi cạnh ghi <b>số nguồn</b> trong danh sách tài liệu tham khảo; <b>kiểu nét</b> cho biết mức kiểm chứng (liền: tôi đã đọc trong nguồn; đứt: nguồn thứ cấp hoặc chỉ đọc tóm tắt; chấm: nguồn yếu); <b>màu</b> cho biết chiều tác động lên ổn định.",
                 f"{fg('rootgraph')} is a graph I <b>compiled from the sources above</b>; it is not an experimental result of this project. The left column holds 9 root causes (grouped as tool–holder, workpiece, machine, operation); the middle column holds machine–tool–material parameters (yellow) and operating parameters (blue); the right column is stability. "
                 "Each edge carries the <b>source number</b> from the reference list; the <b>line style</b> shows the level of verification (solid: I read it in the source; dashed: secondary source or summary only; dotted: weak source); the <b>colour</b> shows the direction of the effect on stability.")))
    s.append(fig("rootgraph", 7.0, T("Quan hệ giữa nguyên nhân gốc, tham số máy–dao–vật liệu, tham số vận hành và ổn định, kèm nguồn của từng quan hệ (số trong ngoặc vuông là số tài liệu tham khảo).",
                                      "Relations among root causes, machine–tool–material parameters, operating parameters and stability, with the source of each relation (bracketed numbers refer to the reference list).")))
    rows = [[hd(T("Nhóm", "Group")), hd(T("Nguyên nhân gốc", "Root cause")), hd(T("Tham số bị ảnh hưởng và chiều", "Affected parameters and direction")), hd(T("Nguồn", "Source")), hd(T("Mức kiểm chứng", "Verification level"))],
            [f(T("Dao – bộ gá", "Tool – holder")), f("R1 " + T("Dao nhô dài", "Long tool overhang")), f(T("FRF: <i>k</i>↓, <i>ω</i><sub>n</sub>↓; cản <i>ζ</i>: không nhất quán giữa các nguồn", "FRF: <i>k</i>↓, <i>ω</i><sub>n</sub>↓; damping <i>ζ</i>: inconsistent across sources")), c_("[14][15]"), f(T("đã đọc (2 bài phay)", "read (2 milling papers)"))],
            [f(T("Phôi", "Workpiece")), f("R2 " + T("Phôi thành mỏng", "Thin-wall part")), f(T("FRF của phôi biến thiên theo vị trí dao khi mất vật liệu", "workpiece FRF varies with tool position as material is removed")), c_("[1][20]"), f(T("[1] đã đọc; [20] chỉ tóm tắt", "[1] read; [20] summary only"))],
            [f(T("Phôi", "Workpiece")), f("R3 " + T("Gá phôi lỏng, đỡ kém", "Loose or poor workholding")), f(T("<i>k</i>↓ phía phôi", "<i>k</i>↓ on the workpiece side")), c_("[17]"), f(T("chỉ blog kỹ thuật", "engineering blog only"))],
            [f(T("Máy", "Machine")), f("R4 " + T("Ổ trục chính mòn, mối nối lỏng", "Worn spindle bearings, loose joints")), f(T("độ cứng động <i>k</i>↓", "dynamic stiffness <i>k</i>↓")), c_("[17] ([18])"), f(T("blog; [18] chỉ liệt kê như dạng lỗi, không về ổn định", "blog; [18] only lists it as a failure mode, not about stability"))],
            [f(T("Máy", "Machine")), f("R5 " + T("Vị trí trục máy (máy lớn)", "Machine axis position (large machines)")), f(T("mode tần thấp dịch theo vị trí", "low-frequency modes shift with position")), c_("[1]"), f(T("đã đọc", "read"))],
            [f(T("Phôi", "Workpiece")), f("R6 " + T("Vật liệu khác danh định", "Material differs from nominal")), f(T("<i>K</i><sub>t</sub>, <i>K</i><sub>r</sub> đổi (vd <i>K</i><sub>t</sub> ≈ 774–984 N/mm² cho nhôm 6061; ≈ 1913–2431 cho thép 45)", "<i>K</i><sub>t</sub>, <i>K</i><sub>r</sub> change (e.g. <i>K</i><sub>t</sub> ≈ 774–984 N/mm² for 6061 aluminium; ≈ 1913–2431 for 45 steel)")), c_("[14]"), f(T("đã đọc", "read"))],
            [f(T("Dao – bộ gá", "Tool – holder")), f("R7 " + T("Mòn mặt sau dao", "Tool flank wear")), f(T("<i>K</i><sub>t</sub>↑ (lực cắt tăng) [16]; <i>C</i><sub>p</sub>↑ (cản quá trình) [1][16][21]. <b>Tác động ròng không đơn điệu</b>", "<i>K</i><sub>t</sub>↑ (cutting force rises) [16]; <i>C</i><sub>p</sub>↑ (process damping) [1][16][21]. <b>Net effect is not monotonic</b>")), c_("[1][16][21]"), f(T("[1], [16] đã đọc; [21] chỉ tóm tắt", "[1], [16] read; [21] summary only"))],
            [f(T("Vận hành", "Operation")), f("R8 " + T("Lập trình chiều sâu quá lớn", "Over-deep programmed depth")), f(T("<i>a</i> tăng; chatter khi <i>a</i> &gt; <i>a</i><sub>lim</sub>", "<i>a</i> rises; chatter when <i>a</i> &gt; <i>a</i><sub>lim</sub>")), c_("[1]"), f(T("đã đọc", "read"))],
            [f(T("Vận hành", "Operation")), f("R9 " + T("Tốc độ ở vùng trũng", "Speed in a lobe valley")), f(T("<i>n</i> đổi độ trễ <i>T</i> = 60/<i>n</i>; trùng tần số răng với mode riêng", "<i>n</i> changes the delay <i>T</i> = 60/<i>n</i>; tooth-passing frequency coincides with a mode")), c_("[1]"), f(T("đã đọc", "read"))]]
    s.append(captioned("rootcause", T("Danh sách nguyên nhân gốc, tham số bị ảnh hưởng và nguồn của từng quan hệ.", "Root causes, affected parameters and the source of each relation."), table(rows, [0.8 * inch, 1.5 * inch, 2.6 * inch, 0.8 * inch, 1.3 * inch])))
    s.append(Spacer(1, 4))
    s.append(P(T("Ngoài ra, đặc trưng dao (đường kính, vật liệu, số răng) cũng làm đổi FRF đầu dao [14]; đây là lựa chọn thiết kế chứ không phải lỗi, nên tôi vẽ nó như một tham số chứ không như nguyên nhân gốc.",
                 "In addition, tool features (diameter, material, tooth count) also change the tool-tip FRF [14]; this is a design choice rather than a fault, so I draw it as a parameter and not as a root cause.")))
    s.append(P(T("<b>Đối chiếu với các thí nghiệm của dự án.</b> “5 nguyên nhân gốc” của thí nghiệm 2 chỉ phủ một phần đồ thị: “độ cứng giảm” gộp lẫn R1, R3, R4; “độ cản giảm” chỉ là một phần của R1; “dao mòn” chỉ mô hình hóa phần <i>K</i><sub>t</sub>↑ của R7 và <b>bỏ qua <i>C</i><sub>p</sub>↑ có tác dụng ổn định</b>; “cắt quá sâu” là R8 và “tốc độ không hợp” là R9. "
                 "Ba nguyên nhân R2, R5, R6 chưa có trong thí nghiệm nào.",
                 "<b>Comparison with the project’s experiments.</b> The “5 root causes” of Experiment 2 cover only part of the graph: “stiffness drop” mixes R1, R3, R4; “damping loss” is only part of R1; “tool wear” models only the <i>K</i><sub>t</sub>↑ part of R7 and <b>ignores the stabilising <i>C</i><sub>p</sub>↑</b>; “depth overload” is R8 and “RPM mismatch” is R9. "
                 "Three causes, R2, R5 and R6, appear in neither experiment.")))

    # 4 ----------------------------------------------------------------------
    s.append(P(T("4. Thí nghiệm 1 – Mô phỏng vật lý và chẩn đoán bằng mạng Bayes", "4. Experiment 1 – Physics simulation and Bayesian-network diagnosis"), h1))
    s.append(P(T("<b>Mục tiêu:</b> kiểm tra xem vật lý của chatter có để lại những “dấu vân tay” phân biệt được hay không, và liệu từ các triệu chứng quan sát được ta có chẩn đoán được nguyên nhân ẩn. "
                 "<b>Cách làm:</b> (i) tính biểu đồ búp ổn định bằng giải tích; (ii) mô phỏng DDE miền thời gian cho 6 kịch bản; (iii) rút 6 triệu chứng có/không từ đặc trưng tín hiệu; (iv) dùng một mạng Bayes nhỏ suy ra nguyên nhân. "
                 "<b>Đo:</b> tỷ lệ chẩn đoán đúng (Top-1, Top-3) trên 120 lần chạy có nhiễu.",
                 "<b>Goal:</b> test whether the physics of chatter leaves distinguishable “fingerprints” and whether the hidden cause can be diagnosed from observable symptoms. "
                 "<b>Method:</b> (i) compute the stability-lobe diagram analytically; (ii) simulate the time-domain DDE for 6 scenarios; (iii) derive 6 yes/no symptoms from signal features; (iv) infer the cause with a small Bayesian network. "
                 "<b>Measured:</b> diagnosis accuracy (Top-1, Top-3) over 120 noisy runs.")))
    s.append(P(T("4.1 Biểu đồ búp ổn định: chọn tốc độ và chiều sâu nào thì an toàn?", "4.1 Stability-lobe diagram: which speed and depth are safe?"), h2))
    s.append(P(T("Biểu đồ <b>búp ổn định</b> cho biết, ở mỗi tốc độ trục chính, chiều sâu cắt lớn nhất mà máy còn êm. "
                 "<b>Dưới đường cong</b> là cắt ổn định; <b>trên đường cong</b> là chatter. Ở một số tốc độ đặc biệt (đỉnh búp) ta cắt sâu được hơn hẳn, còn ở “vùng trũng” chỉ cắt rất nông. "
                 f"{fg('lobes')} được tính giải tích bằng mã Python của dự án (<i>chatter_simulation.py</i>), tái hiện hai ví dụ trong giáo trình Altintas.",
                 "A <b>stability-lobe</b> diagram shows, for each spindle speed, the deepest cut at which the machine still runs smoothly. "
                 "<b>Below the curve</b> cutting is stable; <b>above it</b> chatter occurs. At certain speeds (lobe peaks) a much deeper cut is possible, while in the “valleys” only very shallow cuts are safe. "
                 f"{fg('lobes')} is computed analytically with the project’s Python code (<i>chatter_simulation.py</i>), reproducing two textbook examples from Altintas.")))
    s.append(fig("lobes", 7.0, T("Biểu đồ búp ổn định: (a) máy bào 2 bậc tự do với hai dạng dao động 250 Hz và 150 Hz; (b) phay rãnh và phay biên ăn dao một nửa. Dưới đường cong: cắt ổn định; trên đường cong: chatter.",
                                  "Stability-lobe diagrams: (a) a 2-DOF shaping machine with two vibration modes at 250 Hz and 150 Hz; (b) slotting and half-immersion down-milling. Below a curve: stable cutting; above it: chatter.")))
    s.append(P(T("4.2 Sóng rung theo thời gian", "4.2 Vibration over time"), h2))
    s.append(P(T("Dự án có bộ mô phỏng <b>miền thời gian</b> (<i>rca_chatter_bayesian.py</i>) giải DDE bằng tích phân số (Euler–Cromer): lực cắt tại thời điểm <i>t</i> phụ thuộc độ gợn sóng mà nhát cắt trước (cách đó <i>T</i> = 60/rpm) để lại. "
                 f"{fg('signatures')} cho thấy sóng rung và phổ tần số của ba trường hợp: bình thường, độ cứng giảm 50% (tần số rung dịch từ 250 Hz xuống ≈ 187 Hz đo được) và cắt quá sâu (rung to dần, hình thành chatter).",
                 "The project has a <b>time-domain</b> simulator (<i>rca_chatter_bayesian.py</i>) that integrates the DDE numerically (Euler–Cromer): the force at time <i>t</i> depends on the waviness left by the previous pass (<i>T</i> = 60/rpm earlier). "
                 f"{fg('signatures')} shows the waveform and spectrum of three cases: normal, stiffness −50% (the vibration frequency drops from 250 Hz to a measured ≈ 187 Hz) and excessive depth (vibration grows, i.e. chatter).")))
    s.append(fig("signatures", 6.9, T("“Dấu vân tay” vật lý của từng nguyên nhân: sóng dịch chuyển (trái) và phổ tần số (phải). Thang trục tung khác nhau giữa các hàng.",
                                       "Physical “fingerprint” of each cause: displacement waveform (left) and frequency spectrum (right). Vertical scales differ between rows.")))
    s.append(P(T("4.3 Chẩn đoán bằng mạng Bayes nhỏ", "4.3 Diagnosis with a small Bayesian network"), h2))
    s.append(P(T("Từ mỗi lần mô phỏng ta rút ra 6 “triệu chứng” có/không (rung mạnh? đỉnh chatter? tần số lệch? lực cao? chiều sâu lập trình lớn? tốc độ ở vùng trũng?), "
                 f"rồi dùng <b>mạng Bayes</b> với 6 giả thuyết (bình thường + 5 nguyên nhân) tính xác suất từng giả thuyết theo quy tắc Bayes. {tb('bn')} là kết quả cho 6 kịch bản.",
                 "From each simulation we extract 6 yes/no “symptoms” (strong vibration? chatter peak? frequency shift? high force? large programmed depth? speed in a valley?) "
                 f"and use a <b>Bayesian network</b> with 6 hypotheses (normal + 5 causes) to compute each hypothesis’ probability by Bayes’ rule. {tb('bn')} shows the results for 6 scenarios.")))
    nm = {"Normal": T("Bình thường", "Normal"), "Stiffness": T("Độ cứng −50%", "Stiffness −50%"), "Damping": T("Độ cản giảm (a = 8 mm)", "Damping loss (a = 8 mm)"),
          "ToolWear": T("Dao mòn (K<sub>f</sub> +120%)", "Tool wear (K<sub>f</sub> +120%)"), "Depth": T("Cắt quá sâu (a = 28 mm)", "Excessive depth (a = 28 mm)"),
          "RPM": T("Tốc độ 8.800 rpm", "Spindle 8,800 rpm")}
    dn = {"Normal": T("Bình thường", "Normal"), "Stiffness_Degradation": T("Độ cứng giảm", "Stiffness drop"), "Damping_Loss": T("Độ cản giảm", "Damping loss"),
          "Tool_Wear": T("Dao mòn", "Tool wear"), "Excessive_Cut_Depth": T("Cắt quá sâu", "Excessive depth"), "Unstable_Spindle_RPM": T("Tốc độ không ổn định", "Unstable speed")}
    rows = [[hd(T("Kịch bản (nguyên nhân thật)", "Scenario (true cause)")), hd(T("Rung RMS (µm)", "Vib. RMS (µm)")), hd(T("Tần số đỉnh (Hz)", "Peak freq. (Hz)")),
             hd(T("Chẩn đoán", "Diagnosis")), hd(T("Xác suất", "Prob.")), hd(T("Đúng?", "Correct?"))]]
    for k in ("Normal", "Stiffness", "Damping", "ToolWear", "Depth", "RPM"):
        r = sc[k]
        rows.append([f(nm[k]), c_(dec(r["rms"])), c_(f"{r['dom_freq']:.0f}"), f(dn[r["diagnosed"]]), c_(f"{r['prob']*100:.0f}%"), c_(T("Đúng", "Yes") if r["correct"] else T("<b>Sai</b>", "<b>No</b>"))])
    s.append(captioned("bn", T("Chẩn đoán của mạng Bayes cho 6 kịch bản mô phỏng.", "Bayesian-network diagnosis for 6 simulated scenarios."), table(rows, [1.9 * inch, 0.95 * inch, 0.95 * inch, 1.6 * inch, 0.8 * inch, 0.8 * inch])))
    s.append(Spacer(1, 5))
    s.append(P(T(f"Kịch bản “độ cản giảm” bị chẩn đoán sai: với a = 8 mm, 3.800 rpm, mô phỏng thực tế vẫn <b>ổn định</b> (rung chỉ {dec(sc['Damping']['rms'])} µm, dưới ngưỡng 5 µm) nên mọi triệu chứng đều “không” "
                 f"và mạng Bayes kết luận “bình thường” – hợp lý với dữ liệu, nhưng nhãn lỗi lại là độ cản giảm. Chạy thêm 120 lần mô phỏng có nhiễu (20 lần × 6 kịch bản): <b>Top-1 = {mc['top1']*100:.1f}%</b>, <b>Top-3 = {mc['top3']*100:.0f}%</b>.",
                 f"The “damping loss” scenario is misdiagnosed: at a = 8 mm, 3,800 rpm the simulation actually stays <b>stable</b> (vibration only {dec(sc['Damping']['rms'])} µm, below the 5 µm threshold), so every symptom is “no” "
                 f"and the network concludes “normal” – reasonable given the data, yet the fault label says damping loss. Over 120 additional noisy runs (20 × 6 scenarios): <b>Top-1 = {mc['top1']*100:.1f}%</b>, <b>Top-3 = {mc['top3']*100:.0f}%</b>.")))
    s.append(P(T("4.4 Điều rút ra từ thí nghiệm 1", "4.4 Takeaways from Experiment 1"), h2))
    s.append(bl(T("Mỗi nguyên nhân để lại dấu hiệu khác nhau trong sóng và phổ (tần số dịch xuống khi độ cứng giảm; rung phình to khi cắt quá sâu), nên về nguyên tắc có thể phân biệt từ cảm biến.",
                  "Each cause leaves a different mark in the waveform and spectrum (frequency shifts down when stiffness drops; vibration grows when cutting too deep), so in principle they can be told apart from sensors.")))
    s.append(bl(T(f"Với nguyên nhân là biến ẩn và bằng chứng là triệu chứng quan sát được (đúng như trên máy thật), mạng Bayes cho Top-1 {mc['top1']*100:.0f}% và Top-3 {mc['top3']*100:.0f}%.",
                  f"With hidden causes and observable symptoms as evidence (as on a real machine), the Bayesian network reaches Top-1 {mc['top1']*100:.0f}% and Top-3 {mc['top3']*100:.0f}%.")))
    s.append(bl(T("Giới hạn: chỉ 6 giả thuyết, bảng xác suất và ngưỡng do tác giả đặt tay; ca “độ cản giảm” cho thấy một lỗi có thể không để lại triệu chứng nào nếu điểm vận hành vẫn ổn định (liên hệ mục 3.6).",
                  "Limits: only 6 hypotheses, hand-set probability tables and thresholds; the “damping loss” case shows a fault may leave no symptom if the operating point stays stable (see Section 3.6).")))

    # 5 (Thí nghiệm 2) ------------------------------------------------------------
    s.append(P(T("5. Thí nghiệm 2 – Benchmark RCA với 14 đại lượng", "5. Experiment 2 – RCA benchmark with 14 variables"), h1))
    s.append(P(T("Thí nghiệm 1 chỉ có 6 giả thuyết và triệu chứng nhị phân. Thí nghiệm 2 nhiều biến liên tục hơn: thuật toán thấy <b>14 đại lượng</b> và phải xếp hạng đúng nguyên nhân gốc giữa cả 14. "
                 "Để chạy hàng nghìn thử nghiệm, phần này dùng một <b>mô hình đại số đơn giản</b> thay cho bộ giải thời gian (mã: <i>run_full_paper_benchmark.py</i>). <b>Dữ liệu Thí nghiệm 2 không lấy từ mô phỏng Thí nghiệm 1.</b> "
                 "Như đã phân tích ở mục 3.3, 5 trong 14 cột (k, ζ, K<sub>t</sub>, a<sub>lim</sub>, mức chatter) <b>không đo được trên máy thật</b> nhưng vẫn nằm trong dữ liệu quan sát.",
                 "Experiment 1 has only 6 hypotheses and binary symptoms. Experiment 2 has more continuous variables: the algorithm sees <b>14 quantities</b> and must rank the true root cause among all 14. "
                 "To run thousands of trials it uses a <b>simple algebraic model</b> instead of the time-domain solver (code: <i>run_full_paper_benchmark.py</i>). <b>Experiment 2 data does not come from the Experiment 1 simulation.</b> "
                 "As discussed in Section 3.3, 5 of the 14 columns (k, ζ, K<sub>t</sub>, a<sub>lim</sub>, chatter severity) <b>cannot be measured on a real machine</b> yet are included in the observed data.")))
    s.append(P(T("5.1 Dữ liệu mô phỏng và cách gây lỗi", "5.1 Simulated data and fault injection"), h2))
    s.append(fig("cause_effect", 7.0, T("14 đại lượng và chiều ảnh hưởng (mũi tên đi từ nguyên nhân đến hệ quả). Nét đứt: không đo trực tiếp được trên máy thật.",
                                         "The 14 variables and their direction of influence (arrows run from cause to effect). Dashed outline: not directly measurable on a real machine.")))
    s.append(P(T("<b>Đầu vào</b> – 5 số đã <b>chuẩn hóa</b> (không có đơn vị SI): <i>k</i>, ζ, K<sub>t</sub> quanh 1,0 (độ lệch 3,5%); <i>a</i>, <i>rpm</i> quanh 0,50 (độ lệch 0,025). "
                 "Vì vậy “rpm = 0,5” <b>không</b> phải 0,5 vòng/phút. <b>Đầu ra</b> – bảng N dòng × 14 cột, tính như sau (nhiễu = số ngẫu nhiên nhỏ mô phỏng sai số cảm biến):",
                 "<b>Inputs</b> – 5 <b>normalised</b> numbers (no SI units): <i>k</i>, ζ, K<sub>t</sub> around 1.0 (3.5% spread); <i>a</i>, <i>rpm</i> around 0.50 (0.025 spread). "
                 "So “rpm = 0.5” is <b>not</b> 0.5 revolutions per minute. <b>Outputs</b> – an N × 14 table computed as follows (noise = small random numbers imitating sensor error):")))
    rows = [[hd(T("Cột", "Col.")), hd(T("Đại lượng", "Quantity")), hd(T("Cách tính (rút gọn)", "How it is computed (simplified)"))],
            [f("0–4"), f("k, ζ, K<sub>t</sub>, a, rpm"), f(T("5 đầu vào, cộng nhiễu cảm biến", "the 5 inputs plus sensor noise"))],
            [f("5"), f(T("a<sub>lim</sub> – giới hạn ổn định", "a<sub>lim</sub> – stability limit")), f(T("0,85·k·ζ / (K<sub>t</sub>·độ cứng vật liệu) × (1 − 0,35·sin(4π·rpm)) + nhiễu", "0.85·k·ζ / (K<sub>t</sub>·material hardness) × (1 − 0.35·sin(4π·rpm)) + noise"))],
            [f("6"), f(T("Chatter Severity – mức rung", "Chatter Severity")), f(T("0,25·ln(1 + e<sup>9·(a − a<sub>lim</sub>)</sup>) + nhiễu; tăng nhanh khi a vượt a<sub>lim</sub>", "0.25·ln(1 + e<sup>9·(a − a<sub>lim</sub>)</sup>) + noise; rises fast once a exceeds a<sub>lim</sub>"))],
            [f("7"), f(T("Rung RMS (µm)", "Vibration RMS (µm)")), f(T("0,12 + 1,5 × mức rung + nhiễu", "0.12 + 1.5 × severity + noise"))],
            [f("8"), f(T("Tần số chủ đạo (Hz)", "Dominant frequency (Hz)")), f(T("250·√k + 5 × mức rung + nhiễu", "250·√k + 5 × severity + noise"))],
            [f("9"), f(T("Tỷ số phổ", "Spectral ratio")), f(T("hàm bão hòa theo mức rung (0,08 → ≈ 0,78)", "saturating function of severity (0.08 → ≈ 0.78)"))],
            [f("10"), f(T("Lực cắt trung bình (N)", "Mean cutting force (N)")), f(T("300 × K<sub>t</sub> × độ cứng vật liệu × a + nhiễu", "300 × K<sub>t</sub> × material hardness × a + noise"))],
            [f("11"), f(T("Lực cắt đỉnh (N)", "Peak cutting force (N)")), f(T("lực TB × (1,15 + 1,5 × mức rung) + nhiễu", "mean force × (1.15 + 1.5 × severity) + noise"))],
            [f("12"), f(T("Công suất trục chính (W)", "Spindle power (W)")), f(T("0,75 × lực TB × rpm + 0,12 × lực đỉnh + nhiễu", "0.75 × mean force × rpm + 0.12 × peak force + noise"))],
            [f("13"), f(T("Độ nhám bề mặt Ra (µm)", "Surface roughness Ra (µm)")), f(T("0,35 + 2,5 × rung RMS + 0,0008 × lực đỉnh + nhiễu", "0.35 + 2.5 × vibration RMS + 0.0008 × peak force + noise"))]]
    s.append(table(rows, [0.6 * inch, 2.3 * inch, 4.1 * inch]))
    s.append(P(T("Độ cứng vật liệu phôi dao động ±5% là một yếu tố gây nhiễu thêm.", "Workpiece hardness varies by ±5% as an additional confounder."), cap))
    s.append(fig("pipeline", 7.0, T("Quy trình thí nghiệm. Dữ liệu “bình thường” là mốc so sánh; dữ liệu “bất thường” có đúng một nguyên nhân gốc bị dịch chuyển.",
                                     "Experimental procedure. “Normal” data is the reference; “anomalous” data has exactly one root cause shifted.")))
    s.append(P(T("<b>Lỗi được gây ra thế nào?</b> Mỗi loại lỗi dịch một tham số khoảng 1,2–2,7 lần độ lệch chuẩn tự nhiên (ví dụ k giảm ≈ 5,5%, dao mòn làm K<sub>t</sub> tăng ≈ 6,5%) – lỗi khá nhẹ, dễ bị nhiễu che. Mỗi lần thử gồm 5.000 mẫu bình thường (D_obs) và m mẫu bất thường (D_int, m từ 5 đến 100).",
                 "<b>How are faults injected?</b> Each fault shifts one parameter by about 1.2–2.7 natural standard deviations (e.g. k −5.5%, tool wear raises K<sub>t</sub> by ≈ 6.5%) – fairly subtle and easily masked by noise. Each trial has 5,000 normal samples (D_obs) and m anomalous samples (D_int, m from 5 to 100).")))
    s.append(P(T("5.2 Thuật toán và cách chấm điểm", "5.2 Algorithms and scoring"), h2))
    s.append(P(T("<b>Chấm điểm:</b> thuật toán xếp hạng cả 14 đại lượng từ nghi ngờ nhất đến ít nhất. <b>Top-1</b> = nguyên nhân thật đứng hạng nhất; <b>Top-3 / Top-5</b> = nằm trong 3 / 5 hạng đầu (kỹ sư chỉ cần kiểm tra vài mục đầu danh sách).",
                 "<b>Scoring:</b> each algorithm ranks all 14 variables from most to least suspect. <b>Top-1</b> = the true cause is ranked first; <b>Top-3 / Top-5</b> = it is within the first 3 / 5 (an engineer only needs to inspect a few items).")))
    s.append(P(T("<b>Sáu thuật toán</b> (cách chấm điểm mỗi đại lượng, theo mã trong benchmark):", "<b>The six algorithms</b> (how each variable is scored, as coded in the benchmark):")))
    rows = [[hd(T("Tên", "Name")), hd(T("Cách chấm điểm", "Scoring rule"))],
            [f("<b>SimpleRCA</b>"), f(T("Giá trị lớn nhất trong dữ liệu bất thường vượt mức 95% của dữ liệu bình thường bao nhiêu. Không dùng đồ thị nhân quả.", "How far the largest anomalous value exceeds the 95th percentile of normal data. No causal graph."))],
            [f("<b>BARO</b>"), f(T("Độ lệch trung bình so với trung vị chia cho khoảng tứ phân vị (IQR) – độ lệch “bền vững”. Không dùng đồ thị nhân quả.", "Mean deviation from the median divided by the interquartile range (IQR) – a “robust” deviation. No causal graph."))],
            [f("<b>RCD</b>"), f(T("Sau khi loại ảnh hưởng của các đại lượng cha, xem đại lượng đó còn tương quan với việc “mẫu này bất thường” hay không (kiểm định thống kê).", "After removing the effect of its parent variables, test whether the variable still correlates with “this sample is anomalous” (statistical test)."))],
            [f("<b>RCG</b>"), f(T("Sau khi trừ phần do đại lượng cha giải thích, phần dư dịch chuyển bao nhiêu độ lệch chuẩn.", "After subtracting what the parents explain, how many standard deviations the residual has shifted."))],
            [f("<b>SmoothTraversal</b>"), f(T("Độ lệch của đại lượng trừ 70% độ lệch của đại lượng cha: nếu cha đã lệch nhiều thì nó ít bị nghi hơn.", "Deviation of the variable minus 70% of its parent’s deviation: if the parent already deviates a lot, the variable is less suspect."))],
            [f("<b>BRCD</b>"), f(T("Giống ý tưởng “phần dư dịch chuyển”, nhân thêm hệ số theo số đại lượng chịu ảnh hưởng, rồi quy thành xác suất (softmax) cho 5 nguyên nhân.", "Similar “residual shift” idea, multiplied by a factor for the number of downstream variables, then turned into probabilities (softmax) over the 5 causes."))]]
    s.append(table(rows, [1.5 * inch, 5.5 * inch]))
    s.append(P(T(f"Bốn thuật toán dưới cùng dùng đồ thị nhân quả ở {fg('cause_effect')}. Ví dụ ý tưởng: độ nhám tăng mạnh khi có chatter nhưng chỉ vì rung tăng; phần độ nhám <i>không</i> giải thích được bởi rung vẫn bình thường → không bị nghi. SimpleRCA và BARO chỉ thấy “độ nhám lệch nhiều” nên dễ nhầm triệu chứng với nguyên nhân.",
                 f"The bottom four algorithms use the causal graph in {fg('cause_effect')}. Idea by example: roughness rises sharply during chatter but only because vibration rose; the part of roughness that vibration does <i>not</i> explain stays normal → not suspect. SimpleRCA and BARO just see “roughness deviates a lot” and easily confuse symptoms with causes."), cap))

    # 6 ----------------------------------------------------------------------
    s.append(P(T("5.3 Kết quả", "5.3 Results"), h2))
    s.append(P(T("Mỗi giá trị là trung bình của 500 lần thử (100 lần × 5 loại lỗi), kèm sai số chuẩn (± sau dấu). Số 0,71 nghĩa là “đoán đúng 71 trong 100 lần”. <b>Nhớ rằng đây là kết quả khi các cột không đo được vẫn nằm trong dữ liệu</b> (mục 3.3).",
                 "Each value is the mean of 500 trials (100 × 5 fault types) with the standard error (after ±). A value of 0.71 means “right 71 times out of 100”. <b>Remember that this is with the unmeasurable columns still included in the data</b> (Section 3.3).")))
    s.append(fig("accuracy", 7.0, T("Độ chính xác theo số mẫu bất thường m. Thêm bằng chứng thì mọi thuật toán nhân quả đều tốt lên và hội tụ về ≈ 100%; BARO dừng ở ≈ 0,7 (Top-1); SimpleRCA ≈ 0 ở Top-1.",
                                    "Accuracy versus number of anomalous samples m. With more evidence all causal methods improve and converge to ≈ 100%; BARO plateaus near 0.7 (Top-1); SimpleRCA stays near 0 on Top-1.")))
    ms = [5, 10, 20, 50, 100]
    rows = [[hd(T("Thuật toán", "Algorithm"))] + [hd(f"m = {m}") for m in ms]]
    for a in ALGOS:
        rows.append([f(f"<b>{a}</b>" if a == "BRCD" else a)] + [c_(f"<b>{pct(a, m)}</b>" if a == "BRCD" else pct(a, m)) for m in ms])
    s.append(captioned("top1", T("Top-1 theo số mẫu m.", "Top-1 by number of samples m."), table(rows, [1.6 * inch] + [1.08 * inch] * 5)))
    s.append(Spacer(1, 6))
    s.append(fig("by_fault", 5.8, T("Top-1 theo từng loại lỗi khi chỉ có m = 5 mẫu: “độ cứng giảm” khó nhất; “cắt quá sâu” và “độ cản giảm” dễ nhất.",
                                     "Top-1 per fault type with only m = 5 samples: “stiffness drop” is hardest; “depth overload” and “damping loss” are easiest.")))
    rows = [[hd(T("Thuật toán", "Algorithm")), hd("Top-1"), hd("Top-3"), hd("Top-5")]]
    for a in ALGOS:
        rows.append([f(a)] + [c_(pct(a, 5, k)) for k in ("top1", "top3", "top5")])
    s.append(captioned("crit", T("Ba tiêu chí chấm điểm khi m = 5.", "The three scoring criteria at m = 5."), table(rows, [1.8 * inch, 1.7 * inch, 1.7 * inch, 1.7 * inch])))
    s.append(Spacer(1, 6))
    s.append(P(T("<b>Đọc kết quả:</b>", "<b>Reading the results:</b>")))
    s.append(bl(T("Khi ít bằng chứng (m = 5), BRCD hơn các thuật toán nhân quả khác ≈ 7–8 điểm phần trăm Top-1 (0,71 so với 0,63–0,64) – khoảng cách nhỏ, cùng cỡ vài lần sai số chuẩn; ở m = 10 còn 4–5 điểm và từ m = 50 gần như không khác biệt.",
                  "With little evidence (m = 5) BRCD beats the other causal methods by ≈ 7–8 percentage points of Top-1 (0.71 vs 0.63–0.64) – a small gap, a few standard errors wide; at m = 10 it is 4–5 points and from m = 50 there is almost no difference.")))
    s.append(bl(T("Nếu kỹ sư kiểm tra 3 mục đầu (Top-3), các thuật toán nhân quả đã chứa nguyên nhân thật ≈ 90% ngay với 5 mẫu.",
                  "If an engineer inspects the first 3 items (Top-3), the causal methods already contain the true cause ≈ 90% of the time with only 5 samples.")))
    s.append(bl(T("Hai phương pháp không dùng đồ thị nhân quả kém hẳn – nhiều khả năng do bị triệu chứng (lực đỉnh, độ nhám… dao động lớn) lấn át; SimpleRCA còn kém đi khi thêm mẫu vì giá trị lớn nhất càng dễ rơi vào đại lượng nhiễu lớn.",
                  "The two methods without a causal graph do much worse – most likely because the symptoms (peak force, roughness… with large fluctuations) dominate; SimpleRCA even gets worse with more samples because the maximum value is more likely to land on a high-noise variable.")))
    s.append(P(T("5.4 Xác suất của BRCD thay đổi thế nào khi thu thêm mẫu?", "5.4 How does BRCD’s probability change as samples arrive?"), h2))
    s.append(P(T(f"{fg('convergence')} vẽ xác suất BRCD gán cho nguyên nhân thật khi mẫu bất thường đến lần lượt (t = 1…40; đường là trung bình, vùng mờ là khoảng 25–75% của 60 lần chạy). "
                 f"Đây là xác suất <b>thô</b>, không làm mượt. Sau 40 mẫu, xác suất trung bình chỉ khoảng <b>{dec(p40)}</b> (đoán mò là 0,2) và entropy còn ≈ <b>{dec(h40)}</b> nats (tối đa 1,61): "
                 "xác suất tăng nhưng <b>không</b> tiến tới 1, độ bất định giảm rất ít. Nguyên nhân là softmax trong mã dùng hệ số cố định 0,40, nên con số này chỉ nên coi là <b>điểm xếp hạng</b>, không phải xác suất đã hiệu chỉnh. "
                 "(Hình hội tụ trong phiên bản benchmark cũ có pha thêm prior và nhiễu nên trông “đẹp” hơn; báo cáo này không dùng nó.)",
                 f"{fg('convergence')} plots the probability BRCD assigns to the true cause as anomalous samples arrive one by one (t = 1…40; line = mean, shaded = 25–75% range over 60 runs). "
                 f"This is the <b>raw</b>, unsmoothed probability. After 40 samples the mean probability is only about <b>{dec(p40)}</b> (guessing is 0.2) and the entropy is still ≈ <b>{dec(h40)}</b> nats (maximum 1.61): "
                 "the probability rises but does <b>not</b> approach 1, and uncertainty falls very little. The cause is the softmax in the code with a fixed factor of 0.40, so this number should be read as a <b>ranking score</b>, not a calibrated probability. "
                 "(The convergence plot in the earlier version of the benchmark blended in the prior and added noise, so it looked nicer; this report does not use it.)")))
    s.append(fig("convergence", 7.0, T("Hội tụ của BRCD theo số mẫu t: (a) xác suất gán cho nguyên nhân thật; (b) entropy của phân phối trên 5 nguyên nhân.",
                                        "BRCD convergence versus sample count t: (a) probability assigned to the true cause; (b) entropy of the distribution over the 5 causes.")))

    # 6 ----------------------------------------------------------------------
    s.append(P(T("6. Thảo luận tổng hợp và hạn chế", "6. Discussion and limitations"), h1))
    s.append(P(T("6.1 So sánh hai thí nghiệm", "6.1 Comparing the two experiments"), h2))
    s.append(P(T("Hai thí nghiệm trả lời hai câu hỏi khác nhau và bổ sung cho nhau, nhưng <b>chưa được nối với nhau</b>: dữ liệu của thí nghiệm 2 không sinh từ bộ giải của thí nghiệm 1.",
                 "The two experiments answer different questions and complement each other, but they are <b>not yet connected</b>: Experiment 2’s data is not generated by Experiment 1’s solver.")))
    rows = [[hd(T("Tiêu chí", "Aspect")), hd(T("Thí nghiệm 1: mô phỏng vật lý + mạng Bayes", "Experiment 1: physics + Bayesian network")), hd(T("Thí nghiệm 2: benchmark 14 đại lượng", "Experiment 2: 14-variable benchmark"))],
            [f(T("Mô hình vật lý", "Physics model")), f(T("DDE một dạng dao động, tích phân số; búp ổn định giải tích", "single-mode DDE, numerical integration; analytic stability lobes")), f(T("công thức đại số đơn giản, không có bộ giải thời gian", "simple algebraic formulas, no time-domain solver"))],
            [f(T("Dữ liệu đưa cho RCA", "Data given to RCA")), f(T("6 triệu chứng nhị phân rút từ đặc trưng cảm biến", "6 binary symptoms derived from sensor features")), f(T("bảng N × 14 đại lượng liên tục", "an N × 14 table of continuous variables"))],
            [f(T("Nguyên nhân", "Causes")), f(T("5 nguyên nhân + “bình thường”, là <b>biến ẩn</b>", "5 causes + “normal”, <b>hidden variables</b>")), f(T("5 tham số, nhưng cột của chúng <b>nằm trong dữ liệu</b>", "5 parameters, but their columns <b>are in the data</b>"))],
            [f(T("Phương pháp", "Method")), f(T("mạng Bayes, bảng xác suất đặt tay", "Bayesian network, hand-set probability tables")), f(T("6 thuật toán (BRCD, RCD, RCG, SmoothTraversal, BARO, SimpleRCA), bản đơn giản hóa", "6 algorithms (BRCD, RCD, RCG, SmoothTraversal, BARO, SimpleRCA), simplified versions"))],
            [f(T("Quy mô", "Scale")), f(T("6 kịch bản + 120 lần chạy có nhiễu", "6 scenarios + 120 noisy runs")), f(T("2.500 lần thử mỗi thuật toán", "2,500 trials per algorithm"))],
            [f(T("Kết quả chính", "Main result")), f(T(f"Top-1 {mc['top1']*100:.1f}%, Top-3 {mc['top3']*100:.0f}%", f"Top-1 {mc['top1']*100:.1f}%, Top-3 {mc['top3']*100:.0f}%")), f(T(f"BRCD Top-1 {r5('BRCD')*100:.0f}% (m = 5) → ≈ 100% (m ≥ 50)", f"BRCD Top-1 {r5('BRCD')*100:.0f}% (m = 5) → ≈ 100% (m ≥ 50)"))],
            [f(T("Điểm mạnh", "Strength")), f(T("đúng nguyên tắc quan sát: nguyên nhân ẩn, bằng chứng là triệu chứng", "follows the observability principle: hidden causes, symptoms as evidence")), f(T("quy mô lớn; so sánh được nhiều thuật toán và số mẫu", "large scale; compares many algorithms and sample sizes"))],
            [f(T("Điểm yếu", "Weakness")), f(T("ít giả thuyết; bảng xác suất và ngưỡng đặt tay", "few hypotheses; hand-set tables and thresholds")), f(T("lộ biến không đo được; tự chấm trên mô hình do tôi viết", "exposes unmeasurable variables; scored on a model we wrote"))]]
    s.append(table(rows, [1.2 * inch, 2.9 * inch, 2.9 * inch]))
    s.append(Spacer(1, 4))
    s.append(P(T("Có thể đọc hai kết quả như sau: thí nghiệm 1 cho thấy chẩn đoán từ triệu chứng quan sát được là khả thi trong một bối cảnh nhỏ; thí nghiệm 2 cho thấy các thuật toán nhân quả hơn hẳn phương pháp thống kê đơn giản, nhưng khoảng cách giữa các thuật toán nhân quả nhỏ và con số tuyệt đối bị thổi phồng bởi việc lộ biến không đo được.",
                 "The two results can be read as follows: Experiment 1 shows that diagnosis from observable symptoms is feasible in a small setting; Experiment 2 shows that causal algorithms clearly beat a simple statistical method, but the gap among causal algorithms is small and the absolute numbers are inflated by the exposure of unmeasurable variables.")))
    s.append(P(T("6.2 Hạn chế", "6.2 Limitations"), h2))
    s.append(callout([
        bl(T("<b>Dữ liệu là mô phỏng, không phải máy thật.</b> Công thức do dự án đặt ra và thuật toán được chấm trên chính mô hình đó. Con số 100% không có nghĩa sẽ đạt 100% trên máy CNC thực.",
             "<b>The data is simulated, not from a real machine.</b> The formulas are the project’s own and the algorithms are scored on that same model. 100% does not mean 100% on a real CNC machine.")),
        bl(T("<b>Benchmark Thí nghiệm 2 làm lộ các biến không đo được.</b> k, ζ, K<sub>t</sub>, a<sub>lim</sub> và mức chatter nằm trong dữ liệu quan sát (mục 3.3). Khi nguyên nhân bị gây lỗi, chính cột của nó lệch, nên bài toán dễ hơn nhiều so với suy luận nguyên nhân ẩn từ cảm biến thật. "
             "Chưa có thí nghiệm nào trong báo cáo này đo độ chính xác khi <b>chỉ</b> dùng các đại lượng quan sát được; con số sẽ thấp hơn, nhưng thấp bao nhiêu thì chưa biết.",
             "<b>The Experiment 2 benchmark exposes unmeasurable variables.</b> k, ζ, K<sub>t</sub>, a<sub>lim</sub> and chatter severity are in the observed data (Section 3.3). When a cause is faulted its own column shifts, so the task is far easier than inferring hidden causes from real sensors. "
             "No experiment in this report measures accuracy when <b>only</b> observable quantities are used; the numbers would be lower, but by how much is unknown.")),
        bl(T("<b>Chưa mô hình hóa:</b> điều khiển vòng kín (mục 3.4), vùng lưỡng ổn định và các dạng bifurcation khác (mục 3.5), cản quá trình và tác động kép của dao mòn, động lực học biến thiên theo thời gian, và quan hệ nhiều–nhiều giữa lỗi và tham số (mục 3.6, 3.8).",
             "<b>Not modelled:</b> closed-loop control (Section 3.4), bistable zones and other bifurcations (Section 3.5), process damping and the dual effect of tool wear, time-varying dynamics, and the many-to-many mapping between faults and parameters (Sections 3.6, 3.8).")),
        bl(T("<b>Thí nghiệm 1 và 2 tách rời.</b> Benchmark 14 biến dùng công thức đại số, không dùng bộ giải miền thời gian của Thí nghiệm 1. Mạng Bayes ở Thí nghiệm 1 có bảng xác suất và ngưỡng triệu chứng do tác giả đặt tay, và triệu chứng “chiều sâu lập trình lớn” được suy trực tiếp từ giá trị a đã biết.",
             "<b>Experiments 1 and 2 are separate.</b> The 14-variable benchmark uses algebraic formulas, not Experiment 1’s time-domain solver. The Experiment 1 Bayesian network has hand-set probability tables and symptom thresholds, and the “large programmed depth” symptom is read directly from the known value of a.")),
        bl(T("<b>Các thuật toán RCA là bản cài đặt đơn giản hóa</b> (ví dụ BRCD ở đây là điểm số phần dư dịch chuyển dạng tuyến tính–Gauss rồi softmax), <b>không</b> phải mã gốc của các bài báo, nên không thể coi là tái lập kết quả của ICML 2026. Hệ số +0,12 điểm cho mỗi đại lượng chịu ảnh hưởng là lựa chọn của dự án.",
             "<b>The RCA algorithms are simplified re-implementations</b> (e.g. BRCD here is a linear-Gaussian residual-shift score followed by a softmax), <b>not</b> the original code of the papers, so this is not a reproduction of the ICML 2026 results. The +0.12 bonus per downstream variable is the project’s own design choice.")),
        bl(T("<b>Các thuật toán nhân quả được cấp sẵn đồ thị nhân quả đúng</b>, và hai bộ dữ liệu có mức nhiễu khác nhau (hệ số 1,0 so với 1,1), có thể vô tình cho thuật toán chút “manh mối”.",
             "<b>The causal algorithms are given the correct causal graph</b>, and the two datasets have different noise levels (factor 1.0 vs 1.1), which may leak a little irrelevant “hint”.")),
    ], bg="#FEF2F2", edge="#B91C1C"))

    # 8 ----------------------------------------------------------------------
    s.append(P(T("7. Hướng phát triển: Bayesian causal network + Knowledge Graph + RAG", "7. Next steps: Bayesian causal network + Knowledge Graph + RAG"), h1))
    s.append(P(T("Kết hợp mạng Bayes nhân quả (CBN), knowledge graph (KG) và RAG cho RCA là <b>khả thi và đã có tiền lệ</b> ở lĩnh vực khác: Wehner và cộng sự [7] kết hợp CBN với KG cho RCA trong sản xuất xe điện "
                 "(KG giúp giảm thời gian học và bớt quan hệ nhân quả giả); Ouarrad và cộng sự [8] dùng KG kết hợp RAG với LLM làm lớp quyết định và giải thích cho RCA trên hệ thống HiL ô tô, đạt Top-1 90% và 94% trên hai case study. "
                 "Tôi chưa tìm thấy công trình nào áp dụng trực tiếp cho chatter, và kết quả ở lĩnh vực khác không tự chuyển sang được.",
                 "Combining a causal Bayesian network (CBN), a knowledge graph (KG) and RAG for RCA is <b>feasible and has precedent</b> in other domains: Wehner et al. [7] combine a CBN with a KG for RCA in electric-vehicle manufacturing "
                 "(the KG shortens learning time and reduces spurious causal links); Ouarrad et al. [8] use a KG with RAG and an LLM as the decision and explanation layer for RCA in automotive HiL systems, reaching Top-1 of 90% and 94% on two case studies. "
                 "I have not found work applying it directly to chatter, and results from other domains do not transfer automatically.")))
    s.append(fig("architecture", 7.0, T("Kiến trúc đề xuất. KG cung cấp cấu trúc, mô phỏng vật lý cung cấp xác suất, cảm biến thực cung cấp bằng chứng cho mạng Bayes; RAG và LLM chỉ giải thích và gợi ý.",
                                         "Proposed architecture. The KG provides structure, the physics simulation provides probabilities and real sensors provide evidence for the Bayesian network; RAG and the LLM only explain and suggest.")))
    s.append(P(T("<b>Phân vai:</b>", "<b>Division of roles:</b>")))
    s.append(bl(T("<b>KG:</b> bộ phận, tham số, dạng lỗi (từ FMEA), cảm biến và quan hệ vật lý (như các công thức ở mục 3.2); ràng buộc cấu trúc của mạng Bayes.",
                  "<b>KG:</b> components, parameters, failure modes (from FMEA), sensors and physical relations (such as the formulas in Section 3.2); constrains the structure of the Bayesian network.")))
    s.append(bl(T("<b>Mạng Bayes nhân quả:</b> suy luận định lượng; nguyên nhân là <b>biến ẩn</b>, bằng chứng chỉ là đặc trưng quan sát được. Xác suất có thể học từ bộ giải DDE của Thí nghiệm 1.",
                  "<b>Causal Bayesian network:</b> quantitative inference; causes are <b>hidden variables</b> and evidence is only observable features. Probabilities can be learned from Experiment 1’s DDE solver.")))
    s.append(bl(T("<b>RAG:</b> truy xuất sổ tay máy, nhật ký bảo trì, bài báo để giải thích và đề xuất kiểm tra. <b>LLM</b> chỉ nên là lớp giải thích, không phải bên quyết định cuối.",
                  "<b>RAG:</b> retrieve machine manuals, maintenance logs and papers to explain and suggest checks. The <b>LLM</b> should only be an explanation layer, not the final decision maker.")))
    s.append(P(T("<b>Lộ trình đề xuất:</b>", "<b>Suggested roadmap:</b>")))
    for t in (T("Sửa benchmark Thí nghiệm 2: <b>ẩn</b> các cột không đo được (k, ζ, K<sub>t</sub>, a<sub>lim</sub>, mức chatter) và đo lại độ chính xác.",
                "Fix the Experiment 2 benchmark: <b>hide</b> the unmeasurable columns (k, ζ, K<sub>t</sub>, a<sub>lim</sub>, chatter severity) and re-measure accuracy."),
              T("Dùng bộ giải DDE (Thí nghiệm 1) hoặc biểu đồ búp ổn định để sinh dữ liệu cho benchmark, thay công thức đại tố; thêm cản quá trình và điều kiện lưỡng ổn định.".replace("đại tố", "đại số"),
                "Generate benchmark data from the DDE solver (Experiment 1) or stability lobes instead of the algebraic formulas; add process damping and bistable conditions."),
              T("Dùng đồ thị ở mục 3.8 làm cấu trúc ban đầu: tách rõ hai tầng nguyên nhân (sự kiện lỗi và tham số), mô hình dao mòn với cả <i>K</i><sub>t</sub> lẫn <i>C</i><sub>p</sub>, bổ sung R2, R5, R6; thử đồ thị học từ dữ liệu hoặc đồ thị sai một phần để đo độ bền.",
                "Use the Section 3.8 graph as the starting structure: separate the two cause levels (fault event and parameter), model tool wear with both <i>K</i><sub>t</sub> and <i>C</i><sub>p</sub>, add R2, R5, R6; try a learned or partly wrong graph to test robustness."),
              T("Dựng KG từ FMEA và các quan hệ vật lý, rồi thêm RAG/LLM làm lớp giải thích; đánh giá riêng độ đúng của phần giải thích.",
                "Build the KG from FMEA and physical relations, then add RAG/LLM as an explanation layer; evaluate the correctness of the explanations separately."),
              T("Chạy mã gốc của RCD, BARO, BRCD thay cho bản đơn giản hóa; hiệu chỉnh xác suất của BRCD; thử trên dữ liệu thật có accelerometer, dòng điện trục chính và lực cắt.",
                "Run the original code of RCD, BARO, BRCD instead of simplified versions; calibrate BRCD’s probabilities; test on real data with accelerometers, spindle current and cutting force.")):
        s.append(bl(t))

    s.append(P(T("Tài liệu tham khảo", "References"), h1))
    refs = ["[1] Altintas, Y., Stepan, G., Budak, E., Schmitz, T., Kilic, Z. M. (2020). Chatter Stability of Machining Operations. <i>J. Manuf. Sci. Eng.</i> 142(11), 110801. DOI 10.1115/1.4047391. (" + T("đã đọc toàn văn bản trong thư mục dự án", "full text read from the project folder") + ")",
            "[2] Chatter detection methods in the machining processes: A review. (ResearchGate, researchgate.net/publication/359387627). (" + T("chỉ đọc phần tóm tắt trong kết quả tìm kiếm", "only the search-result summary was read") + ")",
            "[3] Chatter detection by monitoring spindle drive current. (ResearchGate, researchgate.net/publication/226455150). (" + T("chỉ đọc phần tóm tắt", "summary only") + ")",
            "[4] Online chatter suppression in turning by adaptive amplitude modulation of spindle speed variation. (ResearchGate, researchgate.net/publication/327190980). (" + T("chỉ đọc phần tóm tắt", "summary only") + ")",
            "[5] Robust active control based milling chatter suppression with perturbation model via piezoelectric stack actuators. (ScienceDirect, S0888327018307192). (" + T("chỉ đọc phần tóm tắt", "summary only") + ")",
            "[6] On the bistable zone of milling processes. <i>Phil. Trans. R. Soc. A</i> 373(2051), 20140409. (" + T("chỉ đọc phần tóm tắt trong kết quả tìm kiếm", "only the search-result summary was read") + ")",
            "[7] Wehner, C., Kertel, M., Wewerka, J. (2024). Interactive and Intelligent Root Cause Analysis in Manufacturing with Causal Bayesian Networks and Knowledge Graphs. arXiv:2402.00043. (" + T("chỉ đọc phần tóm tắt", "abstract only") + ")",
            "[8] Ouarrad, H., Abboush, M., Rausch, A. (2026). Knowledge-Graph-Guided Retrieval-Augmented LLMs for Explainable Root Cause Analysis in Automotive HiL Validation. arXiv:2608.11277. (" + T("chỉ đọc phần tóm tắt", "abstract only") + ")",
            "[9] Altintas, Y. (2012). <i>Manufacturing Automation</i> (2nd ed.). Cambridge University Press.",
            "[10] Cristello, J. (2022). <i>Machine Tool Vibrations – Assignment 4</i> (ENME 619L01).",
            "[11] Lee, K., Zhou, Z., Kocaoglu, M. (2026). Root Cause Analysis of Failures via Bayesian Root Cause Discovery (BRCD). ICML.",
            "[12] Ikram, A. et al. (2022). Root cause analysis of failures in microservices through causal discovery (RCD). NeurIPS 35.",
            "[13] Pham, L., Ha, H., Zhang, H. (2024). BARO: Robust root cause analysis for microservices via multivariate Bayesian online change point detection. ACM FSE.",
            "[14] Efficient Prediction of Stability Boundaries in Milling Considering the Variation of Tool Features and Workpiece Materials. (PMC10647373). (" + T("đã đọc nội dung bài", "article content read") + ")",
            "[15] Ji, Y., Liu, R. (2024). Research on the influence of cutter overhang length on robotic milling chatter stability. <i>Scientific Reports</i>. (" + T("đã đọc nội dung bài (PMC11496886)", "article content read (PMC11496886)") + ")",
            "[16] Milton, S. Machining dynamics, part 3: tool wear and its influence. Sirris. (" + T("bài tổng hợp ngắn của công ty, đã đọc; dẫn một bài IJAMT 2022 về mòn mặt sau khi phay Ti-6Al-4V thành mỏng, tôi chưa đọc bài đó", "short industry overview, read; it cites a 2022 IJAMT paper on flank wear in milling thin-walled Ti-6Al-4V, which I have not read") + ")",
            "[17] Chatter in Machining: Causes, Effects, and How to Reduce It. JLCCNC blog (jlccnc.com/blog/chatter-in-machining). (" + T("blog kỹ thuật, độ tin cậy thấp; chỉ đọc tóm tắt", "engineering blog, low reliability; summary only") + ")",
            "[18] A novel FMECA method for CNC machine tools based on D-GRA and data envelopment analysis. <i>Scientific Reports</i> (2024). (" + T("chỉ đọc tóm tắt trong kết quả tìm kiếm", "only the search-result summary was read") + ")",
            "[19] Intelligent Fault Diagnosis for CNC Through the Integration of Large Language Models and Domain Knowledge Graphs. (ScienceDirect, S2095809925001948). (" + T("chỉ đọc tóm tắt", "summary only") + ")",
            "[20] Prediction of modal parameters for thin-walled blade milling process considering material removal effect. <i>PLOS ONE</i> (PMC12419624). (" + T("chỉ đọc tóm tắt", "summary only") + ")",
            "[21] Chatter stability of turning and milling with process damping. (ResearchGate, researchgate.net/publication/273136972). (" + T("chỉ đọc tóm tắt trong kết quả tìm kiếm", "only the search-result summary was read") + ")"]
    for t in refs:
        s.append(P(t, S("ref", parent=body, fontSize=8.8, leading=12, leftIndent=12, firstLineIndent=-12, spaceAfter=2)))

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("ArialVN", 8); canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawCentredString(A4[0] / 2, 20, T(f"Trang {doc.page}", f"Page {doc.page}")); canvas.restoreState()

    doc = SimpleDocTemplate(OUT[lang], pagesize=A4, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.65 * inch,
                            title=T("Báo cáo RCA chatter CNC", "RCA of CNC machining chatter"), author="Đặng Hải Đăng")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("OK", OUT[lang])


if __name__ == "__main__":
    setup_fonts()
    os.makedirs(FIG_DIR, exist_ok=True)
    for lg in ("vi", "en"):
        build(lg)
