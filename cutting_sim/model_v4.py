"""Mô hình v4: dao 2SE0250IX150A thật (tham số cố định từ catalog) + đầu dao tham số hóa (học) + rãnh xoắn + lực cạnh.
Nhóm tham số theo chính sách nguồn gốc: FIXED (tra được) / RANGE (paper nêu khoảng) / LEARNED (không có căn cứ -> học từ dữ liệu).
Xem parameter_registry() cho bảng đầy đủ kèm nguồn.
"""
import os, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from milling_model import single_mode  # noqa
from calibrate_cma import feat_vec, cuts_of, real_matrix, cond_matrix  # noqa

HELIX = np.deg2rad(30.0)   # FIXED: Kennametal GOmill GP catalog (đặc điểm dòng 2 rãnh)
NZ = 8                     # số lát dọc trục (hội tụ: nz≥8)
LOC = 38.1e-3              # FIXED: chiều dài cắt = ap của dữ liệu (catalog, trang 5)

# (tên, thấp, cao, giá trị khởi tạo)  — cận tìm kiếm là cận SỐ HỌC, không phải khẳng định từ paper
SPEC = [
    ("ln g_acc", -12, 2, -6), ("ln noise_acc", -12, 0, -4), ("ln g_mic", -6, 6, 0), ("ln noise_mic", -10, 0, -3),
    ("fn_x [Hz]", 500, 4000, 1500), ("fn_y [Hz]", 500, 4000, 1500), ("zeta", 0.005, 0.15, 0.03), ("ln k_modal [N/m]", np.log(5e5), np.log(2e7), np.log(3e6)),
    ("ln Kt [N/m2]", np.log(2e8), np.log(2e9), np.log(7e8)), ("ln Kr/Kt", np.log(0.1), np.log(1.5), np.log(0.3)),
    ("ln Kte [N/m]", np.log(1e3), np.log(1e5), np.log(2e4)), ("ln Kre [N/m]", np.log(1e3), np.log(1e5), np.log(2e4)),
    ("ln U_imb [kg m]", -20, -6, -13), ("acc_rpm_exp", -2, 8, 1), ("mic_rpm_exp", -2, 8, 1), ("ln runout [m]", np.log(1e-6), np.log(3e-5), np.log(5e-6)),
]
NAMES = [s[0] for s in SPEC]; LO = np.array([s[1] for s in SPEC], float); HI = np.array([s[2] for s in SPEC], float); X0 = np.array([s[3] for s in SPEC], float)


def build(cut, th, e=(0.0, 0.0, 0.0)):
    """e = (ε_Kt, ε_runout, ε_rough): xáo trộn log-normal giữa các lần cắt (đã nhân σ)."""
    gl, na, gm, nm, fx, fy, ze, lk, lKt, lkr, lKte, lKre, lU, ea, em, lru = th
    c = dict(cut, a_eff=min(cut["ap"] * 1e-3, LOC))
    p = process_from_cut(c)
    Kt = float(np.exp(lKt + e[0]))
    p = replace(p, Kt=Kt, Kr=float(np.exp(lkr)), Kte=float(np.exp(lKte)), Kre=float(np.exp(lKre)), imb=float(np.exp(lU)),
                runout=float(np.exp(lru + e[1])), helix=HELIX, nz=NZ, D_tool=cut["D"] * 1e-3,
                mod_x=single_mode(fx, ze, float(np.exp(lk))), mod_y=single_mode(fy, ze, float(np.exp(lk))))
    cfg = SensorConfig(acc_gain=float(np.exp(gl)), acc_noise_g=float(np.exp(na)), mic_gain=float(np.exp(gm)), mic_noise_pa=float(np.exp(nm)),
                       acc_rpm_exp=float(ea), mic_rpm_exp=float(em))
    return p, cfg


def one4(args):
    cut, th, seed, sig = args
    r = np.random.default_rng(seed + 31337); e = r.normal(size=3) * (np.zeros(3) if sig is None else sig)
    p, cfg = build(cut, th, e)
    _, _, ft = run(p, seed=seed, t_end=0.6, sensor_cfg=cfg, noise_um=0.02 * float(np.exp(e[2])), skip=0.5)
    return feat_vec(ft)


def sim4(cuts, th, seed, pool, sig=None):
    return np.array(pool.map(one4, [(c, th, seed * 1000 + i, sig) for i, c in enumerate(cuts)]))


def parameter_registry():
    """Bảng nguồn gốc tham số. Cột 'context' = điều kiện mà giá trị CHỈ đúng trong đó; 'verified' = mức xác minh."""
    R = [
        ("tool diameter, flutes, flute length, overall length", "6.35 mm, 2, 38.1 mm, 101.6 mm", "FIXED", "Kennametal 2SE0250IX150A (exact part)", "verified (read from catalog table p.5)", "Kennametal GOmill GP catalog"),
        ("helix angle β", "30°", "FIXED", "Kennametal GOmill GP series (2-flute)", "medium: from series feature list (p.3); not stated per part number", "Kennametal GOmill GP catalog p.3; sensitivity to β will be reported"),
        ("tool substrate/coating", "carbide KC633M, TiAlN PVD", "FIXED", "Kennametal GOmill GP", "medium", "Kennametal GOmill GP catalog p.3, p.5 (no numeric use)"),
        ("N, ae, feed F, direction", "from label.csv", "FIXED", "machine imi_vm20i, lab", "verified (N and F equal mtc Spindle_Speed/Feed_Rate, ratio 1.000)", "Kim et al., Sci. Data 2026 (MSM) + own check"),
        ("axial depth ap = engaged depth", "38.1 mm (= flute length)", "FIXED", "assumes Z0 = workpiece top (NC: G01 Z-1.5 in)", "assumption (not measured)", "NC program of dataset 5/6; catalog flute length"),
        ("workpiece AA 6061; wet cutting (coolant flood)", "—", "FIXED", "laboratory machines imi_vm20i/vmx30ui ONLY (tmf_vf10 is dry)", "verified (stated in paper)", "Kim et al., Sci. Data 2026"),
        ("sampling: accelerometer 51.2 kHz, audio 48 kHz", "—", "FIXED", "imi_vm20i high-frequency accelerometers", "verified (stated in paper)", "Kim et al., Sci. Data 2026"),
        ("analysis window 0.3 s", "own design choice, identical for real and simulated features", "FIXED", "feature extraction", "design choice", "—"),
        ("model structure: regenerative DDE, shear+edge force, ZOA lobes", "—", "FIXED", "milling dynamics", "model structure", "Altintas et al., J. Manuf. Sci. Eng. 142(11):110801 (2020)"),
        ("Kt, Kr, Kte, Kre (AA 6061, this tool, wet)", "search bounds only", "LEARNED", "—", "NO usable source (rejected: steel-like values; dry-cutting AA6061 value unverified)", "—"),
        ("tool-tip FRF: fn_x, fn_y, ζ, modal stiffness k", "bounds from cantilever-beam estimate (numerical only)", "LEARNED", "—", "NO source for this tool/holder/stick-out (Altintas Ex.2 FRF rejected: different machine and tool)", "Aran & Budak 2011 opened: method only, no numbers"),
        ("spindle imbalance U", "search bounds only", "LEARNED", "—", "NO source (rotor mass, balance grade unknown)", "—"),
        ("sensor gains, noise floors, rpm-dependence", "search bounds only", "LEARNED", "—", "NO source (sensors uncalibrated: mic dBFS, accelerometer mm/s^2)", "—"),
        ("runout; between-cut variability σ (Kt, runout, roughness)", "search bounds only", "LEARNED", "—", "NO source", "—"),
        ("chip-thickness cap 0.3 mm; roughness noise base 0.02 µm", "numerical choices", "ASSUMED", "—", "own assumption", "—"),
    ]
    return pd.DataFrame(R, columns=["parameter", "value / bound", "tier", "context (valid only for)", "verification", "source"])


def to_markdown(df):
    h = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.values)
