"""Tác động lên 5 nguyên nhân gốc (root causes) của chatter, với mức nhỏ ("gap" rất nhỏ) -> lớn.

 STIFFNESS      : lỏng gá/ổ trục -> độ cứng các mode ≤1 kHz giảm  (k' = s_k·k, ω∝√k)
 DAMPING        : dao nhô dài/ma sát kẹp giảm -> ζ' = s_ζ·ζ
 TOOL_WEAR      : mòn mặt sau -> Kt tăng, Kr tăng (ploughing)
 DEPTH_OVERLOAD : CAM đặt a quá lớn
 RPM_MISMATCH   : tốc độ trục chính rời điểm tối ưu -> rơi vào thung lũng lobe
"""
from dataclasses import replace
import numpy as np

FAULTS = ["STIFFNESS", "DAMPING", "TOOL_WEAR", "DEPTH_OVERLOAD", "RPM_MISMATCH"]
TRUTH_BN = {"NORMAL": "Normal", "STIFFNESS": "Stiffness_Degradation", "DAMPING": "Damping_Loss",
            "TOOL_WEAR": "Tool_Wear", "DEPTH_OVERLOAD": "Excessive_Cut_Depth", "RPM_MISMATCH": "Unstable_Spindle_RPM"}
# mức tác động (small = gap rất nhỏ / incipient)
LEVELS = {
    "STIFFNESS": {"small": 0.95, "medium": 0.80, "large": 0.55},       # s_k
    "DAMPING": {"small": 0.90, "medium": 0.65, "large": 0.35},         # s_ζ
    "TOOL_WEAR": {"small": 1.05, "medium": 1.25, "large": 1.60},       # s_Kt (Kr tăng 2x tỷ lệ)
    "DEPTH_OVERLOAD": {"small": 1.08, "medium": 1.40, "large": 1.90},  # s_a
    "RPM_MISMATCH": {"small": 0.9915, "medium": 0.9717, "large": 0.9434},  # n=5300 -> 5000 (thung lũng lobe)
}


def apply_fault(proc, fault, level="small", scale=None):
    s = LEVELS[fault][level] if scale is None else scale
    if fault == "STIFFNESS":
        return replace(proc, mod_x=proc.mod_x.modified(k_scale=s), mod_y=proc.mod_y.modified(k_scale=s))
    if fault == "DAMPING":
        return replace(proc, mod_x=proc.mod_x.modified(zeta_scale=s), mod_y=proc.mod_y.modified(zeta_scale=s))
    if fault == "TOOL_WEAR":
        return replace(proc, Kt=proc.Kt * s, Kr=proc.Kr * (1 + 2.0 * (s - 1)))
    if fault == "DEPTH_OVERLOAD":
        return replace(proc, a=proc.a * s)
    if fault == "RPM_MISMATCH":
        return replace(proc, rpm=proc.rpm * s)
    raise ValueError(fault)


def jitter(proc, rng, rel=0.02):
    """Biến thiên tự nhiên giữa các lần cắt: hệ số lực, a, n, độ cứng/cản ±rel."""
    j = lambda: 1 + rng.normal(0, rel)
    p = replace(proc, Kt=proc.Kt * j(), a=proc.a * j(), rpm=proc.rpm * (1 + rng.normal(0, rel / 4)),
                runout=proc.runout * (1 + rng.normal(0, 0.2)))
    return replace(p, mod_x=p.mod_x.modified(k_scale=j(), zeta_scale=j()), mod_y=p.mod_y.modified(k_scale=j(), zeta_scale=j()))
