"""Kiểm thử: tạo FRF từ mode đã biết + nhiễu -> khớp -> so sánh. Đây là kiểm thử khép kín (inverse crime); FRF búa thật sẽ cho sai số lớn hơn."""
import numpy as np
from milling_model import make_modes
from frf_import import fit_modes

mx, _ = make_modes(); f = np.arange(50, 4000, 2.0); H = mx.frf(2 * np.pi * f)
rng = np.random.default_rng(0)
for snr in (np.inf, 0.01, 0.03):
    Hn = H + snr * np.abs(H).max() * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / np.sqrt(2) if np.isfinite(snr) else H
    m, err = fit_modes(f, Hn, fmax=3500)
    b = f <= 3500; Hf = m.frf(2 * np.pi * f); rel = np.linalg.norm(Hf[b] - H[b]) / np.linalg.norm(H[b])
    print(f"nhiễu {snr}: {len(m.fn)} mode, sai số khớp {err:.3f}, sai số so với FRF sạch {rel:.3f}, fn={np.round(m.fn,0)}")
