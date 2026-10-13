"""Nạp FRF đo bằng búa (CSV: f[Hz], Re, Im [m/N] hoặc f, |H|, pha[deg]) -> khớp mode -> `Modes` của milling_model.
Khớp: chọn đỉnh trên -Im(H), tinh chỉnh (fn, ζ) bằng bình phương tối thiểu phi tuyến; phần dư (σ, ν) giải tuyến tính (variable projection).
Chỉ nhận receptance (m/N). Đo bằng gia tốc (accelerance) phải chia (jω)² trước khi nạp: dùng cột `kind`."""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from milling_model import Modes


def load_csv(path, kind="receptance", polar=False):
    d = np.loadtxt(path, delimiter=",", comments="#"); f = d[:, 0]
    H = d[:, 1] * np.exp(1j * np.deg2rad(d[:, 2])) if polar else d[:, 1] + 1j * d[:, 2]
    if kind == "accelerance":
        H = H / (-(2 * np.pi * np.maximum(f, 1e-9)) ** 2)
    elif kind != "receptance":
        raise ValueError(kind)
    return f, H


def _basis(f, fn, ze):
    s = 2j * np.pi * f[:, None]; wn = 2 * np.pi * fn[None]
    den = s ** 2 + 2 * ze[None] * wn * s + wn ** 2
    wd = wn * np.sqrt(np.maximum(1e-12, 1 - ze[None] ** 2))
    # H = Σ (β s + α)/den, β=2σ, α=2(ζ ωn σ − ωd ν)  -> tuyến tính theo (σ, ν)
    A_sig = (2 * s + 2 * ze[None] * wn * 2) / den
    A_nu = (-2 * wd) / den
    return np.hstack([A_sig, A_nu])


def _solve_res(f, H, fn, ze):
    A = _basis(f, fn, ze); A_ri = np.vstack([A.real, A.imag]); b = np.r_[H.real, H.imag]
    w = 1.0 / (np.abs(H).max() + 1e-30)
    x = np.linalg.lstsq(A_ri * w, b * w, rcond=None)[0]; n = len(fn)
    return x[:n], x[n:], A @ x


def _refine(f, H, fn0, ze0, w):
    n = len(fn0)

    def res(p):
        fn, ze = fn0 * np.exp(p[:n]), np.clip(ze0 * np.exp(p[n:]), 1e-4, 0.5)
        _, _, Hm = _solve_res(f, H, fn, ze); r = (Hm - H) * w
        return np.r_[r.real, r.imag]

    lo = np.r_[np.full(n, -0.05), np.full(n, -1.5)]; sol = least_squares(res, np.zeros(2 * n), bounds=(lo, -lo), max_nfev=60)
    return fn0 * np.exp(sol.x[:n]), np.clip(ze0 * np.exp(sol.x[n:]), 1e-4, 0.5), float(np.linalg.norm(sol.fun))


def fit_modes(f, H, fmin=None, fmax=None, max_modes=15, min_gain=0.03):
    """Thêm mode dần tại đỉnh phần dư lớn nhất; dừng khi sai số tương đối giảm < min_gain (chống khớp nhiễu) hoặc đủ max_modes."""
    sel = np.ones_like(f, bool)
    if fmin: sel &= f >= fmin
    if fmax: sel &= f <= fmax
    f, H = f[sel], H[sel]; w = 1.0 / (np.abs(H) + 0.05 * np.abs(H).max())
    fn, ze, prev = np.array([]), np.array([]), np.linalg.norm(H * w)
    while len(fn) < max_modes:
        if len(fn):
            _, _, Hm = _solve_res(f, H, fn, ze); r = np.abs(H - Hm) * w
        else:
            r = np.abs(H) * w
        r = np.convolve(r, np.ones(5) / 5, "same"); k = int(np.argmax(r))
        fn2, ze2, e2 = _refine(f, H, np.r_[fn, f[k]], np.r_[ze, 0.03], w)
        if (prev - e2) / prev < min_gain:
            break
        fn, ze, prev = fn2, ze2, e2
    o = np.argsort(fn); fn, ze = fn[o], ze[o]
    sig, nu, Hm = _solve_res(f, H, fn, ze)
    return Modes(fn, ze, sig, nu), float(np.linalg.norm(Hm - H) / np.linalg.norm(H))
