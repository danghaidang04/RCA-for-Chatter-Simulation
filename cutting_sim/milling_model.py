"""Mô hình phay tái sinh (regenerative milling) - Altintas, Stepan, Budak, Schmitz, Kilic (2020).

Gồm:
  * ModalSet / make_modes : FRF dạng modal (σ, ν, ζ, ω_n) lấy từ Altintas Example 2 (Assignment 4 trong repo)
  * stability_lobes_zoa   : Stability Lobe Diagram theo Zero-Order Approximation (Altintas-Budak 1995)
  * simulate_milling      : mô phỏng miền thời gian DDE 2-DOF (x,y), nhiều răng, chiều dày phoi động,
                            rời khỏi phôi (h<0 -> lực 0), nhiễu, runout tùy chọn.
Khối "Simulink" được dựng bằng các khối LTI của scipy.signal (state-space rời rạc hóa chính xác ZOH).
"""
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import expm

H_MAX = 0.3e-3  # m, chặn trên chiều dày phoi tức thời

# --- Altintas Example 2 (page 160) ----------------------------------------------------------
FREQ_X = np.array([262.16, 503.60, 667.92, 886.78, 2201.5, 2799.4, 3837.7, 4923.6, 6038.7])
ZETA_X = np.array([0.048, 0.096, 0.057, 0.074, 0.015, 0.027, 0.007, 0.019, 0.025])
SIG_X = np.array([1.994606, 1.932445, 13.64547, 0.8632813, -5.011467, -13.94105, -6.874543, -8.824935, 2.564520]) * 1e-6
NU_X = np.array([0.3779091, 2.44616, 4.128093, 7.195716, 4.623555, 8.317728, 6.560925, 6.54481, 1.986276]) * 1e-5
FREQ_Y = np.array([285.53, 587.8, 749.6, 804.9, 1573.7, 2038.1, 2303.3, 2681.0, 2870.5, 3838.8, 4928.9, 6073.4])
ZETA_Y = np.array([0.021, 0.089, 0.027, 0.075, 0.027, 0.016, 0.220, 0.019, 0.014, 0.006, 0.017, 0.016])
SIG_Y = np.array([0.9797268, 13.49004, -31.06831, 30.33849, 1.283945, 1.298625, 0.9518897, 11.90834, -19.12932,
                  -11.19589, -18.59880, -7.608392]) * 1e-6
NU_Y = np.array([0.05959362, 0.4544067, 3.816475, 8.813244, 0.8678961, 1.270846, 3.750897, 2.825781, 4.051860,
                 8.475725, 8.993226, 2.022991]) * 1e-5


@dataclass(frozen=True)
class Modes:
    """Hàm đáp ứng tần số H(s)=Σ (β s+α)/(s²+2ζω s+ω²) cho một trục, đơn vị m/N."""
    fn: np.ndarray
    zeta: np.ndarray
    sig: np.ndarray
    nu: np.ndarray

    @property
    def wn(self):
        return 2 * np.pi * self.fn

    @property
    def wd(self):
        return self.wn * np.sqrt(np.maximum(1e-12, 1 - self.zeta ** 2))

    @property
    def alpha(self):
        return 2 * (self.zeta * self.wn * self.sig - self.wd * self.nu)

    @property
    def beta(self):
        return 2 * self.sig

    def frf(self, omega):
        s = 1j * np.atleast_1d(omega)[:, None]
        num = self.beta[None] * s + self.alpha[None]
        den = s ** 2 + 2 * self.zeta[None] * self.wn[None] * s + self.wn[None] ** 2
        return (num / den).sum(1)

    def modified(self, k_scale=1.0, zeta_scale=1.0, dominant_only=None):
        """Đổi độ cứng (khối lượng modal giữ nguyên: ω∝√k) và cản. Giữ nguyên độ lớn dư (residue) theo ω_d.

        dominant_only: số mode thấp nhất bị tác động (mặc định: tất cả mode ≤ 1 kHz, là các mode chi phối chatter)."""
        sel = self.fn <= 1000.0 if dominant_only is None else (np.arange(len(self.fn)) < dominant_only)
        fn = np.where(sel, self.fn * np.sqrt(k_scale), self.fn)
        ze = np.where(sel, self.zeta * zeta_scale, self.zeta)
        new = Modes(fn, ze, self.sig, self.nu)
        ratio = self.wd / new.wd  # r' = r·ω_d/ω_d'  (khối lượng modal không đổi)
        return Modes(fn, ze, self.sig * ratio, self.nu * ratio)


def single_mode(fn, zeta, k):
    """Một mode đầu dao: H(s)=1/(m(s²+2ζω s+ω²)), m=k/ω² (k: độ cứng modal N/m). Biểu diễn dư dạng (σ,ν)."""
    wn = 2 * np.pi * fn; mm_ = k / wn ** 2; wd = wn * np.sqrt(1 - zeta ** 2)
    return Modes(np.array([fn]), np.array([zeta]), np.array([0.0]), np.array([-1.0 / (2 * mm_ * wd)]))


def make_modes(max_freq=None):
    mx, my = Modes(FREQ_X, ZETA_X, SIG_X, NU_X), Modes(FREQ_Y, ZETA_Y, SIG_Y, NU_Y)
    if max_freq is None:
        return mx, my
    cut = lambda m: Modes(*(a[m.fn <= max_freq] for a in (m.fn, m.zeta, m.sig, m.nu)))
    return cut(mx), cut(my)


@dataclass(frozen=True)
class Process:
    """Điều kiện cắt + hệ số lực. Đơn vị SI (a: m, ft: m/răng)."""
    N: int = 4              # số răng
    Kt: float = 796e6       # N/m²
    Kr: float = 0.212       # tỷ số lực hướng kính/tiếp tuyến
    a: float = 2e-3         # chiều sâu cắt dọc trục (m)
    rpm: float = 6000.0
    ft: float = 0.05e-3     # lượng ăn dao/răng (m)
    phi_st: float = 0.0     # góc vào (slotting)
    phi_ex: float = np.pi   # góc ra
    runout: float = 0.0     # độ lệch tâm dao (m), tùy chọn
    imb: float = 0.0        # mất cân bằng trục chính U=m·e (kg·m): lực quay đồng bộ F=U·Ω² (Ω=2π·rpm/60)
    helix: float = 0.0      # góc xoắn rãnh β (rad); 0 = rãnh thẳng (mô hình cũ)
    nz: int = 1             # số lát cắt dọc trục khi chia nhỏ chiều sâu a
    D_tool: float = 0.0     # đường kính dao (m), dùng cho độ trễ góc do xoắn
    Kte: float = 0.0        # hệ số lực cạnh tiếp tuyến (N/m) — lực không phụ thuộc chiều dày phoi
    Kre: float = 0.0        # hệ số lực cạnh hướng kính (N/m)
    mod_x: Modes = None
    mod_y: Modes = None

    @property
    def tooth_period(self):
        return 60.0 / (self.N * self.rpm)


def immersion_alpha(phi_st, phi_ex, Kr):
    f = lambda p: (0.5 * (np.cos(2 * p) - 2 * Kr * p + Kr * np.sin(2 * p)),
                   0.5 * (-np.sin(2 * p) - 2 * p + Kr * np.cos(2 * p)),
                   0.5 * (-np.sin(2 * p) + 2 * p + Kr * np.cos(2 * p)),
                   0.5 * (-np.cos(2 * p) - 2 * Kr * p - Kr * np.sin(2 * p)))
    return tuple(e - s for e, s in zip(f(phi_ex), f(phi_st)))


# ---------------------------------------------------------------------------------------------
# Stability lobes - ZOA
# ---------------------------------------------------------------------------------------------
def stability_lobes_zoa(proc, fc_range=(300.0, 1200.0), n_fc=1500, n_lobes=25, rpm_max=20000, alim_max=0.060):
    """Trả về (rpm, a_lim[m]) đã sắp xếp theo rpm (tất cả lobe) và đường bao ổn định (lower envelope)."""
    axx, axy, ayx, ayy = immersion_alpha(proc.phi_st, proc.phi_ex, proc.Kr)
    fc = np.linspace(*fc_range, n_fc)
    w = 2 * np.pi * fc
    Hx, Hy = proc.mod_x.frf(w), proc.mod_y.frf(w)
    a0 = Hx * Hy * (axx * ayy - axy * ayx)
    a1 = axx * Hx + ayy * Hy
    lam = -1 / (2 * a0) * (a1 - np.sqrt(a1 ** 2 - 4 * a0))
    lr, li = lam.real, lam.imag
    ok = lr < 0
    kappa = li / np.where(ok, lr, -1)
    psi = np.arctan2(li, lr)
    alim = -2 * np.pi * lr / (proc.N * proc.Kt) * (1 + kappa ** 2)
    rpm_l, a_l, k_l = [], [], []
    for k in range(0, n_lobes):
        T = (np.pi - 2 * psi + 2 * k * np.pi) / w
        rpm = 60 / (proc.N * T)
        m = ok & (T > 0) & (rpm <= rpm_max) & (alim <= alim_max)
        rpm_l.append(rpm[m]); a_l.append(alim[m]); k_l.append(np.full(m.sum(), k))
    rpm_a, a_a, k_a = (np.concatenate(v) for v in (rpm_l, a_l, k_l))
    o = np.argsort(rpm_a)
    return rpm_a[o], a_a[o], k_a[o]


def a_lim_at(rpm_arr, a_arr, rpm_query, bins=400):
    """Nội suy đường bao a_lim(rpm) (cực tiểu giữa các lobe chồng nhau)."""
    edges = np.linspace(rpm_arr.min(), rpm_arr.max(), bins + 1)
    idx = np.clip(np.digitize(rpm_arr, edges) - 1, 0, bins - 1)
    env = np.full(bins, np.nan)
    for b in range(bins):
        if np.any(idx == b):
            env[b] = a_arr[idx == b].min()
    c = 0.5 * (edges[1:] + edges[:-1])
    good = ~np.isnan(env)
    return np.interp(rpm_query, c[good], env[good])


# ---------------------------------------------------------------------------------------------
# Time-domain DDE simulation
# ---------------------------------------------------------------------------------------------
def _modal_ss(modes, dt):
    """Trạng thái modal [x_k, v_k] -> q = Σ (α x + β v). Rời rạc hóa chính xác ZOH."""
    n = len(modes.fn)
    A = np.zeros((2 * n, 2 * n)); B = np.zeros(2 * n); C = np.zeros(2 * n)
    for k in range(n):
        A[2 * k, 2 * k + 1] = 1.0
        A[2 * k + 1, 2 * k] = -modes.wn[k] ** 2
        A[2 * k + 1, 2 * k + 1] = -2 * modes.zeta[k] * modes.wn[k]
        B[2 * k + 1] = 1.0
        C[2 * k], C[2 * k + 1] = modes.alpha[k], modes.beta[k]
    M = np.zeros((2 * n + 1, 2 * n + 1)); M[:2 * n, :2 * n] = A * dt; M[:2 * n, 2 * n] = B * dt
    E = expm(M)
    return np.ascontiguousarray(E[:2 * n, :2 * n]), np.ascontiguousarray(E[:2 * n, 2 * n]), C


try:
    from numba import njit
except Exception:                     # numba không có -> chạy bản Python thuần (chậm hơn ~50x)
    njit = None


def _core_py(n, dt, fs_dummy, Ax, Bx, Cx, Ay, By, Cy, noise, N, rpm, Kt, Kr, a, ft_eff, phi_st, phi_ex, d_int, d_frac, h_max, imb,
             tb2, nz, Kte, Kre):
    """Lõi thời gian. Mỗi răng chia nz lát dọc trục; lát ở độ cao z bị trễ góc ψ(z)=z·tb2 (tb2=2tanβ/D). Lực lát: (Kt·h+Kte)dz, (Kr·Kt·h+Kre)dz."""
    x = np.zeros(n); y = np.zeros(n); Fx = np.zeros(n); Fy = np.zeros(n)
    sx = np.zeros(len(Bx)); sy = np.zeros(len(By))
    w_spin = 2 * np.pi * rpm / 60.0
    pitch = 2 * np.pi / N
    dz = a / nz
    h_min = 1e9
    for i in range(n):
        ic = i - 1 if i > 0 else 0
        if ic > d_int + 1:
            j = ic - d_int
            xd = x[j] * (1 - d_frac) + x[j - 1] * d_frac
            yd = y[j] * (1 - d_frac) + y[j - 1] * d_frac
        else:
            xd = 0.0; yd = 0.0
        fx = 0.0; fy = 0.0
        for k in range(N):
            for q in range(nz):
                z = (q + 0.5) * dz
                phi = (w_spin * i * dt + k * pitch - z * tb2) % (2 * np.pi)
                if phi >= phi_st and phi < phi_ex:
                    s = np.sin(phi); c = np.cos(phi)
                    h = ft_eff[k] * s + (x[ic] - xd) * s + (y[ic] - yd) * c + noise[i, k]
                    h = min(max(h, 0.0), h_max)
                    if h < h_min:
                        h_min = h
                    if h > 0.0:
                        Ft = (Kt * h + Kte) * dz; Fr = (Kr * Kt * h + Kre) * dz
                        fx += -Ft * c - Fr * s
                        fy += Ft * s - Fr * c
        ang = w_spin * i * dt
        fx += imb * w_spin ** 2 * np.cos(ang); fy += imb * w_spin ** 2 * np.sin(ang)
        Fx[i] = fx; Fy[i] = fy
        sx = Ax @ sx + Bx * fx
        sy = Ay @ sy + By * fy
        x[i] = Cx @ sx
        y[i] = Cy @ sy
    return x, y, Fx, Fy, h_min


_core = njit(cache=True)(_core_py) if njit is not None else _core_py


def simulate_milling(proc, t_end=0.4, fs=25600.0, noise_um=0.02, seed=0, return_states=False):
    """Mô phỏng DDE: q'' (modal) = F, chip h_j = g(φ_j)[ft sinφ_j + (Δx sinφ_j + Δy cosφ_j)], Δ=q(t)-q(t-T).
    Vòng lặp thời gian chạy trong lõi numba (_core); noise sinh bằng numpy để tái lập theo seed."""
    dt = 1.0 / fs
    n = int(round(t_end * fs))
    Ax, Bx, Cx = _modal_ss(proc.mod_x, dt)
    Ay, By, Cy = _modal_ss(proc.mod_y, dt)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_um * 1e-6, (n, proc.N))
    d_steps = proc.tooth_period / dt
    d_int = int(np.floor(d_steps)); d_frac = d_steps - d_int
    ft_eff = proc.ft + np.zeros(proc.N)
    ft_eff[0] += proc.runout
    if proc.N > 1:
        ft_eff[1] -= proc.runout
    x, y, Fx, Fy, h_min = _core(n, dt, 0.0, Ax, Bx, Cx, Ay, By, Cy, noise, int(proc.N), float(proc.rpm), float(proc.Kt), float(proc.Kr),
                                float(proc.a), ft_eff, float(proc.phi_st), float(proc.phi_ex), d_int, d_frac, H_MAX, float(proc.imb),
                                float(2 * np.tan(proc.helix) / proc.D_tool) if proc.D_tool > 0 else 0.0, int(proc.nz), float(proc.Kte), float(proc.Kre))
    return {"t": np.arange(n) * dt, "x": x, "y": y, "Fx": Fx, "Fy": Fy, "fs": fs, "h_min": h_min, "rpm": float(proc.rpm)}


def simulate_milling_reference(proc, t_end=0.4, fs=25600.0, noise_um=0.02, seed=0):
    """Bản Python thuần (tham chiếu) để kiểm tra lõi numba cho kết quả giống hệt."""
    global _core
    saved = _core; _core = _core_py
    try:
        return simulate_milling(proc, t_end, fs, noise_um, seed)
    finally:
        _core = saved
