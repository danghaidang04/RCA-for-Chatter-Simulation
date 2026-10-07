"""Họ 'máy' có ràng buộc vật lý + bộ sinh dữ liệu theo từng máy (để huấn luyện/kiểm tra độ bền RCA khi tương tác máy-dao thay đổi).
 Tham số đầu dao suy ra từ MỘT biến gốc (chiều dài nhô L) qua công thức dầm công-xôn (E, rho cacbua: GIẢ ĐỊNH chưa xác minh) nhân hệ số mềm đồ gá:
   fn = 0.5596*sqrt(E/rho)*(d/4)/L^2 * c_f ;  k = 3EI/L^3 * c_k ;  khối lượng modal = k/(2*pi*fn)^2 (không độc lập).
 Hệ số lực Kt: khoảng 0.4-1.7 GPa (cận trên từ năng lượng cắt riêng đo được từ công suất, dao A/nhôm: 1.4-1.7). Độ cản ζ: GIẢ ĐỊNH 0.01-0.05.
 Nền cảm biến mỗi máy: độ lợi, độ nghiêng phổ, và 1-3 vạch tần số cố định (kiểu vạch 1494 Hz quan sát thấy ở mọi dao) — GIẢ ĐỊNH về phân bố.
 kind='train' (trong khoảng) | 'extrap' (NGOÀI khoảng: L, ζ, Kt, độ lợi lệch hơn)."""
import numpy as np
import rca_data as rd
import noise_model as nm

E, RHO, D_TOOL = 600e9, 14.5e3, 6.35e-3


def sample_machine(rng, kind="train"):
    ex = kind == "extrap"
    L = rng.choice([rng.uniform(0.090, 0.110), rng.uniform(0.036, 0.045)]) if ex else rng.uniform(0.050, 0.080)
    c_f, c_k = rng.uniform(0.6, 1.0), rng.uniform(0.4, 1.0)
    fn = 0.5596 * np.sqrt(E / RHO) * (D_TOOL / 4) / L ** 2 * c_f
    k = 3 * E * (np.pi * D_TOOL ** 4 / 64) / L ** 3 * c_k
    zeta = rng.uniform(0.055, 0.09) if ex else rng.uniform(0.01, 0.05)
    Kt = np.exp(rng.uniform(np.log(2.0e9), np.log(2.6e9))) if ex else np.exp(rng.uniform(np.log(0.4e9), np.log(1.7e9)))
    Ke = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
    s = 0.8 if ex else 0.4
    lines = lambda: [(float(rng.uniform(800, 3500)), float(np.exp(rng.uniform(np.log(0.05), np.log(0.8))))) for _ in range(rng.integers(1, 4))]
    return {"kind": kind, "L": float(L), "fn": float(fn), "k": float(k), "zeta": float(zeta), "Kt": float(Kt), "Ke": float(Ke),
            "ga": float(np.exp(rng.normal(0, s))), "gm": float(np.exp(rng.normal(0, s))), "tau_a": float(rng.normal(0, s / 1.5)), "tau_m": float(rng.normal(0, s / 1.5)),
            "lines_a": lines(), "lines_m": lines(), "pj": dict(fn=fn / 1500.0, zeta=zeta / 0.03, k=k / 3e6, Kt=Kt / 7e8, Ke=Ke)}


def nominal_machine():
    return {"kind": "nominal", "ga": 1.0, "gm": 1.0, "tau_a": 0.0, "tau_m": 0.0, "lines_a": [], "lines_m": [], "pj": None}


def apply_machine(sig, mach, rng, fs_a, sr):
    out = []
    for x, fs, g, tau, lines in ((sig[0], fs_a, mach["ga"], mach["tau_a"], mach["lines_a"]), (sig[1], fs_a, mach["ga"], mach["tau_a"], mach["lines_a"]), (sig[2], sr, mach["gm"], mach["tau_m"], mach["lines_m"])):
        n = len(x); f = np.fft.rfftfreq(n, 1 / fs); y = np.fft.irfft(np.fft.rfft(x) * (np.maximum(f, 50.0) / 1000.0) ** (tau / 2), n) * g
        t = np.arange(n) / fs; rms = y.std()
        for fl, ar in lines: y = y + ar * rms * np.sqrt(2) * np.sin(2 * np.pi * fl * t + rng.uniform(0, 2 * np.pi))
        out.append(y)
    return out


def machine_job(args):
    """Một máy: n_win cửa sổ đánh giá x 6 lớp + n_cal cửa sổ khỏe hiệu chuẩn (để tính nền tương đối)."""
    mach, mrows, cal_rows, seed = args; g = rd._G; rng = np.random.default_rng(seed); n_a, n_m, fs_a, sr = g["n_a"], g["n_m"], g["fs_a"], g["sr"]
    u = rng.normal(0, np.sqrt(g["gen"].between)); X, Y = [], []
    for mrow in mrows:
        b = g["gen"].signals(mrow, u, rng, n_a, n_m); d0 = rd.dde_signals(mrow, "NORMAL", 1.0, int(rng.integers(1 << 30)), fs_a, sr, n_a, n_m, mach["pj"])
        for k, c in enumerate(rd.CLASSES):
            if c == "NORMAL": sig = b
            else:
                lo, hi = rd.SEV[c]; sev = rng.choice([-1, 1]) * rng.uniform(0.03, 0.15) + 1.0 if c == "RPM_MISMATCH" else rng.uniform(lo, hi)
                d1 = rd.dde_signals(mrow, c, sev, int(rng.integers(1 << 30)), fs_a, sr, n_a, n_m, mach["pj"]); n = min(len(d1[0]), len(b[0])), min(len(d1[2]), len(b[2]))
                sig = (b[0][:n[0]] + g["gacc"] * (d1[0][:n[0]] - d0[0][:n[0]]), b[1][:n[0]] + g["gacc"] * (d1[1][:n[0]] - d0[1][:n[0]]), b[2][:n[1]] + g["gmic"] * (d1[2][:n[1]] - d0[2][:n[1]]))
            s2 = apply_machine(sig, mach, rng, fs_a, sr); X.append(rd.rca_features(s2[0], s2[1], s2[2], fs_a, sr, mrow)); Y.append(k)
    XC = []
    for mrow in cal_rows:
        b = g["gen"].signals(mrow, u, rng, n_a, n_m); s2 = apply_machine(b, mach, rng, fs_a, sr); XC.append(rd.rca_features(s2[0], s2[1], s2[2], fs_a, sr, mrow))
    return np.array(X), np.array(Y), np.array(XC)
