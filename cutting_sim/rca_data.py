"""Bộ sinh dataset RCA có nhãn.
 NỀN (healthy)  : mô hình nhiễu phổ (đường bao + hài) hồi quy phi tuyến + biến thiên phân cấp, KHỚP TỪ DỮ LIỆU THẬT (noise_model.py).
 NGUYÊN NHÂN GỐC: DDE (dao 2SE0250IX150A, xoắn 30°, lực cạnh). Tín hiệu chênh lệch Δ = DDE(lỗi) − DDE(bình thường) tại CÙNG điều kiện
                  được nhân hệ số khuếch đại (g_acc, g_mic) rồi CỘNG vào nền. Hệ số g học từ dữ liệu thật (nhãn 1) ở rca_gain.py.
Tham số vật lý DDE đặt ở giá trị danh định (SPEC X0 của model_v4) — GIẢ ĐỊNH, không học được (xem parameter_registry).
"""
import os
from multiprocessing import Pool

import numpy as np
from scipy.signal import resample_poly
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold

import noise_model as nm
from model_v4 import X0, build, HELIX  # noqa
from common import run, replace  # noqa

OUT = nm.OUT
CLASSES = ["NORMAL", "STIFFNESS", "DAMPING", "TOOL_WEAR", "DEPTH_OVERLOAD", "RPM_MISMATCH"]
BANDS = np.array([50, 150, 300, 600, 1000, 1500, 2200, 3200, 5000, 8000, 12000.0])
# mức độ nặng của từng nguyên nhân: GIẢ ĐỊNH (không có nguồn); khoảng chọn theo tính hợp lý vật lý, ghi rõ trong báo cáo
SEV = {"STIFFNESS": (0.4, 0.9), "DAMPING": (0.3, 0.8), "TOOL_WEAR": (1.1, 1.6), "DEPTH_OVERLOAD": (1.5, 4.0), "RPM_MISMATCH": (0.85, 1.15)}


class NoiseGen:
    def fit(self, meta, TH, fs_a, sr):
        self.fs_a, self.sr, self.K_H = fs_a, sr, nm.K_H
        cid = (meta[:, 0] * 1000 + meta[:, 1]).astype(int)
        X = nm.design(meta); self.lam = 1.0
        self.zsl = [1, 2, 3, 4, 5]; Z = X[:, self.zsl]; self.zmu, self.zsd = Z.mean(0), Z.std(0) + 1e-9; Zs = (Z - self.zmu) / self.zsd
        self.beta = np.linalg.solve(X.T @ X + self.lam * np.eye(X.shape[1]), X.T @ TH)
        self.kr = KernelRidge(alpha=0.3, kernel="rbf", gamma=0.15).fit(Zs, TH - X @ self.beta)
        oof = np.zeros_like(TH)
        for a_, b_ in GroupKFold(5).split(X, groups=cid):                         # phần dư ngoài mẫu
            bt = np.linalg.solve(X[a_].T @ X[a_] + self.lam * np.eye(X.shape[1]), X[a_].T @ TH[a_])
            k_ = KernelRidge(alpha=0.3, kernel="rbf", gamma=0.15).fit(Zs[a_], TH[a_] - X[a_] @ bt); oof[b_] = X[b_] @ bt + k_.predict(Zs[b_])
        res = TH - oof; _, _, Vt = np.linalg.svd(res - res.mean(0), full_matrices=False); self.V = Vt[:12].T
        sc = res @ self.V; ug = np.unique(cid)
        cm = np.array([sc[cid == g].mean(0) for g in ug]); within = np.mean([sc[cid == g].var(0, ddof=1) for g in ug if (cid == g).sum() > 1], axis=0)
        nbar = np.mean([(cid == g).sum() for g in ug]); self.within = within; self.between = np.maximum(cm.var(0, ddof=1) - within / nbar, 1e-12)
        self.lo, self.hi = np.percentile(TH, 1, axis=0), np.percentile(TH, 99, axis=0)
        return self

    def theta(self, mrow, u, e):
        m = mrow.copy(); m[3] = 0.0                                               # nền KHỎE: nhãn=0
        x = nm.design(m[None])[0]; z = ((x[self.zsl] - self.zmu) / self.zsd)[None]
        return np.clip(x @ self.beta + self.kr.predict(z)[0] + (u + e) @ self.V.T, self.lo, self.hi)

    def signals(self, mrow, u, rng, n_a, n_m):
        e = rng.normal(0, np.sqrt(self.within)); th = self.theta(mrow, u, e); f0 = mrow[5] / 60.0
        ka = len(nm.knots(self.fs_a)) + self.K_H; km = len(nm.knots(self.sr)) + self.K_H
        return (nm.synth(th[:ka], self.fs_a, n_a, f0, rng), nm.synth(th[ka:2 * ka], self.fs_a, n_a, f0, rng), nm.synth(th[2 * ka:2 * ka + km], self.sr, n_m, f0, rng))


def dde_signals(mrow, cls, sev, seed, fs_a, sr, n_a, n_m, pj=None):
    """DDE (0.6 s) -> acc_x, acc_y, mic ở fs gốc, cửa sổ 0.3 s cuối."""
    cut = dict(N=mrow[5], ap=38.1, ae=mrow[6], F=mrow[7], Z=int(mrow[8]), D=6.35, direction="UP" if mrow[4] else "DOWN")
    th = X0.copy()
    if pj:                                   # tham số máy thật lệch so với danh định (kiểm tra độ bền)
        th[4] *= pj['fn']; th[5] *= pj['fn']; th[6] *= pj['zeta']; th[7] += np.log(pj['k']); th[8] += np.log(pj['Kt']); th[10] += np.log(pj['Ke']); th[11] += np.log(pj['Ke'])
    if cls == "STIFFNESS":
        th[4] *= np.sqrt(sev); th[5] *= np.sqrt(sev); th[7] += np.log(sev)           # k'=s·k, fn ∝ √k
    elif cls == "DAMPING":
        th[6] *= sev
    elif cls == "TOOL_WEAR":
        th[8] += np.log(sev); th[10] += np.log(1 + 3 * (sev - 1)); th[11] += np.log(1 + 3 * (sev - 1))   # lực cạnh tăng theo mòn (giả định)
    elif cls == "DEPTH_OVERLOAD":
        cut["ae"] = min(cut["ae"] * sev, 3.0)
    elif cls == "RPM_MISMATCH":
        cut["N"] = cut["N"] * sev
    p, cfg = build(cut, th)
    cfg = replace(cfg, acc_noise_g=0.0, mic_noise_pa=0.0, acc_gain=1.0, mic_gain=1.0, acc_rpm_exp=0.0, mic_rpm_exp=0.0)
    sim, sig, _ = run(p, seed=seed, t_end=0.6, sensor_cfg=cfg, noise_um=0.02, skip=0.5)
    h = len(sig["t"]) // 2
    up_a = lambda x: resample_poly(x[h:], int(round(fs_a)) // 100, 25600 // 100)[:n_a]
    up_m = lambda x: resample_poly(x[h:], int(round(sr)) // 1600, 25600 // 1600)[:n_m]
    return up_a(sig["acc_x"]), up_a(sig["acc_y"]), up_m(sig["mic"])


def band_feats(x, fs):
    f = np.fft.rfftfreq(len(x), 1 / fs); P = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) ** 2
    return np.array([np.log10(P[(f >= BANDS[k]) & (f < min(BANDS[k + 1], fs / 2))].sum() + 1e-30) for k in range(len(BANDS) - 1)])


def rca_features(a0, a1, mic, fs_a, sr, mrow):
    base = nm.feats(a0, a1, mic, fs_a, sr, mrow[5], int(mrow[8]))
    kurt = lambda x: float(np.mean((x - x.mean()) ** 4) / (np.var(x) ** 2 + 1e-30))
    return np.r_[base, band_feats(a0, fs_a), band_feats(mic, sr), kurt(a0), kurt(mic), np.log(mrow[5]), np.log(mrow[6]), np.log(mrow[7]), mrow[4]]


FEAT_NAMES = (["log10 acc", "mic dB", "chat_acc", "chat_mic"] + [f"accB{k}" for k in range(10)] + [f"micB{k}" for k in range(10)] + ["kurt_acc", "kurt_mic", "logN", "logae", "logF", "up"])
_G = {}


def _init(gen, fs_a, sr, n_a, n_m, gacc, gmic):
    _G.update(gen=gen, fs_a=fs_a, sr=sr, n_a=n_a, n_m=n_m, gacc=gacc, gmic=gmic)


def make_row_lat(args):
    """Một điều kiện (hàng meta) + một lần lặp -> mẫu cho cả 6 lớp, dùng chung DDE bình thường và độ lệch giữa đường cắt u."""
    mrow, rep, seed, u, classes = args[:5]; mode = args[5] if len(args) > 5 else "nominal"; part = args[6] if len(args) > 6 else "full"
    g = _G; rng = np.random.default_rng(seed)
    pj = None
    if mode == "shift":
        rp = np.random.default_rng(seed + 999); lu = lambda a, b: float(np.exp(rp.uniform(np.log(a), np.log(b))))
        pj = dict(fn=lu(0.75, 1.33), zeta=lu(0.5, 2.0), k=lu(0.6, 1.7), Kt=lu(0.6, 1.6), Ke=lu(0.4, 2.5))
    d0 = dde_signals(mrow, "NORMAL", 1.0, seed, g["fs_a"], g["sr"], g["n_a"], g["n_m"], pj); out = []; lat = []
    for c in classes:
        b = g["gen"].signals(mrow, u, rng, g["n_a"], g["n_m"]); lv = np.zeros(5)
        if c == "NORMAL":
            sig = b
        else:
            lo, hi = SEV[c]
            if c == "RPM_MISMATCH":
                a_, b_ = (0.03, 0.15) if part == "full" else ((0.03, 0.09) if part == "low" else (0.09, 0.15)); sev = rng.choice([-1, 1]) * rng.uniform(a_, b_) + 1.0
            else:
                mid = 0.5 * (lo + hi); a_, b_ = (lo, hi) if part == "full" else ((lo, mid) if part == "low" else (mid, hi)); sev = rng.uniform(a_, b_)
            d1 = dde_signals(mrow, c, sev, seed, g["fs_a"], g["sr"], g["n_a"], g["n_m"], pj); lv[CLASSES.index(c) - 1] = np.log(sev)   # nhân tố ẩn thật (log hệ số)
            n = min(len(d1[0]), len(b[0])), min(len(d1[2]), len(b[2]))
            sig = (b[0][:n[0]] + g["gacc"] * (d1[0][:n[0]] - d0[0][:n[0]]), b[1][:n[0]] + g["gacc"] * (d1[1][:n[0]] - d0[1][:n[0]]), b[2][:n[1]] + g["gmic"] * (d1[2][:n[1]] - d0[2][:n[1]]))
        out.append(rca_features(sig[0], sig[1], sig[2], g["fs_a"], g["sr"], mrow)); lat.append(lv)
    return np.array(out), np.array(lat)


def make_row(args):
    return make_row_lat(args)[0]


def load():
    d = np.load(f"{OUT}/real_windows.npz"); meta = d["meta"]; fs_a, sr = float(d["fs_a"]), float(d["sr"]); n_a, n_m = d["acc0"].shape[1], d["mic"].shape[1]
    TH = np.array([np.concatenate([nm.fit_window(d[k][i], fs, meta[i, 5] / 60.0) for k, fs in (("acc0", fs_a), ("acc1", fs_a), ("mic", sr))]) for i in range(len(meta))])
    return d, meta, TH, fs_a, sr, n_a, n_m
