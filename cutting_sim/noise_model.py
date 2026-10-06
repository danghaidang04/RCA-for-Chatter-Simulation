"""Mô hình nhiễu phổ: tín hiệu = nhiễu TRẮNG + nhiễu HỒNG (1/f^α) + các HÀI của tần số quay f0=N/60.
 1) fit_window: ước lượng thông số từ phổ Welch của TỪNG cửa sổ thật (mỗi kênh: acc0, acc1, mic).
 2) hồi quy ridge: log-thông-số ~ điều kiện cắt (log N, log ae, log F, hướng UP, nhãn); phần dư ~ Gaussian (co rút) = biến thiên tự nhiên.
 3) synth: sinh tín hiệu giả theo thông số bốc từ mô hình -> trích 4 đặc trưng giống hệt dữ liệu thật.
 4) cổng C2ST: khớp trên 70% đường cắt, kiểm trên 30% còn lại (tách theo đường cắt), so thật vs giả tại cùng điều kiện.
Chạy: python noise_model.py
"""
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch
from sklearn.covariance import LedoitWolf
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

from features import nonsync_ratio

K_H = 8                     # số hài mô hình hóa (k·f0, k=1..8)
OUT = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "output")


def knots(fs):
    ny = fs / 2
    return np.r_[np.logspace(np.log10(40.0), np.log10(0.7 * ny), 12), np.array([0.8, 0.87, 0.91, 0.94, 0.96, 0.98]) * ny]


def _hat(f, kn):
    """Ma trận cơ sở hat (nội suy tuyến tính theo log f) -> đường bao phổ = B @ giá_trị_nút."""
    lf, lk = np.log(np.maximum(f, 1e-9)), np.log(kn); B = np.zeros((len(f), len(kn)))
    for j in range(len(kn)):
        e = np.zeros(len(kn)); e[j] = 1.0; B[:, j] = np.interp(lf, lk, e)
    return B


def fit_window(x, fs, f0):
    """Tham số = [log10 PSD tại các nút (đường bao phổ)] + [log10 rms của hài k·f0, k=1..K_H]."""
    x = x.astype(np.float64) - np.mean(x)
    f, P = welch(x, fs, nperseg=2048, noverlap=1024); df = f[1] - f[0]
    kn = knots(fs); near = np.abs(f - np.round(f / f0) * f0) <= 1.6 * df
    m = (f >= 20) & ~near
    B = _hat(f[m], kn)
    # bảo toàn năng lượng: giá trị nút = trung bình TUYẾN TÍNH của PSD trong dải hat quanh nút (khớp log sẽ đánh giá thấp các đỉnh)
    wsum = B.sum(0); node = np.log10((B * P[m][:, None]).sum(0) / np.maximum(wsum, 1e-12) + 1e-30)
    ok = wsum > 0.5
    if not ok.all():
        node[~ok] = np.interp(np.log(kn[~ok]), np.log(kn[ok]), node[ok])
    env = lambda ff: 10 ** (_hat(ff, kn) @ node)
    amps = []
    for k in range(1, K_H + 1):
        fk = k * f0
        if fk > 0.45 * fs:
            amps.append(-12.0); continue
        sel = np.abs(f - fk) <= 1.6 * df
        pw = max(np.sum((P[sel] - env(f[sel])) * df), 1e-24)
        amps.append(0.5 * np.log10(pw))
    return np.r_[node, amps]


def synth(theta, fs, n, f0, rng):
    kn = knots(fs); node = theta[:len(kn)]; amps = 10 ** theta[len(kn):len(kn) + K_H]
    f = np.fft.rfftfreq(n, 1 / fs); S = 10 ** (_hat(np.maximum(f, 1e-3), kn) @ node); S[f < 30] = 0.0
    X = np.sqrt(n * S * fs / 2) * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.sqrt(2); X[0] = 0
    x = np.fft.irfft(X, n)
    t = np.arange(n) / fs
    for k in range(K_H):
        if (k + 1) * f0 < 0.45 * fs:
            x += amps[k] * np.sqrt(2) * np.sin(2 * np.pi * (k + 1) * f0 * t + rng.uniform(0, 2 * np.pi))
    return x


def feats(a0, a1, mic, fs_a, sr, N, Z):
    rms = lambda v: float(np.sqrt(np.mean((v - v.mean()) ** 2)))
    return np.array([np.log10(np.sqrt(rms(a0) ** 2 + rms(a1) ** 2) + 1e-12), 20 * np.log10(rms(mic) + 1e-9),
                     nonsync_ratio(a0, fs_a, N, Z), nonsync_ratio(mic, sr, N, Z)])


def design(meta):
    # cột: ds,cut,win,label,up,N,ae,F,Z
    lN = np.log(meta[:, 5]) - 9.2; up = meta[:, 4]; lab = (meta[:, 3] > 0).astype(float)
    return np.c_[np.ones(len(meta)), lN, np.log(meta[:, 6]), np.log(meta[:, 7]), up, lab, lN ** 2, up * lN, lab * lN]


def main():
    d = np.load(f"{OUT}/real_windows.npz"); meta = d["meta"]; fs_a = float(d["fs_a"]); sr = float(d["sr"])
    sigs = [d["acc0"], d["acc1"], d["mic"]]; fss = [fs_a, fs_a, sr]; n_a, n_m = d["acc0"].shape[1], d["mic"].shape[1]
    nwin = len(meta); print(f"{nwin} cửa sổ, fs_acc={fs_a:.0f}, fs_mic={sr:.0f}", flush=True)
    TH = np.array([np.concatenate([fit_window(sigs[c][i], fss[c], meta[i, 5] / 60.0) for c in range(3)]) for i in range(nwin)])    # (n, 3*(3+K_H))
    Rf = np.array([feats(sigs[0][i], sigs[1][i], sigs[2][i], fs_a, sr, meta[i, 5], int(meta[i, 8])) for i in range(nwin)])
    cid = (meta[:, 0] * 1000 + meta[:, 1]).astype(int); ucut = np.unique(cid)
    import os
    SEED = int(os.environ.get("SEED", "0")); rng = np.random.default_rng(SEED)
    cut_ae = {c: (meta[cid == c][0, 6], meta[cid == c][0, 3] > 0) for c in ucut}
    strata = {}
    for c, k in cut_ae.items(): strata.setdefault(k, []).append(c)
    test_cuts = set()
    for k, cs in strata.items():                                   # chia tầng theo (ae, nhãn): mỗi tầng ≥3 đường cắt mới lấy ~30% làm kiểm
        if len(cs) >= 3:
            test_cuts |= set(rng.choice(cs, size=max(1, int(round(0.3 * len(cs)))), replace=False))
    te = np.array([c in test_cuts for c in cid])
    print(f"khớp trên {np.sum(~te)} cửa sổ ({len(ucut) - len(test_cuts)} đường cắt), kiểm trên {np.sum(te)} cửa sổ ({len(test_cuts)} đường cắt)", flush=True)
    X = design(meta); lam = 1.0
    if os.environ.get("HARM", "0") == "1":
        kn_a = knots(fs_a); ia = np.where((kn_a >= 1400) & (kn_a <= 2500))[0]; band = TH[:, ia].mean(1)        # công suất acc0 (log) quanh 1.4-2.5 kHz
        ftp = meta[:, 8] * meta[:, 5] / 60.0; best = (-1e9, None)
        for fn in np.arange(1300, 2800, 20.0):
            Xc_ = np.c_[X, np.cos(2 * np.pi * fn / ftp), np.sin(2 * np.pi * fn / ftp)]
            b_ = np.linalg.solve(Xc_[~te].T @ Xc_[~te] + lam * np.eye(Xc_.shape[1]), Xc_[~te].T @ band[~te]); sse = np.sum((band[~te] - Xc_[~te] @ b_) ** 2)
            r2 = 1 - sse / np.sum((band[~te] - band[~te].mean()) ** 2)
            if r2 > best[0]: best = (r2, fn)
        r2_0 = 1 - np.sum((band[~te] - X[~te] @ np.linalg.solve(X[~te].T @ X[~te] + lam * np.eye(X.shape[1]), X[~te].T @ band[~te])) ** 2) / np.sum((band[~te] - band[~te].mean()) ** 2)
        print(f"mode ước lượng fn={best[1]:.0f} Hz: R² công suất 1.4–2.5 kHz {r2_0:.3f} -> {best[0]:.3f} (trên tập khớp)", flush=True)
        X = np.c_[X, np.cos(2 * np.pi * best[1] / ftp), np.sin(2 * np.pi * best[1] / ftp)]
    beta = np.linalg.solve(X[~te].T @ X[~te] + lam * np.eye(X.shape[1]), X[~te].T @ TH[~te])
    import os
    NONLIN = os.environ.get('NONLIN', '0') == '1'
    Z = X[:, [1, 2, 3, 4, 5] + ([9, 10] if X.shape[1] > 9 else [])]; zmu, zsd = Z[~te].mean(0), Z[~te].std(0) + 1e-9; Zs = (Z - zmu) / zsd
    if NONLIN:
        base = X[~te] @ beta; kr = KernelRidge(alpha=0.3, kernel='rbf', gamma=0.15).fit(Zs[~te], TH[~te] - base)
        predict = lambda i: X[i] @ beta + kr.predict(Zs[i:i + 1])[0]
        # phần dư NGOÀI MẪU (leave-cuts-out): tránh nội suy làm phương sai giữa đường cắt về 0
        idx = np.where(~te)[0]; oof = np.zeros((len(idx), TH.shape[1]))
        for a_, b_ in GroupKFold(5).split(idx, groups=cid[idx]):
            ia, ib = idx[a_], idx[b_]
            bt = np.linalg.solve(X[ia].T @ X[ia] + lam * np.eye(X.shape[1]), X[ia].T @ TH[ia])
            k_ = KernelRidge(alpha=0.3, kernel='rbf', gamma=0.15).fit(Zs[ia], TH[ia] - X[ia] @ bt)
            oof[b_] = X[ib] @ bt + k_.predict(Zs[ib])
        res = TH[~te] - oof
    else:
        predict = lambda i: X[i] @ beta; res = TH[~te] - X[~te] @ beta
    U_, sv, Vt = np.linalg.svd(res - res.mean(0), full_matrices=False); KC = 12; V = Vt[:KC].T          # KC thành phần chính của phần dư
    sc = res @ V; gid_tr = cid[~te]; ug = np.unique(gid_tr)
    cm = np.array([sc[gid_tr == g].mean(0) for g in ug]); within = np.mean([sc[gid_tr == g].var(0, ddof=1) for g in ug if (gid_tr == g).sum() > 1], axis=0)
    nbar = np.mean([(gid_tr == g).sum() for g in ug]); between = np.maximum(cm.var(0, ddof=1) - within / nbar, 1e-12)
    print("phương sai phần dư: giữa đường cắt / trong đường cắt (thành phần 1-4):", np.round(between[:4], 3), np.round(within[:4], 3), flush=True)
    cov = None
    lo_, hi_ = np.percentile(TH[~te], 1, axis=0), np.percentile(TH[~te], 99, axis=0)      # kẹp tham số trong khoảng quan sát được
    out = {}
    for name, use_noise in (("regression mean only", False), ("hierarchical: between-cut + within-cut", True)):
        S, G = [], []
        for r in range(5):
            r_ = np.random.default_rng(100 + r); ucut_te = {c: r_.normal(0, np.sqrt(between)) for c in np.unique(cid[te])}
            for i in np.where(te)[0]:
                delta = (ucut_te[cid[i]] + r_.normal(0, np.sqrt(within))) @ V.T if use_noise else 0.0
                th = np.clip(predict(i) + delta, lo_, hi_)
                f0 = meta[i, 5] / 60.0; ka = len(knots(fs_a)) + K_H; km = len(knots(sr)) + K_H
                a0 = synth(th[:ka], fs_a, n_a, f0, r_); a1 = synth(th[ka:2 * ka], fs_a, n_a, f0, r_); mc = synth(th[2 * ka:2 * ka + km], sr, n_m, f0, r_)
                S.append(feats(a0, a1, mc, fs_a, sr, meta[i, 5], int(meta[i, 8]))); G.append(cid[i])
        S = np.array(S); G = np.array(G)
        Xc = np.vstack([np.c_[Rf[te], X[te][:, 1:4]], np.c_[S, np.tile(X[te][:, 1:4], (5, 1))]]); y = np.r_[np.zeros(te.sum()), np.ones(len(S))]
        gg = np.r_[cid[te], G]; mu, sd = Xc.mean(0), Xc.std(0) + 1e-9; accs = []
        for s in range(3):
            clf = RandomForestClassifier(200, min_samples_leaf=3, random_state=s, n_jobs=4, class_weight="balanced")
            accs.append(cross_val_score(clf, (Xc - mu) / sd, y, groups=gg, cv=GroupKFold(min(5, len(np.unique(gg)))), scoring="balanced_accuracy").mean())
        out[name] = [float(np.mean(accs)), float(np.std(accs))]
        print(f"GATE (held-out cuts) {name:32s} C2ST(balanced acc; 0.5 = không phân biệt được) = {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)
        print("   feature  real mean±sd | synth mean±sd")
        for j, nm in enumerate(["log10 acc", "mic dB", "chat_acc", "chat_mic"]):
            print(f"   {nm:10s} {Rf[te][:, j].mean():7.3f}±{Rf[te][:, j].std():.3f} | {S[:, j].mean():7.3f}±{S[:, j].std():.3f}", flush=True)
    json.dump({"c2st_heldout": out, "n_test_cuts": len(test_cuts)}, open(f"{OUT}/noise_model_gate.json", "w"), indent=1)
    np.savez(f"{OUT}/noise_model_params.npz", beta=beta, cov=cov, names=np.array(["acc0", "acc1", "mic"]))


if __name__ == "__main__":
    main()
