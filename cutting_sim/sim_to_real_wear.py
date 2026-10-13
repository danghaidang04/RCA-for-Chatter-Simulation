"""Khung A, phép thử THẬT: học trước bằng MÔ PHỎNG lực (mòn dao) rồi tinh chỉnh bằng k mẫu thật (LUH, máy chưa thấy).
 Mô phỏng: DDE phay 4 rãnh, tham số máy-dao và điều kiện cắt NGẪU NHIÊN (GIẢ ĐỊNH, phạm vi rộng), mòn VB tăng Kt, lực cạnh, độ cản (hệ số ngẫu nhiên: GIẢ ĐỊNH).
 Đặc trưng: chỉ lực x,y (mean/std/kurtosis/7 băng/resultant), chuẩn hóa TƯƠNG ĐỐI theo nền khỏe (VB<=10) của từng dao (mô phỏng và thật).
 So sánh: sim-only | scratch(k) | sim->tinh chỉnh(k) | tham chiếu: học từ 2 máy thật. Tinh chỉnh = hồi quy logistic có chính quy hóa kéo về trọng số tiền huấn luyện."""
import glob, json, os, sys
import h5py, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.signal import welch
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import process_from_cut, replace  # noqa
from milling_model import simulate_milling, single_mode  # noqa
from sensors import apply_sensors, SensorConfig  # noqa

H5 = "../data/luh_raw/h5"; BANDS = [0, 25, 100, 300, 1000, 3000, 8000, 12500]; OUT = "output"


def force_feats(fx, fy, fs):
    out = []
    for x in (fx, fy):
        d = x - x.mean(); f, P = welch(d, fs, nperseg=4096)
        out += [x.mean(), d.std(), kurtosis(d)] + [np.log10(P[(f >= BANDS[i]) & (f < BANDS[i + 1])].sum() + 1e-12) for i in range(len(BANDS) - 1)]
    R = np.hypot(fx - fx.mean(), fy - fy.mean()); return np.array(out + [R.mean(), R.std(), np.log10(R.var() + 1e-12)])


def real_features():
    cache = f"{OUT}/luh_force_xy.npz"
    if os.path.exists(cache): d = np.load(cache); return d["X"], pd.DataFrame(d["lab"], columns=["machine", "tool", "run", "wear"])
    X, L = [], []
    for p in sorted(glob.glob(f"{H5}/*.h5")):
        with h5py.File(p) as f:
            fx, fy = [f["signals_sensor"][f"force_sensor_{a}"][0].astype(np.float64) for a in "xy"]; L.append([float(f["labels"][k][0, 0]) for k in ("machine", "tool", "run", "wear")])
        X.append(force_feats(fx, fy, 25000.0))
    X = np.array(X); np.savez(cache, X=X, lab=np.array(L)); return X, pd.DataFrame(L, columns=["machine", "tool", "run", "wear"])


def rel_by_tool(X, wear, groups):
    Z = np.zeros_like(X)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]; base = idx[wear[idx] <= 10]
        if len(base) < 3: base = idx[np.argsort(wear[idx])[:3]]
        mu, sd = X[base].mean(0), X[base].std(0) + 1e-6 * (np.abs(X[base]).mean(0) + 1); Z[idx] = (X[idx] - mu) / sd
    return Z


def sim_tool(seed):
    """Một cặp máy-dao mô phỏng: 45 lần chạy với VB tăng dần -> (đặc trưng thô, VB)."""
    r = np.random.default_rng(seed); lu = lambda a, b: float(np.exp(r.uniform(np.log(a), np.log(b))))
    D = float(r.choice([6, 8, 10, 12])); N = float(r.uniform(3000, 9000)); fz = r.uniform(0.03, 0.12); Z = 4
    cut = dict(N=N, ap=r.uniform(2, 8), ae=r.uniform(0.15, 0.6) * D, F=fz * Z * N / 60.0, Z=Z, D=D, direction="DOWN")
    fn, ze, k = lu(800, 3500), r.uniform(0.01, 0.05), lu(1e6, 3e7); Kt0, Ke0 = lu(0.8e9, 2.5e9), lu(2e3, 4e4)
    al, be, ga = r.uniform(0.1, 0.6), r.uniform(0.5, 4.0), r.uniform(0.0, 1.0)            # tác động của mòn: GIẢ ĐỊNH, ngẫu nhiên
    cfg = SensorConfig(dyno_fn=lu(1500, 4000), dyno_zeta=r.uniform(0.02, 0.1), dyno_noise_n=lu(0.2, 2.0))
    vb = np.sort(r.uniform(3, 160, 45)); vb[:3] = np.sort(r.uniform(3, 10, 3)); feats = []
    for i, v in enumerate(vb):
        w = v / 100.0
        p = process_from_cut(cut); p = replace(p, Kt=Kt0 * (1 + al * w), Kr=0.3, Kte=Ke0 * (1 + be * w), Kre=0.7 * Ke0 * (1 + be * w), helix=np.deg2rad(30), nz=8, D_tool=D * 1e-3, runout=r.uniform(2e-6, 2e-5),
                                  mod_x=single_mode(fn, ze * (1 + ga * w), k), mod_y=single_mode(fn * r.uniform(0.95, 1.05), ze * (1 + ga * w), k))
        sim = simulate_milling(p, t_end=1.0, fs=25600.0, noise_um=0.02, seed=int(r.integers(1 << 30))); sig = apply_sensors(sim, cfg, seed=int(r.integers(1 << 30)))
        feats.append(force_feats(sig["F_x"][6000:], sig["F_y"][6000:], 25600.0))
    return np.array(feats), vb


def fit_prior(X, y, w0, lam):
    """Hồi quy logistic cân bằng lớp, chính quy hóa lam*||w-w0||^2 (w0=0: học từ đầu; w0=trọng số mô phỏng: tinh chỉnh)."""
    Xb = np.c_[X, np.ones(len(X))]; wt = np.where(y == 1, 0.5 / max(1, (y == 1).sum()), 0.5 / max(1, (y == 0).sum())) * len(y)
    def f(w):
        z = Xb @ w; l = np.sum(wt * (np.logaddexp(0, z) - y * z)) + lam * np.sum((w - w0) ** 2); p = 1 / (1 + np.exp(-z)); return l, Xb.T @ (wt * (p - y)) + 2 * lam * (w - w0)
    return minimize(f, w0.copy(), jac=True, method="L-BFGS-B").x


def main(n_sim=150):
    Xr, lab = real_features(); wear, mach = lab.wear.values, lab.machine.values.astype(int); grp = (lab.machine.values * 10 + lab.tool.values).astype(int); THR = float(os.environ.get('THR', '100')); y = (wear >= THR).astype(int)
    if os.environ.get('ABSMEAN', '1') == '1':                      # bất biến theo dấu quy ước trục lực: dùng |lực trung bình| (cột 0 = x_mean, 10 = y_mean)
        Xr = Xr.copy(); Xr[:, [0, 10]] = np.abs(Xr[:, [0, 10]])
    Zr = rel_by_tool(Xr, wear, grp)
    cache = f"{OUT}/sim_wear_features.npz"
    if os.path.exists(cache): d = np.load(cache); Xs, vbs, gs = d["X"], d["vb"], d["g"]
    else:
        with Pool(10) as p: res = p.map(sim_tool, range(n_sim))
        Xs = np.vstack([r[0] for r in res]); vbs = np.concatenate([r[1] for r in res]); gs = np.repeat(np.arange(n_sim), 45); np.savez(cache, X=Xs, vb=vbs, g=gs)
    if os.environ.get('ABSMEAN', '1') == '1': Xs = Xs.copy(); Xs[:, [0, 10]] = np.abs(Xs[:, [0, 10]])
    Zs = rel_by_tool(Xs, vbs, gs); ys = (vbs >= THR).astype(int); ok = np.isfinite(Zs).all(1) & (np.abs(Zs).max(1) < 1e4); Zs, ys = Zs[ok], ys[ok]
    print(f"mô phỏng: {len(Zs)} lần chạy, {int(ys.mean() * 100)}% mòn | thật: {len(Zr)} lần chạy, {int(y.mean() * 100)}% mòn", flush=True)
    sc_s = StandardScaler().fit(Zs); lr_s = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(sc_s.transform(Zs), ys)
    res = {}; rng = np.random.default_rng(0); KS = (2, 4, 8, 16, 32); LAMS = (0.3, 1.0, 3.0)
    for tm in (1, 2, 3):
        te = mach == tm; sc_t = StandardScaler().fit(Zr[te]); Zt = sc_t.transform(Zr[te]); yt = y[te]          # chuẩn hóa theo thống kê KHÔNG NHÃN của máy đích
        w_sim = np.r_[lr_s.coef_[0], lr_s.intercept_[0]]; r_ = {}
        r_["sim-only"] = roc_auc_score(yt, np.c_[Zt, np.ones(len(Zt))] @ w_sim); r_["rms-rule"] = roc_auc_score(yt, Zr[te][:, 20])
        sc_src = StandardScaler().fit(Zr[~te]); src = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(sc_src.transform(Zr[~te]), y[~te]); r_["real-source (2 machines, all labels)"] = roc_auc_score(yt, src.predict_proba(sc_src.transform(Zr[te]))[:, 1])
        for k in KS:
            for lam in LAMS:
                a_s, a_f = [], []
                for _ in range(40):
                    idx = rng.choice(len(Zt), k, replace=False)
                    if y[te][idx].min() == y[te][idx].max(): continue
                    rest = np.setdiff1d(np.arange(len(Zt)), idx); w_f = fit_prior(Zt[idx], yt[idx], w_sim, lam); w_0 = fit_prior(Zt[idx], yt[idx], np.zeros_like(w_sim), lam)
                    Xb = np.c_[Zt[rest], np.ones(len(rest))]; a_f.append(roc_auc_score(yt[rest], Xb @ w_f)); a_s.append(roc_auc_score(yt[rest], Xb @ w_0))
                r_[f"k={k} lam={lam} scratch"] = float(np.mean(a_s)) if a_s else float("nan"); r_[f"k={k} lam={lam} sim->finetune"] = float(np.mean(a_f)) if a_f else float("nan")
        res[tm] = r_
    json.dump(res, open(f"{OUT}/sim_to_real_wear{'_absmean' if os.environ.get('ABSMEAN', '1') == '1' else ''}{'' if THR == 100 else f'_thr{int(THR)}'}.json", "w"), indent=1)
    print("\nAUROC (VB>=100) trên máy ĐÍCH chưa thấy; đoán bừa 0.5 (trung bình 40 lần lấy k mẫu có nhãn)")
    print("rms-rule: " + ", ".join(f"M{m} {res[m]['rms-rule']:.3f}" for m in res)); print("sim-only: " + ", ".join(f"M{m} {res[m]['sim-only']:.3f}" for m in res), "| tham chiếu 2 máy thật: " + ", ".join(f"M{m} {res[m]['real-source (2 machines, all labels)']:.3f}" for m in res))
    print("\nk  lam   scratch  sim->finetune   (trung bình 3 máy đích)")
    for k in KS:
        for lam in LAMS:
            s = np.nanmean([res[m][f"k={k} lam={lam} scratch"] for m in res]); f = np.nanmean([res[m][f"k={k} lam={lam} sim->finetune"] for m in res]); print(f"{k:2d} {lam:4.1f}   {s:.3f}    {f:.3f}")


if __name__ == "__main__":
    main()
