"""Học hệ số khuếch đại (g_acc, g_mic) của tín hiệu chênh lệch DDE: khớp phân bố đặc trưng của mẫu LỖI sinh ra với cửa sổ BẤT THƯỜNG thật (nhãn 1) bằng MMD (lưới)."""
import json
from multiprocessing import Pool
import numpy as np
import rca_data as rd

_S = {}


def _init2(fs_a, sr):
    _S.update(fs_a=fs_a, sr=sr)


def feat_pair(args):
    b, d, gacc, gmic, mrow = args
    s = _S; n = min(len(b[0]), len(d[0])), min(len(b[2]), len(d[2]))
    return rd.rca_features(b[0][:n[0]] + gacc * d[0][:n[0]], b[1][:n[0]] + gacc * d[1][:n[0]], b[2][:n[1]] + gmic * d[2][:n[1]], s["fs_a"], s["sr"], mrow)


def mmd(X, Y):
    Z = np.vstack([X, Y]); d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1); med = np.median(d2[d2 > 0]) + 1e-9; n = len(X); t = 0
    for m in (0.25, 1.0, 4.0):
        K = np.exp(-d2 / (m * med)); t += K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()
    return t / 3


def base_pair(args):
    mrow, cls, seed, u = args; g = rd._G; rng = np.random.default_rng(seed)
    b = g["gen"].signals(mrow, u, rng, g["n_a"], g["n_m"]); d0 = rd.dde_signals(mrow, "NORMAL", 1.0, seed, g["fs_a"], g["sr"], g["n_a"], g["n_m"])
    lo, hi = rd.SEV[cls]; sev = rng.uniform(lo, hi) if cls != "RPM_MISMATCH" else rng.choice([-1, 1]) * rng.uniform(0.03, 0.15) + 1.0
    d1 = rd.dde_signals(mrow, cls, sev, seed, g["fs_a"], g["sr"], g["n_a"], g["n_m"])
    n = min(len(d1[0]), len(b[0])), min(len(d1[2]), len(b[2]))
    return [x[:n[0]] for x in b[:2]] + [b[2][:n[1]]], [d1[0][:n[0]] - d0[0][:n[0]], d1[1][:n[0]] - d0[1][:n[0]], d1[2][:n[1]] - d0[2][:n[1]]]


def main():
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr)
    real1 = np.where(meta[:, 3] == 1)[0]; real0 = np.where(meta[:, 3] == 0)[0]
    Rf = lambda idx: np.array([rd.rca_features(d["acc0"][i], d["acc1"][i], d["mic"][i], fs_a, sr, meta[i]) for i in idx])
    F1, F0 = Rf(real1), Rf(real0); sd = np.vstack([F0, F1]).std(0) + 1e-9; mu = np.vstack([F0, F1]).mean(0)
    faults = [c for c in rd.CLASSES if c != "NORMAL"]; jobs = []
    for r in range(2):
        for i in real1:
            for k, c in enumerate(faults): jobs.append((meta[i], c, 1000 * r + 10 * int(i) + k, np.zeros(12)))
    with Pool(10, initializer=rd._init, initargs=(gen, fs_a, sr, n_a, n_m, 0.0, 0.0)) as pool:
        BD = pool.map(base_pair, jobs, chunksize=8)
    print(f"{len(jobs)} mẫu lỗi cơ sở", flush=True)
    grid_a = 10.0 ** np.arange(-9, -3.4, 0.5); grid_m = 10.0 ** np.arange(-7, -1.4, 0.5); L = np.zeros((len(grid_a), len(grid_m)))
    with Pool(10, initializer=_init2, initargs=(fs_a, sr)) as pool:
        for ia, ga in enumerate(grid_a):
            for im, gm in enumerate(grid_m):
                Fs = np.array(pool.map(feat_pair, [(b, dd, ga, gm, jobs[j][0]) for j, (b, dd) in enumerate(BD)], chunksize=40))
                L[ia, im] = mmd((Fs - mu) / sd, (F1 - mu) / sd)
            print(f"g_acc=1e{np.log10(ga):.1f}: min MMD {L[ia].min():.4f} tại g_mic=1e{np.log10(grid_m[L[ia].argmin()]):.1f}", flush=True)
    ia, im = np.unravel_index(L.argmin(), L.shape)
    out = {"g_acc": float(grid_a[ia]), "g_mic": float(grid_m[im]), "mmd_best": float(L.min()), "mmd_worst": float(L.max()), "mmd_median": float(np.median(L))}
    json.dump({**out, "grid_acc": grid_a.tolist(), "grid_mic": grid_m.tolist(), "loss": L.tolist()}, open(f"{rd.OUT}/rca_gain.json", "w"), indent=1)
    print("GAIN", out)


if __name__ == "__main__":
    main()
