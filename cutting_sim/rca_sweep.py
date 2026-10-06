"""Đường cong độ khó: RCA theo hệ số khuếch đại lỗi (0.3x, 1x, 3x giá trị hiệu chuẩn) + recall từng lớp. Dataset nhỏ (R=3)."""
import json
from multiprocessing import Pool
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rca_data as rd


def main():
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr); g = json.load(open(f"{rd.OUT}/rca_gain_effect.json"))
    cid = (meta[:, 0] * 1000 + meta[:, 1]).astype(int); ucut = np.unique(cid); rng = np.random.default_rng(0); split = {}; by = {}
    for c in ucut: by.setdefault(meta[cid == c][0, 6], []).append(c)
    for ae, cs in by.items():
        cs = list(rng.permutation(cs)); n = len(cs); a = int(round(0.6 * n)); b = int(round(0.2 * n))
        for k, c in enumerate(cs): split[c] = "train" if k < a else ("val" if k < a + b else "test")
    out = {}; scales = [0.3, 1.0, 3.0]
    for s in scales:
        rng2 = np.random.default_rng(1); jobs = []
        for r in range(3):
            u_cut = {c: rng2.normal(0, np.sqrt(gen.between)) for c in ucut}
            for i in range(len(meta)): jobs.append((meta[i], r, 100000 * r + i, u_cut[cid[i]], rd.CLASSES))
        with Pool(10, initializer=rd._init, initargs=(gen, fs_a, sr, n_a, n_m, g["g_acc"] * s, g["g_mic"] * s)) as pool:
            F = np.array(pool.map(rd.make_row, jobs, chunksize=8))
        nr, nc, nf = F.shape; X = F.reshape(-1, nf); y = np.tile(np.arange(nc), nr); grp = np.repeat([cid[j[2] % 100000] for j in jobs], nc); sp = np.array([split[c] for c in grp])
        ok = np.isfinite(X).all(1); X, y, sp = X[ok], y[ok], sp[ok]; tr, te = sp == "train", sp == "test"; sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
        out[s] = {}
        for name, m in (("Gradient boosting", HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, random_state=0)), ("Random forest", RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=10)),
                        ("QDA (Bayes)", QDA(reg_param=0.2)), ("Logistic regression", LogisticRegression(C=1, max_iter=2000))):
            m.fit(Xs[tr], y[tr]); pr = m.predict(Xs[te]); rec = [float(np.mean(pr[y[te] == k] == k)) for k in range(nc)]
            out[s][name] = {"top1": float(np.mean(pr == y[te])), "recall": rec}
        print(f"gain x{s}: " + " | ".join(f"{n} {v['top1']:.3f}" for n, v in out[s].items()), flush=True)
    print("\nrecall từng lớp (Gradient boosting) theo mức khuếch đại:"); print("lớp".ljust(16), *[f"x{s}".rjust(7) for s in scales])
    for k, c in enumerate(rd.CLASSES): print(c.ljust(16), *[f"{out[s]['Gradient boosting']['recall'][k]:7.2f}" for s in scales])
    json.dump({str(k): v for k, v in out.items()}, open(f"{rd.OUT}/rca_sweep.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(6.5, 4)); 
    for name in out[1.0]: ax.plot(scales, [out[s][name]["top1"] for s in scales], "o-", label=name)
    ax.set_xscale("log"); ax.set_xlabel("fault gain relative to calibrated"); ax.set_ylabel("test top-1"); ax.axhline(1 / 6, color="k", ls=":"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{rd.OUT}/rca_sweep.png", dpi=130)


if __name__ == "__main__":
    main()
