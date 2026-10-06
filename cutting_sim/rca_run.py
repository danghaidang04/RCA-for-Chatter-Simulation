"""Sinh dataset RCA (nền nhiễu khớp từ dữ liệu thật + lỗi DDE) -> chia train/val/test THEO ĐƯỜNG CẮT (tầng theo ae) -> nhiều thuật toán RCA."""
import json
from multiprocessing import Pool
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rca_data as rd

R = 5


def main():
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr)
    g = json.load(open(f"{rd.OUT}/rca_gain_effect.json")); print("gain", g, flush=True)
    cid = (meta[:, 0] * 1000 + meta[:, 1]).astype(int); ucut = np.unique(cid)
    # chia theo đường cắt, tầng theo ae: 60/20/20
    rng = np.random.default_rng(0); split = {}
    by_ae = {}
    for c in ucut: by_ae.setdefault(meta[cid == c][0, 6], []).append(c)
    for ae, cs in by_ae.items():
        cs = list(rng.permutation(cs)); n = len(cs); ntr = int(round(0.6 * n)); nva = int(round(0.2 * n))
        for k, c in enumerate(cs): split[c] = "train" if k < ntr else ("val" if k < ntr + nva else "test")
    jobs, tags = [], []
    for r in range(R):
        u_cut = {c: rng.normal(0, np.sqrt(gen.between)) for c in ucut}
        for i in range(len(meta)):
            jobs.append((meta[i], r, 100000 * r + i, u_cut[cid[i]], rd.CLASSES)); tags.append((cid[i], r))
    with Pool(10, initializer=rd._init, initargs=(gen, fs_a, sr, n_a, n_m, g["g_acc"], g["g_mic"])) as pool:
        F = pool.map(rd.make_row, jobs, chunksize=8)
    F = np.array(F); n_rows, nc, nf = F.shape; print("dataset:", F.shape, flush=True)
    X = F.reshape(-1, nf); y = np.tile(np.arange(nc), n_rows); grp = np.repeat([t[0] for t in tags], nc); sp = np.array([split[c] for c in grp])
    ok = np.isfinite(X).all(1); X, y, grp, sp = X[ok], y[ok], grp[ok], sp[ok]
    tr, va, te = sp == "train", sp == "val", sp == "test"
    print(f"mẫu: train {tr.sum()} (cắt {len(set(grp[tr]))}), val {va.sum()} (cắt {len(set(grp[va]))}), test {te.sum()} (cắt {len(set(grp[te]))}); lớp {rd.CLASSES}", flush=True)
    np.savez_compressed(f"{rd.OUT}/rca_dataset.npz", X=X, y=y, grp=grp, split=sp, names=np.array(rd.FEAT_NAMES), classes=np.array(rd.CLASSES))
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    grids = {
        "QDA (Bayes)": [lambda a=a: QDA(reg_param=a) for a in (0.05, 0.2, 0.5)],
        "Gaussian Naive Bayes": [lambda: GaussianNB()],
        "Logistic regression": [lambda C=C: LogisticRegression(C=C, max_iter=2000) for C in (0.1, 1, 10)],
        "kNN": [lambda k=k: KNeighborsClassifier(k) for k in (5, 15, 40)],
        "Random forest": [lambda m=m: RandomForestClassifier(300, min_samples_leaf=m, random_state=0, n_jobs=10) for m in (1, 3, 10)],
        "Gradient boosting": [lambda lr=lr: HistGradientBoostingClassifier(learning_rate=lr, max_iter=200, random_state=0) for lr in (0.03, 0.1)],
        "MLP": [lambda a=a: MLPClassifier((64, 32), alpha=a, max_iter=600, early_stopping=True, random_state=0) for a in (1e-3, 1e-1)],
    }
    res, models = {}, {}
    top3 = lambda P, yy: float(np.mean([yy[i] in np.argsort(-P[i])[:3] for i in range(len(yy))]))
    for name, mk in grids.items():
        best = None
        for f in mk:
            m = f().fit(Xs[tr], y[tr]); acc = float(np.mean(m.predict(Xs[va]) == y[va]))
            if best is None or acc > best[0]: best = (acc, m)
        m = best[1]; P = m.predict_proba(Xs[te]); pr = P.argmax(1); models[name] = m
        conf = np.zeros((nc, nc))
        for a_, b_ in zip(y[te], pr): conf[a_, b_] += 1
        res[name] = {"val_top1": best[0], "test_top1": float(np.mean(pr == y[te])), "test_top3": top3(P, y[te]), "test_macroF1": float(f1_score(y[te], pr, average="macro")),
                     "detect_auroc": float(roc_auc_score(y[te] > 0, 1 - P[:, 0])), "recall": (np.diag(conf) / conf.sum(1)).tolist(), "confusion": (conf / conf.sum(1, keepdims=True)).tolist()}
        print(f"{name:22s} val {best[0]:.3f} | TEST top1 {res[name]['test_top1']:.3f} top3 {res[name]['test_top3']:.3f} macroF1 {res[name]['test_macroF1']:.3f} AUROC(detect) {res[name]['detect_auroc']:.3f}", flush=True)
    print("chance top1 = %.3f, top3 = %.3f" % (1 / nc, 3 / nc))
    # kiểm tra trên dữ liệu THẬT: cửa sổ bình thường nên bị phân loại NORMAL (độ đặc hiệu), cửa sổ bất thường: phân bố dự đoán (không có đáp án)
    RF = np.array([rd.rca_features(d["acc0"][i], d["acc1"][i], d["mic"][i], fs_a, sr, meta[i]) for i in range(len(meta))]); RFs = sc.transform(RF)
    real = {}
    for name, m in models.items():
        p = m.predict(RFs); real[name] = {"label0_pred_NORMAL": float(np.mean(p[meta[:, 3] == 0] == 0)), "label1_pred_dist": np.bincount(p[meta[:, 3] == 1], minlength=nc).tolist()}
        print(f"  [thật] {name:22s} cửa sổ nhãn 0 -> NORMAL: {real[name]['label0_pred_NORMAL']:.2f} | nhãn 1 dự đoán: {dict(zip(rd.CLASSES, real[name]['label1_pred_dist']))}")
    json.dump({"gain": g, "split_cuts": {k: len(set(grp[sp == k])) for k in ("train", "val", "test")}, "results": res, "real_check": real}, open(f"{rd.OUT}/rca_results.json", "w"), indent=1)
    names = list(res); fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); x = np.arange(len(names)); w = 0.38
    ax[0].bar(x - w / 2, [res[n]["test_top1"] for n in names], w, label="top-1"); ax[0].bar(x + w / 2, [res[n]["test_top3"] for n in names], w, label="top-3")
    ax[0].axhline(1 / nc, color="k", ls=":", lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8); ax[0].legend(); ax[0].set_title("RCA test accuracy (held-out cuts); dotted = chance top-1")
    bn = max(names, key=lambda n: res[n]["test_top1"]); C = np.array(res[bn]["confusion"]); ax[1].imshow(C, vmin=0, vmax=1, cmap="Blues")
    for i in range(nc):
        for j in range(nc): ax[1].text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax[1].set_xticks(range(nc)); ax[1].set_xticklabels(rd.CLASSES, rotation=40, ha="right", fontsize=7); ax[1].set_yticks(range(nc)); ax[1].set_yticklabels(rd.CLASSES, fontsize=7); ax[1].grid(False); ax[1].set_title(f"confusion: {bn}")
    fig.tight_layout(); fig.savefig(f"{rd.OUT}/rca_results.png", dpi=130)


if __name__ == "__main__":
    main()
