"""Kiểm tra độ bền của RCA (chống 'inverse crime'):
 S1 train danh định -> test danh định (đường cơ sở)       S2 train danh định -> test với tham số DDE LỆCH
 S3 train ngẫu nhiên hóa tham số DDE -> test lệch (khắc phục)   S4 train lỗi NHẸ -> test lỗi NẶNG (độ nặng chưa thấy)
 Mỗi cài đặt: 5 cách chia theo đường cắt (tầng theo ae) x 2 hạt giống thuật toán; báo trung bình ± sd.
 Thuật toán: GB, RF, MLP, QDA, Logistic, GaussianNB, và MẠNG BAYES (bằng chứng rời rạc, bảng xác suất điều kiện học từ train)."""
import json
from multiprocessing import Pool
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
import rca_data as rd

R = 3


def gen_set(gen, meta, cid, fs_a, sr, n_a, n_m, g, mode, part, seed0):
    rng = np.random.default_rng(seed0); ucut = np.unique(cid); jobs = []
    for r in range(R):
        u_cut = {c: rng.normal(0, np.sqrt(gen.between)) for c in ucut}
        for i in range(len(meta)): jobs.append((meta[i], r, seed0 + 100000 * r + i, u_cut[cid[i]], rd.CLASSES, mode, part))
    with Pool(10, initializer=rd._init, initargs=(gen, fs_a, sr, n_a, n_m, g["g_acc"], g["g_mic"])) as pool:
        F = np.array(pool.map(rd.make_row, jobs, chunksize=8))
    return F  # (R*n, 6, nf), thứ tự: rep-major theo hàng meta


def models(seed):
    return {"Gradient boosting": lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, random_state=seed),
            "Random forest": lambda: RandomForestClassifier(300, min_samples_leaf=3, random_state=seed, n_jobs=10),
            "MLP": lambda: MLPClassifier((64, 32), alpha=1e-1, max_iter=600, early_stopping=True, random_state=seed),
            "QDA (Gaussian Bayes)": lambda: QDA(reg_param=0.2), "Gaussian Naive Bayes": lambda: GaussianNB(), "Logistic regression": lambda: LogisticRegression(C=1, max_iter=2000)}


def fit_eval(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr); a, b = sc.transform(Xtr), sc.transform(Xte); out = {}
    for n, mk in models(seed).items():
        out[n] = float(np.mean(mk().fit(a, ytr).predict(b) == yte))
    # MẠNG BAYES: bằng chứng rời rạc (5 mức theo phân vị của dữ liệu NORMAL train), CPT học bằng đếm (Laplace)
    kb = KBinsDiscretizer(5, encode="ordinal", strategy="quantile", subsample=None).fit(Xtr[ytr == 0]); Dtr = np.clip(kb.transform(Xtr), 0, 4).astype(int); Dte = np.clip(kb.transform(Xte), 0, 4).astype(int)
    out["Bayesian network (learned CPTs)"] = float(np.mean(CategoricalNB(alpha=1.0, min_categories=5).fit(Dtr, ytr).predict(Dte) == yte))
    return out


def main():
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr); g = json.load(open(f"{rd.OUT}/rca_gain_effect.json"))
    cid = (meta[:, 0] * 1000 + meta[:, 1]).astype(int); n = len(meta); nc = len(rd.CLASSES)
    sets = {"nominal": ("nominal", "full", 11), "shifted": ("shift", "full", 22), "randomized": ("shift", "full", 33), "sev_low": ("nominal", "low", 44), "sev_high": ("nominal", "high", 55)}
    F = {}
    for k, (mode, part, s0) in sets.items():
        F[k] = gen_set(gen, meta, cid, fs_a, sr, n_a, n_m, g, mode, part, s0); print("sinh xong", k, F[k].shape, flush=True)
    nf = F["nominal"].shape[2]; ucut = np.unique(cid); by = {}
    for c in ucut: by.setdefault(meta[cid == c][0, 6], []).append(c)
    scen = {"S1 train nominal -> test nominal": ("nominal", "nominal"), "S2 train nominal -> test SHIFTED DDE params": ("nominal", "shifted"),
            "S3 train randomized DDE -> test SHIFTED": ("randomized", "shifted"), "S4 train mild faults -> test severe faults": ("sev_low", "sev_high")}
    res = {k: {} for k in scen}
    for split_seed in range(5):
        rng = np.random.default_rng(split_seed); tr_c, te_c = set(), set()
        for ae, cs in by.items():
            cs = list(rng.permutation(cs)); nt = max(1, int(round(0.3 * len(cs)))) if len(cs) >= 3 else 0
            te_c |= set(cs[:nt]); tr_c |= set(cs[nt:])
        row_cut = np.tile(cid, R); mtr = np.isin(row_cut, list(tr_c)); mte = np.isin(row_cut, list(te_c))
        flat = lambda A, m: (A[m].reshape(-1, nf), np.tile(np.arange(nc), m.sum()))
        for sname, (a, b) in scen.items():
            Xtr, ytr = flat(F[a], mtr); Xte, yte = flat(F[b], mte); ok1, ok2 = np.isfinite(Xtr).all(1), np.isfinite(Xte).all(1)
            for sd in range(2):
                for algo, acc in fit_eval(Xtr[ok1], ytr[ok1], Xte[ok2], yte[ok2], sd).items(): res[sname].setdefault(algo, []).append(acc)
        print("chia", split_seed, "xong", flush=True)
    algos = list(next(iter(res.values())))
    print("\ntop-1 trên test (trung bình ± sd qua 5 cách chia x 2 hạt giống); đoán bừa = 0.167")
    print("thuật toán".ljust(32), *[s.split(" ")[0].rjust(12) for s in scen])
    for a in algos: print(a.ljust(32), *[f"{np.mean(res[s][a]):.3f}±{np.std(res[s][a]):.3f}".rjust(12) for s in scen])
    json.dump({s: {a: [float(np.mean(v)), float(np.std(v))] for a, v in r.items()} for s, r in res.items()}, open(f"{rd.OUT}/rca_robust.json", "w"), indent=1)


if __name__ == "__main__":
    main()
