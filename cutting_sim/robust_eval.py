"""Độ bền RCA khi tương tác máy-dao thay đổi: huấn luyện trên máy danh định vs trên nhiều máy (họ ràng buộc vật lý),
đặc trưng THÔ vs TƯƠNG ĐỐI theo nền khỏe của chính máy, kiểm tra trên máy CHƯA THẤY trong khoảng và NGOÀI khoảng; có chọn 'không chắc'.
Cảnh báo: toàn bộ trong mô phỏng; 'máy' do họ tham số giả định sinh ra; chưa phải kiểm chứng trên máy thật."""
import json, sys
from multiprocessing import Pool
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import rca_data as rd
import machine_family as mf

NW, NCAL = 48, 40


def rel_features(X, XC, mrows_x, mrows_c):
    """Đặc trưng 0..25 -> độ lệch so với nền khỏe của máy (ridge theo điều kiện, chuẩn hóa sd phần dư); 26..29 (điều kiện) giữ nguyên."""
    C = lambda M: np.c_[np.ones(len(M)), np.log(M[:, 5]) - 9.2, np.log(M[:, 6]), np.log(M[:, 7]), M[:, 4]]
    A = C(mrows_c); B = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ XC[:, :26]); sd = (XC[:, :26] - A @ B).std(0) + 1e-6
    return np.c_[(X[:, :26] - C(mrows_x) @ B) / sd, X[:, 26:]]


def build(machs, meta, seed0, pool):
    jobs = []
    for i, m in enumerate(machs):
        r = np.random.default_rng(seed0 + i); jobs.append((m, meta[r.integers(0, len(meta), NW)], meta[r.integers(0, len(meta), NCAL)], seed0 + 10 * i))
    out = pool.map(mf.machine_job, jobs, chunksize=1); RAW, REL, Y, G = [], [], [], []
    for i, (X, y, XC) in enumerate(out):
        mx = np.repeat(jobs[i][1], 6, axis=0); RAW.append(X); REL.append(rel_features(X, XC, mx, jobs[i][2])); Y.append(y); G.append(np.full(len(y), i))
    return np.vstack(RAW), np.vstack(REL), np.concatenate(Y), np.concatenate(G)


def main(seed=0, ntr=40, nva=10, nte=16, nex=16):
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr); g = json.load(open(f"{rd.OUT}/rca_gain_effect.json"))
    rng = np.random.default_rng(seed)
    sets = {"nominal_train": [mf.nominal_machine() for _ in range(ntr)], "rand_train": [mf.sample_machine(rng, "train") for _ in range(ntr)], "val": [mf.sample_machine(rng, "train") for _ in range(nva)],
            "test_in": [mf.sample_machine(rng, "train") for _ in range(nte)], "test_extrap": [mf.sample_machine(rng, "extrap") for _ in range(nex)]}
    D = {}
    with Pool(10, initializer=rd._init, initargs=(gen, fs_a, sr, n_a, n_m, g["g_acc"], g["g_mic"])) as pool:
        for k, ms in sets.items(): D[k] = build(ms, meta, 1000 * seed + 7 * len(k), pool); print("sinh", k, D[k][0].shape, flush=True)
    names = {"nominal_train": "nominal", "rand_train": "randomized"}; res = {}
    for tr_key, tr_name in (("nominal_train", "train on ONE nominal machine"), ("rand_train", "train on MANY machines")):
        for fi, ft in ((0, "raw features"), (1, "healthy-relative features")):
            Xtr, ytr = D[tr_key][fi], D[tr_key][2]; sc = StandardScaler().fit(Xtr)
            for cn, mk in (("GB", lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=150, random_state=seed)), ("RF", lambda: RandomForestClassifier(200, min_samples_leaf=3, random_state=seed, n_jobs=10))):
                m = mk().fit(sc.transform(Xtr), ytr)
                for tk in ("val", "test_in", "test_extrap"):
                    P = m.predict_proba(sc.transform(D[tk][fi])); pr = P.argmax(1); y = D[tk][2]
                    res[f"{tr_name} | {ft} | {cn} | {tk}"] = {"top1": float(np.mean(pr == y)), "macroF1": float(f1_score(y, pr, average="macro")), "top3": float(np.mean([y[i] in np.argsort(-P[i])[:3] for i in range(len(y))]))}
                    if tk == "val": thr_P = P; thr_y = y
                # chọn ngưỡng 'không chắc' trên máy val: độ chính xác >= 0.9 trên phần được trả lời
                best = None
                for th in np.linspace(0.3, 0.99, 36):
                    ans = thr_P.max(1) >= th
                    if ans.mean() > 0.05 and np.mean(thr_P.argmax(1)[ans] == thr_y[ans]) >= 0.9: best = th; break
                for tk in ("test_in", "test_extrap"):
                    P = m.predict_proba(sc.transform(D[tk][fi])); y = D[tk][2]; ans = P.max(1) >= (best if best else 1.1)
                    res[f"{tr_name} | {ft} | {cn} | {tk}"].update({"coverage": float(ans.mean()), "acc_answered": float(np.mean(P.argmax(1)[ans] == y[ans])) if ans.any() else float("nan"), "thr": float(best) if best else None})
    json.dump(res, open(f"{rd.OUT}/robust_eval_seed{seed}.json", "w"), indent=1)
    print("\nseed", seed, "(đoán bừa top-1 = 0.167)")
    print("cấu hình".ljust(66), "val".rjust(6), "test_in".rjust(8), "extrap".rjust(8), " | không-chắc: cover/acc (test_in -> extrap)")
    for key in [k for k in res if k.endswith("| val")]:
        base = key[:-6]; a = res[base + " | val"]; b = res[base + " | test_in"]; c = res[base + " | test_extrap"]
        print(base.ljust(66), f"{a['top1']:.3f}".rjust(6), f"{b['top1']:.3f}".rjust(8), f"{c['top1']:.3f}".rjust(8), f" | {b['coverage']:.2f}/{b['acc_answered']:.2f} -> {c['coverage']:.2f}/{c['acc_answered']:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
