"""Thử hướng: chuyển giao thật có suy giảm khi tổ hợp máy-dao ĐÍCH xa các tổ hợp huấn luyện không?
Đại diện động lực học = phổ lực (7 băng x 3 kênh, log10) trung bình trên các lần chạy khỏe (VB<=10) của mỗi (máy,dao). KHÔNG phải FRF -> chỉ là proxy.
Với mỗi dao đích (9): huấn luyện trên dao của MÁY KHÁC (6 dao): (a) tất cả, (b) 3 dao gần nhất, (c) 3 dao xa nhất theo proxy; đo AUROC trên mọi lần chạy của dao đích."""
import json, warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import luh_transfer as lt
warnings.filterwarnings("ignore")


def main():
    lab, X = lt.load(); M, T, W = lab.machine.values, lab.tool.values, lab.wear.values; pairs = sorted({(m, t) for m, t in zip(M, T)})
    band = [c for ch in range(3) for c in range(10 * ch + 3, 10 * ch + 10)]                      # 21 cột băng tần
    idx = {p: np.where((M == p[0]) & (T == p[1]))[0] for p in pairs}; prox = {p: X[idx[p][np.argsort(W[idx[p]])[:3]]][:, band].mean(0) for p in pairs}      # 3 lần chạy mòn ít nhất (như rel())
    skipped = [p for p in pairs if p not in prox]; print("bỏ (không có):", skipped); pairs = [p for p in pairs if p in prox]
    FS = {"raw": X, "tool-relative": lt.rel(X, lab, "tool")}; out = {}
    for thr in (100, 60):
        y = (W >= thr).astype(int)
        for fn, XX in FS.items():
            for mt in ("GB", "LR"):
                rows = []
                for tgt in pairs:
                    te = (M == tgt[0]) & (T == tgt[1])
                    if y[te].min() == y[te].max(): continue
                    cand = [p for p in pairs if p[0] != tgt[0]]; d = {p: float(np.linalg.norm(prox[p] - prox[tgt])) for p in cand}; order = sorted(cand, key=d.get)
                    r = {"target": [int(tgt[0]), int(tgt[1])], "dist_nearest": d[order[0]], "dist_mean": float(np.mean(list(d.values())))}
                    for name, sel in (("all", cand), ("near3", order[:3]), ("far3", order[-3:])):
                        tr = np.array([(m, t) in set(sel) for m, t in zip(M, T)])
                        sc = StandardScaler().fit(XX[tr]); model = HistGradientBoostingClassifier(max_iter=150, random_state=0) if mt == "GB" else LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")
                        r[name] = float(roc_auc_score(y[te], model.fit(sc.transform(XX[tr]), y[tr]).predict_proba(sc.transform(XX[te]))[:, 1]))
                    rows.append(r)
                rho = spearmanr([r["dist_mean"] for r in rows], [r["all"] for r in rows])
                out[f"thr{thr}|{fn}|{mt}"] = {"rows": rows, "spearman_dist_vs_auroc_all": [float(rho.statistic), float(rho.pvalue)]}
                print(f"VB>={thr:3d} {fn:13s} {mt}: n={len(rows)} AUROC all={np.mean([r['all'] for r in rows]):.3f} near3={np.mean([r['near3'] for r in rows]):.3f} far3={np.mean([r['far3'] for r in rows]):.3f} | Spearman(khoảng cách, AUROC)={rho.statistic:+.2f} (p={rho.pvalue:.2f})  AUROC thấp nhất={min(r['all'] for r in rows):.3f}")
    json.dump(out, open("output/luh_dynamics_shift.json", "w"), indent=1)


if __name__ == "__main__":
    main()
