"""Mức đáy của C2ST: so hai lần cắt THẬT cùng điều kiện (N, ae, F, hướng) nhưng khác dataset (5 vs 6), cùng quy trình C2ST (balanced accuracy, tách fold theo đường cắt)."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
OUT = "output"
df = pd.read_csv(f"{OUT}/real_features.csv"); df = df[df.label == 0].copy()
df["key"] = df.N.astype(int).astype(str) + "_" + df.ae.astype(str) + "_" + df.F.round(1).astype(str) + "_" + df.direction
a = df[df.dataset == 5]; b = df[df.dataset == 6]
common = sorted(set(a.key) & set(b.key)); print("điều kiện (label 0) có ở cả dataset 5 và 6:", len(common), "| số cửa sổ:", int(a.key.isin(common).sum()), "(ds5),", int(b.key.isin(common).sum()), "(ds6)")
if common:
    A = a[a.key.isin(common)]; B = b[b.key.isin(common)]
    F = ["acc_rms_g", "mic_db_fs", "chatter_acc", "chatter_mic"]
    def mat(d): return np.c_[np.log10(d.acc_rms_g), d.mic_db_fs, d.chatter_acc, d.chatter_mic, np.log(d.N), np.log(d.ae), np.log(d.F)]
    X = np.vstack([mat(A), mat(B)]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]; g = np.r_[(A.dataset * 1000 + A.cut).values, (B.dataset * 1000 + B.cut).values]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); accs = []
    for s in range(5):
        for gk in (GroupKFold(min(5, len(np.unique(g)))),):
            clf = RandomForestClassifier(200, min_samples_leaf=3, random_state=s, n_jobs=4, class_weight="balanced")
            accs.append(cross_val_score(clf, X, y, groups=g, cv=gk, scoring="balanced_accuracy").mean())
    print(f"C2ST giữa 2 lần cắt thật cùng điều kiện (ds5 vs ds6): balanced acc = {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print("trung bình đặc trưng ds5 vs ds6 (cùng điều kiện):")
    for j, n in enumerate(["log10 acc", "mic dB", "chat_acc", "chat_mic"]):
        m = mat(A)[:, j], mat(B)[:, j]; print(f"   {n:10s} ds5 {m[0].mean():7.3f}±{m[0].std():.3f} | ds6 {m[1].mean():7.3f}±{m[1].std():.3f}")
