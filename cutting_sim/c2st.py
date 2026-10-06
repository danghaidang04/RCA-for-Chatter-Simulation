"""C2ST (Lopez-Paz & Oquab 2017) cho mô hình v2: bộ phân loại thật/mô phỏng, tách fold theo ĐƯỜNG CẮT.
Độ chính xác ≈ 0.5 <=> không phân biệt được. Dùng RandomForest trên [đặc trưng, điều kiện cắt] (điều kiện đưa vào để kiểm cả quan hệ phụ thuộc điều kiện).
Chạy: python c2st.py
"""
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from calibrate_v2 import *

R = 5


def c2st(real, sim, creal, grp, seed=0):
    X = np.vstack([np.c_[real, creal], np.c_[sim, np.tile(creal, (R, 1))]]); y = np.r_[np.zeros(len(real)), np.ones(len(sim))]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    gg = np.r_[grp, np.tile(grp, R)]                      # nhóm theo ĐƯỜNG CẮT: các cửa sổ cùng một đường cắt luôn cùng fold
    accs = []
    for s in range(3):
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=seed + s, n_jobs=1, class_weight="balanced")
        accs.append(cross_val_score(clf, (X - mu) / sd, y, groups=gg, cv=GroupKFold(n_splits=5), scoring="balanced_accuracy").mean())
    return float(np.mean(accs)), float(np.std(accs))


def main():
    cal = json.load(open(f"{OUT}/calibration_v2.json"))
    thA = np.array(list(cal["machine"].values())); sig = np.array(cal["sigma"]); thB = np.array(cal["faults_theta"])
    df = pd.read_csv(f"{OUT}/real_features.csv"); df["age"] = tool_age(df); out = {}
    with Pool(10) as pool:
        for label, cls in ((0, "normal"), (1, "anomaly")):
            d = df[df.label == label].reset_index(drop=True); cuts = cuts_of(d); real = real_matrix(d); cr = cond_matrix(cuts)
            variants = [("uncalibrated", X0_M, np.zeros(7), False), ("machine-fit", thA, np.zeros(7), False), ("machine+σ", thA, sig, False)]
            if label == 1:
                variants.append(("machine+σ+fault model", thA, sig, True))
            for name, th, s, use_f in variants:
                S = np.vstack([sim_v2(cuts, th, fault_list(thB, d.age.values, np.random.default_rng(900 + r)) if use_f else None, 900 + r, pool, s) for r in range(R)])
                acc, sd_ = c2st(real, S, cr, (d.dataset * 1000 + d.cut).values)
                out[f"{cls}/{name}"] = {"c2st_acc": acc, "std": sd_}
                print(f"{cls:8s} {name:24s} C2ST acc = {acc:.3f} ± {sd_:.3f}", flush=True)
    json.dump(out, open(f"{OUT}/c2st_v2.json", "w"), indent=1)
    keys = list(out); fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(keys))
    ax.bar(x, [out[k]["c2st_acc"] for k in keys], yerr=[out[k]["std"] for k in keys], color=["tab:blue" if k.startswith("normal") else "tab:red" for k in keys])
    ax.axhline(0.5, color="k", ls=":"); ax.set_xticks(x); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=7); ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("C2ST accuracy (0.5 = indistinguishable)"); fig.tight_layout(); fig.savefig(f"{OUT}/c2st_v2.png", dpi=130)


if __name__ == "__main__":
    main()
