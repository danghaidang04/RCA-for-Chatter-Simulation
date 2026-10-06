"""RCA (miền MSM: acc + mic) trên mô phỏng v2 với nguyên nhân gốc lấy từ MÔ HÌNH ĐÃ HỌC (không còn mức small/medium/large tay đặt).
Tiến triển dần (TOOL_WEAR, STIFFNESS): tuổi dao s ~ U(0,1). Sai lệch cài đặt (DAMPING, DEPTH, RPM): độ lệch cố định đã học.
So sánh mô phỏng chưa hiệu chỉnh (máy mặc định) và đã hiệu chỉnh (máy + σ học bằng CMA-ES).
Bộ chẩn đoán: posterior Bayes P(cause|x)∝P(x|cause)P(cause), P(x|cause) Gaussian (QDA), tiên nghiệm đều; train/test dùng seed độc lập.
Chạy: python exp5_rca_realistic.py
"""
import json
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from calibrate_v2 import *

CAUSES6 = ["NORMAL"] + FAULTS
REPS = 4


def gen(cuts, th, sig, cause, thB, seed, pool):
    rng = np.random.default_rng(seed)
    ages = rng.uniform(0, 1, len(cuts))
    faults = None if cause == "NORMAL" else [effects(cause, thB, float(s)) for s in ages]
    return sim_v2(cuts, th, faults, seed, pool, sig)


def run_domain(name, th, sig, cuts, thB, pool):
    X, y = [], []
    for k, c in enumerate(CAUSES6):
        for rep in range(REPS):
            F = gen(cuts, th, sig, c, thB, 10 + rep, pool); X.append(F); y += [k] * len(F)
    X = np.vstack(X); y = np.array(y); mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = QDA(reg_param=0.1, priors=np.ones(6) / 6).fit((X - mu) / sd, y)
    res = {}; conf = np.zeros((6, 6))
    for k, c in enumerate(CAUSES6):
        F = gen(cuts, th, sig, c, thB, 500, pool); P = clf.predict_proba((F - mu) / sd)
        for pr in P.argmax(1): conf[k, pr] += 1
        res[c] = {"top1": float(np.mean(P.argmax(1) == k)), "top3": float(np.mean([k in np.argsort(-p)[:3] for p in P]))}
        print(f"[{name}] {c:15s} top1={res[c]['top1']:.2f} top3={res[c]['top3']:.2f}", flush=True)
    res["_confusion_rows_true_cols_pred"] = (conf / conf.sum(1, keepdims=True)).tolist()
    res["_macro_top1"] = float(np.mean([v["top1"] for kk, v in res.items() if not kk.startswith("_")]))
    return res


def main():
    cal = json.load(open(f"{OUT}/calibration_v2.json"))
    thA = np.array(list(cal["machine"].values())); sig = np.array(cal["sigma"]); thB = np.array(cal["faults_theta"])
    df = pd.read_csv(f"{OUT}/real_features.csv"); cuts = cuts_of(df); out = {}
    with Pool(10) as pool:
        out["uncalibrated"] = run_domain("uncalibrated", X0_M, np.zeros(7), cuts, thB, pool)
        out["calibrated"] = run_domain("calibrated", thA, sig, cuts, thB, pool)
    json.dump(out, open(f"{OUT}/exp5_rca_realistic.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); w = 0.38; x = np.arange(6)
    ax[0].bar(x - w / 2, [out["uncalibrated"][n]["top1"] for n in CAUSES6], w, label="uncalibrated sim")
    ax[0].bar(x + w / 2, [out["calibrated"][n]["top1"] for n in CAUSES6], w, label="calibrated sim (CMA-ES)")
    ax[0].axhline(1 / 6, color="k", ls=":", lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(CAUSES6, rotation=30, ha="right", fontsize=8); ax[0].set_ylabel("Top-1"); ax[0].legend(fontsize=8)
    C = np.array(out["calibrated"]["_confusion_rows_true_cols_pred"]); im = ax[1].imshow(C, vmin=0, vmax=1, cmap="Blues")
    ax[1].set_xticks(range(6)); ax[1].set_xticklabels(CAUSES6, rotation=40, ha="right", fontsize=7); ax[1].set_yticks(range(6)); ax[1].set_yticklabels(CAUSES6, fontsize=7)
    for i in range(6):
        for j in range(6): ax[1].text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax[1].set_title("confusion (calibrated): rows = true"); ax[1].grid(False); fig.tight_layout(); fig.savefig(f"{OUT}/exp5_rca_realistic.png", dpi=130)


if __name__ == "__main__":
    main()
