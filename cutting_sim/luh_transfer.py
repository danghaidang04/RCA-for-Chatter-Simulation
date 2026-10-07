"""THÍ NGHIỆM THẬT (bộ LUH, Sci. Data; 3 máy x 3 dao, lực 25 kHz + tín hiệu máy 500 Hz): đoán 'dao mòn' (VB>=100 µm) và chuyển giao giữa MÁY.
 E1 dao mới trên cùng máy (leave-one-tool-out)  E2 máy chưa thấy (leave-one-machine-out, huấn luyện 2 máy)  E3 huấn luyện chỉ 1 máy -> máy khác
 Đặc trưng: THÔ vs TƯƠNG ĐỐI (chuẩn hóa theo các lần chạy đầu đời của chính máy / chính dao: dữ liệu khỏe, không nhãn mòn)."""
import glob, json, os
import h5py, numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool

H5 = "../data/luh_raw/h5"; BANDS = [0, 25, 100, 300, 1000, 3000, 8000, 12500]
MKEYS = ['torque_spindle', 'torque_axis_x', 'torque_axis_y', 'torque_axis_z', 'position_control_deviation_axis_x', 'position_control_deviation_axis_y']


def feats(path):
    with h5py.File(path) as f:
        lab = {k: float(f["labels"][k][0, 0]) for k in ("machine", "tool", "run", "wear", "cumulated_tool_contact_time")}
        F = [f["signals_sensor"][f"force_sensor_{a}"][0].astype(np.float64) for a in "xyz"]; M = {k: (f["signals_machine"][k][0].astype(np.float64) if k in f["signals_machine"] else None) for k in MKEYS}
    out = []
    for x in F:
        d = x - x.mean(); fq, P = welch(d, 25000.0, nperseg=4096)
        out += [x.mean(), d.std(), kurtosis(d)] + [np.log10(P[(fq >= BANDS[i]) & (fq < BANDS[i + 1])].sum() + 1e-12) for i in range(len(BANDS) - 1)]
    R = np.sqrt(sum((x - x.mean()) ** 2 for x in F)); out += [R.mean(), R.std(), np.log10(R.var() + 1e-12)]
    for k in MKEYS: out += [M[k].mean(), M[k].std()] if M[k] is not None else [np.nan, np.nan]
    return lab, np.array(out)


def load():
    files = sorted(glob.glob(f"{H5}/*.h5")); cache = "output/luh_features.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True); return pd.DataFrame(d["lab"], columns=d["cols"]), d["X"]
    with Pool(10) as p: res = p.map(feats, files)
    lab = pd.DataFrame([r[0] for r in res]); X = np.vstack([r[1] for r in res]); keep = ~np.isnan(X).any(0); print("bỏ", int((~keep).sum()), "đặc trưng không có ở mọi máy"); X = X[:, keep]; np.savez(cache, lab=lab.values, cols=np.array(lab.columns), X=X); return lab, X


def rel(X, lab, level):
    """Chuẩn hóa theo nền khỏe (VB<=10 µm) của chính máy ('machine') hoặc chính dao ('tool')."""
    Z = np.zeros_like(X); keys = ["machine"] if level == "machine" else ["machine", "tool"]
    for _, g in lab.groupby(keys):
        idx = g.index.values; base = idx[lab.wear.values[idx] <= 10]
        if len(base) < 3: base = idx[np.argsort(lab.wear.values[idx])[:3]]      # thiếu lần chạy đầu đời: dùng 3 lần mòn ít nhất của nhóm (dao 8 máy 3 bắt đầu ở VB 34 µm)
        mu, sd = X[base].mean(0), X[base].std(0) + 1e-6 * (np.abs(X[base]).mean(0) + 1); Z[idx] = (X[idx] - mu) / sd
    return Z


def evaluate(X, lab, tr_mask, te_mask, seed=0):
    y = (lab.wear.values >= 100).astype(int); sc = StandardScaler().fit(X[tr_mask]); A, B = sc.transform(X[tr_mask]), sc.transform(X[te_mask]); out = {}
    for n, m in (("GB", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed)), ("LR", LogisticRegression(C=0.3, max_iter=3000))):
        out[n] = roc_auc_score(y[te_mask], m.fit(A, y[tr_mask]).predict_proba(B)[:, 1])
    r = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=seed).fit(A, lab.wear.values[tr_mask]); out["MAE_VB"] = mean_absolute_error(lab.wear.values[te_mask], r.predict(B))
    return out


def main():
    lab, X = load(); print("đặc trưng", X.shape, "| máy", lab.machine.value_counts().to_dict(), "| tỉ lệ mòn(VB>=100): %.2f" % (lab.wear >= 100).mean())
    feats_ = {"raw": X, "machine-relative": rel(X, lab, "machine"), "tool-relative": rel(X, lab, "tool")}; M, T = lab.machine.values, lab.tool.values; res = {}
    for fn, XX in feats_.items():
        e1 = [evaluate(XX, lab, (M == m) & (T != t), (M == m) & (T == t)) for m in (1, 2, 3) for t in sorted(set(T[M == m]))]
        e2 = [evaluate(XX, lab, M != m, M == m) for m in (1, 2, 3)]
        e3 = [evaluate(XX, lab, M == a, M == b) for a in (1, 2, 3) for b in (1, 2, 3) if a != b]
        res[fn] = {k: {m: float(np.mean([r[m] for r in v])) for m in ("GB", "LR", "MAE_VB")} for k, v in (("E1 new tool, same machine", e1), ("E2 unseen machine, trained on 2 machines", e2), ("E3 unseen machine, trained on 1 machine", e3))}
        res[fn]["_E2_per_machine_GB_AUROC"] = [round(r["GB"], 3) for r in e2]
    print("\nAUROC (VB>=100) | MAE VB [µm]   (đoán bừa AUROC=0.5)")
    print("đặc trưng".ljust(18), "kịch bản".ljust(44), "GB AUROC", "LR AUROC", "GB MAE_VB")
    for fn, v in res.items():
        for k, r in v.items():
            if k.startswith("_"): continue
            print(fn.ljust(18), k.ljust(44), f"{r['GB']:.3f}".rjust(8), f"{r['LR']:.3f}".rjust(8), f"{r['MAE_VB']:.1f}".rjust(9))
    print("E2 theo máy bị loại (GB AUROC):", {fn: v["_E2_per_machine_GB_AUROC"] for fn, v in res.items()})
    json.dump(res, open("output/luh_transfer.json", "w"), indent=1)


if __name__ == "__main__":
    main()
