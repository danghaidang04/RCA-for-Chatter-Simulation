"""Kiểm tra nhanh 2 ràng buộc rút từ dữ liệu thật (không dùng nguồn ngoài):
 (1) Năng lượng cắt riêng u = (công suất cắt - công suất không tải) / tốc độ bóc vật liệu  [W/(mm^3/s) = N/mm^2] từ mtc.csv (watt)
 (2) Tần số cộng hưởng và độ cản thô từ phổ khỏe, đối chiếu công thức dầm công-xôn (E, rho của cacbua: GIẢ ĐỊNH CHƯA XÁC MINH)."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.signal import welch, find_peaks, medfilt
ROOT = "../data/msm/imi_vm20i"; OUT = "output"; res = {}

print("=== (1) năng lượng cắt riêng từ công suất ===")
for ds in (5, 6, 3, 14, 28):
    lab = pd.read_csv(f"{ROOT}/dataset{ds}/label.csv"); m = pd.read_csv(glob.glob(f"{ROOT}/dataset{ds}/*_mtc.csv")[0], encoding="utf-8-sig")
    t = m.time.values; w = m.watt.values; sp = m.Spindle_Speed.values
    incut = np.zeros(len(t), bool)
    for r in lab.itertuples(): incut |= (t >= r.start - 0.3) & (t <= r.end + 0.3)
    rows = []
    for r in lab.itertuples():
        d = r.end - r.start; sel = (t >= r.start + 0.15 * d) & (t <= r.end - 0.15 * d)
        idle = (~incut) & (np.abs(sp - r.N) < 0.02 * r.N)
        if sel.sum() < 3 or idle.sum() < 5: continue
        dP = w[sel].mean() - np.median(w[idle]); mrr = r.ap * r.ae * r.F
        rows.append((r.N, r.ap, r.ae, r.F, r.direction, r.label, dP, mrr, dP / mrr))
    d_ = pd.DataFrame(rows, columns=["N", "ap", "ae", "F", "dir", "label", "dP", "mrr", "u"])
    if len(d_) < 5: print(f"dataset {ds}: không đủ cửa sổ không tải"); continue
    A = np.c_[np.ones(len(d_)), d_.mrr]; coef, *_ = np.linalg.lstsq(A, d_.dP, rcond=None); pred = A @ coef; r2 = 1 - ((d_.dP - pred) ** 2).sum() / ((d_.dP - d_.dP.mean()) ** 2).sum()
    tool = f"{lab.Tool.iloc[0]}/{lab.Workpiece.iloc[0]}"
    print(f"dataset {ds:2d} ({tool:14s}) {len(d_):2d} cắt | ΔP trung vị {d_.dP.median():7.0f} W | MRR trung vị {d_.mrr.median():7.1f} mm3/s | u=ΔP/MRR trung vị {d_.u.median():6.2f} N/mm2 [Q1 {d_.u.quantile(.25):.2f}, Q3 {d_.u.quantile(.75):.2f}] | hồi quy ΔP=a+u·MRR: u={coef[1]:.2f}, a={coef[0]:.0f} W, R2={r2:.2f}")
    res[f"ds{ds}"] = {"tool": tool, "n": len(d_), "u_median": float(d_.u.median()), "u_slope": float(coef[1]), "R2": float(r2)}

print("\n=== (2) cộng hưởng từ phổ khỏe vs công thức dầm ===")
def spec(npz, sel_fn):
    d = np.load(f"{OUT}/{npz}"); meta = d["meta"]; fs = float(d["fs_a"]); idx = np.where(sel_fn(meta))[0]
    P = np.mean([welch(d["acc0"][i] - d["acc0"][i].mean(), fs, nperseg=4096, noverlap=2048)[1] for i in idx], axis=0); f = welch(d["acc0"][idx[0]], fs, nperseg=4096, noverlap=2048)[0]
    return f, P, len(idx)
E, RHO = 600e9, 14.5e3          # GIẢ ĐỊNH chưa xác minh cho cacbua
for name, npz, fn, dmm in (("tool A khỏe (ds5-6, label 0)", "real_windows.npz", lambda m: m[:, 3] == 0, 6.35), ("tool D khỏe (ds14)", "real_windows_defect.npz", lambda m: m[:, 0] == 14, 12.7), ("tool D* hỏng (ds3)", "real_windows_defect.npz", lambda m: m[:, 0] == 3, 12.7)):
    f, P, n = spec(npz, fn); band = (f >= 600) & (f <= 5000); Ps = np.convolve(P, np.ones(5) / 5, mode="same")
    pk, pr = find_peaks(np.log10(Ps[band]), prominence=0.25, distance=8); fb = f[band]
    top = sorted(zip(pr["prominences"], fb[pk]), reverse=True)[:4]
    print(f"{name}: {n} cửa sổ | đỉnh nổi bật (Hz): {[f'{x[1]:.0f}' for x in top]}")
    d = dmm * 1e-3; c = np.sqrt(E / RHO) * d / 4
    for fr, _ in [(x[1], 0) for x in top[:2]]:
        L = np.sqrt(0.5596 * c / fr); print(f"    nếu f1={fr:.0f} Hz thì chiều dài nhô L≈{L*1e3:.0f} mm (dầm công-xôn, E={E/1e9:.0f} GPa, ρ={RHO:.0f}; GIẢ ĐỊNH) | hợp lý nếu trong [LOC, OAL]")
json.dump(res, open(f"{OUT}/quick_constraints.json", "w"), indent=1)
