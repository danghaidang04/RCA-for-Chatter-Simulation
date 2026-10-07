"""Đỉnh cộng hưởng + độ cản thô (độ rộng nửa công suất) từ phổ trung bình của cửa sổ KHỎE; tách mode chung (xuất hiện ở cả 2 dao trên cùng máy) và riêng từng dao."""
import json
import numpy as np
from scipy.signal import welch, find_peaks, peak_widths
OUT = "output"
def avg_psd(npz, sel):
    d = np.load(f"{OUT}/{npz}"); meta = d["meta"]; fs = float(d["fs_a"]); idx = np.where(sel(meta))[0]
    P = 0
    for i in idx:
        f, p = welch(d["acc0"][i] - d["acc0"][i].mean(), fs, nperseg=8192, noverlap=4096); P = P + p
    return f, P / len(idx), len(idx)
def modes(f, P, lo=700, hi=4000):
    k = (f >= lo) & (f <= hi); Ps = np.convolve(P[k], np.ones(3) / 3, mode="same"); ff = f[k]; df = ff[1] - ff[0]
    pk, pr = find_peaks(Ps, prominence=0.15 * np.median(Ps) + 0.3 * np.std(Ps), distance=int(25 / df))
    wd = peak_widths(Ps, pk, rel_height=0.5)[0] * df
    out = [(float(ff[p]), float(w), float(w / (2 * ff[p])), float(pr_)) for p, w, pr_ in zip(pk, wd, pr["prominences"])]
    return sorted(out, key=lambda x: -x[3])[:6]
res = {}
for name, npz, sel in (("A_khoe (ds5-6)", "real_windows.npz", lambda m: m[:, 3] == 0), ("D_khoe (ds14)", "real_windows_defect.npz", lambda m: m[:, 0] == 14), ("D*_hong (ds3)", "real_windows_defect.npz", lambda m: m[:, 0] == 3)):
    f, P, n = avg_psd(npz, sel); M = modes(f, P); res[name] = M
    print(f"{name}: {n} cửa sổ"); [print(f"   fn≈{a:6.0f} Hz | rộng nửa công suất {b:5.1f} Hz | ζ thô≈{c:.3f}") for a, b, c, _ in M]
def match(X, Y, tol=0.03): return [(x[0], y[0]) for x in X for y in Y if abs(x[0] - y[0]) / x[0] < tol]
print("\nđỉnh chung giữa dao A và dao D (cùng máy imi_vm20i, dung sai 3%):", [(round(a), round(b)) for a, b in match(res["A_khoe (ds5-6)"], res["D_khoe (ds14)"])])
print("đỉnh chung giữa D khỏe và D* hỏng:", [(round(a), round(b)) for a, b in match(res["D_khoe (ds14)"], res["D*_hong (ds3)"])])
json.dump(res, open(f"{OUT}/quick_modes.json", "w"), indent=1)
