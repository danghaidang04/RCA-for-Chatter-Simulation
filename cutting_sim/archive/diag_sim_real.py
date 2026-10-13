import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sim_to_real_wear import *
Xr, lab = real_features(); wear, grp = lab.wear.values, (lab.machine.values * 10 + lab.tool.values).astype(int); y = (wear >= 100).astype(int); Zr = rel_by_tool(Xr, wear, grp)
d = np.load("output/sim_wear_features.npz"); Zs = rel_by_tool(d["X"], d["vb"], d["g"]); ys = (d["vb"] >= 100).astype(int); ok = np.isfinite(Zs).all(1) & (np.abs(Zs).max(1) < 1e4)
names = [f"{a}_{n}" for a in "xy" for n in ["mean", "std", "kurt"] + [f"band{i}" for i in range(7)]] + ["res_mean", "res_std", "res_logvar"]
print("đặc trưng (tương đối): AUROC đơn lẻ cho 'mòn' — mô phỏng vs thật (0.5=không phân biệt; <0.5 = ngược chiều)")
print(f"{'đặc trưng':12s} {'sim':>6s} {'thật':>6s}  {'cùng chiều?':>12s}")
for j, n in enumerate(names):
    a = roc_auc_score(ys[ok], Zs[ok, j]); b = roc_auc_score(y, Zr[:, j]); flag = "" if (a - 0.5) * (b - 0.5) > 0 else "<-- NGƯỢC"
    print(f"{n:12s} {a:6.2f} {b:6.2f}  {flag:>12s}")
