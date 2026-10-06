import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
OUT = "output"; d = np.load(f"{OUT}/real_windows.npz"); meta = d["meta"]; fs_a = float(d["fs_a"]); sr = float(d["sr"])
sel = [np.where((meta[:, 3] == 0) & (meta[:, 5] == 9000))[0][0], np.where((meta[:, 3] == 0) & (meta[:, 5] == 12000))[0][0],
       np.where((meta[:, 3] == 1) & (meta[:, 5] == 9000))[0][0], np.where((meta[:, 3] == 1) & (meta[:, 5] >= 10000))[0][0]]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for i in sel:
    lab = f"N={int(meta[i,5])} {'UP' if meta[i,4] else 'DOWN'} label{int(meta[i,3])}"
    f, P = welch(d["acc0"][i] - d["acc0"][i].mean(), fs_a, nperseg=2048); ax[0].loglog(f[1:], P[1:], label=lab, lw=0.8)
    f, P = welch(d["mic"][i] - d["mic"][i].mean(), sr, nperseg=2048); ax[1].loglog(f[1:], P[1:], label=lab, lw=0.8)
ax[0].set_title("accelerometer acc0 PSD [g²/Hz]"); ax[1].set_title("microphone PSD [FS²/Hz]")
for a in ax: a.set_xlabel("Hz"); a.legend(fontsize=7); a.grid(True, which="both", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/real_psd_examples.png", dpi=120)
