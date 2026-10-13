"""Vẽ loss (MMD²) theo thế hệ CMA-ES từ output/calibration.log (best trong thế hệ + best tích lũy)."""
import re, sys
from common import *

log = open(f"{OUT}/calibration.log").read()
fig, ax = plt.subplots(figsize=(8, 4))
colors = {"A machine": "tab:blue", "C sigma": "tab:green", "B anomaly": "tab:red"}
off = 0
for stage in ("A machine", "C sigma", "B anomaly"):
    L = 0
    for sd in (1, 2, 3):
        v = [float(m.group(2)) for m in re.finditer(rf"\[{stage}\] s{sd} (\d+) best=([\d.]+)", log)]
        if not v:
            continue
        x = np.arange(off + 1, off + len(v) + 1)
        ax.plot(x, np.minimum.accumulate(v), "-", lw=1.8, color=colors[stage], alpha=0.5 + 0.2 * (sd == 1),
                label=f"{stage} (best so far, seeds 1-3)" if sd == 1 else None)
        ax.plot(x, v, ".", ms=2, color=colors[stage], alpha=0.3)
        off += len(v); L += len(v)
    if L: ax.axvline(off + 0.5, color="k", lw=0.5, ls=":")
ax.set_xlabel("generation (cumulative across stages)"); ax.set_ylabel("MMD² loss"); ax.legend(fontsize=7)
ax.set_title("CMA-ES calibration loss vs. real MSM data")
fig.tight_layout(); fig.savefig(f"{OUT}/calibration_loss_curve.png", dpi=130); print("saved", len(v), "gens last stage")
