"""Hình cho thí nghiệm học liên tục: (a) giữ lại các máy đã học theo độ dài chuỗi; (b) chuyển sang máy mới theo số ca xác nhận k (L=6)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); r = json.load(open(os.path.join(HERE, "report", "continual_results.json")))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4)); Ls = sorted(int(k) for k in r["by_len"])
ARMS = [("static_base", "static base", "#6b6b6b", "--"), ("naive_ema", "naive overwrite (EMA)", "#c0392b", "-"), ("anchored_ema_g0.02", "anchored to base", "#d98c1f", "-"), ("hierarchical_overlay", "base + per-machine overlay", "#1b6ca8", "-"), ("pseudo_label_ema", "overwrite with pseudo-labels", "#8a5fbf", ":"), ("specialist_ceiling", "specialist (ceiling)", "#3a9d8f", "--")]
for key, lab, c, ls in ARMS: ax[0].errorbar(Ls, [r["by_len"][str(L)]["retention"][key][0] for L in Ls], yerr=[r["by_len"][str(L)]["retention"][key][1] for L in Ls], marker="o", ms=4, color=c, ls=ls, label=lab, capsize=2)
ax[0].set_xlabel("machines learned in sequence L"); ax[0].set_ylabel("top-1 on the machines already learned"); ax[0].set_xticks(Ls); ax[0].set_title("Retention", fontsize=9); ax[0].axhline(0.2, color="k", ls=":", lw=0.8); ax[0].legend(fontsize=6.5, frameon=False, loc="lower left")
F = r["by_len"]["6"]["forward"]; ks = [int(k) for k in F]
FW = [("static_base", "static base", "#6b6b6b", "--"), ("naive_ema", "naive overwrite", "#c0392b", "-"), ("anchored_ema_g0.02", "anchored to base", "#d98c1f", "-"), ("hierarchical", "base + overlay (hierarchical)", "#1b6ca8", "-"), ("adapt_from_base", "adapt from base (no history)", "#3a9d8f", "-"), ("pseudo_label_ema", "pseudo-labels", "#8a5fbf", ":"), ("specialist_ceiling", "specialist (ceiling)", "#999999", "--")]
for key, lab, c, ls in FW: ax[1].errorbar(ks, [F[str(k)][key][0] for k in ks], yerr=[F[str(k)][key][1] for k in ks], marker="o", ms=4, color=c, ls=ls, label=lab, capsize=2)
ax[1].set_xlabel("confirmed cases per cause on the NEW machine, k"); ax[1].set_ylabel("top-1 on the new machine"); ax[1].set_title("Transfer to a new machine (L = 6 learned)", fontsize=9); ax[1].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "report", "continual.png"), dpi=150, bbox_inches="tight"); print("ok")
for L in (3, 6):
    d = r["by_len"][str(L)]; print(L, "hier vs adapt_from_base k=10:", [round(x, 3) for x in d["forward"]["10"]["hierarchical"]], [round(x, 3) for x in d["forward"]["10"]["adapt_from_base"]], "naive", [round(x, 3) for x in d["forward"]["10"]["naive_ema"]])
    print(L, "retention naive", [round(x, 3) for x in d["retention"]["naive_ema"]], "overlay", [round(x, 3) for x in d["retention"]["hierarchical_overlay"]])
