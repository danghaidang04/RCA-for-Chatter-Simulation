"""Sinh toàn bộ hình của paper từ JSON trong cutting_sim/output (và mô phỏng trực tiếp cho hình dạng sóng). Chạy: python make_figures.py"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
O = "../cutting_sim/output"; F = "figures"; os.makedirs(F, exist_ok=True)
sys.path.insert(0, "../cutting_sim"); sys.path.insert(0, "..")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})
AL = ["BRCD", "RCD", "RCG", "SmoothTraversal", "BARO", "SimpleRCA"]; COL = dict(zip(AL, ["#1b6ca8", "#8a5fbf", "#3a9d8f", "#d98c1f", "#c0392b", "#6b6b6b"]))
SENS = ["log10_acc", "mic_dB", "chatter_acc", "chatter_mic", "acc_low", "acc_res", "acc_high", "mic_low", "mic_high"]
SL = ["log acc rms", "mic level", "non-sync acc", "non-sync mic", "acc 50-600 Hz", "acc 1-3.2 kHz", "acc >3.2 kHz", "mic low", "mic high"]


def fig_teaser():
    d = json.load(open(f"{O}/rca_causal.json"))["results"]; ks = list(d)
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); w = 0.38; x = np.arange(len(AL))
    for j, (k, lab, c) in enumerate(zip(ks, ["Same generator", "Shifted generator parameters"], ["#1b6ca8", "#c0392b"])):
        v = [np.mean([d[k][a]["10"][f][0] for f in d[k][a]["10"]]) for a in AL]; ax.bar(x + (j - 0.5) * w, v, w, label=lab, color=c)
        for xi, vi in zip(x, v): ax.text(xi + (j - 0.5) * w, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=7)
    ax.axhline(1 / 14, color="k", ls=":", lw=1); ax.text(5.6, 1 / 14 + 0.015, "chance", fontsize=7, ha="right"); ax.set_xticks(x); ax.set_xticklabels(["BRCD", "RCD", "RCG", "Smooth\nTraversal", "BARO", "SimpleRCA"]); ax.set_ylabel("Top-1 accuracy (m=10)"); ax.set_ylim(0, 1.08); ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.savefig(f"{F}/f_teaser.png")


def fig_heat():
    d = json.load(open(f"{O}/rca_causal.json"))["results"]; fl = ["STIFFNESS", "DAMPING", "TOOL_WEAR", "DEPTH_OVERLOAD", "RPM_MISMATCH"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    for a, k, t in zip(axs, list(d), ["Same generator", "Shifted generator"]):
        M = np.array([[d[k][al]["10"][f][0] for f in fl] for al in AL]); im = a.imshow(M, vmin=0, vmax=1, cmap="viridis"); a.set_title(t, fontsize=9)
        a.set_xticks(range(5)); a.set_xticklabels(["stiffness", "damping", "wear", "depth", "rpm"], rotation=30, ha="right"); a.set_yticks(range(6)); a.set_yticklabels(AL); a.grid(False)
        for i in range(6):
            for j in range(5): a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=axs, shrink=0.8, label="Top-1 (m=10)"); fig.savefig(f"{F}/f_cause_heatmap.png")


def fig_ablation():
    ab = json.load(open(f"{O}/rca_ablation.json")); V = [("soft|physics|x1.0", "physics graph"), ("soft|shuffled|x1.0", "shuffled graph"), ("soft|none|x1.0", "no edges"), ("physical|physics|x1.0", "label-free nodes")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True); w = 0.2
    for a, sc, t in zip(axs, ["in", "shift"], ["Same generator", "Shifted generator"]):
        for j, (k, lab) in enumerate(V):
            m = [np.mean(ab[f"{k}|{sc}|m10"][al]) for al in AL]; s = [np.std(ab[f"{k}|{sc}|m10"][al]) for al in AL]
            a.bar(np.arange(6) + (j - 1.5) * w, m, w, yerr=s, capsize=1.5, label=lab, color=["#1b6ca8", "#7fb3d5", "#b9d3e6", "#e0a030"][j])
        a.axhline(1 / 14, color="k", ls=":", lw=1); a.set_xticks(range(6)); a.set_xticklabels(["BRCD", "RCD", "RCG", "Smooth", "BARO", "Simple"]); a.set_title(t, fontsize=9)
    axs[0].set_ylabel("Top-1 (m=10, 3 seeds)"); axs[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left"); fig.savefig(f"{F}/f_ablation.png")


def fig_m_curve():
    d = json.load(open(f"{O}/rca_causal.json"))["results"]; ms = ["5", "10", "20", "50"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    for a, k, t in zip(axs, list(d), ["Same generator", "Shifted generator"]):
        for al in AL: a.plot([int(m) for m in ms], [np.mean([d[k][al][m][f][0] for f in d[k][al][m]]) for m in ms], "o-", ms=3, label=al, color=COL[al])
        a.axhline(1 / 14, color="k", ls=":", lw=1); a.set_xscale("log"); a.set_xticks([5, 10, 20, 50]); a.set_xticklabels(["5", "10", "20", "50"]); a.set_xlabel("anomalous samples m"); a.set_title(t, fontsize=9)
    axs[0].set_ylabel("Top-1"); axs[1].legend(frameon=False, fontsize=7, ncol=2); fig.savefig(f"{F}/f_m_curve.png")


def fig_c2st():
    names = [(0, "regression mean only", "Linear\nmean only"), (0, "hierarchical: between-cut + within-cut", "Linear\n+ hierarchical"), (1, "regression mean only", "Kernel ridge\nmean only"), (1, "hierarchical: between-cut + within-cut", "Kernel ridge\n+ hierarchical\n(final)")]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, (nl, k, lab) in enumerate(names):
        v = [json.load(open(f"{O}/noise_model_gate_nonlin{nl}_seed{s}.json"))["c2st_heldout"][k][0] for s in range(3)]; ax.bar(i, np.mean(v), yerr=np.std(v), capsize=3, color="#1b6ca8" if i == 3 else "#9db8d2"); ax.scatter([i] * 3, v, color="k", s=8, zorder=3)
    ax.axhline(0.5, color="k", ls="--", lw=1); ax.text(3.45, 0.505, "indistinguishable", fontsize=7, ha="right"); ax.set_xticks(range(4)); ax.set_xticklabels([n[2] for n in names], fontsize=7); ax.set_ylim(0.4, 1.0); ax.set_ylabel("C2ST balanced accuracy"); fig.savefig(f"{F}/f_c2st.png")


def fig_signature():
    rs = json.load(open(f"{O}/real_defect_check.json")); fa = ["STIFFNESS", "DAMPING", "TOOL_WEAR", "DEPTH_OVERLOAD", "RPM_MISMATCH"]
    M = np.array([[rs["real_shift"][n] for n in SENS]] + [[rs["sim_shift"][f][n] for n in SENS] for f in fa]); ci = np.array([rs["real_shift_ci95"][n] for n in SENS])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(9); axs[0].bar(x, M[0], color="#c0392b", alpha=0.85); axs[0].errorbar(x, M[0], yerr=[M[0] - ci[:, 0], ci[:, 1] - M[0]], fmt="none", ecolor="k", capsize=2)
    axs[0].axhline(0, color="k", lw=0.8); axs[0].set_xticks(x); axs[0].set_xticklabels(SL, rotation=45, ha="right", fontsize=7); axs[0].set_ylabel("shift [healthy s.d.]"); axs[0].set_title("Real defective tool D* (95% bootstrap CI)", fontsize=9)
    im = axs[1].imshow(M, cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto"); axs[1].set_yticks(range(6)); axs[1].set_yticklabels(["real D*", "stiffness", "damping", "wear", "depth", "rpm"]); axs[1].set_xticks(x); axs[1].set_xticklabels(SL, rotation=45, ha="right", fontsize=7); axs[1].grid(False); axs[1].set_title("Real versus simulated faults", fontsize=9)
    fig.colorbar(im, ax=axs[1], shrink=0.8, label="s.d."); fig.savefig(f"{F}/f_signature.png")


def fig_chipped():
    d = json.load(open(f"{O}/defect_mechanism_check.json")); rs = json.load(open(f"{O}/real_defect_check.json"))["real_shift"]; sel = ["chatter_acc", "chatter_mic", "acc_low", "mic_low", "acc_high"]
    keys = [f"runout{s}um_gain{g}x" for s in (20, 80) for g in (1, 10)]; fig, ax = plt.subplots(figsize=(7, 3.0)); w = 0.15; x = np.arange(len(sel))
    ax.bar(x - 2 * w, [rs[n] for n in sel], w, color="#c0392b", label="real D*")
    for j, k in enumerate(keys): ax.bar(x + (j - 1) * w, [d[k]["shift"][n] for n in sel], w, label=k.replace("runout", "runout ").replace("um_gain", " um, gain ").replace("x", "x") , color=["#9db8d2", "#6a9bc3", "#3f7ca8", "#1b5a86"][j])
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["non-sync acc", "non-sync mic", "acc low", "mic low", "acc >3.2 kHz"], fontsize=8); ax.set_ylim(-5, 14); ax.set_ylabel("shift [healthy s.d.]"); ax.legend(frameon=False, fontsize=6.5, ncol=2); ax.set_title("Chipped-flute (runout) mechanism versus the real defective tool", fontsize=9); fig.savefig(f"{F}/f_chipped.png")


def fig_luh():
    d = json.load(open(f"{O}/luh_baselines.json")); M = ["RMS-threshold", "GB", "LR", "CORAL+LR", "DANN"]; fig, axs = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True); w = 0.27
    for a, fs, t in zip(axs, ["raw", "machine-relative", "tool-relative"], ["Raw features", "Machine-relative", "Tool-relative"]):
        for j in range(3): a.bar(np.arange(5) + (j - 1) * w, [d[f"{fs}|{m}"][j]["auroc_mean"] for m in M], w, label=f"target M{j + 1}", color=["#1b6ca8", "#3a9d8f", "#d98c1f"][j])
        a.axhline(0.5, color="k", ls=":", lw=1); a.set_xticks(range(5)); a.set_xticklabels(["RMS\nrule", "GB", "LR", "CORAL\n+LR", "DANN"], fontsize=8); a.set_title(t, fontsize=9); a.set_ylim(0.3, 1.02)
    axs[0].set_ylabel("AUROC (VB>=100 um)"); axs[0].legend(frameon=False, fontsize=7, loc="lower left"); fig.savefig(f"{F}/f_luh_baselines.png")


def fig_thr():
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.9), sharey=True)
    for a, (suf, t) in zip(axs, [("", "VB>=100"), ("_thr60", "VB>=60"), ("_thr40", "VB>=40")]):
        r = json.load(open(f"{O}/sim_to_real_wear_absmean{suf}.json")); ms = ["1", "2", "3"]; x = np.arange(3)
        a.bar(x - 0.2, [r[m]["rms-rule"] for m in ms], 0.4, label="one-feature rule", color="#9a9a9a"); a.bar(x + 0.2, [r[m]["sim-only"] for m in ms], 0.4, label="simulation only", color="#1b6ca8")
        a.axhline(0.5, color="k", ls=":", lw=1); a.set_xticks(x); a.set_xticklabels(["M1", "M2", "M3"]); a.set_title(t + " um (label-free)", fontsize=9); a.set_ylim(0.3, 1.02)
    axs[0].set_ylabel("AUROC"); axs[0].legend(frameon=False, fontsize=7, loc="lower left"); fig.savefig(f"{F}/f_wear_threshold.png")


def fig_ft():
    r = json.load(open(f"{O}/sim_to_real_wear_absmean.json")); KS = [2, 4, 8, 16, 32]; fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(KS, [np.nanmean([r[m][f"k={k} lam=1.0 scratch"] for m in r]) for k in KS], "o-", label="from scratch", color="#6b6b6b"); ax.plot(KS, [np.nanmean([r[m][f"k={k} lam=1.0 sim->finetune"] for m in r]) for k in KS], "s-", label="sim -> fine-tune", color="#1b6ca8")
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.set_xlabel("real labelled runs k"); ax.set_ylabel("AUROC (mean of 3 machines)"); ax.legend(frameon=False); fig.savefig(f"{F}/f_finetune.png")


def fig_dyn():
    d = json.load(open(f"{O}/luh_dynamics_shift.json")); fig, axs = plt.subplots(1, 2, figsize=(8, 3.0), sharey=True)
    for a, k, t in zip(axs, ["thr100|raw|GB", "thr60|raw|GB"], ["VB>=100", "VB>=60"]):
        r = d[k]["rows"]; x = [q["dist_mean"] for q in r]; y = [q["all"] for q in r]; a.scatter(x, y, color="#1b6ca8"); rho = d[k]["spearman_dist_vs_auroc_all"]; a.set_title(f"{t}: Spearman {rho[0]:+.2f} (p={rho[1]:.2f})", fontsize=9); a.set_xlabel("proxy distance of target to training tools")
    axs[0].set_ylabel("AUROC, trained on other machines"); axs[0].set_ylim(0.9, 1.005); fig.savefig(f"{F}/f_dynamics_shift.png")


def fig_prov():
    from model_v4 import parameter_registry
    r = parameter_registry(); c = r.tier.value_counts(); order = ["FIXED", "RANGE", "LEARNED", "ASSUMED"]; fig, ax = plt.subplots(figsize=(4.4, 2.8))
    v = [int(c.get(k, 0)) for k in order]; ax.barh(order[::-1], v[::-1], color=["#c0392b", "#d98c1f", "#9db8d2", "#1b6ca8"]); [ax.text(vi + 0.1, i, str(vi), va="center") for i, vi in enumerate(v[::-1])]; ax.set_xlabel("number of parameter groups"); ax.set_title("Provenance tier of simulator parameters", fontsize=9); fig.savefig(f"{F}/f_provenance.png")


def fig_frf():
    from milling_model import make_modes
    from frf_import import fit_modes
    mx, _ = make_modes(); f = np.arange(50, 4000, 2.0); H = mx.frf(2 * np.pi * f); rng = np.random.default_rng(0); fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.0), sharey=True)
    for a, nz in zip(axs, [0.01, 0.03]):
        Hn = H + nz * np.abs(H).max() * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / np.sqrt(2); m, err = fit_modes(f, Hn, fmax=3500); Hf = m.frf(2 * np.pi * f); b = f <= 3500; rel = np.linalg.norm(Hf[b] - H[b]) / np.linalg.norm(H[b])
        a.semilogy(f, np.abs(Hn), color="#bbbbbb", lw=0.8, label="noisy FRF"); a.semilogy(f, np.abs(H), color="k", lw=1, label="true"); a.semilogy(f[b], np.abs(Hf[b]), color="#c0392b", lw=1, label="fitted"); a.set_xlabel("frequency [Hz]"); a.set_title(f"noise {nz * 100:.0f}% of peak: {len(m.fn)} modes, shape error {rel * 100:.1f}%", fontsize=9)
    axs[0].set_ylabel("|H| [m/N]"); axs[0].legend(frameon=False, fontsize=7); fig.savefig(f"{F}/f_frf_fit.png")


def fig_waveforms():
    from model_v4 import build, X0
    from common import run
    fig, axs = plt.subplots(4, 2, figsize=(9.5, 5.6), sharex="col")
    for col, (ae, t) in enumerate([(0.0635, "small radial depth ae=0.06 mm"), (3.0, "large radial depth ae=3 mm")]):
        cut = dict(N=10000.0, ap=38.1, ae=ae, F=0.08 * 2 * 10000 / 60.0, Z=2, D=6.35, direction="DOWN"); p, cfg = build(cut, X0); sim, sig, _ = run(p, seed=1, t_end=0.6, sensor_cfg=cfg, noise_um=0.02, skip=0.5)
        k = (sig["t"] > 0.52) & (sig["t"] < 0.56); tt = (sig["t"][k] - 0.52) * 1e3
        for r, (key, lab) in enumerate([("vib_x", "tool-work vib. [um]"), ("acc_x", "accel. [g]"), ("mic", "mic [Pa]"), ("F_x", "force F_x [N]")]):
            axs[r, col].plot(tt, sig[key][k], lw=0.7, color="#1b6ca8"); axs[r, col].set_ylabel(lab if col == 0 else "", fontsize=8)
        axs[0, col].set_title(t, fontsize=9); axs[3, col].set_xlabel("time [ms]")
    fig.savefig(f"{F}/f_waveforms.png")


def fig_gain():
    s = json.load(open(f"{O}/rca_sweep.json")); ks = ["0.3", "1.0", "3.0"]; fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for n, c in zip(s["1.0"], ["#1b6ca8", "#3a9d8f", "#d98c1f", "#6b6b6b"]): ax.plot([0.3, 1, 3], [s[k][n]["top1"] for k in ks], "o-", label=n, color=c)
    ax.set_xscale("log"); ax.set_xticks([0.3, 1, 3]); ax.set_xticklabels(["x0.3", "x1 (calibrated)", "x3"]); ax.axhline(1 / 6, color="k", ls=":", lw=1); ax.set_ylabel("6-class top-1 (simulated test)"); ax.set_xlabel("fault gain"); ax.legend(frameon=False, fontsize=7); fig.savefig(f"{F}/f_gain_sweep.png")


def fig_graph():
    c = json.load(open(f"{O}/rca_causal.json")); nodes, edges = c["nodes"], c["edges"]; roots = nodes[:5]; sens = nodes[5:]
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis("off"); pos = {n: (0, 4 - i) for i, n in enumerate(roots)}; pos.update({n: (3, 4.5 - i * 0.55) for i, n in enumerate(sens)})
    for u, v in edges: ax.annotate("", pos[v], pos[u], arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#7a7a7a", shrinkA=26, shrinkB=34, connectionstyle="arc3,rad=0.0"))
    for n in roots: ax.text(*pos[n], n.replace("_est", "").replace("Spindle", "Spindle "), ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="#fff3d9", ec="#8a6d1c"))
    for n, l in zip(sens, SL): ax.text(*pos[n], l, ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle="round,pad=0.25", fc="#e8eef7", ec="#35506b"))
    ax.text(0, 4.75, "root-cause nodes (5)", ha="center", fontsize=8, style="italic"); ax.text(3, 4.95, "sensor nodes (9)", ha="center", fontsize=8, style="italic"); ax.set_xlim(-0.8, 3.8); ax.set_ylim(-0.4, 5.1)
    fig.savefig(f"{F}/f_graph.png")


def fig_planned():
    """PLACEHOLDER: dữ liệu GIẢ minh họa hình dạng đường cong k; KHÔNG phải kết quả thật (đánh dấu đỏ trên hình và trong PLACEHOLDERS.md)."""
    k = np.array([0, 2, 4, 8, 16, 32]); fig, ax = plt.subplots(figsize=(4.8, 3.0)); ax.plot(k, [0.46, 0.55, 0.62, 0.70, 0.77, 0.82], "o-", color="#1b6ca8", label="sim + fine-tune (placeholder)"); ax.plot(k[1:], [0.30, 0.44, 0.58, 0.72, 0.80], "s-", color="#6b6b6b", label="real only (placeholder)")
    ax.text(0.5, 0.5, "PLACEHOLDER\nnot real data", transform=ax.transAxes, color="red", fontsize=22, alpha=0.35, ha="center", va="center", rotation=20, fontweight="bold"); ax.set_xlabel("real labelled cuts k"); ax.set_ylabel("top-1 (leave-one-configuration-out)"); ax.legend(frameon=False, fontsize=7, loc="lower right"); ax.set_ylim(0.2, 1.0)
    fig.savefig(f"{F}/f_planned_kcurve.png")


def fig_k10():
    a = json.load(open(f"{O}/rca_causal.json"))["results"]; b = json.load(open(f"{O}/rca_causal_k10.json"))["results"]; sc = list(a)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.0), sharey=True); w = 0.38
    for ax_, s, t in zip(axs, sc, ["Same generator", "Shifted generator"]):
        for j, (R, lab, c) in enumerate([(a, "nominal parameters", "#9db8d2"), (b, "stiffness x10 (stable regime)", "#1b6ca8")]):
            v = [np.mean([R[s][al]["10"][f][0] for f in R[s][al]["10"]]) for al in AL]; ax_.bar(np.arange(6) + (j - 0.5) * w, v, w, label=lab, color=c)
        ax_.axhline(1 / 14, color="k", ls=":", lw=1); ax_.set_xticks(range(6)); ax_.set_xticklabels(["BRCD", "RCD", "RCG", "Smooth", "BARO", "Simple"]); ax_.set_title(t, fontsize=9)
    axs[0].set_ylabel("Top-1 (m=10)"); axs[0].legend(frameon=False, fontsize=7, loc="lower left"); fig.savefig(f"{F}/f_k10.png")



def fig_architecture():
    """Sơ đồ kiến trúc lớn (đặt trên trang nằm ngang): Input -> Simulation -> RCA. Chữ ngắn, cỡ lớn để còn đọc được khi in."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(14, 9.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 10); ax.axis("off")
    C = {"in": ("#e8eef7", "#35506b"), "sim": ("#eaf3e6", "#4a6b35"), "rca": ("#fff3d9", "#8a6d1c"), "ev": ("#f7e6e6", "#8a3535"), "real": ("#efe9f7", "#5b4a7a")}
    FS = 11.2

    def box(x, y, w, h, t, k, fs=FS):
        fc, ec = C[k]; ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.5)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.28)

    def panel(x, y, w, h, title, k):
        ec = C[k][1]; ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=ec, lw=2.2, ls="--")); ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="center", fontsize=16, fontweight="bold", color=ec)

    def arrow(x1, y1, x2, y2, c="#35506b", lw=2.0):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=lw, color=c, shrinkA=0, shrinkB=0, mutation_scale=16))

    def chip(x, y, t, c):
        ax.text(x, y, t, fontsize=8.2, ha="center", va="center", color="white", bbox=dict(boxstyle="round,pad=0.2", fc=c, ec="none"))

    panel(0.1, 0.25, 3.7, 9.6, "1  INPUT", "in"); panel(4.1, 0.25, 5.05, 9.6, "2  SIMULATION", "sim"); panel(9.45, 0.25, 3.45, 9.6, "3  RCA", "rca")
    # INPUT
    box(0.3, 8.0, 3.3, 1.15, "Cutting condition\nn, a$_e$, a$_p$, f$_t$, direction", "in")
    box(0.3, 6.5, 3.3, 1.3, "Tool and machine\nD, flutes, helix; modal f$_n$, ζ, k\nor measured FRF → mode fit", "in")
    box(0.3, 5.25, 3.3, 1.05, "Force and sensor constants\nK$_t$, K$_r$, K$_{te}$, K$_{re}$, U, gains", "in")
    box(0.3, 3.85, 3.3, 1.2, "Fault specification\nstiffness · damping · wear\ndepth · spindle speed", "in")
    box(0.3, 2.3, 3.3, 1.3, "Real healthy cuts (MSM)\n73 cuts, 3 channels\n→ PSD envelope + harmonics", "real")
    box(0.3, 0.55, 3.3, 1.45, "Provenance registry\ntier of every parameter", "in")
    for i, (t, c) in enumerate([("FIXED", "#1b5a86"), ("RANGE", "#3a9d8f"), ("LEARNED", "#d98c1f"), ("ASSUMED", "#c0392b")]): chip(0.75 + i * 0.86, 0.85, t, c)
    # SIMULATION
    box(4.3, 7.4, 4.65, 1.75, "Regenerative milling DDE\nmodal state space (exact ZOH)\nhelical slices, delayed chip thickness\nshear + edge forces, imbalance", "sim")
    box(4.3, 5.7, 4.65, 1.4, "Sensor chain\ndisplacement · accelerometer\nmicrophone · dynamometer", "sim")
    box(4.3, 4.45, 4.65, 1.0, "Fault effect  Δ = faulty − healthy\n(same condition, same noise seed)", "sim")
    box(4.3, 2.55, 2.4, 1.6, "Healthy baseline\nfrom real spectra\nkernel ridge +\nvariation model", "real", fs=10.4)
    ax.text(7.85, 3.35, "g · Δ is added\nto the baseline", ha="center", va="center", fontsize=10.5, style="italic", color="#4a6b35")
    box(4.3, 1.35, 4.65, 0.95, "Labelled window = baseline + g · Δ\n(g from real effect sizes)", "sim")
    box(4.3, 0.45, 4.65, 0.7, "30 features → 14 nodes", "sim")
    arrow(6.6, 7.4, 6.6, 7.1); arrow(6.6, 5.7, 6.6, 5.45); arrow(8.4, 4.45, 8.4, 2.3); arrow(5.5, 2.55, 5.5, 2.3); arrow(6.6, 1.35, 6.6, 1.15)
    # RCA
    box(9.65, 7.85, 3.05, 1.3, "Data for RCA\nD$_{obs}$: healthy windows\nD$_{int}$: m anomalous", "rca")
    box(9.65, 6.3, 3.05, 1.3, "14 nodes: 5 root causes\n+ 9 sensors, 19 edges\nsoft sensors (GB)", "rca")
    box(9.65, 4.3, 3.05, 1.7, "Six RCA algorithms\nBRCD · RCD · RCG\nSmooth Traversal\nBARO · SimpleRCA", "rca")
    box(9.65, 3.2, 3.05, 0.85, "Ranking of 14 nodes\ntop-1 / 3 / 5", "rca")
    box(9.65, 1.6, 3.05, 1.3, "Synthetic tests\nshared vs shifted generator\ngraph · node · gain ablations", "ev", fs=10.4)
    box(9.65, 0.45, 3.05, 0.9, "Real-data checks\nC2ST · defective tool · LUH", "real", fs=10.4)
    arrow(11.15, 7.85, 11.15, 7.6); arrow(11.15, 6.3, 11.15, 6.0); arrow(11.15, 4.3, 11.15, 4.05); arrow(11.15, 3.2, 11.15, 2.9)
    # liên kết giữa các khối
    for y in (8.55, 7.15, 5.77, 4.45): ax.plot([3.6, 3.95], [y, y], color="#35506b", lw=2.0)
    ax.plot([3.95, 3.95], [4.45, 8.55], color="#35506b", lw=2.0); arrow(3.95, 8.3, 4.3, 8.3)
    arrow(3.6, 2.95, 4.3, 3.35)
    ax.plot([8.95, 9.3, 9.3], [0.8, 0.8, 8.5], color="#8a6d1c", lw=2.0); arrow(9.3, 8.5, 9.65, 8.5, c="#8a6d1c")
    ax.text(6.5, 9.93, "Provenance-tracked simulation and root cause analysis pipeline", ha="center", fontsize=17, fontweight="bold")
    fig.savefig(f"{F}/f_architecture.png", dpi=170, bbox_inches="tight")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("fig_"):
            try: f(); plt.close("all"); print("ok", n)
            except Exception as e: print("LỖI", n, repr(e)[:200])
