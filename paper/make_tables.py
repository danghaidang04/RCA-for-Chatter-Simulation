"""Sinh các bảng LaTeX TRỰC TIẾP từ file kết quả trong ../cutting_sim/output (không chép tay số). Chạy: python make_tables.py
Quy ước (theo research-paper-writing/experiments.md): booktabs, không đường dọc, mũi tên chiều tốt trong tiêu đề cột, số thập phân nhất quán,
tô nhẹ + in đậm giá trị tốt nhất, nhóm cột bằng multicolumn + cmidrule."""
import json, os, sys
import numpy as np, pandas as pd
O = "../cutting_sim/output"; sys.path.insert(0, "../cutting_sim")
ALG = ["BRCD", "RCD", "RCG", "SmoothTraversal", "BARO", "SimpleRCA"]; ALN = {"SmoothTraversal": "Smooth Traversal"}
nm = lambda a: ALN.get(a, a)
BEST = r"\cellcolor{gray!20}\textbf{%s}"


def cells(vals, dec, higher=True, tol=1e-9):
    """Chuỗi ô cho một cột: giá trị tốt nhất (có thể đồng hạng) được tô nhẹ và in đậm."""
    v = np.array(vals, float); b = np.nanmax(v) if higher else np.nanmin(v)
    return [(BEST % f"{x:.{dec}f}") if abs(x - b) <= tol else f"{x:.{dec}f}" for x in v]


def table(path, spec, head, rows, notes=""):
    open(path, "w").write("\\begin{tabular}{" + spec + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n" + notes)


# ---------- 1. Causal RCA: top-1 theo m ----------
c = json.load(open(f"{O}/rca_causal.json")); sc = list(c["results"]); ms = ["5", "10", "20", "50"]
top1 = lambda s, a, m: np.mean([c["results"][s][a][m][f][0] for f in c["results"][s][a][m]])
cols = [[top1(s, a, m) for a in ALG] for s in sc for m in ms]; C = [cells(col, 2) for col in cols]
rows = [f"{nm(a)} & " + " & ".join(C[j][i] for j in range(8)) + r" \\" for i, a in enumerate(ALG)]
rows.append("\\midrule\nChance (1/14) & " + " & ".join(["0.07"] * 8) + r" \\")
table("tables/tab_causal.tex", "l cccc cccc", "& \\multicolumn{4}{c}{Same generator} & \\multicolumn{4}{c}{Shifted generator parameters} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nAlgorithm & $m{=}5$ & $10$ & $20$ & $50$ & $m{=}5$ & $10$ & $20$ & $50$ \\\\", rows)

# ---------- 2. Classifiers (simulated) ----------
r = json.load(open(f"{O}/rca_results.json"))["results"]; names = sorted(r, key=lambda n: -r[n]["test_top1"]); K = ["test_top1", "test_top3", "test_macroF1", "detect_auroc"]
C = [cells([r[n][k] for n in names], 3) for k in K]
table("tables/tab_cls.tex", "lcccc", "Classifier & Top-1 $\\uparrow$ & Top-3 $\\uparrow$ & Macro-F1 $\\uparrow$ & AUROC$_{\\text{fault vs.\\ normal}}$ $\\uparrow$ \\\\", [f"{n} & " + " & ".join(C[j][i] for j in range(4)) + r" \\" for i, n in enumerate(names)])

# ---------- 3. Sweep gain ----------
s = json.load(open(f"{O}/rca_sweep.json")); algs = list(s["1.0"]); KS = ("0.3", "1.0", "3.0"); C = [cells([s[k][a]["top1"] for a in algs], 3) for k in KS]
table("tables/tab_sweep.tex", "lccc", "Classifier & gain $\\times0.3$ $\\uparrow$ & $\\times1$ (calibrated) $\\uparrow$ & $\\times3$ $\\uparrow$ \\\\", [f"{a} & " + " & ".join(C[j][i] for j in range(3)) + r" \\" for i, a in enumerate(algs)])

# ---------- 4. C2ST ----------
names = {(0, "regression mean only"): ("Linear", "mean only"), (0, "hierarchical: between-cut + within-cut"): ("Linear", "+ between/within-cut variation"),
         (1, "regression mean only"): ("Kernel ridge", "mean only"), (1, "hierarchical: between-cut + within-cut"): ("Kernel ridge", "+ between/within-cut variation (final)")}
vals = {k: [json.load(open(f"{O}/noise_model_gate_nonlin{k[0]}_seed{sd}.json"))["c2st_heldout"][k[1]][0] for sd in range(3)] for k in names}
best = min(np.mean(v) for v in vals.values()); rows = []
for k, (a, b) in names.items():
    v = vals[k]; t = f"{np.mean(v):.3f} $\\pm$ {np.std(v):.3f}"; rows.append(f"{a} & {b} & " + ((BEST % t) if abs(np.mean(v) - best) < 1e-12 else t) + r" \\")
table("tables/tab_c2st.tex", "llc", "Mean model & Variation model & C2ST balanced accuracy $\\downarrow$ \\\\", rows)

# ---------- 5. Registry ----------
reg = pd.read_csv(f"{O}/parameter_registry.csv")[["parameter", "tier", "verification"]]; G = {"β": "$\\beta$", "ζ": "$\\zeta$", "σ": "$\\sigma$", "µ": "$\\mu$", "≈": "$\\approx$", "≤": "$\\leq$", "≥": "$\\geq$", "°": "$^\\circ$", "×": "$\\times$", "→": "$\\to$", "Δ": "$\\Delta$", "²": "$^2$", "·": "$\\cdot$", "∝": "$\\propto$", "−": "-", "—": "---", "–": "--"}
def esc(t):
    t = str(t).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#").replace("^", "\\^{}")
    for k, v in G.items(): t = t.replace(k, v)
    return t
rows = []; prev = None
for p, t, v in reg.values:
    if prev is not None and t != prev: rows.append("\\midrule")
    rows.append(f"{esc(p)} & {esc(t)} & {esc(v)} \\\\"); prev = t
table("tables/tab_registry.tex", "p{5.2cm}p{1.8cm}p{6.5cm}", "Parameter & Tier & Verification \\\\", rows)

# ---------- 6. Ablation (3 hạt giống), m=10 ----------
ab = json.load(open(f"{O}/rca_ablation.json"))
NAME = {"soft|physics|x1.0": "Supervised nodes, physics graph (main)", "soft|shuffled|x1.0": "Supervised nodes, shuffled graph", "soft|none|x1.0": "Supervised nodes, no edges",
        "physical|physics|x1.0": "Label-free nodes, physics graph", "soft|physics|x0.3": "Supervised, physics graph, gain $\\times0.3$", "soft|physics|x3.0": "Supervised, physics graph, gain $\\times3$"}
rows = []
for sc_, title in (("in", "Same generator"), ("shift", "Shifted generator parameters")):
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\")
    for k, lab in NAME.items(): rows.append(f"\\quad {lab} & " + " & ".join(f"{np.mean(ab[f'{k}|{sc_}|m10'][a]):.2f}\\,$\\pm$\\,{np.std(ab[f'{k}|{sc_}|m10'][a]):.2f}" for a in ALG) + " \\\\")
    rows.append("\\midrule")
table("tables/tab_ablation.tex", "p{5.6cm}cccccc", "Variant & BRCD & RCD & RCG & Smooth & BARO & Simple \\\\", rows[:-1])

# ---------- 7. Real defective tool vs simulated ----------
rs = json.load(open(f"{O}/real_defect_check.json")); order = ["log10_acc", "mic_dB", "chatter_acc", "chatter_mic", "acc_low", "acc_res", "acc_high", "mic_low", "mic_high"]
LAB = {"log10_acc": "log acc.\\ rms", "mic_dB": "mic.\\ level", "chatter_acc": "non-sync.\\ index (acc)", "chatter_mic": "non-sync.\\ index (mic)", "acc_low": "acc.\\ 50--600\\,Hz", "acc_res": "acc.\\ 1--3.2\\,kHz", "acc_high": "acc.\\ $>$3.2\\,kHz", "mic_low": "mic.\\ 50--600\\,Hz", "mic_high": "mic.\\ $>$3.2\\,kHz"}
fa = ["STIFFNESS", "DAMPING", "TOOL_WEAR", "DEPTH_OVERLOAD", "RPM_MISMATCH"]
cc = lambda x, d=1: (f"\\cellcolor{{red!18}}{x:+.{d}f}" if x >= 2 else (f"\\cellcolor{{blue!15}}{x:+.{d}f}" if x <= -2 else f"{x:+.{d}f}"))
rows = [f"{LAB[n]} & {cc(rs['real_shift'][n])} [{rs['real_shift_ci95'][n][0]:+.1f}, {rs['real_shift_ci95'][n][1]:+.1f}] & " + " & ".join(cc(rs['sim_shift'][f][n]) for f in fa) + " \\\\" for n in order]
rows.append("\\midrule\nCosine with real shift & -- & " + " & ".join(f"{rs['cosine_with_sim'][f]:.2f}" for f in fa) + " \\\\")
table("tables/tab_realshift.tex", "lccccc c".replace(" ", ""), "& Real tool D$^*$ & \\multicolumn{5}{c}{Simulated fault} \\\\\n\\cmidrule(lr){3-7}\nSensor node [s.d.] & [95\\% CI] & Stiff. & Damp. & Wear & Depth & RPM \\\\", rows[:-2] + rows[-2:])

# ---------- 8. LUH baselines (raw) ----------
lb = json.load(open(f"{O}/luh_baselines.json")); MET = ["RMS-threshold", "GB", "LR", "CORAL+LR", "DANN"]; MN = {"RMS-threshold": "One-feature RMS rule (no labels)", "GB": "Gradient boosting", "LR": "Logistic regression", "CORAL+LR": "CORAL + LR$^{\\dagger}$", "DANN": "DANN$^{\\dagger}$"}
sim0 = json.load(open(f"{O}/sim_to_real_wear_absmean.json")); simrow = [sim0[m]["sim-only"] for m in ("1", "2", "3")]
rows = []
for fs, ft in (("raw", "Raw features"), ("machine-relative", "Machine-relative features"), ("tool-relative", "Tool-relative features")):
    C = [cells([lb[f"{fs}|{m}"][j]["auroc_mean"] for m in MET], 3) for j in range(3)]
    rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{ft}}}}} \\\\")
    rows += [f"\\quad {MN[m]} & " + " & ".join(C[j][i] for j in range(3)) + r" \\" for i, m in enumerate(MET)]
    rows.append("\\midrule")
rows.append("Simulation only, no real labels (tool-relative $x,y$) & " + " & ".join(f"{x:.3f}" for x in simrow) + r" \\")
table("tables/tab_luhbase.tex", "lccc", "Method (target machine $\\rightarrow$) & M1 $\\uparrow$ & M2 $\\uparrow$ & M3 $\\uparrow$ \\\\", rows, "\\vspace{1pt}\\\\{\\footnotesize $^{\\dagger}$ uses unlabelled target-machine features (transductive).}\n")

# ---------- 9. wear threshold: sim vs rule ----------
rows = []
for suf, t in (("", "100"), ("_thr60", "60"), ("_thr40", "40")):
    r_ = json.load(open(f"{O}/sim_to_real_wear_absmean{suf}.json")); ms_ = ["1", "2", "3"]; a = [r_[m]["rms-rule"] for m in ms_]; b = [r_[m]["sim-only"] for m in ms_]
    mb = max(np.mean(a), np.mean(b))
    f = lambda v: " & ".join(f"{x:.3f}" for x in v) + " & " + ((BEST % f"{np.mean(v):.3f}") if abs(np.mean(v) - mb) < 1e-12 else f"{np.mean(v):.3f}")
    rows.append(f"\\multirow{{2}}{{*}}{{$\\ge{t}$}} & One-feature rule & {f(a)} \\\\\n & Simulation only & {f(b)} \\\\")
    if t != "40": rows.append("\\midrule")
table("tables/tab_thr.tex", "llcccc", "VB threshold [$\\mu$m] & Method & M1 $\\uparrow$ & M2 $\\uparrow$ & M3 $\\uparrow$ & Mean $\\uparrow$ \\\\", rows)

# ---------- 10. RCA nodes ----------
NODES_ROOT = [("Stiffness\\_est", "soft sensor", "log stiffness factor", "gradient boosting on 30 spectral features"), ("Damping\\_est", "soft sensor", "log damping factor", "same"),
              ("ToolWear\\_est", "soft sensor", "log wear factor ($K_t$)", "same"), ("CutDepth\\_est", "soft sensor", "log radial-depth factor", "same"), ("SpindleRPM\\_est", "controller", "log$(N/N_\\text{plan})$", "true value + 0.5\\% read noise")]
table("tables/tab_nodes.tex", "llll", "Root-cause node & Source & Quantity & Construction \\\\", [" & ".join(x) + r" \\" for x in NODES_ROOT])

# ---------- 11. simulator nominal parameters ----------
from model_v4 import SPEC
conv = {"ln g_acc": ("$g_{\\mathrm{acc}}$", lambda x: f"{np.exp(x):.2e}", "--"), "ln noise_acc": ("acc. noise floor", lambda x: f"{np.exp(x):.3f}", "g"), "ln g_mic": ("$g_{\\mathrm{mic}}$", lambda x: f"{np.exp(x):.2f}", "--"), "ln noise_mic": ("mic. noise floor", lambda x: f"{np.exp(x):.3f}", "Pa"),
        "fn_x [Hz]": ("natural freq.\\ $f_x$", lambda x: f"{x:.0f}", "Hz"), "fn_y [Hz]": ("natural freq.\\ $f_y$", lambda x: f"{x:.0f}", "Hz"), "zeta": ("damping ratio $\\zeta$", lambda x: f"{x:.3f}", "--"), "ln k_modal [N/m]": ("modal stiffness $k$", lambda x: f"{np.exp(x):.1e}", "N/m"),
        "ln Kt [N/m2]": ("tangential coeff.\\ $K_t$", lambda x: f"{np.exp(x):.1e}", "N/m$^2$"), "ln Kr/Kt": ("ratio $K_r/K_t$", lambda x: f"{np.exp(x):.2f}", "--"), "ln Kte [N/m]": ("edge coeff.\\ $K_{te}$", lambda x: f"{np.exp(x):.1e}", "N/m"), "ln Kre [N/m]": ("edge coeff.\\ $K_{re}$", lambda x: f"{np.exp(x):.1e}", "N/m"),
        "ln U_imb [kg m]": ("imbalance $U$", lambda x: f"{np.exp(x):.1e}", "kg\\,m"), "acc_rpm_exp": ("acc. noise rpm exponent", lambda x: f"{x:.1f}", "--"), "mic_rpm_exp": ("mic. noise rpm exponent", lambda x: f"{x:.1f}", "--"), "ln runout [m]": ("runout", lambda x: f"{np.exp(x) * 1e6:.1f}", "$\\mu$m")}
rows = []
for n, lo, hi, x0 in SPEC:
    lab, f, u = conv[n]; rng = (f(lo) + "--" + f(hi)) if not n.startswith("ln") or True else ""; rows.append(f"{lab} & {f(x0)} & {u} & {f(lo)} to {f(hi)} \\\\")
table("tables/tab_simparams.tex", "lccl", "Parameter & Nominal & Unit & Numerical search bounds \\\\", rows)
print("tables written")

# ---------- 12. Operating regime (regime_check.json) ----------
rg = json.load(open(f"{O}/regime_check.json")); rows = []; prev = None
for q in rg:
    if prev is not None and q["kscale"] != prev: rows.append("\\midrule")
    unstable = (q["zoa_alim_mm"] is not None) and q["zoa_alim_mm"] < q["ap_mm"]
    al = "$>200$" if q["zoa_alim_mm"] is None else (f"\\cellcolor{{red!18}}{q['zoa_alim_mm']:.1f}" if unstable else f"{q['zoa_alim_mm']:.1f}")
    rows.append(f"{q['kscale']:g} & {q['ae_mm']:g} & {al} & {q['x_rms_um']:.1f} & {q['Fx_rms_N']:.0f} \\\\"); prev = q["kscale"]
table("tables/tab_regime.tex", "ccccc", "Stiffness scale & $a_e$ [mm] & ZOA $a_{\\lim}$ [mm] $\\uparrow$ & $x$ rms [$\\mu$m] $\\downarrow$ & $F_x$ rms [N] \\\\", rows, "")

# ---------- 13. Robustness: nominal vs stiffer (x10) parameters, m=10 ----------
k1 = json.load(open(f"{O}/rca_causal.json"))["results"]; k10 = json.load(open(f"{O}/rca_causal_k10.json"))["results"]
t10 = lambda R, sc, a: np.mean([R[sc][a]["10"][f][0] for f in R[sc][a]["10"]])
S1, S2 = list(k1)[0], list(k1)[1]; cols = [[t10(R, s, a) for a in ALG] for R in (k1, k10) for s in (S1, S2)]; C = [cells(c_, 2) for c_ in cols]
rows = [f"{nm(a)} & " + " & ".join(C[j][i] for j in range(4)) + r" \\" for i, a in enumerate(ALG)] + ["\\midrule\nChance (1/14) & 0.07 & 0.07 & 0.07 & 0.07 \\\\"]
table("tables/tab_k10.tex", "lcccc", "& \\multicolumn{2}{c}{Nominal parameters (above stability limit)} & \\multicolumn{2}{c}{Stiffness $\\times10$ (stable regime)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nAlgorithm & Same generator & Shifted & Same generator & Shifted \\\\", rows)
