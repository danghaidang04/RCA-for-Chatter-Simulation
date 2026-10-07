"""Sinh các bảng LaTeX TRỰC TIẾP từ file kết quả trong ../cutting_sim/output (không chép tay số). Chạy: python make_tables.py"""
import json, numpy as np, pandas as pd
O = "../cutting_sim/output"
ALG = ["BRCD", "RCD", "RCG", "SmoothTraversal", "BARO", "SimpleRCA"]
c = json.load(open(f"{O}/rca_causal.json")); sc = list(c["results"]); ms = ["5", "10", "20", "50"]
def top1(s, a, m): return np.mean([c["results"][s][a][m][f][0] for f in c["results"][s][a][m]])
rows = []
for a in ALG:
    rows.append(f"{a} & " + " & ".join(f"{top1(sc[0], a, m):.2f}" for m in ms) + " & " + " & ".join(f"{top1(sc[1], a, m):.2f}" for m in ms) + r" \\")
open("tables/tab_causal.tex", "w").write("\\begin{tabular}{l cccc cccc}\n\\toprule\n & \\multicolumn{4}{c}{Same generator (in-distribution)} & \\multicolumn{4}{c}{Shifted DDE parameters} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nAlgorithm & $m{=}5$ & 10 & 20 & 50 & $m{=}5$ & 10 & 20 & 50 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
r = json.load(open(f"{O}/rca_results.json"))["results"]; rows = []
for n, v in sorted(r.items(), key=lambda x: -x[1]["test_top1"]): rows.append(f"{n} & {v['test_top1']:.3f} & {v['test_top3']:.3f} & {v['test_macroF1']:.3f} & {v['detect_auroc']:.3f} \\\\")
open("tables/tab_cls.tex", "w").write("\\begin{tabular}{lcccc}\n\\toprule\nClassifier & Top-1 $\\uparrow$ & Top-3 $\\uparrow$ & Macro-F1 $\\uparrow$ & AUROC (fault vs.\\ normal) $\\uparrow$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
s = json.load(open(f"{O}/rca_sweep.json")); algs = list(s["1.0"]); rows = [f"{a} & " + " & ".join(f"{s[k][a]['top1']:.3f}" for k in ("0.3", "1.0", "3.0")) + r" \\" for a in algs]
open("tables/tab_sweep.tex", "w").write("\\begin{tabular}{lccc}\n\\toprule\nClassifier & gain $\\times 0.3$ & $\\times 1$ (calibrated) & $\\times 3$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# C2ST: giá trị gõ từ log console của phiên làm việc (CHỈ lần chạy cuối được lưu trong noise_model_gate.json) -> đánh dấu trong experiment_log.md
runs = {"Linear regression, mean only": [0.986], "Linear regression + between/within-cut variation": [0.876],
        "Kernel ridge + stratified split, mean only (3 splits)": [0.851, 0.846, 0.895], "Kernel ridge + hierarchical variation (3 splits)": [0.817, 0.778, 0.866]}
rows = [f"{k} & {np.mean(v):.3f}" + (f" $\\pm$ {np.std(v):.3f}" if len(v) > 1 else "") + r" \\" for k, v in runs.items()]
open("tables/tab_c2st.tex", "w").write("\\begin{tabular}{lc}\n\\toprule\nReal-noise model variant & C2ST balanced accuracy (0.5 = indistinguishable) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
reg = pd.read_csv(f"{O}/parameter_registry.csv")[["parameter", "tier", "verification"]]; G = {"β": "$\\beta$", "ζ": "$\\zeta$", "σ": "$\\sigma$", "µ": "$\\mu$", "≈": "$\\approx$", "≤": "$\\leq$", "≥": "$\\geq$", "°": "$^\\circ$", "×": "$\\times$", "→": "$\\to$", "Δ": "$\\Delta$", "²": "$^2$", "·": "$\\cdot$", "∝": "$\\propto$", "−": "-", "—": "---", "–": "--"}
def esc(t):
    t = str(t).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#").replace("^", "\\^{}")
    for k, v in G.items(): t = t.replace(k, v)
    return t
rows = [f"{esc(p)} & {esc(t)} & {esc(v)} \\\\" for p, t, v in reg.values]
open("tables/tab_registry.tex", "w").write("\\begin{tabular}{p{5.2cm}p{1.8cm}p{6.5cm}}\n\\toprule\nParameter & Tier & Verification \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# --- ablation (3 hạt giống), m=10
ab = json.load(open(f"{O}/rca_ablation.json")); AL = ["BRCD", "RCD", "RCG", "SmoothTraversal", "BARO", "SimpleRCA"]
NAME = {"soft|physics|x1.0": "Supervised nodes, physics graph, gain $\\times1$ (main setting)", "soft|shuffled|x1.0": "Supervised nodes, \\emph{shuffled} graph", "soft|none|x1.0": "Supervised nodes, \\emph{no edges}",
        "physical|physics|x1.0": "\\emph{Label-free} nodes, physics graph", "soft|physics|x0.3": "Supervised nodes, physics graph, gain $\\times0.3$", "soft|physics|x3.0": "Supervised nodes, physics graph, gain $\\times3$"}
rows = []
for sc, title in (("in", "Same generator"), ("shift", "Shifted DDE parameters")):
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\")
    for k, lab in NAME.items(): rows.append(f"{lab} & " + " & ".join(f"{np.mean(ab[f'{k}|{sc}|m10'][a]):.2f}$\\pm${np.std(ab[f'{k}|{sc}|m10'][a]):.2f}" for a in AL) + " \\\\")
    rows.append("\\midrule")
open("tables/tab_ablation.tex", "w").write("\\begin{tabular}{p{5.6cm}cccccc}\n\\toprule\nVariant & BRCD & RCD & RCG & Smooth & BARO & Simple \\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
# --- dao hỏng thật vs mô phỏng
rs = json.load(open(f"{O}/real_defect_check.json")); order = ["log10_acc", "mic_dB", "chatter_acc", "chatter_mic", "acc_low", "acc_res", "acc_high", "mic_low", "mic_high"]
LAB = {"log10_acc": "log acc.\\ rms", "mic_dB": "mic.\\ level", "chatter_acc": "non-sync.\\ (acc)", "chatter_mic": "non-sync.\\ (mic)", "acc_low": "acc.\\ 50--600\\,Hz", "acc_res": "acc.\\ 1--3.2\\,kHz", "acc_high": "acc.\\ $>$3.2\\,kHz", "mic_low": "mic.\\ 50--600\\,Hz", "mic_high": "mic.\\ $>$3.2\\,kHz"}
fa = ["STIFFNESS", "DAMPING", "TOOL_WEAR", "DEPTH_OVERLOAD", "RPM_MISMATCH"]
rows = [f"{LAB[n]} & {rs['real_shift'][n]:+.1f} [{rs['real_shift_ci95'][n][0]:+.1f}, {rs['real_shift_ci95'][n][1]:+.1f}] & " + " & ".join(f"{rs['sim_shift'][f][n]:+.1f}" for f in fa) + " \\\\" for n in order]
rows.append("\\midrule\ncosine with real shift & -- & " + " & ".join(f"{rs['cosine_with_sim'][f]:.2f}" for f in fa) + " \\\\")
open("tables/tab_realshift.tex", "w").write("\\begin{tabular}{lcccccc}\n\\toprule\n & Real defective tool & Stiff. & Damp. & Wear & Depth & RPM \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("tables written")
