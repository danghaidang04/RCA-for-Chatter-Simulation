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
print("tables written")
