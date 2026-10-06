"""Nhóm thực nghiệm 2: tác động nhỏ lên từng root cause, xem 4 loại cảm biến thay đổi thế nào.
Chạy: python exp2_faults.py
"""
import json
from common import *
from faults import FAULTS, LEVELS, apply_fault

A0, N0 = 4.0e-3, 5300.0   # danh định: đỉnh lobe (a_lim~7.1 mm); thung lũng tại 5000 và 5500 rpm
KEYS = ["vib_rms_um", "acc_rms_g", "mic_spl_db", "F_mean_n", "F_rms_dyn_n", "F_peak_n", "chatter_vib", "chatter_acc", "chatter_mic"]


def main():
    base = baseline_process(); base = replace(base, rpm=N0, a=A0)
    sim0, sig0, f0 = run(base, seed=5)
    plot_four_sensors(sig0, base, "Exp.2 — NOMINAL (no fault)", f"{OUT}/exp2_nominal.png", feats=f0)
    res = {"nominal": f0}
    # n lặp lại để có sai số (seed khác -> nhiễu nhám/cảm biến khác)
    def mean_feats(p, seeds=range(3)):
        fs_ = [run(p, seed=s)[2] for s in seeds]
        return {k: float(np.mean([f[k] for f in fs_])) for k in fs_[0]}
    f0m = mean_feats(base)
    rel = {}
    for fault in FAULTS:
        rel[fault] = {}
        for level in ("small", "medium", "large"):
            p = apply_fault(base, fault, level)
            ft = mean_feats(p)
            res[f"{fault}_{level}"] = ft
            rel[fault][level] = {k: ft[k] / f0m[k] if k.startswith(("vib", "acc_rms", "F_", "mic_rms")) else ft[k] - f0m[k] for k in KEYS}
            print(fault, level, {k: round(ft[k], 2) for k in ("vib_rms_um", "mic_spl_db", "F_mean_n", "chatter_vib", "f_dom_vib")})
        # hình 4 cảm biến: nominal vs fault (medium) chồng phổ + thời gian
        p = apply_fault(base, fault, "medium")
        _, sig, ft = run(p, seed=5)
        plot_four_sensors(sig, p, f"Exp.2 — {fault} (medium)", f"{OUT}/exp2_{fault.lower()}_medium.png", feats=ft)
        _, sigs, fts = run(apply_fault(base, fault, "small"), seed=5)
        plot_four_sensors(sigs, apply_fault(base, fault, "small"), f"Exp.2 — {fault} (SMALL gap)", f"{OUT}/exp2_{fault.lower()}_small.png", feats=fts)

    # heatmap thay đổi tương đối của đặc trưng (4 sensor) theo fault x level
    cols = ["vib_rms_um", "acc_rms_g", "mic_spl_db", "F_mean_n", "F_rms_dyn_n", "chatter_vib"]
    names = ["Vib RMS", "Acc RMS", "Mic SPL", "F mean", "F dyn RMS", "chatter idx"]
    rows, lab = [], []
    for fault in FAULTS:
        for level in ("small", "medium", "large"):
            r = []
            for k in cols:
                a, b = res[f"{fault}_{level}"][k], f0m[k]
                r.append((a - b) if k in ("mic_spl_db", "chatter_vib") else 100 * (a / b - 1))
            rows.append(r); lab.append(f"{fault[:9]}-{level}")
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(np.sign(M) * np.log10(1 + np.abs(M)), cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            u = " dB" if cols[j] == "mic_spl_db" else ("" if cols[j] == "chatter_vib" else "%")
            ax.text(j, i, f"{M[i, j]:+.1f}{u}" if cols[j] != "chatter_vib" else f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False); ax.set_title("Exp.2 — change vs nominal (sign·log10(1+|Δ|)); numbers = % (dB for SPL)")
    fig.tight_layout(); fig.savefig(f"{OUT}/exp2_feature_changes.png"); plt.close(fig)
    json.dump({"nominal_mean": f0m, "runs": res}, open(f"{OUT}/exp2_features.json", "w"), indent=1)


if __name__ == "__main__":
    main()
