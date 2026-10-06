"""Bước cuối: chạy thuật toán RCA (Bayesian Network của repo: rca_chatter_bayesian.BayesianNetworkRCA)
trên đặc trưng rút ra từ 4 cảm biến mô phỏng, cho 5 nguyên nhân gốc + Normal, ở 3 mức tác động.
Ngưỡng bằng chứng được học từ dữ liệu NORMAL (μ+3σ), không đặt tay.
Chạy: python exp4_rca.py
"""
import json, sys, os
from common import *
from faults import FAULTS, TRUTH_BN, apply_fault, jitter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rca_chatter_bayesian import BayesianNetworkRCA  # noqa: E402

A0, N0 = 4.0e-3, 5300.0
N_NORMAL, N_PER = 30, 12


def main():
    base = replace(baseline_process(), rpm=N0, a=A0)
    r, a, _ = lobes(base)
    env_min = float(np.min(a_lim_at(r, a, np.linspace(3000, 12000, 600), bins=1000)))
    rng = np.random.default_rng(11)

    def sample(proc, seed):
        p = jitter(proc, rng)
        _, _, ft = run(p, seed=seed, t_end=0.6)
        return p, ft

    normals = [sample(base, s) for s in range(N_NORMAL)]
    stat = {k: (np.mean([f[k] for _, f in normals]), np.std([f[k] for _, f in normals]) + 1e-9)
            for k in normals[0][1]}
    print('NORMAL stats', {k: (round(m, 3), round(sd, 3)) for k, (m, sd) in stat.items() if k.startswith(('f_res', 'vib_rms', 'F_mean', 'mic_spl'))})
    z = lambda ft, k: (ft[k] - stat[k][0]) / stat[k][1]

    def evidence(p, ft):
        a_lim_n = float(a_lim_at(r, a, p.rpm, bins=1000))
        return {
            "High_Vibration": bool(z(ft, "vib_rms_um") > 3 and z(ft, "mic_spl_db") > 2),
            "Chatter_Regenerative_Peak": bool(ft["chatter_vib"] > 0.3 or ft["chatter_acc"] > 0.8),
            "Resonance_Frequency_Shift": bool(abs(z(ft, "f_res_acc")) > 4 or abs(z(ft, "f_res_vib")) > 4),
            "High_Cutting_Force": bool(z(ft, "F_mean_n") > 3),
            "Programmed_Depth_High": bool(p.a > 1.06 * A0),           # a trong G-code vượt kế hoạch công nghệ >6% (jitter ±2%)
            "RPM_In_Lobe_Valley": bool(a_lim_n < 1.05 * env_min),     # n đặt trùng thung lũng lobe
        }

    bn = BayesianNetworkRCA()
    rows, ev_tab = [], {}
    for truth in ["NORMAL"] + FAULTS:
        for level in (["none"] if truth == "NORMAL" else ["small", "medium", "large"]):
            h1 = h3 = 0
            evs = []
            for s in range(N_PER):
                proc = base if truth == "NORMAL" else apply_fault(base, truth, level)
                p, ft = sample(proc, 1000 + s)
                ev = evidence(p, ft); evs.append(ev)
                post = bn.compute_posterior(ev); rk = list(post)
                h1 += rk[0] == TRUTH_BN[truth]; h3 += TRUTH_BN[truth] in rk[:3]
            ev_tab[f"{truth}_{level}"] = {k: float(np.mean([e[k] for e in evs])) for k in evs[0]}
            rows.append({"truth": truth, "level": level, "top1": h1 / N_PER, "top3": h3 / N_PER})
            print(f"{truth:15s} {level:7s} top1={h1/N_PER:.2f} top3={h3/N_PER:.2f}  evidence rate:",
                  {k[:10]: round(v, 2) for k, v in ev_tab[f'{truth}_{level}'].items()})
    json.dump({"rows": rows, "evidence_rate": ev_tab, "env_min_mm": env_min * 1e3}, open(f"{OUT}/exp4_rca.json", "w"), indent=1)

    # hình: Top-1 theo fault x level
    fig, ax = plt.subplots(figsize=(8, 4))
    names = ["NORMAL"] + FAULTS; lv = ["none", "small", "medium", "large"]
    M = np.full((len(names), 4), np.nan)
    for rw in rows:
        M[names.index(rw["truth"]), lv.index(rw["level"])] = rw["top1"]
    im = ax.imshow(M, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(lv); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(M.shape[0]):
        for j in range(4):
            if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center")
    ax.grid(False); ax.set_title("RCA (Bayesian network) Top-1 accuracy from 4-sensor evidence"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(f"{OUT}/exp4_rca_top1.png"); plt.close(fig)


if __name__ == "__main__":
    main()
