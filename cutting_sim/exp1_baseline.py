"""Nhóm thực nghiệm 1: (a) Stability Lobe -> chọn (a, n); (b) mô phỏng cắt; (c) 4 biểu đồ cảm biến.
Chạy: python exp1_baseline.py
"""
import json, time
from common import *


def main():
    proc = baseline_process()
    r, a, k = lobes(proc)
    # ----- (a) kiểm chứng chéo lobe với mô phỏng miền thời gian trên lưới (n, a)
    rpms = np.arange(3000, 12001, 750)
    a_grid = np.arange(0.5e-3, 8.01e-3, 0.5e-3)
    chat = np.zeros((len(a_grid), len(rpms)))
    t0 = time.time()
    for j, n in enumerate(rpms):
        for i, ad in enumerate(a_grid):
            _, _, ft = run(replace(proc, rpm=float(n), a=float(ad)), seed=1, t_end=0.5)
            chat[i, j] = ft["chatter_vib"]
    print(f"grid {chat.shape} in {time.time()-t0:.0f}s")
    env_rpm = np.linspace(3000, 12000, 800)
    env = a_lim_at(r, a, env_rpm, bins=1000)

    unstable_sim = chat > 0.5
    unstable_zoa = a_grid[:, None] > a_lim_at(r, a, rpms.astype(float), bins=1000)[None, :]
    agree = float(np.mean(unstable_sim == unstable_zoa))
    print("agreement ZOA vs time-domain:", agree)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    sel = (r > 2500) & (r < 12500)
    ax.plot(r[sel], a[sel] * 1e3, ".", ms=1.0, color="0.75", label="all lobes (ZOA)")
    ax.plot(env_rpm, env * 1e3, color="k", lw=1.8, label="stability boundary a_lim(n)")
    R, A = np.meshgrid(rpms, a_grid * 1e3)
    ax.scatter(R[~unstable_sim], A[~unstable_sim], c="tab:green", s=22, marker="o", label="time-domain: stable")
    ax.scatter(R[unstable_sim], A[unstable_sim], c="tab:red", s=26, marker="x", label="time-domain: chatter")
    ax.set(xlabel="spindle speed n [rpm]", ylabel="axial depth a [mm]", ylim=(0, 8.2), xlim=(2800, 12200),
           title=f"Stability lobes (ZOA, Altintas Ex.2 FRF) vs. DDE time-domain simulation — agreement {agree*100:.0f}%")
    ax.legend(loc="upper left", fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/exp1_lobes_validation.png"); plt.close(fig)

    # ----- (b) chọn 3 điểm vận hành từ lobe: ổn định / sát biên / chatter, cùng n
    n0 = 6000.0
    al = float(a_lim_at(r, a, n0, bins=1000))
    cases = {"stable": 0.5 * al, "borderline": 1.05 * al, "chatter": 1.8 * al}
    summary = {"a_lim_at_6000rpm_mm": al * 1e3, "grid_agreement": agree, "cases": {}}
    for name, ad in cases.items():
        sim, sig, ft = run(replace(proc, rpm=n0, a=ad), seed=3)
        plot_four_sensors(sig, replace(proc, rpm=n0, a=ad), f"Exp.1 — {name.upper()} cut", f"{OUT}/exp1_{name}.png", feats=ft)
        summary["cases"][name] = {"a_mm": ad * 1e3, **ft}
        print(name, {k: round(v, 3) for k, v in ft.items()})
    json.dump(summary, open(f"{OUT}/exp1_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
