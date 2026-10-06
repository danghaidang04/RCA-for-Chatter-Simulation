import os, sys
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from milling_model import Process, make_modes, simulate_milling, stability_lobes_zoa, a_lim_at  # noqa: E402
from sensors import apply_sensors, SensorConfig  # noqa: E402
from features import extract, spectrum  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
C = {"x": "#1f77b4", "y": "#d62728", "mic": "#2ca02c", "acc": "#9467bd", "F": "#ff7f0e"}

T_END, FS, T_SKIP = 0.6, 25600.0, 0.4


def baseline_process(**kw):
    mx, my = make_modes()
    return Process(mod_x=mx, mod_y=my, phi_st=np.pi / 2, phi_ex=np.pi, runout=5e-6, **kw)  # 50% immersion, down-milling


def lobes(proc, **kw):
    return stability_lobes_zoa(proc, fc_range=(200, 6500), n_fc=20000, n_lobes=60, alim_max=0.08, **kw)


def run(proc, seed=0, t_end=T_END, sensor_cfg=SensorConfig(), noise_um=0.02, skip=T_SKIP):
    sim = simulate_milling(proc, t_end=t_end, fs=FS, seed=seed, noise_um=noise_um)
    sig = apply_sensors(sim, sensor_cfg, seed=seed)
    return sim, sig, extract(sig, proc.rpm, proc.N, skip)


def plot_four_sensors(sig, proc, title, path, t_win=(0.50, 0.54), spec_fmax=5000.0, feats=None):
    """4 loại biểu đồ (hàng) x [miền thời gian | phổ]."""
    fs, t = sig["fs"], sig["t"]
    m = (t >= t_win[0]) & (t <= t_win[1])
    i0 = int(T_SKIP * len(t))
    fig, ax = plt.subplots(4, 2, figsize=(12, 9.5), gridspec_kw={"width_ratios": [1.5, 1]})
    rows = [("Relative tool-work vibration", "µm", [("vib_x", "x", C["x"]), ("vib_y", "y", C["y"])]),
            ("Accelerometer", "g", [("acc_x", "ax", C["acc"]), ("acc_y", "ay", "#8c564b")]),
            ("Microphone", "Pa", [("mic", "p", C["mic"])]),
            ("Force dynamometer (workpiece)", "N", [("F_x", "Fx", C["F"]), ("F_y", "Fy", "#7f7f7f")])]
    f0 = proc.rpm / 60.0
    for r, (name, unit, chans) in enumerate(rows):
        for key, lab, col in chans:
            ax[r, 0].plot(t[m] * 1e3, sig[key][m], color=col, lw=0.9, label=lab)
            f, X = spectrum(sig[key][i0:], fs)
            s = f <= spec_fmax
            ax[r, 1].semilogy(f[s], X[s] + 1e-12, color=col, lw=0.8, label=lab)
        ax[r, 0].set_ylabel(f"{name}\n[{unit}]"); ax[r, 0].legend(loc="upper right", ncol=2, fontsize=7)
        ax[r, 1].set_ylabel("|X| [" + unit + "]")
        for k in range(1, int(spec_fmax / (proc.N * f0)) + 1):  # hài tooth-passing
            ax[r, 1].axvline(k * proc.N * f0, color="k", lw=0.4, ls=":", alpha=0.6)
    ax[3, 0].set_xlabel("time [ms]"); ax[3, 1].set_xlabel("frequency [Hz]  (dotted = tooth-passing harmonics)")
    sub = (f"n={proc.rpm:.0f} rpm, a={proc.a*1e3:.2f} mm, ae/D=50% down-milling")
    if feats:
        sub += f"   |  vib RMS={feats['vib_rms_um']:.2f} µm, SPL={feats['mic_spl_db']:.0f} dB, F̄={feats['F_mean_n']:.0f} N, chatter idx={feats['chatter_vib']:.2f}"
    fig.suptitle(f"{title}\n{sub}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path); plt.close(fig)


def process_from_cut(cut, **kw):
    """Điều kiện cắt thật (một dòng label.csv) -> Process. Phay thuận/nghịch theo ae/D, ft = F/(N·Z).
    cut: dict(N rpm, ap mm, ae mm, F mm/s, Z, D mm, direction 'UP'|'DOWN')."""
    mx, my = make_modes()
    D, ae = cut["D"], cut["ae"]
    span = float(np.arccos(1 - 2 * ae / D))                      # góc ăn dao
    st, ex = (0.0, span) if cut["direction"] == "UP" else (np.pi - span, np.pi)
    ft = cut["F"] * 60.0 / (cut["N"] * cut["Z"]) * 1e-3          # m/răng
    kw.setdefault("D_tool", D * 1e-3)
    return Process(N=int(cut["Z"]), rpm=float(cut["N"]), a=cut.get("a_eff", cut["ap"] * 1e-3), ft=ft, phi_st=st, phi_ex=ex,
                   runout=5e-6, mod_x=mx, mod_y=my, **kw)
