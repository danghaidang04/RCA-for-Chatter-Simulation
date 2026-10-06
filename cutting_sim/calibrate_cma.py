"""Hiệu chỉnh mô phỏng bằng CMA-ES trên dữ liệu thật MSM: khớp phân bố đặc trưng (MMD).

Giai đoạn A (máy)   : 8 tham số khớp trên đường cắt NORMAL (label 0)
   [ln gain_acc, ln noise_acc(g), ln gain_mic, ln noise_mic(Pa), ln s_k, ln s_zeta, ln s_Kt, ln a_eff/19.05mm]
Giai đoạn B (lỗi)   : phân bố tác động của 5 root cause + 'none' (không giải thích được) khớp trên ANOMALY (label 1)
   [m_c (5), logit trọng số w_c (6), σ]  -- magnitude = exp(m_c + σ·ε), kẹp trong BOUNDS
Đặc trưng: log10 acc_rms_g, mic dB (re 1 Pa trong mô phỏng / dBFS ở dữ liệu thật; chênh lệch hấp thụ bởi gain_mic),
           chatter_acc, chatter_mic (tỷ lệ năng lượng phổ phi đồng bộ).
"""
import json, os, sys
from multiprocessing import Pool

import cma
import pandas as pd
from common import *
from faults import FAULTS, apply_fault

T_SIM = 0.60          # mô phỏng 0.6 s, phân tích nửa sau = cửa sổ 0.3 s (= cửa sổ đặc trưng thật, real_features.W)
LOG_AEFF0 = np.log(19.05e-3)
NAMES_M = ["ln g_acc", "ln noise_acc", "ln g_mic", "ln noise_mic", "ln s_k", "ln s_zeta", "ln s_Kt", "ln a_eff", "ln U_imb", "acc_rpm_exp", "mic_rpm_exp"]
LO_M = np.array([-12, -12, -6, -10, -0.7, -0.7, -0.7, -2.5, -20, -2, -2]); HI_M = np.array([2, 0, 6, 0, 0.7, 0.7, 0.7, 0.3, -6, 8, 8])
X0_M = np.array([-6, -5, 0, -5, 0, 0, 0, -1.0, -13, 1.0, 1.0])
BOUNDS = {"STIFFNESS": (-0.8, 0.0), "DAMPING": (-1.2, 0.0), "TOOL_WEAR": (0.0, 0.6),
          "DEPTH_OVERLOAD": (0.0, 0.8), "RPM_MISMATCH": (-0.2, 0.0)}
CAUSES = FAULTS + ["NONE"]


def feat_vec(ft):
    return np.array([np.log10(ft["acc_rms_g"] + 1e-12), ft["mic_spl_db"] - 94.0, ft["chatter_acc"], ft["chatter_mic"]])


def real_matrix(df):
    return np.c_[np.log10(df.acc_rms_g), df.mic_db_fs, df.chatter_acc, df.chatter_mic]


NAMES_S = ["σ_k", "σ_zeta", "σ_Kt", "σ_a_eff", "σ_rpm", "σ_runout", "σ_rough"]
SIG_MAX = np.array([0.30, 0.30, 0.30, 0.30, 0.05, 1.0, 1.0])   # cận trên (tương đối, log-normal)


def _one(args):
    cut, th, fault, seed, sig = args
    gl, na, gm, nm, lk, lz, lkt, la, lu, ea, em = th
    r = np.random.default_rng(seed + 31337)
    e = r.normal(size=7) * (sig if sig is not None else np.zeros(7))     # biến thiên giữa các lần cắt (log-normal)
    cut = dict(cut, a_eff=float(np.exp(la + e[3]) * 19.05e-3), N=float(cut["N"] * np.exp(e[4])))
    p = process_from_cut(cut)
    p = replace(p, Kt=p.Kt * np.exp(lkt + e[2]), runout=p.runout * np.exp(e[5]), imb=float(np.exp(lu + e[5])),
                mod_x=p.mod_x.modified(np.exp(lk + e[0]), np.exp(lz + e[1])), mod_y=p.mod_y.modified(np.exp(lk + e[0]), np.exp(lz + e[1])))
    if isinstance(fault, dict):            # mô hình v2: tác động tổng quát (nhân lên tham số)
        p = replace(p, Kt=p.Kt * fault.get("Kt", 1.0), Kr=p.Kr * fault.get("Kr", 1.0), a=p.a * fault.get("a", 1.0), rpm=p.rpm * fault.get("rpm", 1.0))
        if "k" in fault or "zeta" in fault:
            p = replace(p, mod_x=p.mod_x.modified(fault.get("k", 1.0), fault.get("zeta", 1.0)), mod_y=p.mod_y.modified(fault.get("k", 1.0), fault.get("zeta", 1.0)))
    elif fault is not None and fault[0] != "NONE":
        p = apply_fault(p, fault[0], scale=fault[1])
    cfg = SensorConfig(acc_gain=float(np.exp(gl)), acc_noise_g=float(np.exp(na)), mic_gain=float(np.exp(gm)), mic_noise_pa=float(np.exp(nm)), acc_rpm_exp=float(ea), mic_rpm_exp=float(em))
    _, _, ft = run(p, seed=seed, t_end=T_SIM, sensor_cfg=cfg, noise_um=0.02 * float(np.exp(e[6])), skip=0.5)
    return feat_vec(ft)


def cond_matrix(cuts):
    """Điều kiện cắt (chuẩn hóa) ghép vào đặc trưng: MMD chung P(điều kiện, đặc trưng) ép khớp cả quan hệ phụ thuộc điều kiện."""
    return np.array([[np.log(c["N"]), np.log(c["ae"]), np.log(c["F"]), float(c["direction"] == "UP")] for c in cuts])


def mmd2(X, Y, scale, CX=None, CY=None, cscale=None):
    X, Y = X / scale, Y / scale
    if CX is not None:
        X = np.c_[X, 2.0 * CX / cscale]; Y = np.c_[Y, 2.0 * CY / cscale]      # trọng số 2: điều kiện quan trọng như đặc trưng
    Z = np.vstack([X, Y]); d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0]) + 1e-9; n = len(X); tot = 0.0
    for m_ in (0.25, 1.0, 4.0):                                               # đa băng thông: nhạy cả với phương sai lẫn trung bình
        K = np.exp(-d2 / (m_ * med))
        tot += K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()
    return float(tot / 3)


def sim_features(cuts, th, sampler, seed, pool, sig=None):
    rng = np.random.default_rng(seed)
    jobs = [(c, th, sampler(rng) if sampler else None, seed * 1000 + i, sig) for i, c in enumerate(cuts)]
    return np.array(pool.map(_one, jobs))


def anomaly_sampler(th):
    m = th[:5]; w = np.exp(th[5:11] - th[5:11].max()); w /= w.sum(); sig = abs(th[11]) + 0.02

    def sample(rng):
        k = rng.choice(6, p=w)
        if k == 5:
            return ("NONE", 1.0)
        lo, hi = BOUNDS[FAULTS[k]]
        return (FAULTS[k], float(np.exp(np.clip(m[k] + sig * rng.normal(), lo, hi))))
    return sample


def run_cma(fun, u0, maxiter, popsize, sigma0=0.25, seed=1, log=""):
    es = cma.CMAEvolutionStrategy(u0, sigma0, {"bounds": [0, 1], "popsize": popsize, "seed": seed, "maxiter": maxiter, "tolfun": 1e-4, "tolx": 1e-3, "tolstagnation": 10, "verbose": -9})
    hist = []
    while not es.stop():
        us = es.ask(); fs = [fun(u) for u in us]; es.tell(us, fs); hist.append(float(min(fs)))
        print(log, es.countiter, f"best={min(fs):.4f}", flush=True)
    return es.result.xbest, es.result.fbest, hist


def best_of(fun, u0, maxiter, popsize, log, seeds=(1, 2, 3)):
    """Chạy CMA-ES với nhiều hạt giống độc lập; trả về nghiệm tốt nhất + tất cả nghiệm (để đánh giá độ nhận dạng)."""
    sols = []
    for sd in seeds:
        u, f, h = run_cma(fun, u0, maxiter, popsize, seed=sd, log=f"{log} s{sd}")
        sols.append((u, f, h))
    k = int(np.argmin([f for _, f, _ in sols]))
    return sols[k][0], sols[k][1], sols[k][2], sols


def cuts_of(df):
    return [dict(N=r.N, ap=r.ap, ae=r.ae, F=r.F, Z=r.Z, D=r.D, direction=r.direction) for r in df.itertuples()]


def main(maxiter=100, popsize=16):
    csv = f"{OUT}/real_features.csv"
    df = pd.read_csv(csv)
    dn, da = df[df.label == 0].reset_index(drop=True), df[df.label == 1].reset_index(drop=True)
    Rn, Ra = real_matrix(dn), real_matrix(da)
    scale = Rn.std(0) + 1e-6                                   # chuẩn hóa theo NORMAL: nhạy hơn với sai lệch nhỏ
    cn, ca = cond_matrix(cuts_of(dn)), cond_matrix(cuts_of(da)); cs = np.vstack([cn, ca]).std(0) + 1e-6
    print(f"real: {len(dn)} normal, {len(da)} anomaly cuts")
    with Pool(10) as pool:
        decode = dec = lambda u: LO_M + u * (HI_M - LO_M)
        fA = lambda u: mmd2(sim_features(cuts_of(dn), decode(u), None, 0, pool), Rn, scale, cn, cn, cs)
        u0 = (X0_M - LO_M) / (HI_M - LO_M)
        loss0 = fA(u0)
        ub, fb, hA, solsA = best_of(fA, u0, maxiter, popsize, "[A machine]")
        th = decode(ub)
        print("MACHINE", dict(zip(NAMES_M, np.round(th, 3))), f"loss {loss0:.4f} -> {fb:.4f}")
        # C: độ lệch chuẩn (biến thiên giữa các lần cắt) của từng biến, khớp trên NORMAL
        decS = lambda u: u * SIG_MAX
        fC = lambda u: mmd2(sim_features(cuts_of(dn), th, None, 2, pool, sig=decS(u)), Rn, scale, cn, cn, cs)
        uS0 = np.full(7, 0.15)
        lossC0 = fC(uS0)
        ubC, fbC, hC, solsC = best_of(fC, uS0, maxiter, popsize, "[C sigma]")
        sigma = decS(ubC)
        print("SIGMA", dict(zip(NAMES_S, np.round(sigma, 3))), f"loss {lossC0:.4f} -> {fbC:.4f}")
        # B: anomaly
        lo = np.r_[[BOUNDS[f][0] for f in FAULTS], [-3] * 6, 0.0]; hi = np.r_[[BOUNDS[f][1] for f in FAULTS], [3] * 6, 0.6]
        decB = lambda u: lo + u * (hi - lo)
        fB = lambda u: mmd2(sim_features(cuts_of(da), th, anomaly_sampler(decB(u)), 1, pool, sig=sigma), Ra, scale, ca, ca, cs)
        uB0 = (np.r_[[-0.2, -0.3, 0.2, 0.3, -0.05], np.zeros(6), 0.15] - lo) / (hi - lo)
        lossB0 = fB(uB0)
        ubB, fbB, hB, solsB = best_of(fB, uB0, maxiter, popsize, "[B anomaly]")
        thB = decB(ubB); w = np.exp(thB[5:11] - thB[5:11].max()); w /= w.sum()
        print("ANOMALY m", dict(zip(FAULTS, np.round(thB[:5], 3))), "weights", dict(zip(CAUSES, np.round(w, 3))), f"sigma {thB[11]:.2f}", f"loss {lossB0:.4f} -> {fbB:.4f}")
        # chẩn đoán: mô phỏng tốt nhất vs thật
        Sn = sim_features(cuts_of(dn), th, None, 5, pool, sig=sigma); Sa = sim_features(cuts_of(da), th, anomaly_sampler(thB), 6, pool, sig=sigma)
    json.dump({"machine": dict(zip(NAMES_M, th.tolist())), "machine_loss": [loss0, fb], "sigma": dict(zip(NAMES_S, sigma.tolist())), "sigma_loss": [lossC0, fbC], "hist_C": hC, "anomaly_theta": thB.tolist(),
               "anomaly_m": dict(zip(FAULTS, thB[:5].tolist())), "anomaly_weights": dict(zip(CAUSES, w.tolist())), "anomaly_sigma": float(thB[11]),
               "anomaly_loss": [lossB0, fbB], "seeds_machine": [dec(u).tolist() for u, _, _ in solsA], "seeds_machine_loss": [f for _, f, _ in solsA], "seeds_sigma": [decS(u).tolist() for u, _, _ in solsC], "seeds_sigma_loss": [f for _, f, _ in solsC], "hist_A": hA, "hist_B": hB}, open(f"{OUT}/calibration_result.json", "w"), indent=1)
    np.savez(f"{OUT}/calibration_features.npz", Rn=Rn, Ra=Ra, Sn=Sn, Sa=Sa)
    fn = ["log10 acc RMS [g]", "mic level [dB]", "chatter idx (acc)", "chatter idx (mic)"]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6))
    for j in range(4):
        for r_, (R, S, nm) in enumerate(((Rn, Sn, "normal"), (Ra, Sa, "anomaly"))):
            ax[r_, j].hist(R[:, j], bins=12, alpha=.55, label="real", color="tab:blue", density=True)
            ax[r_, j].hist(S[:, j], bins=12, alpha=.55, label="sim (fitted)", color="tab:orange", density=True)
            ax[r_, j].set_title(f"{nm}: {fn[j]}", fontsize=9)
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/calibration_fit.png")


if __name__ == "__main__":
    main()
