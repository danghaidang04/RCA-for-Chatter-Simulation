"""Hiệu chỉnh v2: mô hình nguyên nhân gốc theo bản chất vật lý (thay cho bộ trộn ngẫu nhiên v1).

 Tiến triển dần theo TUỔI DAO s∈[0,1] (thứ tự cắt tích lũy trong dữ liệu):
   TOOL_WEAR : Kt·(1 + a_w·s^b),  Kr·(1 + c_w·s^b)
   STIFFNESS : k·exp(−a_k·s^b)
 Sai lệch cài đặt CỐ ĐỊNH suốt công việc (không xấu dần):
   DAMPING : ζ·exp(−m_ζ)      DEPTH_OVERLOAD : a·exp(+m_a)      RPM_MISMATCH : n·exp(−m_n)
 Mỗi đường cắt bất thường thuộc một nguyên nhân (trọng số w, 6 lớp gồm NONE), nguyên nhân là biến ẩn.
Giai đoạn: A (máy, 11 tham số) -> C (biến thiên giữa các lần cắt: chỉ σ_Kt, σ_runout, σ_rough) -> B (mô hình nguyên nhân, 13 tham số).
Mỗi giai đoạn lưu checkpoint vào output/calibration_v2.json.   Chạy: python calibrate_v2.py
"""
import json

from calibrate_cma import *
from calibrate_cma import _one

SIGV = np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.5, 0.5])          # chỉ giữ 3 σ có lý do vật lý; còn lại = 0
NAMES_B = ["a_w", "c_w", "a_k", "b", "m_zeta", "m_a", "m_n", "w_STIFF", "w_DAMP", "w_WEAR", "w_DEPTH", "w_RPM", "w_NONE"]
LO_B = np.array([0, 0, 0, 0.3, 0, 0, 0] + [-3] * 6); HI_B = np.array([1.5, 3, 1, 3, 1.2, 0.8, 0.2] + [3] * 6)
CAUSE_ORDER = ["STIFFNESS", "DAMPING", "TOOL_WEAR", "DEPTH_OVERLOAD", "RPM_MISMATCH", "NONE"]


def tool_age(df):
    """Tuổi dao chuẩn hóa [0,1] theo thời gian cắt tích lũy, thứ tự (dataset, đường cắt). Các cửa sổ cùng một đường cắt có cùng tuổi."""
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "msm", "imi_vm20i")
    cuts = df[["dataset", "cut"]].drop_duplicates().sort_values(["dataset", "cut"]).reset_index(drop=True)
    dur = []
    for r in cuts.itertuples():
        lab = pd.read_csv(f"{root}/dataset{r.dataset}/label.csv"); dur.append(float(lab.end[r.cut] - lab.start[r.cut]))
    cuts["age"] = np.cumsum(dur) / np.sum(dur)
    return df.merge(cuts, on=["dataset", "cut"], how="left")["age"].values


def effects(cause, thB, s):
    a_w, c_w, a_k, b, m_z, m_a, m_n = thB[:7]; sb = s ** b
    if cause == "TOOL_WEAR":
        return {"Kt": 1 + a_w * sb, "Kr": 1 + c_w * sb}
    if cause == "STIFFNESS":
        return {"k": float(np.exp(-a_k * sb))}
    if cause == "DAMPING":
        return {"zeta": float(np.exp(-m_z))}
    if cause == "DEPTH_OVERLOAD":
        return {"a": float(np.exp(m_a))}
    if cause == "RPM_MISMATCH":
        return {"rpm": float(np.exp(-m_n))}
    return {}


def fault_list(thB, ages, rng, forced=None):
    w = np.exp(thB[7:13] - thB[7:13].max()); w /= w.sum()
    out = []
    for s in ages:
        c = forced if forced is not None else CAUSE_ORDER[rng.choice(6, p=w)]
        out.append(effects(c, thB, float(s)))
    return out


def sim_v2(cuts, th, faults, seed, pool, sig):
    jobs = [(c, th, faults[i] if faults is not None else None, seed * 1000 + i, sig) for i, c in enumerate(cuts)]
    return np.array(pool.map(_one, jobs))


def save(d):
    json.dump(d, open(f"{OUT}/calibration_v2.json", "w"), indent=1)


def main(maxiter=80, popsize=12):
    df = pd.read_csv(f"{OUT}/real_features.csv"); df["age"] = tool_age(df)
    dn, da = df[df.label == 0].reset_index(drop=True), df[df.label == 1].reset_index(drop=True)
    Rn, Ra = real_matrix(dn), real_matrix(da); scale = Rn.std(0) + 1e-6
    cn, ca = cond_matrix(cuts_of(dn)), cond_matrix(cuts_of(da)); cs = np.vstack([cn, ca]).std(0) + 1e-6
    cutn, cuta = cuts_of(dn), cuts_of(da)
    res = {"n_normal": len(dn), "n_anomaly": len(da)}
    with Pool(10) as pool:
        dec = lambda u: LO_M + u * (HI_M - LO_M)
        fA = lambda u: mmd2(sim_v2(cutn, dec(u), None, 0, pool, None), Rn, scale, cn, cn, cs)
        u0 = (X0_M - LO_M) / (HI_M - LO_M); lossA0 = fA(u0)
        ub, fb, hA, solsA = best_of(fA, u0, maxiter, popsize, "[A machine]", seeds=(1, 2)); th = dec(ub)
        print("MACHINE", dict(zip(NAMES_M, np.round(th, 3))), f"loss {lossA0:.4f} -> {fb:.4f}", flush=True)
        res.update(machine=dict(zip(NAMES_M, th.tolist())), machine_loss=[lossA0, fb], hist_A=hA,
                   seeds_machine=[dec(u).tolist() for u, _, _ in solsA], seeds_machine_loss=[f for _, f, _ in solsA]); save(res)
        decS = lambda u: u * SIGV
        fC = lambda u: mmd2(sim_v2(cutn, th, None, 2, pool, decS(u)), Rn, scale, cn, cn, cs)
        uS0 = np.full(7, 0.15); lossC0 = fC(uS0)
        ubC, fbC, hC, solsC = best_of(fC, uS0, maxiter, popsize, "[C sigma]", seeds=(1, 2)); sig = decS(ubC)
        print("SIGMA", dict(zip(["k", "zeta", "Kt", "a", "rpm", "runout", "rough"], np.round(sig, 3))), f"loss {lossC0:.4f} -> {fbC:.4f}", flush=True)
        res.update(sigma=sig.tolist(), sigma_loss=[lossC0, fbC], hist_C=hC, seeds_sigma=[decS(u).tolist() for u, _, _ in solsC]); save(res)
        ages = da.age.values; decB = lambda u: LO_B + u * (HI_B - LO_B)

        def fB(u):
            thB = decB(u); rng = np.random.default_rng(1)                      # số ngẫu nhiên chung (CRN) cho mọi ứng viên
            return mmd2(sim_v2(cuta, th, fault_list(thB, ages, rng), 1, pool, sig), Ra, scale, ca, ca, cs)
        uB0 = (np.array([0.5, 1.0, 0.3, 1.0, 0.3, 0.3, 0.05] + [0.0] * 6) - LO_B) / (HI_B - LO_B); lossB0 = fB(uB0)
        ubB, fbB, hB, solsB = best_of(fB, uB0, maxiter, popsize, "[B faults]", seeds=(1, 2)); thB = decB(ubB)
        w = np.exp(thB[7:13] - thB[7:13].max()); w /= w.sum()
        print("FAULTS", dict(zip(NAMES_B[:7], np.round(thB[:7], 3))), "weights", dict(zip(CAUSE_ORDER, np.round(w, 3))), f"loss {lossB0:.4f} -> {fbB:.4f}", flush=True)
        res.update(faults=dict(zip(NAMES_B, thB.tolist())), faults_theta=thB.tolist(), fault_weights=dict(zip(CAUSE_ORDER, w.tolist())),
                   faults_loss=[lossB0, fbB], hist_B=hB, seeds_faults=[decB(u).tolist() for u, _, _ in solsB], seeds_faults_loss=[f for _, f, _ in solsB]); save(res)
        Sn = sim_v2(cutn, th, None, 5, pool, sig); Sa = sim_v2(cuta, th, fault_list(thB, ages, np.random.default_rng(6)), 6, pool, sig)
    np.savez(f"{OUT}/calibration_v2_features.npz", Rn=Rn, Ra=Ra, Sn=Sn, Sa=Sa, age=ages)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
