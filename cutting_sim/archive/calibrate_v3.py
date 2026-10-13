"""Hiệu chỉnh v3 — đúng quy trình: input thật (điều kiện cắt) -> DDE + cảm biến (+nhiễu) -> output; loss GHÉP CẶP theo điều kiện với output thật.
 Giai đoạn 1 (phần xác định): tham số máy/cảm biến (11) khớp TRUNG BÌNH theo nhóm điều kiện: Σ_j mean_g ((μ_sim−μ_real)/s_j)².
 Giai đoạn 2 (nhiễu): σ giữa các lần cắt (7) khớp ĐỘ LỆCH CHUẨN TRONG NHÓM điều kiện: Σ_j (log sd_sim − log sd_real)².
 Cổng: C2ST trên dữ liệu NORMAL (sau cổng mới học nguyên nhân gốc).
Chạy: python calibrate_v3.py
"""
import json
from calibrate_v2 import *
from c2st import c2st

R_SIM = 2


def groups(df):
    key = df.N.astype(int).astype(str) + "_" + df.ae.astype(str) + "_" + df.F.round(1).astype(str) + "_" + df.direction
    return pd.factorize(key)[0]


def group_stats(M, gid, G):
    cnt = np.bincount(gid, minlength=G)[:, None]
    mu = np.stack([np.bincount(gid, M[:, j], minlength=G) for j in range(M.shape[1])], 1) / cnt
    var = np.stack([np.bincount(gid, (M[:, j] - mu[gid, j]) ** 2, minlength=G) for j in range(M.shape[1])], 1) / np.maximum(cnt - 1, 1)
    return mu, np.sqrt(var.mean(0))


def main(maxiter=60, popsize=14):
    df = pd.read_csv(f"{OUT}/real_features.csv"); dn = df[df.label == 0].reset_index(drop=True)
    cuts = cuts_of(dn); Rn = real_matrix(dn); gid = groups(dn); G = gid.max() + 1
    mu_r, sd_r = group_stats(Rn, gid, G); s_j = Rn.std(0)
    print(f"{len(dn)} cửa sổ, {G} nhóm điều kiện; sd trong nhóm (thật) = {np.round(sd_r, 3)}; sd tổng = {np.round(s_j, 3)}", flush=True)
    res = {}
    with Pool(10) as pool:
        def sims(th, sig, seed):
            return np.vstack([sim_v2(cuts, th, None, seed + r, pool, sig) for r in range(R_SIM)]), np.tile(gid, R_SIM)
        dec = lambda u: LO_M + u * (HI_M - LO_M)

        def f1(u):
            S, g2 = sims(dec(u), None, 0); mu_s, _ = group_stats(S, g2, G)
            return float(np.mean(((mu_s - mu_r) / s_j) ** 2))
        u0 = (X0_M - LO_M) / (HI_M - LO_M); l0 = f1(u0)
        ub, fb, h1, sols = best_of(f1, u0, maxiter, popsize, "[1 mean-structure]", seeds=(1,)); th = dec(ub)
        print("MACHINE", dict(zip(NAMES_M, np.round(th, 3))), f"loss {l0:.4f} -> {fb:.4f}", flush=True)
        res.update(machine=dict(zip(NAMES_M, th.tolist())), machine_loss=[l0, fb], hist_A=h1); json.dump(res, open(f"{OUT}/calibration_v3.json", "w"), indent=1)

        decS = lambda u: u * np.array([0.3, 0.3, 0.5, 0.3, 0.05, 1.0, 1.0])

        def f2(u):
            S, g2 = sims(th, decS(u), 7); _, sd_s = group_stats(S, g2, G)
            return float(np.mean((np.log(sd_s + 0.1 * sd_r) - np.log(sd_r * 1.0 + 0.1 * sd_r)) ** 2))
        uS0 = np.full(7, 0.1); l20 = f2(uS0)
        uc, fc, h2, _ = best_of(f2, uS0, maxiter, popsize, "[2 within-group noise]", seeds=(1,)); sig = decS(uc)
        S, g2 = sims(th, sig, 9); _, sd_s = group_stats(S, g2, G)
        print("SIGMA", np.round(sig, 3), f"loss {l20:.4f} -> {fc:.4f}", "sd trong nhóm sim", np.round(sd_s, 3), flush=True)
        res.update(sigma=sig.tolist(), sigma_loss=[l20, fc], hist_C=h2); json.dump(res, open(f"{OUT}/calibration_v3.json", "w"), indent=1)
        # cổng C2ST (chỉ NORMAL)
        grp = (dn.dataset * 1000 + dn.cut).values; cr = cond_matrix(cuts)
        for name, t_, s_ in (("uncalibrated", X0_M, np.zeros(7)), ("v3: mean-structure", th, np.zeros(7)), ("v3: + noise", th, sig)):
            Ssim = np.vstack([sim_v2(cuts, t_, None, 900 + r, pool, s_) for r in range(5)])
            acc, sd_ = c2st(Rn, Ssim, cr, grp); res.setdefault("c2st_gate_normal", {})[name] = [acc, sd_]
            print(f"GATE normal {name:20s} C2ST = {acc:.3f} ± {sd_:.3f}", flush=True)
        mu_s, _ = group_stats(sims(th, sig, 11)[0], g2, G)
        print("tương quan trung bình nhóm thật vs sim:", np.round([np.corrcoef(mu_r[:, j], mu_s[:, j])[0, 1] for j in range(4)], 3), flush=True)
        json.dump(res, open(f"{OUT}/calibration_v3.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
