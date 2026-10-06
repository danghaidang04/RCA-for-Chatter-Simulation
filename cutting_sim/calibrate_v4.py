"""Hiệu chỉnh v4 (mô hình dao thật + xoắn + lực cạnh). Input = điều kiện cắt thật; loss ghép cặp theo nhóm điều kiện.
 GĐ1: 15 tham số học được (runout cố định 5 µm — độ nhạy ~0) khớp TRUNG BÌNH nhóm.  GĐ2: σ_Kt, σ_rough khớp sd TRONG nhóm.
 Cổng: C2ST trên NORMAL. Chạy: python calibrate_v4.py"""
import json, os
from model_v4 import *
from calibrate_v3 import groups, group_stats
from calibrate_cma import best_of, mmd2
from c2st import c2st

R_SIM = 2
ACTIVE = np.arange(15)          # chỉ số tham số học (15 = runout cố định)


def main(maxiter=60, popsize=14):
    df = pd.read_csv(f"{OUT}/real_features.csv"); dn = df[df.label == 0].reset_index(drop=True)
    import os
    if os.environ.get("DIR_FILTER"):                       # hướng C: chỉ một hướng phay (thuận) để tách cơ chế
        dn = dn[dn.direction == os.environ["DIR_FILTER"]].reset_index(drop=True)
    print(f"khớp trên {len(dn)} cửa sổ, hướng={os.environ.get('DIR_FILTER', 'tất cả')}", flush=True)
    cuts = cuts_of(dn); Rn = real_matrix(dn); gid = groups(dn); G = gid.max() + 1
    mu_r, sd_r = group_stats(Rn, gid, G); s_j = Rn.std(0); res = {}
    with Pool(10) as pool:
        def sims(th, sig, seed):
            return np.vstack([sim4(cuts, th, seed + r, pool, sig) for r in range(R_SIM)]), np.tile(gid, R_SIM)
        dec = lambda u: np.r_[LO[ACTIVE] + u * (HI[ACTIVE] - LO[ACTIVE]), X0[15:]]

        def f1(u):
            S, g2 = sims(dec(u), None, 0); mu_s, _ = group_stats(S, g2, G)
            return float(np.mean(((mu_s - mu_r) / s_j) ** 2))
        u0 = (X0[ACTIVE] - LO[ACTIVE]) / (HI[ACTIVE] - LO[ACTIVE]); l0 = f1(u0)
        ub, fb, h1, _ = best_of(f1, u0, maxiter, popsize, "[1 mean-structure]", seeds=(1,)); th = dec(ub)
        print("THETA", {n: round(float(v), 3) for n, v in zip(NAMES, th)}, f"loss {l0:.4f} -> {fb:.4f}", flush=True)
        res.update(theta=th.tolist(), names=NAMES, loss1=[l0, fb], hist1=h1); json.dump(res, open(f"{OUT}/calibration_v4{os.environ.get('TAG', '')}.json", "w"), indent=1)
        decS = lambda u: u * np.array([0.3, 0.0, 0.5])

        def f2(u):
            S, g2 = sims(th, decS(u), 7); _, sd_s = group_stats(S, g2, G)
            return float(np.mean((np.log(sd_s + 0.1 * sd_r) - np.log(1.1 * sd_r)) ** 2))
        uS0 = np.array([0.15, 0.0, 0.15]); l20 = f2(uS0)
        uc, fc, h2, _ = best_of(f2, uS0, maxiter, popsize, "[2 within-group noise]", seeds=(1,)); sig = decS(uc)
        S, g2 = sims(th, sig, 9); _, sd_s = group_stats(S, g2, G)
        print("SIGMA", np.round(sig, 3), f"loss {l20:.4f} -> {fc:.4f}", "sd trong nhóm: thật", np.round(sd_r, 3), "sim", np.round(sd_s, 3), flush=True)
        res.update(sigma=sig.tolist(), loss2=[l20, fc], hist2=h2); json.dump(res, open(f"{OUT}/calibration_v4{os.environ.get('TAG', '')}.json", "w"), indent=1)
        grp = (dn.dataset * 1000 + dn.cut).values; cr = cond_matrix(cuts)
        for name, t_, s_ in (("default (initial θ)", X0, None), ("v4 mean-structure", th, None), ("v4 + noise", th, sig)):
            Ssim = np.vstack([sim4(cuts, t_, 900 + r, pool, s_) for r in range(5)])
            acc, sd_ = c2st(Rn, Ssim, cr, grp); res.setdefault("c2st_gate_normal", {})[name] = [acc, sd_]
            print(f"GATE normal {name:20s} C2ST = {acc:.3f} ± {sd_:.3f}", flush=True)
        mu_s, _ = group_stats(sims(th, sig, 11)[0], g2, G)
        print("tương quan trung bình nhóm thật vs sim (4 đặc trưng):", np.round([np.corrcoef(mu_r[:, j], mu_s[:, j])[0, 1] for j in range(4)], 3), flush=True)
        json.dump(res, open(f"{OUT}/calibration_v4{os.environ.get('TAG', '')}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
