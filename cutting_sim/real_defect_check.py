"""Kiểm chứng THẬT: dao hỏng có chủ ý (D*, dataset 3, nhãn 2) so với dao khỏe cùng loại (D, dataset 14), thép AISI 4140.
 1) Trích 30 đặc trưng -> 9 nút cảm biến cho mỗi cửa sổ 0.3 s.  2) Mô hình điều kiện từ dữ liệu KHỎE (ridge theo log N, log ae, log F, hướng), phần dư ngoài mẫu theo đường cắt.
 3) Mẫu lệch của dao hỏng: trung bình z = (nút - dự đoán)/sd_phần_dư theo nút, kèm khoảng tin cậy bootstrap theo đường cắt.
 4) So với mẫu lệch của từng nguyên nhân MÔ PHỎNG (cosine): nếu mô phỏng đúng, dao hỏng thật phải giống TOOL_WEAR nhất.
 Lưu ý: dao D (12.7 mm, 4 rãnh) khác dao A đã hiệu chỉnh nên so sánh chỉ ở mức MẪU LỆCH TƯƠNG ĐỐI trong đơn vị độ lệch chuẩn khỏe."""
import json
import numpy as np
from sklearn.model_selection import GroupKFold
import rca_data as rd
import rca_causal as rc


def feats_of(npz):
    d = np.load(f"{rd.OUT}/{npz}"); meta = d["meta"]; fs_a, sr = float(d["fs_a"]), float(d["sr"])
    F = np.array([rd.rca_features(d["acc0"][i], d["acc1"][i], d["mic"][i], fs_a, sr, meta[i]) for i in range(len(meta))]); return meta, rc.sens_nodes(F)


def cond(meta):
    lN = np.log(meta[:, 5]) - 8.0; return np.c_[np.ones(len(meta)), lN, lN ** 2, np.log(meta[:, 6]), np.log(meta[:, 7]), meta[:, 4]]


def main():
    meta, Z = feats_of("real_windows_defect.npz"); healthy = meta[:, 0] == 14; bad = meta[:, 0] == 3
    Xh, Zh, Xb, Zb = cond(meta[healthy]), Z[healthy], cond(meta[bad]), Z[bad]; cid = (meta[healthy][:, 0] * 1000 + meta[healthy][:, 1]).astype(int)
    lam = 1.0; fit = lambda X, Y: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y); oof = np.zeros_like(Zh)
    for a_, b_ in GroupKFold(5).split(Xh, groups=cid): oof[b_] = Xh[b_] @ fit(Xh[a_], Zh[a_])
    sd = (Zh - oof).std(0) + 1e-9; beta = fit(Xh, Zh); zb = (Zb - Xb @ beta) / sd; zh = (Zh - oof) / sd
    cb = (meta[bad][:, 0] * 1000 + meta[bad][:, 1]).astype(int); ub = np.unique(cb); rng = np.random.default_rng(0)
    boot = np.array([zb[np.isin(cb, rng.choice(ub, len(ub)))].mean(0) for _ in range(500)])
    real_vec = zb.mean(0)
    print("MẪU LỆCH THẬT của dao hỏng D* (đơn vị sd khỏe; [khoảng tin cậy 95% bootstrap theo đường cắt]):")
    for j, n in enumerate(rc.SENS): print(f"  {n:12s} {real_vec[j]:+6.2f}  [{np.percentile(boot[:, j], 2.5):+.2f}, {np.percentile(boot[:, j], 97.5):+.2f}]")
    # mẫu lệch mô phỏng của từng nguyên nhân (tại cùng điều kiện, chuẩn hóa theo sd khỏe mô phỏng)
    dm, metaA, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(metaA, TH, fs_a, sr); g = json.load(open(f"{rd.OUT}/rca_gain_effect.json")); cidA = (metaA[:, 0] * 1000 + metaA[:, 1]).astype(int)
    Fn, Ln = rc.gen_set(gen, metaA, cidA, fs_a, sr, n_a, n_m, g, "nominal", 11); Fl = Fn.reshape(-1, Fn.shape[-1]); cl = np.tile(np.arange(6), len(Fn)); Zs = rc.sens_nodes(Fl)
    sdn = Zs[cl == 0].std(0) + 1e-9; base = Zs[cl == 0].mean(0); sim_vec = {c: ((Zs[cl == k] - base) / sdn).mean(0) for k, c in enumerate(rd.CLASSES) if k > 0}
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print("\nđộ tương đồng cosine giữa mẫu lệch THẬT (dao hỏng) và mẫu lệch từng nguyên nhân MÔ PHỎNG:")
    sims = {c: cos(real_vec, v) for c, v in sim_vec.items()}
    for c, v in sorted(sims.items(), key=lambda x: -x[1]): print(f"  {c:16s} {v:+.2f}")
    bc = {c: [] for c in sims}
    for b in boot:
        for c, v in sim_vec.items(): bc[c].append(cos(b, v))
    print("tỉ lệ bootstrap mà từng nguyên nhân là giống nhất:", {c: float(np.mean([max(bc, key=lambda k: bc[k][i]) == c for i in range(len(boot))])) for c in sims})
    json.dump({"real_shift": dict(zip(rc.SENS, real_vec.tolist())), "cosine_with_sim": sims}, open(f"{rd.OUT}/real_defect_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
