"""Nút cảm biến THẬT của dao hỏng D* (MSM dataset 3) và dao khỏe D (dataset 14), tính như cutting_sim/real_defect_check.py:
phần dư ngoài mẫu (theo đường cắt) của mô hình ridge điều kiện -> đơn vị độ lệch chuẩn khỏe của DAO THẬT. Ghi data/real_nodes.npz."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CS = os.path.join(ROOT, "cutting_sim"); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CS); sys.path.insert(0, ROOT); os.chdir(CS)


def main():
    from sklearn.model_selection import GroupKFold
    import real_defect_check as rdc
    meta, Z = rdc.feats_of("real_windows_defect.npz"); healthy = meta[:, 0] == 14; bad = meta[:, 0] == 3
    Xh, Zh, Xb, Zb = rdc.cond(meta[healthy]), Z[healthy], rdc.cond(meta[bad]), Z[bad]; cid = (meta[healthy][:, 0] * 1000 + meta[healthy][:, 1]).astype(int)
    lam = 1.0; fit = lambda X, Y: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y); oof = np.zeros_like(Zh)
    for a_, b_ in GroupKFold(5).split(Xh, groups=cid): oof[b_] = Xh[b_] @ fit(Xh[a_], Zh[a_])
    sd = (Zh - oof).std(0) + 1e-9; beta = fit(Xh, Zh); zb = (Zb - Xb @ beta) / sd; zh = (Zh - oof) / sd
    cb = (meta[bad][:, 0] * 1000 + meta[bad][:, 1]).astype(int)
    np.savez(os.path.join(HERE, "data", "real_nodes.npz"), z_bad=zb, cut_bad=cb, z_healthy=zh, cut_healthy=cid)
    print("D* windows", zb.shape, "cuts", len(np.unique(cb)), "| healthy D windows", zh.shape, "cuts", len(np.unique(cid)))


if __name__ == "__main__":
    main()
