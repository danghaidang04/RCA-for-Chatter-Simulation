"""Vì sao mạng Bayes trong ứng dụng chỉ đạt ~0.5? Kiểm tra THÔNG TIN có trong dữ liệu mô phỏng (cùng giao thức rút cùng điều kiện, bộ ổn định k10):
 (a) Bayes điều kiện trên 9 nút (mô hình của ứng dụng); (b) Bayes điều kiện trên 24 đặc trưng phổ; (c) bộ phân loại GB theo cửa sổ (24 đặc trưng + điều kiện), cộng log xác suất qua m=10 cửa sổ."""
import json, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import cond_experiment as ce
import experiment as ex

HERE = os.path.dirname(os.path.abspath(__file__))


def cond_bayes(Atr, Xtr, ytr, lam=1.0, ridge=1e-3):
    p, k = Atr.shape[1], Xtr.shape[1]; B = np.array([np.linalg.solve(Atr[ytr == c].T @ Atr[ytr == c] + lam * np.eye(p), Atr[ytr == c].T @ Xtr[ytr == c]) for c in range(6)])
    R = Xtr - np.einsum("np,npk->nk", Atr, B[ytr]); S = R.T @ R / (len(Xtr) - 6 * p) + ridge * np.eye(k) * np.trace(R.T @ R / len(Xtr)) / k; return B, np.linalg.inv(S)


def main():
    d = np.load(os.path.join(HERE, "data", "sim_sets_k10.npz"), allow_pickle=True); split = d["split"]; F = d["F_nom"]; Fs = d["F_shift"]
    meta = np.load(os.path.join(os.path.dirname(HERE), "cutting_sim", "output", "real_windows.npz"))["meta"]; idx = np.arange(len(split)) % len(meta)
    A = ex.cond_design(meta[idx, 5], meta[idx, 6], meta[idx, 7], meta[idx, 4]); key = np.array([f"{meta[i, 5]:.0f}|{meta[i, 6]}|{meta[i, 4]:.0f}" for i in idx]); tr = np.where(split == "train")[0]; te = np.where(split == "test")[0]; gte = ce.groups_of(te, key)
    out = {}
    for name, sel in (("9 nodes", None), ("24 spectral features", slice(0, 24))):
        from bn import to_nodes
        X = to_nodes(F) if sel is None else F[..., sel]; Xs = to_nodes(Fs) if sel is None else Fs[..., sel]; sd0 = X[tr, 0].std(0); X = X / sd0; Xs = Xs / sd0
        B, P = cond_bayes(np.repeat(A[tr], 6, axis=0), X[tr].reshape(-1, X.shape[-1]), np.tile(np.arange(6), len(tr)))
        for tag, XX in (("nominal", X), ("shifted", Xs)):
            rng = np.random.default_rng(6 if tag == "nominal" else 7); D, As, L = ce.draw_same_condition(XX, A, gte, 150, rng); pred = []
            for dd, a in zip(D, As):
                mu = np.einsum("p,cpk->ck", a, B)[1:]; diff = dd[:, None, :] - mu[None]; ll = -0.5 * np.einsum("mck,kj,mcj->c", diff, P, diff); pred.append(ll.argmax() + 1)
            out[f"bayes_{name}_{tag}"] = float((np.array(pred) == L).mean())
    # (c) GB por cửa sổ
    X = F[..., :24]; Xtr = np.c_[X[tr].reshape(-1, 24), np.repeat(A[tr][:, 1:], 6, axis=0)]; ytr = np.tile(np.arange(6), len(tr)); gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtr, ytr)
    for tag, XX in (("nominal", F[..., :24]), ("shifted", Fs[..., :24])):
        rng = np.random.default_rng(6 if tag == "nominal" else 7); D, As, L = ce.draw_same_condition(XX, A, gte, 150, rng); pred = []; single = []
        for dd, a in zip(D, As):
            P_ = gb.predict_proba(np.c_[dd, np.tile(a[1:], (len(dd), 1))])[:, 1:]; pred.append(np.log(P_ + 1e-6).sum(0).argmax() + 1); single.append(P_.argmax(1)[0] + 1)
        out[f"gb_window_{tag}_m10"] = float((np.array(pred) == L).mean()); out[f"gb_window_{tag}_single"] = float((np.array(single) == L).mean())
    json.dump(out, open(os.path.join(HERE, "report", "separability.json"), "w"), indent=1); [print(f"{k:34s} {v:.3f}") for k, v in out.items()]


if __name__ == "__main__":
    main()
