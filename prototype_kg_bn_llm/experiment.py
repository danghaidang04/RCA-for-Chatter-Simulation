"""Thực nghiệm của prototype: (1) trọng số cạnh + khoảng tin cậy + kiểm tra kết nối KG; (2) KG có đóng góp vào suy luận không (kg / full / shuffled);
(3) hiệu chỉnh xác suất (ECE); (4) bộ chấm điểm chỉ dùng trọng số cạnh; (5) ngưỡng 'độ lệch khỏi mô hình'. Ghi report/experiment_results.json, report/edges.csv, hình.
Dữ liệu: data/sim_sets.npz (make_data.py). Mọi đánh giá tách theo ĐƯỜNG CẮT. Bộ sinh lệch (shifted) chỉ dùng để KIỂM TRA, không dùng để khớp."""
import csv
import json
import os

import numpy as np

from bn import CondGaussianBN, STATES, ece, kg_mask, score_matched_filter, score_pagerank, shuffled_mask, to_nodes
from kg import CAUSES, FEATURES, KnowledgeGraph

HERE = os.path.dirname(os.path.abspath(__file__)); M_DRAW = 10; N_DRAW = 300; SUF = os.environ.get("TAG", "")      # TAG=_k10: bộ dữ liệu/kết quả ở chế độ ổn định


COND_REL = os.environ.get("COND_REL", "1" if SUF == "_k10" else "0") == "1"      # chuẩn hóa theo nền khỏe của CHÍNH điều kiện (hồi quy nền khỏe theo điều kiện); mặc định bật cho bộ ổn định (ứng dụng web)
REF = {}


def cond_design(N, ae, F, up):
    """Ma trận thiết kế của nền khỏe theo điều kiện (cùng cho huấn luyện và ứng dụng)."""
    N, ae, F, up = (np.asarray(v, float) for v in (N, ae, F, up)); lN = np.log(N) - 9.2
    return np.c_[np.ones_like(lN), lN, lN ** 2, np.log(ae), np.log(F), up, up * lN]


def load_sim():
    d = np.load(os.path.join(HERE, "data", f"sim_sets{SUF}.npz"), allow_pickle=True); split = d["split"]; row_cut = d["row_cut"]
    Xn, Xs = to_nodes(d["F_nom"]), to_nodes(d["F_shift"]); tr = split == "train"
    if not COND_REL:
        mu, sd = Xn[tr, 0].mean(0), Xn[tr, 0].std(0); return (Xn - mu) / sd, (Xs - mu) / sd, split, row_cut
    meta = np.load(os.path.join(os.path.dirname(HERE), "cutting_sim", "output", "real_windows.npz"))["meta"]; idx = np.arange(len(split)) % len(meta); A = cond_design(meta[idx, 5], meta[idx, 6], meta[idx, 7], meta[idx, 4])
    Y = Xn[tr, 0]; beta = np.linalg.solve(A[tr].T @ A[tr] + 1.0 * np.eye(A.shape[1]), A[tr].T @ Y); sd = (Y - A[tr] @ beta).std(0); REF.update(beta=beta.tolist(), sd=sd.tolist())
    mu_row = (A @ beta)[:, None, :]; return (Xn - mu_row) / sd, (Xs - mu_row) / sd, split, row_cut


def flat(X, rows):
    """Cửa sổ của các hàng được chọn -> (n*6, 9) và nhãn 0..5."""
    A = X[rows]; return A.reshape(-1, 9), np.tile(np.arange(6), len(A))


def draws(X, rows, m, n, rng):
    """n lần rút, mỗi lần m cửa sổ (từ các hàng test) của từng nguyên nhân 1..5 -> (5n, m, 9), nhãn."""
    D, L = [], []
    for c in range(1, 6):
        for _ in range(n): D.append(X[rng.choice(rows, m, replace=True), c, :]); L.append(c)
    return np.array(D), np.array(L)


def rank_eval(model, W, D, L, T=None):
    """top-1 và ECE của hậu nghiệm trên 5 nguyên nhân + top-1 của hai bộ chấm điểm chỉ dùng trọng số cạnh."""
    post = np.array([model.posterior(d, states=range(1, 6), T=T)[1] for d in D]); pred = post.argmax(1) + 1; conf = post.max(1)
    mf = np.array([score_matched_filter(W, d).argmax() + 1 for d in D]); pr = np.array([score_pagerank(W, d).argmax() + 1 for d in D])
    return {"top1": float((pred == L).mean()), "ece": ece(conf, pred == L), "mean_conf": float(conf.mean()), "top1_matched_filter": float((mf == L).mean()), "top1_pagerank": float((pr == L).mean())}


def bootstrap_weights(X, y, cut, B=300, seed=0):
    """Trọng số cạnh của mô hình đầy đủ với bootstrap theo ĐƯỜNG CẮT (huấn luyện)."""
    rng = np.random.default_rng(seed); uc = np.unique(cut); Ws = []
    idx_by = {c: np.where(cut == c)[0] for c in uc}
    for _ in range(B):
        pick = rng.choice(uc, len(uc), replace=True); ii = np.concatenate([idx_by[c] for c in pick]); Xb, yb = X[ii], y[ii]
        mu0 = Xb[yb == 0].mean(0); sd = Xb.std(0) + 1e-9; Ws.append(np.array([(Xb[yb == c].mean(0) - mu0) / sd for c in range(1, 6)]))
    Ws = np.array(Ws); return Ws.mean(0), np.percentile(Ws, 2.5, axis=0), np.percentile(Ws, 97.5, axis=0)


def fit_variant(Xn, split, mask, seed=0):
    tr = np.where(split == "train")[0]; val = np.where(split == "val")[0]; X, y = flat(Xn, tr); m = CondGaussianBN(mask).fit(X, y)
    Dv, Lv = draws(Xn, val, M_DRAW, 80, np.random.default_rng(100 + seed)); m.fit_temperature(Dv, Lv); return m


def run():
    Xn, Xs, split, row_cut = load_sim(); kg = KnowledgeGraph(); M, S = kg_mask(kg); te = np.where(split == "test")[0]
    rng = np.random.default_rng(7); Dn, Ln = draws(Xn, te, M_DRAW, N_DRAW, rng); Ds, Ls = draws(Xs, te, M_DRAW, N_DRAW, np.random.default_rng(8)); res = {"edges_in_kg": int(M.sum()), "m": M_DRAW, "n_draws_per_cause": N_DRAW, "chance": 0.2}
    # ---- kiểm tra cạnh bằng dữ liệu (mô hình đầy đủ, bootstrap theo đường cắt) ----
    tr = np.where(split == "train")[0]; X, y = flat(Xn, tr); cut = np.repeat(row_cut[tr], 6); mean, lo, hi = bootstrap_weights(X, y, cut); rows = []
    for i, c in enumerate(CAUSES):
        for k, f in enumerate(FEATURES):
            in_kg = bool(M[i, k]); sig = (lo[i, k] > 0) or (hi[i, k] < 0); sgn = S[i, k]; agree = (sgn == 0) or (np.sign(mean[i, k]) == sgn)
            if in_kg and not sig: status = "KG edge not supported (CI contains 0): candidate for removal"
            elif in_kg and sig and not agree: status = "KG edge supported but SIGN CONFLICT with KG prediction"
            elif in_kg: status = "KG edge supported"
            elif sig and abs(mean[i, k]) >= 0.5: status = "NOT in KG, significant and large: candidate for addition (expert review)"
            else: status = "not in KG, not needed"
            rows.append({"cause": c, "feature": f, "in_kg": in_kg, "kg_sign": int(sgn), "weight": float(mean[i, k]), "ci_lo": float(lo[i, k]), "ci_hi": float(hi[i, k]), "status": status})
    with open(os.path.join(HERE, "report", f"edges{SUF}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cnt = {}
    for r in rows: cnt[r["status"]] = cnt.get(r["status"], 0) + 1
    res["edge_status_counts"] = cnt; print("edge status:", cnt)
    sig_ = (lo > 0) | (hi < 0); keep = (M > 0) & sig_; add = (M == 0) & sig_ & (np.abs(mean) >= 0.5)
    variants = {"kg": [M], "kg_pruned": [keep.astype(float)], "kg_validated": [(keep | add).astype(float)], "full": [np.ones((5, 9))], "shuffled": [shuffled_mask(M, s) for s in (0, 1, 2)]}; res["edges_per_variant"] = {k: float(np.mean([m.sum() for m in v])) for k, v in variants.items()}; models = {}
    for name, masks in variants.items():
        rows = []
        for i, mk in enumerate(masks):
            mod = fit_variant(Xn, split, mk, seed=i); W = mod.weights
            rows.append({"T": mod.T, "nominal": rank_eval(mod, W, Dn, Ln), "shifted": rank_eval(mod, W, Ds, Ls), "nominal_T1": rank_eval(mod, W, Dn, Ln, T=1.0)["ece"], "shifted_T1": rank_eval(mod, W, Ds, Ls, T=1.0)["ece"]})
            if i == 0: models[name] = mod
        agg = lambda k1, k2: [float(np.mean([r[k1][k2] for r in rows])), float(np.std([r[k1][k2] for r in rows]))]
        res[name] = {"T": [float(np.mean([r["T"] for r in rows]))], **{f"{s}_{k}": agg(s, k) for s in ("nominal", "shifted") for k in ("top1", "ece", "mean_conf", "top1_matched_filter", "top1_pagerank")},
                     "ece_without_temperature_nominal": float(np.mean([r["nominal_T1"] for r in rows])), "ece_without_temperature_shifted": float(np.mean([r["shifted_T1"] for r in rows]))}
        print(f"{name:9s} T={res[name]['T'][0]:7.1f} | nominal top1 {res[name]['nominal_top1'][0]:.3f} ece {res[name]['nominal_ece'][0]:.3f} | shifted top1 {res[name]['shifted_top1'][0]:.3f} ece {res[name]['shifted_ece'][0]:.3f} | no-T ece nominal {res[name]['ece_without_temperature_nominal']:.3f}", flush=True)
    # ---- ngưỡng độ lệch khỏi mô hình (Mahalanobis) hiệu chỉnh trên rút nominal test, mô hình 'kg' ----
    mk = models["kg"]; best = lambda d: int(mk.posterior(d)[1].argmax()); d2 = lambda d, c: float(len(d) * (d.mean(0) - mk.mu[c]) @ mk.P @ (d.mean(0) - mk.mu[c]))
    ref = np.array([d2(d, best(d)) for d in Dn]); q95 = float(np.percentile(ref, 95)); sh = np.array([d2(d, best(d)) for d in Ds])
    node_sd = np.std(np.array([d.mean(0) - mk.mu[best(d)] for d in Dn]), axis=0)
    res["misfit"] = {"q95_nominal": q95, "flag_rate_nominal": float((ref > q95).mean()), "flag_rate_shifted": float((sh > q95).mean()), "node_sd_ref": node_sd.tolist()}
    print("misfit q95 (nominal)", round(q95, 2), "| flag rate shifted", round(res["misfit"]["flag_rate_shifted"], 3))
    res["masks"] = {n: v[0].tolist() for n, v in variants.items() if n != "shuffled"}
    res["cond_ref"] = dict(REF) if COND_REL else None
    json.dump(res, open(os.path.join(HERE, "report", f"experiment_results{SUF}.json"), "w"), indent=1)
    return res, models, (Xn, Xs, split, row_cut), (mean, lo, hi, M, S)


def figures(res, W3, M, S):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mean, lo, hi = W3; fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    im = ax[0].imshow(mean, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto"); ax[0].set_yticks(range(5)); ax[0].set_yticklabels(CAUSES); ax[0].set_xticks(range(9)); ax[0].set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=8); ax[0].set_title("Edge weights (healthy s.d.), fully connected fit", fontsize=9)
    for i in range(5):
        for k in range(9):
            if M[i, k]: ax[0].add_patch(plt.Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=1.6))
    fig.colorbar(im, ax=ax[0], shrink=0.8); ax[0].grid(False)
    sig = ((lo > 0) | (hi < 0)).astype(float); ax[1].imshow(sig, cmap="Greys", vmin=0, vmax=1.6, aspect="auto"); ax[1].set_yticks(range(5)); ax[1].set_yticklabels(CAUSES); ax[1].set_xticks(range(9)); ax[1].set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=8); ax[1].grid(False)
    ax[1].set_title("Bootstrap CI excludes 0 (dark); boxes = KG edges", fontsize=9)
    for i in range(5):
        for k in range(9):
            if M[i, k]: ax[1].add_patch(plt.Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False, ec="#c0392b", lw=1.6))
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "report", f"edge_weights{SUF}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    res, models, data, W = run(); figures(res, W[:3], W[3], W[4]); print("done")
