"""Mô hình cho ứng dụng web: mạng Bayes có ĐIỀU KIỆN làm biến cha, đánh giá theo giao thức rút cùng điều kiện (như dùng thật trên một máy: 10 cửa sổ cùng điều kiện).
Dữ liệu: data/sim_sets_k10.npz (chế độ ổn định). Mặt nạ cạnh: kg_validated từ experiment_results_k10.json (cùng bước kiểm chứng cạnh bằng dữ liệu). Ghi report/cond_results_k10.json."""
import json
import os

import numpy as np

import experiment as ex
from bn import CondBN, CondGaussianBN, ece, shuffled_mask, kg_mask, to_nodes
from kg import KnowledgeGraph

HERE = os.path.dirname(os.path.abspath(__file__)); SUF = "_k10"; M = 10


def load():
    d = np.load(os.path.join(HERE, "data", f"sim_sets{SUF}.npz"), allow_pickle=True); split = d["split"]; Fn, Fs = to_nodes(d["F_nom"]), to_nodes(d["F_shift"])
    meta = np.load(os.path.join(os.path.dirname(HERE), "cutting_sim", "output", "real_windows.npz"))["meta"]; idx = np.arange(len(split)) % len(meta)
    A = ex.cond_design(meta[idx, 5], meta[idx, 6], meta[idx, 7], meta[idx, 4]); key = np.array([f"{meta[i, 5]:.0f}|{meta[i, 6]}|{meta[i, 4]:.0f}" for i in idx]); sd0 = Fn[split == "train", 0].std(0)
    return Fn / sd0, Fs / sd0, A, key, split, sd0, d["row_cut"]


def groups_of(rows, key, min_rows=3):
    g = {k: rows[key[rows] == k] for k in np.unique(key[rows])}; return {k: v for k, v in g.items() if len(v) >= min_rows}


def draw_same_condition(X, A, groups, n, rng, m=M):
    """n lần rút mỗi nguyên nhân: chọn một điều kiện, lấy m hàng (cửa sổ) của điều kiện đó. Trả về (D, a, nhãn)."""
    ks = list(groups); D, As, L = [], [], []
    for c in range(1, 6):
        for _ in range(n):
            g = groups[ks[rng.integers(len(ks))]]; r = rng.choice(g, m, replace=True); D.append(X[r, c, :]); As.append(A[r[0]]); L.append(c)
    return D, np.array(As), np.array(L)


def evaluate(model, D, As, L, T=None):
    post = []
    for d, a in zip(D, As):
        mm = model.at(a); mm.T = model.T if T is None else T; post.append(mm.posterior(d - model.baseline(a), states=range(1, 6))[1])
    post = np.array(post); pred = post.argmax(1) + 1; return {"top1": float((pred == L).mean()), "ece": ece(post.max(1), pred == L), "per_class": [float((pred[L == c] == c).mean()) for c in range(1, 6)]}


def fit_T(model, D, As, L):
    grid = np.exp(np.linspace(0, np.log(2000), 40)); LL = []
    for d, a in zip(D, As): mm = model.at(a); LL.append(mm.loglik(d - model.baseline(a))[1:])
    LL = np.array(LL); y = L - 1; best = (1e18, 1.0)
    for T in grid:
        z = LL / T; z -= z.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True)); nll = -lp[np.arange(len(y)), y].mean()
        if nll < best[0]: best = (nll, T)
    model.T = float(best[1]); return model.T


def main():
    Fn, Fs, A, key, split, sd0, row_cut = load(); res = json.load(open(os.path.join(HERE, "report", f"experiment_results{SUF}.json"))); M_kg, S = kg_mask(KnowledgeGraph()); tr = np.where(split == "train")[0]; va = np.where(split == "val")[0]; te = np.where(split == "test")[0]
    gtr, gva, gte = groups_of(tr, key), groups_of(va, key, 1), groups_of(te, key); print("same-condition groups: val", len(gva), "test", len(gte))
    flat = lambda X: (np.repeat(A[tr], 6, axis=0), X[tr].reshape(-1, 9), np.tile(np.arange(6), len(tr)))
    variants = {"kg": [M_kg], "kg_validated": [np.array(res["masks"]["kg_validated"])], "full": [np.ones((5, 9))], "shuffled": [shuffled_mask(M_kg, s) for s in (0, 1, 2)]}; out = {"m": M, "groups_test": len(gte)}; models = {}
    rng = np.random.default_rng(5); Dv, Av, Lv = draw_same_condition(Fn, A, gva, 120, rng); Dn, An, Ln = draw_same_condition(Fn, A, gte, 150, np.random.default_rng(6)); Ds, As_, Ls = draw_same_condition(Fs, A, gte, 150, np.random.default_rng(7))
    for name, masks in variants.items():
        rr = []
        for i, mk in enumerate(masks):
            mod = CondBN(mk).fit(*flat(Fn)); fit_T(mod, Dv, Av, Lv); rr.append({"T": mod.T, "nominal": evaluate(mod, Dn, An, Ln), "shifted": evaluate(mod, Ds, As_, Ls), "nominal_T1": evaluate(mod, Dn, An, Ln, T=1.0)["ece"]})
            if i == 0: models[name] = mod
        out[name] = {"edges": float(np.mean([m.sum() for m in masks])), "T": float(np.mean([r["T"] for r in rr])), "nominal_top1": float(np.mean([r["nominal"]["top1"] for r in rr])), "nominal_ece": float(np.mean([r["nominal"]["ece"] for r in rr])), "shifted_top1": float(np.mean([r["shifted"]["top1"] for r in rr])), "shifted_ece": float(np.mean([r["shifted"]["ece"] for r in rr])),
                     "nominal_per_class": rr[0]["nominal"]["per_class"], "ece_without_temperature": float(np.mean([r["nominal_T1"] for r in rr]))}
        print(f"{name:13s} edges {out[name]['edges']:4.0f} T={out[name]['T']:6.1f} | nominal top1 {out[name]['nominal_top1']:.3f} ece {out[name]['nominal_ece']:.3f} | shifted top1 {out[name]['shifted_top1']:.3f} ece {out[name]['shifted_ece']:.3f} | per class {np.round(out[name]['nominal_per_class'], 2).tolist()}", flush=True)
    mod = models["kg_validated"]; best = lambda d, a: int(mod.at(a).posterior(d - mod.baseline(a))[1].argmax())
    def d2(d, a):
        mm = mod.at(a); dr = d - mod.baseline(a); c = best(d, a); x = dr.mean(0) - mm.mu[c]; return float(len(d) * x @ mm.P @ x), x
    ref = [d2(d, a) for d, a in zip(Dn, An)]; q95 = float(np.percentile([r[0] for r in ref], 95)); shf = [d2(d, a)[0] for d, a in zip(Ds, As_)]; node_sd = np.std([r[1] for r in ref], axis=0)
    out["misfit"] = {"q95_nominal": q95, "flag_rate_nominal": float(np.mean([r[0] > q95 for r in ref])), "flag_rate_shifted": float(np.mean([x > q95 for x in shf])), "node_sd_ref": node_sd.tolist()}; print("misfit q95", round(q95, 1), "| flag rate shifted", round(out["misfit"]["flag_rate_shifted"], 3))
    # ---- bảng cạnh: trọng số trung bình (đơn vị sd của mô hình) + khoảng tin cậy bootstrap theo ĐƯỜNG CẮT, mô hình nối đầy đủ ----
    from kg import CAUSES, FEATURES
    ucut = np.unique(row_cut[tr]); by = {c: tr[row_cut[tr] == c] for c in ucut}; brng = np.random.default_rng(11); Wb = []
    for _ in range(100):
        rows = np.concatenate([by[c] for c in brng.choice(ucut, len(ucut))]); m_ = CondBN(np.ones((5, 9))).fit(np.repeat(A[rows], 6, axis=0), Fn[rows].reshape(-1, 9), np.tile(np.arange(6), len(rows)))
        Wb.append(np.einsum("cpk,p->ck", m_.beta[1:] - m_.beta[0], A[rows].mean(0)) / m_.sd)
    Wb = np.array(Wb); wm, wlo, whi = Wb.mean(0), np.percentile(Wb, 2.5, axis=0), np.percentile(Wb, 97.5, axis=0)
    out["edges_table"] = {f"{c}|{f}": {"weight": float(wm[i, k]), "ci_lo": float(wlo[i, k]), "ci_hi": float(whi[i, k]), "in_kg": bool(M_kg[i, k]), "kg_sign": int(S[i, k]), "in_model": bool(variants["kg_validated"][0][i, k])} for i, c in enumerate(CAUSES) for k, f in enumerate(FEATURES)}
    out["model"] = {"mask": variants["kg_validated"][0][0].tolist(), "beta": mod.beta.tolist(), "Sigma": mod.Sigma.tolist(), "T": mod.T, "sd0": sd0.tolist()}
    json.dump(out, open(os.path.join(HERE, "report", f"cond_results{SUF}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
