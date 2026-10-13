"""Học liên tục qua nhiều "máy": đo QUÊN kiến thức nền và các cơ chế chống quên (mô phỏng; mỗi máy có tham số DDE cố định, data/machines.npz).

 Trạng thái cần giữ = vectơ dịch chuyển trung bình s[c,k] của 5 nguyên nhân trên 9 nút (đơn vị độ lệch chuẩn khỏe của máy), chỉ ở các cạnh KG đã kiểm chứng.
 NỀN (base) = s_base học từ bộ sinh danh định ("máy tham chiếu/nhà máy"). Hiệp phương sai Sigma cố định lấy từ mô hình nền.
 Quy tắc cập nhật dùng CA ĐÃ XÁC NHẬN (x, c). Các cánh tay:
   static_base      : không học.
   naive_ema        : s_c <- s_c + eta (x - s_c); cập nhật chạy liên tục, GHI ĐÈ (đường cơ sở gây quên).
   anchored_ema(g)  : thêm lực kéo về nền: - g (s_c - s_base_c) (kiểu L2-SP / neo tiên nghiệm).
   pooled_replay    : giữ mọi ca đã thấy (tham chiếu "lưu hết"; trung bình giữa các máy).
   hierarchical     : s_máy = w_g + delta_máy; w_g tổng hợp có trọng số độ chính xác từ các máy (nền làm tiên nghiệm), delta co về 0; máy mới bắt đầu từ w_g.
   pseudo_label_ema : như naive_ema nhưng nhãn lấy từ chính dự đoán của mô hình (không xác nhận) -> kiểm tra tự củng cố/trôi.
   adapt_from_base  : tham chiếu: máy mới học bằng EMA bắt đầu từ nền (không học các máy trước).
"""
import json
import os

import numpy as np

import experiment as ex
from bn import to_nodes

HERE = os.path.dirname(os.path.abspath(__file__)); ETA = 0.05; N_SEQ = 30; KS = [0, 2, 5, 10, 20]; M_DRAW = 10


def load_machines():
    d = np.load(os.path.join(HERE, "data", "machines.npz"), allow_pickle=True); X = to_nodes(d["F"]); split = d["split"]; tr = split == "train"; out = []
    for m in range(len(X)):                                          # chuẩn hóa mỗi máy theo cửa sổ KHỎE của chính nó (rẻ, không cần nhãn lỗi)
        mu, sd = X[m][tr, 0].mean(0), X[m][tr, 0].std(0); out.append((X[m] - mu) / sd)
    return np.array(out), split, json.loads(str(d["pj"]))


def base_state():
    res = json.load(open(os.path.join(HERE, "report", "experiment_results.json"))); mask = np.array(res["masks"]["kg_validated"]); Xn, Xs, split, _ = ex.load_sim(); model = ex.fit_variant(Xn, split, mask)
    return (model.mu[1:] - model.mu[0]) * mask, mask, model.P, model.Sigma


def top1(S, X, rows, rng, P, n=100):
    """Top-1 trên 5 nguyên nhân: m=10 cửa sổ mỗi lần rút; S: (5,9)."""
    D = np.array([X[rng.choice(rows, M_DRAW), c + 1, :] for c in range(5) for _ in range(n)]); y = np.repeat(np.arange(5), n)
    diff = D[:, None, :, :] - S[None, :, None, :]; ll = -0.5 * np.einsum("ncmi,ij,ncmj->nc", diff, P, diff); return float((ll.argmax(1) == y).mean())


def cases_of(X, rows, rng, n_per_class):
    """Dòng ca xác nhận (x, c): n_per_class cửa sổ mỗi nguyên nhân, xáo trộn."""
    cs = [(X[rng.choice(rows)][c + 1], c) for c in range(5) for _ in range(n_per_class)]; rng.shuffle(cs); return cs


class Ema:
    def __init__(s, base, mask, gamma=0.0, eta=ETA): s.S = base.copy(); s.base = base; s.mask = mask; s.g = gamma; s.eta = eta
    def observe(s, x, c): s.S[c] = (s.S[c] + s.eta * (x - s.S[c]) - s.g * (s.S[c] - s.base[c])) * s.mask[c]
    def state(s): return s.S


class Pooled:
    def __init__(s, base, mask, n0=10): s.base = base; s.mask = mask; s.n0 = n0; s.sum = np.zeros_like(base); s.n = np.zeros(5)
    def observe(s, x, c): s.sum[c] += x * s.mask[c]; s.n[c] += 1
    def state(s): return (s.n0 * s.base + s.sum) / (s.n0 + s.n)[:, None] * s.mask


class Hier:
    """Nền chung w_g + lớp phủ từng máy. w_g tính lại sau mỗi máy từ trung bình từng máy (tiên nghiệm = nền, độ chính xác tau0)."""
    def __init__(s, base, mask, sigma2, tau0=0.5): s.base = base; s.mask = mask; s.s2 = sigma2; s.tau0 = tau0; s.means = []; s.ns = []; s.wg = base.copy(); s.tau2 = 0.3 ** 2

    def consolidate(s, cases):
        m = np.zeros((5, 9)); n = np.zeros(5)
        for x, c in cases: m[c] += x; n[c] += 1
        m = m / n[:, None] * s.mask; s.means.append(m); s.ns.append(n.mean())
        if len(s.means) >= 2:
            M = np.array(s.means); v = M.var(0, ddof=1) - np.mean([s.s2 / nn for nn in s.ns], axis=0); s.tau2 = float(np.clip(v[s.mask > 0].mean(), 0.05 ** 2, 1.0))
        prec0 = 1 / s.tau0 ** 2; num = prec0 * s.base; den = prec0 * np.ones((5, 9))
        for m_, nn in zip(s.means, s.ns): p = 1 / (s.tau2 + s.s2 / nn); num = num + p * m_; den = den + p
        s.wg = num / den * s.mask; return s.wg

    def machine_state(s, cases):
        """w_g + delta (co về w_g theo độ chính xác); với 0 ca thì chính là w_g."""
        if not cases: return s.wg
        m = np.zeros((5, 9)); n = np.zeros(5)
        for x, c in cases: m[c] += x; n[c] += 1
        m = m / np.maximum(n, 1)[:, None]; w = (n[:, None] / s.s2) / (n[:, None] / s.s2 + 1 / s.tau2); return (s.wg + w * (m - s.wg)) * s.mask


def run(seq, new, X, split, base, mask, P, sigma2, gammas, rng):
    tr = np.where(split == "train")[0]; te = np.where(split == "test")[0]
    arms = {"static_base": None, "naive_ema": Ema(base, mask), "pooled_replay": Pooled(base, mask), "pseudo_label_ema": Ema(base, mask)}
    for g in gammas: arms[f"anchored_ema_g{g}"] = Ema(base, mask, gamma=g)
    hier = Hier(base, mask, sigma2); stream_by_m = {}
    for j in seq:
        cs = cases_of(X[j], tr, rng, N_SEQ); stream_by_m[j] = cs
        for x, c in cs:
            for name, a in arms.items():
                if a is None or name == "pseudo_label_ema": continue
                a.observe(x, c)
            a = arms["pseudo_label_ema"]; ll = -0.5 * np.einsum("cj,jk,ck->c", x - a.S, P, x - a.S); a.observe(x, int(ll.argmax()))     # nhãn giả: dự đoán của chính mô hình
        hier.consolidate(cs)
    res = {"retention": {}, "forward": {k: {} for k in KS}, "fidelity": {}}
    # --- giữ lại (retention) trên các máy đã học: không thích nghi thêm ---
    spec = {j: np.array([X[j][tr][:, c + 1, :].mean(0) for c in range(5)]) * mask for j in seq}
    for name, a in arms.items():
        S = base if a is None else a.state(); res["retention"][name] = float(np.mean([top1(S, X[j], te, rng, P) for j in seq]))
    res["retention"]["hierarchical_overlay"] = float(np.mean([top1(hier.machine_state(stream_by_m[j]), X[j], te, rng, P) for j in seq])); res["retention"]["hierarchical_global_only"] = float(np.mean([top1(hier.wg, X[j], te, rng, P) for j in seq]))
    res["retention"]["specialist_ceiling"] = float(np.mean([top1(spec[j], X[j], te, rng, P) for j in seq]))
    # --- độ trung thành cạnh nền: cosine và tỉ lệ dấu giữ nguyên ---
    big = (np.abs(base) >= 0.3)
    for name, a in list(arms.items()) + [("hierarchical_global", type("H", (), {"state": lambda s_: hier.wg})())]:
        S = base if a is None else a.state(); res["fidelity"][name] = {"cosine": float((S * mask).ravel() @ (base * mask).ravel() / (np.linalg.norm(S * mask) * np.linalg.norm(base * mask) + 1e-12)), "sign_kept": float((np.sign(S[big]) == np.sign(base[big])).mean())}
    # --- chuyển tiếp sang máy MỚI với k ca xác nhận mỗi nguyên nhân ---
    for k in KS:
        acc = {}
        for j in new:
            for rep in range(5):
                cs = cases_of(X[j], tr, rng, k) if k > 0 else []
                for name, a in arms.items():
                    if a is None: S = base
                    elif isinstance(a, Pooled): b = Pooled(base, mask); b.sum, b.n = a.sum.copy(), a.n.copy(); [b.observe(x, c) for x, c in cs]; S = b.state()
                    elif name == "pseudo_label_ema":
                        b = Ema(base, mask); b.S = a.S.copy()
                        for x, c in cs: ll = -0.5 * np.einsum("cj,jk,ck->c", x - b.S, P, x - b.S); b.observe(x, int(ll.argmax()))
                        S = b.S
                    else: b = Ema(base, mask, gamma=a.g); b.S = a.S.copy(); [b.observe(x, c) for x, c in cs]; S = b.S
                    acc.setdefault(name, []).append(top1(S, X[j], te, rng, P))
                acc.setdefault("hierarchical", []).append(top1(hier.machine_state(cs), X[j], te, rng, P))
                b = Ema(base, mask); [b.observe(x, c) for x, c in cs]; acc.setdefault("adapt_from_base", []).append(top1(b.S, X[j], te, rng, P))
                acc.setdefault("specialist_ceiling", []).append(top1(np.array([X[j][tr][:, c + 1, :].mean(0) for c in range(5)]) * mask, X[j], te, rng, P))
        res["forward"][k] = {n: float(np.mean(v)) for n, v in acc.items()}
    res["tau2"] = hier.tau2; return res


def edge_support(X, split, mask, machines, K_frac=0.6, B=60, rng=None):
    """Mỗi (nguyên nhân, nút): đếm số máy mà khoảng tin cậy bootstrap loại trừ 0 và |dịch chuyển| >= 0.5. Chỉ ĐỀ XUẤT (chờ chuyên gia duyệt): cạnh ngoài mask được nhiều máy ủng hộ; cạnh trong mask ít máy ủng hộ."""
    rng = rng or np.random.default_rng(0); tr = np.where(split == "train")[0]; cnt = np.zeros((5, 9))
    for j in machines:
        for c in range(5):
            Y = X[j][tr][:, c + 1, :]; bm = np.array([Y[rng.choice(len(Y), len(Y))].mean(0) for _ in range(B)]); lo, hi = np.percentile(bm, 2.5, axis=0), np.percentile(bm, 97.5, axis=0); m = Y.mean(0)
            cnt[c] += ((lo > 0) | (hi < 0)) & (np.abs(m) >= 0.5)
    need = int(np.ceil(K_frac * len(machines))); add = (mask == 0) & (cnt >= need); drop = (mask > 0) & (cnt < max(1, len(machines) // 3))
    return {"machines": len(machines), "need": need, "n_support_matrix": cnt.tolist(), "candidates_to_add": int(add.sum()), "candidates_to_drop": int(drop.sum()), "edges_in_base": int(mask.sum()), "add_list": [[int(i), int(k)] for i, k in zip(*np.where(add))], "drop_list": [[int(i), int(k)] for i, k in zip(*np.where(drop))]}


def machine_check(X, split, base, mask, P, rng):
    """Các máy có thật sự khác nhau không? So khoảng cách GIỮA máy với dao động bootstrap TRONG máy; và top-1 của nền trên từng máy so với chuyên gia của máy đó."""
    tr = np.where(split == "train")[0]; te = np.where(split == "test")[0]; S = [np.array([X[j][tr][:, c + 1, :].mean(0) for c in range(5)]) * mask for j in range(len(X))]
    between = np.mean([np.linalg.norm(S[i] - S[j]) for i in range(len(X)) for j in range(i + 1, len(X))]); within = []
    for j in range(len(X)):
        B = []
        for _ in range(60): r = rng.choice(tr, len(tr)); B.append(np.array([X[j][r][:, c + 1, :].mean(0) for c in range(5)]) * mask)
        within.append(np.mean([np.linalg.norm(B[a] - B[b]) for a, b in zip(range(30), range(30, 60))]))
    return {"between_machine_distance": float(between), "within_machine_bootstrap_distance": float(np.mean(within)), "ratio": float(between / np.mean(within)),
            "base_top1_per_machine": [top1(base, X[j], te, rng, P) for j in range(len(X))], "specialist_top1_per_machine": [top1(S[j], X[j], te, rng, P) for j in range(len(X))], "base_distance_to_machine": [float(np.linalg.norm(S[j] - base)) for j in range(len(X))]}


def main():
    X, split, pj = load_machines(); base, mask, P, Sigma = base_state(); sigma2 = np.tile(np.diag(Sigma), (5, 1)); gammas = [0.005, 0.01, 0.02, 0.05]; rng = np.random.default_rng(0); K = len(X)
    chk = machine_check(X, split, base, mask, P, rng); print("machines:", K, "| machine check:", {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in chk.items()}, flush=True)
    new = list(range(K - 3, K)); pool = list(range(K - 3)); out = {"machine_check": chk, "pj": pj, "new_machines": new, "n_seq_cases_per_class": N_SEQ, "eta": ETA, "by_len": {}}
    out["edge_support"] = edge_support(X, split, mask, pool); print("edge proposals (support by >= %d of %d machines):" % (out["edge_support"]["need"], len(pool)), {k: out["edge_support"][k] for k in ("candidates_to_add", "candidates_to_drop", "edges_in_base")}, flush=True)
    for L in (1, 3, 6):
        if L > len(pool): continue
        runs = [run(list(np.random.default_rng(100 * L + s).permutation(pool)[:L]), new, X, split, base, mask, P, sigma2, gammas, np.random.default_rng(10 + s)) for s in range(3)]
        agg = lambda get: [float(np.mean([get(r) for r in runs])), float(np.std([get(r) for r in runs]))]
        out["by_len"][str(L)] = {"retention": {n: agg(lambda r, n=n: r["retention"][n]) for n in runs[0]["retention"]}, "forward": {str(k): {n: agg(lambda r, n=n, k=k: r["forward"][k][n]) for n in runs[0]["forward"][k]} for k in KS},
                                 "fidelity": {n: {m: agg(lambda r, n=n, m=m: r["fidelity"][n][m]) for m in ("cosine", "sign_kept")} for n in runs[0]["fidelity"]}, "tau2": agg(lambda r: r["tau2"])}
        d = out["by_len"][str(L)]; print(f"\n=== sequence length L={L} machines learned (new machines {new}) ===")
        print("RETENTION:", {n: round(v[0], 3) for n, v in d["retention"].items() if n in ("static_base", "naive_ema", "pooled_replay", "pseudo_label_ema", "anchored_ema_g0.02", "hierarchical_overlay", "hierarchical_global_only", "specialist_ceiling")})
        for k in (0, 5, 10, 20): print(f"FORWARD k={k:2d}:", {n: round(v[0], 3) for n, v in d["forward"][str(k)].items() if n in ("static_base", "naive_ema", "pooled_replay", "pseudo_label_ema", "anchored_ema_g0.02", "hierarchical", "adapt_from_base", "specialist_ceiling")})
        print("FIDELITY (cos / sign kept):", {n: (round(v["cosine"][0], 3), round(v["sign_kept"][0], 2)) for n, v in d["fidelity"].items() if n in ("naive_ema", "pseudo_label_ema", "anchored_ema_g0.02", "hierarchical_global")}, flush=True)
    json.dump(out, open(os.path.join(HERE, "report", "continual_results.json"), "w"), indent=1); return out


if __name__ == "__main__":
    main()
