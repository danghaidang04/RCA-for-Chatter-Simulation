"""LUH leave-one-machine-out (E2): baseline chuẩn + CORAL + DANN, 10 hạt giống, CI bootstrap theo lần chạy.
Mục tiêu 'dao mòn' = VB>=100 µm. DANN/CORAL dùng đặc trưng máy đích KHÔNG nhãn (transductive) -> ghi rõ trong bảng.
CI bootstrap ở mức lần chạy ĐÁNH GIÁ THẤP độ không chắc chắn (các lần chạy cùng dao tương quan)."""
import json, warnings
import numpy as np, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.linalg import fractional_matrix_power
import luh_transfer as lt

warnings.filterwarnings("ignore"); torch.set_num_threads(4); SEEDS = range(10)


def coral(A, B, eps=1.0):
    Ca = np.cov(A, rowvar=False) + eps * np.eye(A.shape[1]); Cb = np.cov(B, rowvar=False) + eps * np.eye(A.shape[1])
    return (A @ np.real(fractional_matrix_power(Ca, -0.5))) @ np.real(fractional_matrix_power(Cb, 0.5))


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None


def dann(A, y, B, seed, epochs=200):
    torch.manual_seed(seed); d = A.shape[1]
    fe = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU()); cl = nn.Linear(32, 1); dm = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam([*fe.parameters(), *cl.parameters(), *dm.parameters()], 1e-3, weight_decay=1e-3)
    At, Bt, yt = (torch.tensor(a, dtype=torch.float32) for a in (A, B, y)); bce = nn.BCEWithLogitsLoss()
    pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6), dtype=torch.float32)
    for ep in range(epochs):
        lam = 2 / (1 + np.exp(-10 * ep / epochs)) - 1; fe.train(); fa, fb = fe(At), fe(Bt)
        loss = nn.functional.binary_cross_entropy_with_logits(cl(fa).squeeze(1), yt, pos_weight=pw)
        dl = torch.cat([dm(GRL.apply(fa, lam)), dm(GRL.apply(fb, lam))]).squeeze(1); dt = torch.cat([torch.zeros(len(At)), torch.ones(len(Bt))])
        loss = loss + 0.5 * bce(dl, dt); opt.zero_grad(); loss.backward(); opt.step()
    fe.eval(); return torch.sigmoid(cl(fe(Bt)).squeeze(1)).detach().numpy()


def predict(method, A, y, B, seed, raw_B=None):
    if method == "GB": return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed).fit(A, y).predict_proba(B)[:, 1]
    if method == "LR": return LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(A, y).predict_proba(B)[:, 1]
    if method == "CORAL+LR": return LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(coral(A, B), y).predict_proba(B)[:, 1]
    if method == "DANN": return dann(A, y, B, seed)
    raise ValueError(method)


def main():
    lab, X = lt.load(); y = (lab.wear.values >= 100).astype(int); M = lab.machine.values; rng = np.random.default_rng(0)
    FS = {"raw": X, "machine-relative": lt.rel(X, lab, "machine"), "tool-relative": lt.rel(X, lab, "tool")}; res = {}
    ires = [i for i, n in enumerate(["F" for _ in range(X.shape[1])])]                                          # chỉ số cột (để lấy RMS hợp lực)
    rms_col = 3 * (3 + 7)                                                                                  # cột R.mean() = cột đầu sau 3 kênh x (3+7) đặc trưng
    for fn, XX in FS.items():
        for mt in ("RMS-threshold", "GB", "LR", "CORAL+LR", "DANN"):
            per = []
            for m in (1, 2, 3):
                tr, te = M != m, M == m; sc = StandardScaler().fit(XX[tr]); A, B = sc.transform(XX[tr]), sc.transform(XX[te])
                if mt == "RMS-threshold":
                    s = XX[te][:, rms_col]; scores = [s] * 1; au = [roc_auc_score(y[te], s)]
                else:
                    scores = [predict(mt, A, y[tr], B, sd) for sd in SEEDS]; au = [roc_auc_score(y[te], s) for s in scores]
                ens = np.mean(scores, 0); n = te.sum(); bs = [roc_auc_score(y[te][i], ens[i]) for i in (rng.integers(0, n, n) for _ in range(300)) if 0 < y[te][i].sum() < len(i)]
                per.append({"machine": m, "auroc_mean": float(np.mean(au)), "auroc_sd_seeds": float(np.std(au)), "ci95_runs": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]})
            res[f"{fn}|{mt}"] = per; print(f"{fn:17s} {mt:14s} " + "  ".join(f"M{p['machine']}: {p['auroc_mean']:.3f}±{p['auroc_sd_seeds']:.3f} [{p['ci95_runs'][0]:.2f},{p['ci95_runs'][1]:.2f}]" for p in per), flush=True)
    json.dump(res, open("output/luh_baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
