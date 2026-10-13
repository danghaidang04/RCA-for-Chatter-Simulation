"""Mạng Bayes Gaussian có điều kiện, cấu trúc lấy từ KG.

 Biến gốc C ∈ {HEALTHY, 5 nguyên nhân}; 9 nút cảm biến x (đã chuẩn hóa theo độ lệch chuẩn khỏe của mô phỏng).
 p(x | C=c) = N(mu_c, Sigma): hiệp phương sai CHUNG đầy đủ (kiểu LDA) để xử lý tương quan giữa các nút.
 Cạnh c -> k (trọng số w_ck = (mu_ck - mu_0k)/sd_k) chỉ tự do khi KG có cạnh đó; nếu không, mu_ck bị buộc bằng mu_0k (biến thể 'kg').
 Hai biến thể đối chứng: 'full' (mọi cạnh tự do) và 'shuffled' (cạnh ngẫu nhiên cùng bậc ra) để kiểm tra KG có đóng góp vào suy luận hay không.
 Hậu nghiệm cho m cửa sổ: tích likelihood chia cho nhiệt độ T (T chỉnh trên tập val) vì các cửa sổ cùng đường cắt tương quan.
"""
import numpy as np

from kg import CAUSES, FEATURES, KnowledgeGraph

STATES = ["HEALTHY"] + CAUSES


def to_nodes(F):
    """30 đặc trưng -> 9 nút cảm biến (cùng công thức với cutting_sim/rca_causal.sens_nodes)."""
    acc, mic = F[..., 4:14], F[..., 14:24]
    return np.stack([F[..., 0], F[..., 1], F[..., 2], F[..., 3], acc[..., 0:3].mean(-1), acc[..., 4:7].mean(-1), acc[..., 7:10].mean(-1), mic[..., 0:3].mean(-1), mic[..., 6:10].mean(-1)], axis=-1)


def kg_mask(kg=None):
    """Ma trận 5x9 (nguyên nhân x nút): 1 nếu KG có cạnh; ma trận dấu dự đoán (0 = không xác định)."""
    kg = kg or KnowledgeGraph(); e = kg.compile_edges(); M = np.zeros((5, 9)); S = np.zeros((5, 9))
    for (c, f), v in e.items(): M[CAUSES.index(c), FEATURES.index(f)] = 1; S[CAUSES.index(c), FEATURES.index(f)] = v["sign"]
    return M, S


def shuffled_mask(M, seed):
    rng = np.random.default_rng(seed); out = np.zeros_like(M)
    for i in range(M.shape[0]): out[i, rng.choice(M.shape[1], int(M[i].sum()), replace=False)] = 1
    return out


class CondGaussianBN:
    def __init__(self, mask=None, ridge=1e-3):
        """mask: 5x9 (None = đầy đủ). Hàng i = nguyên nhân i; mu_c,k = mu_0,k nếu mask[i,k] = 0."""
        self.mask = np.ones((5, 9)) if mask is None else np.asarray(mask, float); self.ridge = ridge; self.T = 1.0

    def fit(self, X, y):
        """X: (n, 9) nút đã chuẩn hóa; y: (n,) trạng thái 0..5 (0 = khỏe)."""
        self.mu = np.zeros((6, 9)); self.mu[0] = X[y == 0].mean(0)
        for c in range(1, 6):
            mc = X[y == c].mean(0); self.mu[c] = np.where(self.mask[c - 1] > 0, mc, self.mu[0])
        R = X - self.mu[y]; self.Sigma = R.T @ R / max(len(X) - 6, 1) + self.ridge * np.eye(9); self.P = np.linalg.inv(self.Sigma); self.sd = np.sqrt(np.diag(self.Sigma)); return self

    @property
    def weights(self):
        """Trọng số cạnh w[c,k] (5x9) = (mu_ck - mu_0k)/sd_k, bằng 0 nếu cạnh bị KG chặn."""
        return (self.mu[1:] - self.mu[0]) / self.sd

    def loglik(self, D):
        """log p(D | state) cho từng trạng thái (6,), D: (m, 9). Bỏ hằng số chung."""
        out = np.zeros(6)
        for c in range(6):
            r = D - self.mu[c]; out[c] = -0.5 * np.einsum("ij,jk,ik->", r, self.P, r)
        return out

    def posterior(self, D, states=None, T=None):
        """Hậu nghiệm (đều tiên nghiệm) trên các trạng thái cho phép (mặc định cả 6)."""
        T = self.T if T is None else T; idx = list(range(6)) if states is None else list(states); ll = self.loglik(D)[idx] / T
        p = np.exp(ll - ll.max()); return idx, p / p.sum()

    def contributions(self, D, c, ref=0):
        """Phân rã log-likelihood ratio của trạng thái c so với ref theo từng nút: tổng đúng bằng loglik[c]-loglik[ref]."""
        m = len(D); xbar = D.mean(0); d = self.mu[c] - self.mu[ref]; mid = 0.5 * (self.mu[c] + self.mu[ref])
        return m * (self.P @ d) * (xbar - mid)

    def fit_temperature(self, draws, labels, grid=None):
        """Chọn T (1 số thực) tối thiểu NLL trên các lần rút val: draws[i] là (m,9), labels[i] ∈ 1..5."""
        grid = np.exp(np.linspace(0, np.log(2000), 40)) if grid is None else grid; LL = np.array([self.loglik(D)[1:] for D in draws]); y = np.asarray(labels) - 1
        best = (1e18, 1.0)
        for T in grid:
            z = LL / T; z = z - z.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True)); nll = -lp[np.arange(len(y)), y].mean()
            if nll < best[0]: best = (nll, T)
        self.T = float(best[1]); return self.T


# ---------------- bộ chấm điểm chỉ dùng trọng số cạnh (không dùng hiệp phương sai) ----------------
def score_matched_filter(W, D):
    """s_c = <w_c, zbar> / ||w_c||: tương quan giữa mẫu lệch quan sát và vectơ trọng số của nguyên nhân c."""
    z = D.mean(0); n = np.linalg.norm(W, axis=1) + 1e-9; return (W @ z) / n


def score_pagerank(W, D, damping=0.85, iters=200):
    """Kiểu CausalRCA: PageRank trên đồ thị có trọng số, cạnh bị đảo (nút cảm biến -> nguyên nhân) với trọng số |w|;
    vectơ cá nhân hóa là độ lớn dịch chuyển quan sát của nút cảm biến. Nguyên nhân là nút cụt: khối lượng của chúng quay về vectơ cá nhân hóa."""
    z = np.abs(D.mean(0)); A = np.abs(W).T; p0 = np.r_[z / (z.sum() + 1e-12), np.zeros(5)]; T = np.zeros((14, 14))
    rows = A.sum(1, keepdims=True); T[:9, 9:] = np.where(rows > 0, A / (rows + 1e-12), 0.0)
    dangling = (T.sum(1) == 0).astype(float); r = p0.copy()
    for _ in range(iters): r = (1 - damping) * p0 + damping * (r @ T + (r @ dangling) * p0)
    return r[9:]


def ece(conf, correct, bins=10):
    conf = np.asarray(conf); correct = np.asarray(correct, float); e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b) if a > 0 else (conf >= a) & (conf <= b)
        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


class CondBN:
    """Mạng Bayes Gauss có ĐIỀU KIỆN cắt làm biến cha: mu_c(cond) = A(cond) @ beta_c. Mặt nạ cạnh KG áp dụng cho PHẦN CHÊNH so với nền khỏe
    (beta_c[:,k] = beta_0[:,k] nếu không có cạnh c->k). Hiệp phương sai Sigma chung. `at(a)` trả về mô hình TƯƠNG ĐỐI (trung bình trừ nền khỏe của điều kiện) dùng như CondGaussianBN."""
    def __init__(self, mask=None, lam=1.0, ridge=1e-3, k=9): self.k = k; self.mask = np.ones((5, k)) if mask is None else np.asarray(mask, float); self.lam = lam; self.ridge = ridge; self.T = 1.0

    def fit(self, A, X, y):
        p = A.shape[1]; k = X.shape[1]; self.k = k
        if self.mask.shape[1] != k: self.mask = np.ones((5, k))
        fit1 = lambda a, x: np.linalg.solve(a.T @ a + self.lam * np.eye(p), a.T @ x); self.beta = np.zeros((6, p, k)); self.beta[0] = fit1(A[y == 0], X[y == 0])
        for c in range(1, 6):
            b = fit1(A[y == c], X[y == c]); self.beta[c] = np.where(self.mask[c - 1][None, :] > 0, b, self.beta[0])
        R = X - np.einsum("np,npk->nk", A, self.beta[y]); self.Sigma = R.T @ R / max(len(X) - 6 * p, 1) + self.ridge * np.eye(k); self.P = np.linalg.inv(self.Sigma); self.sd = np.sqrt(np.diag(self.Sigma)); return self

    def at(self, a):
        """Mô hình tương đối tại điều kiện a (vectơ thiết kế): mu_0 = 0, mu_c = (beta_c - beta_0)^T a."""
        m = CondGaussianBN(self.mask); m.mu = np.array([np.zeros(self.k)] + [(self.beta[c] - self.beta[0]).T @ a for c in range(1, 6)]); m.Sigma = self.Sigma; m.P = self.P; m.sd = self.sd; m.T = self.T; return m

    def baseline(self, a): return self.beta[0].T @ a
