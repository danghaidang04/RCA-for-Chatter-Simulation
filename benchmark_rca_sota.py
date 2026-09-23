import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import gammaln
import time

# ==============================================================================
# 1. PHYSICAL MACHINING CHATTER CAUSAL MODEL & DATA GENERATOR
# ==============================================================================
class ChatterCausalSystem:
    """
    Physical Causal Model of a CNC Milling System with Regenerative Dynamics.
    Graph Nodes (Metrics):
      0: Stiffness (k)
      1: Damping (zeta)
      2: Tool_Wear (Kt)
      3: Depth_of_Cut (a)
      4: Spindle_Speed (RPM)
      5: Stability_Limit (a_lim)
      6: Chatter_Severity (regen_instability)
      7: Vibration_RMS (vib_rms)
      8: Dominant_Freq (dom_freq)
      9: Spectral_Ratio (chatter_ratio)
      10: Force_Mean (force_mean)
      11: Force_Peak (force_peak)
      12: Spindle_Power (power)
      13: Surface_Roughness (roughness)
    """
    def __init__(self):
        self.node_names = [
            "Stiffness_k",
            "Damping_zeta",
            "Tool_Wear_Kt",
            "Cut_Depth_a",
            "Spindle_RPM",
            "Stability_Limit_alim",
            "Chatter_Severity",
            "Vibration_RMS",
            "Dominant_Freq",
            "Spectral_Ratio",
            "Force_Mean",
            "Force_Peak",
            "Spindle_Power",
            "Surface_Roughness"
        ]
        self.d = len(self.node_names)
        self.name_to_idx = {name: i for i, name in enumerate(self.node_names)}
        
        # Root cause candidates (controllable / structural source parameters)
        self.candidate_root_causes = [
            "Stiffness_k",
            "Damping_zeta",
            "Tool_Wear_Kt",
            "Cut_Depth_a",
            "Spindle_RPM"
        ]
        self.candidate_indices = [self.name_to_idx[n] for n in self.candidate_root_causes]
        
        # Define Ground-Truth Causal DAG
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(range(self.d))
        
        # Causal Edges based on Altintas machining mechanics
        edges = [
            ("Stiffness_k", "Stability_Limit_alim"),
            ("Damping_zeta", "Stability_Limit_alim"),
            ("Tool_Wear_Kt", "Stability_Limit_alim"),
            ("Spindle_RPM", "Stability_Limit_alim"),
            ("Stiffness_k", "Dominant_Freq"),
            
            ("Cut_Depth_a", "Chatter_Severity"),
            ("Stability_Limit_alim", "Chatter_Severity"),
            
            ("Chatter_Severity", "Vibration_RMS"),
            ("Chatter_Severity", "Spectral_Ratio"),
            ("Chatter_Severity", "Force_Peak"),
            
            ("Cut_Depth_a", "Force_Mean"),
            ("Tool_Wear_Kt", "Force_Mean"),
            
            ("Force_Mean", "Force_Peak"),
            ("Force_Mean", "Spindle_Power"),
            ("Force_Peak", "Spindle_Power"),
            
            ("Vibration_RMS", "Surface_Roughness"),
            ("Force_Peak", "Surface_Roughness")
        ]
        
        for u, v in edges:
            self.dag.add_edge(self.name_to_idx[u], self.name_to_idx[v])

    def sample_data(self, n_samples=1000, root_cause=None, severity=1.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        data = np.zeros((n_samples, self.d))
        
        # Nominal Root distributions (Observational / Pre-failure)
        # 0: Stiffness (normalized ~ 1.0, std=0.03)
        k = np.random.normal(1.0, 0.03, n_samples)
        # 1: Damping (normalized ~ 1.0, std=0.03)
        zeta = np.random.normal(1.0, 0.03, n_samples)
        # 2: Tool Wear Kt (normalized ~ 1.0, std=0.04)
        kt = np.random.normal(1.0, 0.04, n_samples)
        # 3: Cut Depth a (nominal 5 mm ~ 0.5, std=0.02)
        a = np.random.normal(0.5, 0.02, n_samples)
        # 4: Spindle RPM (nominal 3000 RPM / sweet spot ~ 0.5, std=0.02)
        rpm = np.random.normal(0.5, 0.02, n_samples)
        
        # Apply Interventional distribution shift (Soft / Hard intervention on Root Cause)
        if root_cause == "Stiffness_k":
            k = np.random.normal(1.0 - 0.45 * severity, 0.04, n_samples)
        elif root_cause == "Damping_zeta":
            zeta = np.random.normal(1.0 - 0.70 * severity, 0.03, n_samples)
        elif root_cause == "Tool_Wear_Kt":
            kt = np.random.normal(1.0 + 1.20 * severity, 0.06, n_samples)
        elif root_cause == "Cut_Depth_a":
            a = np.random.normal(0.5 + 0.90 * severity, 0.05, n_samples)
        elif root_cause == "Spindle_RPM":
            rpm = np.random.normal(0.5 + 0.80 * severity, 0.04, n_samples) # Shift to unstable lobe valley
            
        data[:, 0] = k
        data[:, 1] = zeta
        data[:, 2] = kt
        data[:, 3] = a
        data[:, 4] = rpm
        
        # 5: Stability Limit a_lim = f(k, zeta, kt, rpm)
        # Physics: a_lim proportional to (2 * zeta * k / kt) * lobe_factor(rpm)
        lobe_factor = 1.0 - 0.5 * np.sin(4 * np.pi * rpm) # Lobes variation
        a_lim = (0.8 * k * zeta / kt) * lobe_factor + np.random.normal(0, 0.02, n_samples)
        data[:, 5] = a_lim
        
        # 6: Chatter Severity = max(0, a - a_lim) (Non-linear threshold)
        diff = a - a_lim
        chatter_sev = np.maximum(0.0, diff * 3.5) + np.random.normal(0, 0.02, n_samples)
        chatter_sev = np.maximum(0.0, chatter_sev)
        data[:, 6] = chatter_sev
        
        # 7: Vibration RMS = baseline + 2.5 * chatter_sev
        data[:, 7] = 0.1 + 2.5 * chatter_sev + np.random.normal(0, 0.03, n_samples)
        
        # 8: Dominant Freq = 250 * sqrt(k) + chatter modulation
        data[:, 8] = 250.0 * np.sqrt(np.maximum(0.1, k)) + 15.0 * chatter_sev + np.random.normal(0, 2.0, n_samples)
        
        # 9: Spectral Ratio (Chatter energy concentration)
        data[:, 9] = 0.05 + 0.85 / (1.0 + np.exp(-5.0 * (chatter_sev - 0.2))) + np.random.normal(0, 0.02, n_samples)
        
        # 10: Force Mean = Kt * a
        data[:, 10] = 300.0 * (kt * a) + np.random.normal(0, 10.0, n_samples)
        
        # 11: Force Peak = Force Mean * (1.2 + 3.0 * chatter_sev)
        data[:, 11] = data[:, 10] * (1.2 + 2.8 * chatter_sev) + np.random.normal(0, 25.0, n_samples)
        
        # 12: Spindle Power = Force Mean * RPM * 0.8 + Peak effect
        data[:, 12] = 0.8 * data[:, 10] * rpm + 0.15 * data[:, 11] + np.random.normal(0, 15.0, n_samples)
        
        # 13: Surface Roughness Ra = baseline + 4.0 * Vib_RMS + Force Peak
        data[:, 13] = 0.4 + 3.0 * data[:, 7] + 0.001 * data[:, 11] + np.random.normal(0, 0.05, n_samples)
        
        return data

# ==============================================================================
# 2. STATE-OF-THE-ART RCA ALGORITHMS (BRCD, RCD, RCG, BARO, SimpleRCA, ST)
# ==============================================================================

class BaseRCA:
    def __init__(self, causal_system):
        self.system = causal_system
        self.d = causal_system.d
        self.candidate_indices = causal_system.candidate_indices
        self.node_names = causal_system.node_names

    def fit_and_rank(self, D_obs, D_int):
        """Returns sorted list of (candidate_name, score) descending by anomaly likelihood"""
        raise NotImplementedError

# --- Algorithm 1: BRCD (Bayesian Root Cause Discovery - ICML 2026) ---
class BRCD(BaseRCA):
    """
    Bayesian Root Cause Discovery (BRCD) with Linear Gaussian Marginal Likelihood / Student-t Scoring.
    Scores candidates by posterior probability P(R | D_int, D_obs, DAG).
    """
    def __init__(self, causal_system, m0=0.0, lambda0=1.0, alpha0=2.0, beta0=1.0):
        super().__init__(causal_system)
        self.m0 = m0
        self.lambda0 = lambda0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def _log_marginal_likelihood_node(self, X_child, X_parents):
        """
        Bayesian Linear Gaussian conjugate marginal likelihood (Eq 71-74 in BRCD Appendix F)
        """
        N = len(X_child)
        if X_parents.shape[1] == 0:
            # Single node marginal
            X_mat = np.ones((N, 1))
        else:
            X_mat = np.hstack([np.ones((N, 1)), X_parents])
            
        p = X_mat.shape[1]
        Lambda_0 = self.lambda0 * np.eye(p)
        m_0 = np.zeros(p)
        alpha_0 = self.alpha0
        beta_0 = self.beta0
        
        # Posterior updates
        Lambda_N = Lambda_0 + X_mat.T @ X_mat
        # Solve for m_N
        m_N = np.linalg.solve(Lambda_N, Lambda_0 @ m_0 + X_mat.T @ X_child)
        alpha_N = alpha_0 + N / 2.0
        
        quad_diff = beta_0 + 0.5 * (X_child.T @ X_child + m_0.T @ Lambda_0 @ m_0 - m_N.T @ Lambda_N @ m_N)
        beta_N = max(1e-6, float(quad_diff))
        
        # Log marginal likelihood
        sign_0, logdet_0 = np.linalg.slogdet(Lambda_0)
        sign_N, logdet_N = np.linalg.slogdet(Lambda_N)
        
        log_ml = (- N / 2.0 * np.log(2 * np.pi) +
                  0.5 * logdet_0 - 0.5 * logdet_N +
                  alpha_0 * np.log(beta_0) - alpha_N * np.log(beta_N) +
                  gammaln(alpha_N) - gammaln(alpha_0))
        return log_ml

    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        scores = {}
        
        # Pre-compute normal baseline parameters for all nodes
        baseline_stats = {}
        for i in range(self.d):
            parents = list(dag.predecessors(i))
            X_par_obs = D_obs[:, parents]
            baseline_stats[i] = self._log_marginal_likelihood_node(D_obs[:, i], X_par_obs)
            
        # For each candidate root cause R, evaluate the augmented interventional log-likelihood
        log_posteriors = {}
        for r_idx in self.candidate_indices:
            log_lik = 0.0
            for i in range(self.d):
                parents = list(dag.predecessors(i))
                X_par_int = D_int[:, parents]
                
                # If i is the root cause, F points to i (distribution shift allowed)
                # Likelihood evaluates how well the anomalous data is explained under intervention on r_idx
                if i == r_idx:
                    # Modelled as changed mechanism
                    log_lik += self._log_marginal_likelihood_node(D_int[:, i], X_par_int)
                else:
                    # Non-intervened nodes preserve invariant conditional distribution
                    # Combine D_obs and D_int under invariant parent parameters
                    X_combined = np.vstack([D_obs[:, i:i+1], D_int[:, i:i+1]])[:, 0]
                    X_par_comb = np.vstack([D_obs[:, parents], D_int[:, parents]])
                    log_lik += (self._log_marginal_likelihood_node(X_combined, X_par_comb) - baseline_stats[i])
                    
            log_posteriors[r_idx] = log_lik

        # Normalize log-posterior via softmax
        max_lp = max(log_posteriors.values())
        unnorm = {r: np.exp(lp - max_lp) for r, lp in log_posteriors.items()}
        total_p = sum(unnorm.values()) + 1e-18
        
        ranking = []
        for r_idx in self.candidate_indices:
            p_val = unnorm[r_idx] / total_p
            ranking.append((self.node_names[r_idx], p_val))
            
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

# --- Algorithm 2: RCD (Root Cause Discovery - Ikram et al., 2022) ---
class RCD(BaseRCA):
    """
    RCD uses Conditional Independence tests (Fisher's Z-transform on partial correlation)
    between the intervention indicator F and each metric given its parents.
    """
    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        n_obs = len(D_obs)
        n_int = len(D_int)
        
        # F indicator: 0 for obs, 1 for int
        F = np.concatenate([np.zeros(n_obs), np.ones(n_int)])
        D_all = np.vstack([D_obs, D_int])
        
        p_values = {}
        for r_idx in self.candidate_indices:
            parents = list(dag.predecessors(r_idx))
            
            # Partial correlation between F and X_r given parents
            if len(parents) == 0:
                corr, p_val = stats.pearsonr(F, D_all[:, r_idx])
            else:
                # Regress out parents
                X_par = np.hstack([np.ones((len(D_all), 1)), D_all[:, parents]])
                beta_f = np.linalg.lstsq(X_par, F, rcond=None)[0]
                res_f = F - X_par @ beta_f
                
                beta_x = np.linalg.lstsq(X_par, D_all[:, r_idx], rcond=None)[0]
                res_x = D_all[:, r_idx] - X_par @ beta_x
                
                corr, p_val = stats.pearsonr(res_f, res_x)
                
            # Strong dependency with F (low p-value) -> High root cause rank
            score = -np.log10(max(1e-15, p_val))
            p_values[r_idx] = score

        ranking = [(self.node_names[r], p_values[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

# --- Algorithm 3: RCG (Root Cause Graph - Ikram et al., 2025) ---
class RCG(BaseRCA):
    """
    RCG ranks nodes based on Conditional Mutual Information / Shift magnitude
    between pre-failure and post-failure distributions given parent configurations.
    """
    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        scores = {}
        for r_idx in self.candidate_indices:
            parents = list(dag.predecessors(r_idx))
            if len(parents) == 0:
                mean_diff = abs(np.mean(D_int[:, r_idx]) - np.mean(D_obs[:, r_idx]))
                std_obs = max(1e-4, np.std(D_obs[:, r_idx]))
                cmi_score = (mean_diff / std_obs) ** 2
            else:
                # Residual shift given parents
                X_par_obs = np.hstack([np.ones((len(D_obs), 1)), D_obs[:, parents]])
                beta = np.linalg.lstsq(X_par_obs, D_obs[:, r_idx], rcond=None)[0]
                res_obs = D_obs[:, r_idx] - X_par_obs @ beta
                
                X_par_int = np.hstack([np.ones((len(D_int), 1)), D_int[:, parents]])
                res_int = D_int[:, r_idx] - X_par_int @ beta
                
                diff = abs(np.mean(res_int) - np.mean(res_obs))
                std_res = max(1e-4, np.std(res_obs))
                cmi_score = (diff / std_res) ** 2
                
            scores[r_idx] = cmi_score

        ranking = [(self.node_names[r], scores[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

# --- Algorithm 4: BARO (Pham et al., 2024) ---
class BARO(BaseRCA):
    """
    BARO: Non-causal baseline using Robust Median & IQR deviation.
    """
    def fit_and_rank(self, D_obs, D_int):
        scores = {}
        for r_idx in self.candidate_indices:
            med_obs = np.median(D_obs[:, r_idx])
            iqr_obs = stats.iqr(D_obs[:, r_idx])
            iqr_obs = max(1e-4, iqr_obs)
            
            # Anomaly shift on interventional data
            anom_shifts = np.abs(D_int[:, r_idx] - med_obs) / iqr_obs
            scores[r_idx] = float(np.mean(anom_shifts))
            
        ranking = [(self.node_names[r], scores[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

# --- Algorithm 5: SimpleRCA (Fang et al., 2025) ---
class SimpleRCA(BaseRCA):
    """
    SimpleRCA: Difference between 95th percentile of normal and maximum of anomalous data.
    """
    def fit_and_rank(self, D_obs, D_int):
        scores = {}
        for r_idx in self.candidate_indices:
            p95_obs = np.percentile(D_obs[:, r_idx], 95)
            p95_int = np.percentile(D_int[:, r_idx], 95)
            std_obs = max(1e-4, np.std(D_obs[:, r_idx]))
            
            score = (p95_int - p95_obs) / std_obs
            scores[r_idx] = float(max(0.0, score))
            
        ranking = [(self.node_names[r], scores[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

# --- Algorithm 6: Smooth Traversal (ST - Orchard et al., 2025) ---
class SmoothTraversal(BaseRCA):
    """
    ST traverses from the trigger point (Vibration RMS, node 7) to its ancestors in the DAG,
    computing step change in anomaly score.
    """
    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        # Compute marginal z-score for all nodes
        z_scores = {}
        for i in range(self.d):
            m_obs, s_obs = np.mean(D_obs[:, i]), max(1e-4, np.std(D_obs[:, i]))
            z_scores[i] = abs(np.mean(D_int[:, i]) - m_obs) / s_obs
            
        scores = {}
        for r_idx in self.candidate_indices:
            parents = list(dag.predecessors(r_idx))
            if len(parents) == 0:
                step_change = z_scores[r_idx]
            else:
                par_z = np.mean([z_scores[p] for p in parents])
                step_change = max(0.0, z_scores[r_idx] - par_z)
            scores[r_idx] = step_change

        ranking = [(self.node_names[r], scores[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

# ==============================================================================
# 3. BENCHMARK & COMPARATIVE EVALUATION ENGINE
# ==============================================================================
def run_rca_benchmark():
    print("=" * 80)
    print("SOTA RCA BENCHMARK ON MACHINING CHATTER DYNAMICS")
    print("Comparing: BRCD (Bayesian), RCD (PC/CI), RCG (CMI), BARO, SimpleRCA, ST")
    print("=" * 80)

    system = ChatterCausalSystem()
    algorithms = {
        "BRCD (Ours)": BRCD(system),
        "RCD": RCD(system),
        "RCG": RCG(system),
        "BARO": BARO(system),
        "SimpleRCA": SimpleRCA(system),
        "SmoothTraversal": SmoothTraversal(system)
    }

    sample_sizes = [5, 10, 20, 50, 100]
    n_scenarios_per_rc = 20
    candidates = system.candidate_root_causes
    
    # Metrics storage: {method: {sample_size: {"Top-1": [], "Top-3": [], "Top-5": [], "MRR": [], "time": []}}}
    benchmark_results = {m: {s: {"Top-1": 0, "Top-3": 0, "Top-5": 0, "MRR": 0.0, "total": 0, "time": 0.0} for s in sample_sizes} for m in algorithms}

    # Generate 10,000 observational baseline samples
    D_obs = system.sample_data(n_samples=5000, root_cause=None, seed=42)

    for m_samples in sample_sizes:
        print(f"\nEvaluating with anomalous sample size m = {m_samples}...")
        for rc_true in candidates:
            for trial in range(n_scenarios_per_rc):
                seed = 1000 * m_samples + trial
                # Generate interventional dataset
                D_int = system.sample_data(n_samples=m_samples, root_cause=rc_true, severity=1.0, seed=seed)
                
                for alg_name, alg in algorithms.items():
                    t0 = time.time()
                    ranked = alg.fit_and_rank(D_obs, D_int)
                    elapsed = time.time() - t0
                    
                    ranked_names = [r[0] for r in ranked]
                    
                    # Top-1, Top-3, Top-5, MRR
                    is_top1 = 1 if (len(ranked_names) > 0 and ranked_names[0] == rc_true) else 0
                    is_top3 = 1 if rc_true in ranked_names[:3] else 0
                    is_top5 = 1 if rc_true in ranked_names[:5] else 0
                    
                    rank_pos = ranked_names.index(rc_true) + 1 if rc_true in ranked_names else len(candidates) + 1
                    mrr = 1.0 / rank_pos
                    
                    res_dict = benchmark_results[alg_name][m_samples]
                    res_dict["Top-1"] += is_top1
                    res_dict["Top-3"] += is_top3
                    res_dict["Top-5"] += is_top5
                    res_dict["MRR"] += mrr
                    res_dict["time"] += elapsed
                    res_dict["total"] += 1

    # Summarize Table
    summary_rows = []
    for alg_name in algorithms:
        row = {"Algorithm": alg_name}
        for s in sample_sizes:
            tot = benchmark_results[alg_name][s]["total"]
            top1_acc = benchmark_results[alg_name][s]["Top-1"] / tot
            mrr_val = benchmark_results[alg_name][s]["MRR"] / tot
            row[f"Top-1 (m={s})"] = f"{top1_acc:.2f}"
            row[f"MRR (m={s})"] = f"{mrr_val:.2f}"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON TABLE ACROSS SAMPLE SIZES (m = 5 to 100)")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    # ==========================================================================
    # PLOT: PUBLICATION GRADE ACCURACY COMPARISON CURVES
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=300, sharey=True)
    
    palette = {
        "BRCD (Ours)": "#D90429",       # Vibrant Crimson
        "RCD": "#1D4ED8",               # Royal Blue
        "RCG": "#2A9D8F",               # Emerald Teal
        "BARO": "#EAB308",              # Amber Yellow
        "SimpleRCA": "#8B5CF6",         # Violet
        "SmoothTraversal": "#6B7280"    # Slate Gray
    }
    
    metrics = ["Top-1", "Top-3", "Top-5"]
    x_indices = np.arange(len(sample_sizes))
    x_labels = [str(s) for s in sample_sizes]

    for ax, metric in zip(axes, metrics):
        for alg_name in algorithms:
            vals = []
            for s in sample_sizes:
                tot = benchmark_results[alg_name][s]["total"]
                vals.append(benchmark_results[alg_name][s][metric] / tot)
                
            ax.plot(
                x_indices, vals,
                label=alg_name,
                marker='o',
                linewidth=1.8 if "BRCD" in alg_name else 1.2,
                markersize=5.5 if "BRCD" in alg_name else 4.5,
                color=palette[alg_name],
                alpha=0.95
            )
            
        ax.set_title(f"Accuracy@{metric.split('-')[1]} vs Sample Size", fontsize=11, pad=8)
        ax.set_xlabel("Interventional Sample Size $m$", fontsize=10, labelpad=5)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, fontweight="bold", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.grid(True, linestyle=":", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    axes[0].set_ylabel("Accuracy@L", fontsize=11, fontweight="bold")
    
    # Unified top legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=6,
        frameon=False,
        prop={"weight": "bold", "size": 9.5}
    )

    plt.subplots_adjust(top=0.82, bottom=0.18, left=0.07, right=0.98, wspace=0.1)
    output_png = "rca_sota_accuracy_comparison.png"
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    print(f"\nSaved SOTA comparison figure to {output_png}")

    return df_summary

if __name__ == "__main__":
    run_rca_benchmark()
