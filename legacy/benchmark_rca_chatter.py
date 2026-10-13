import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
import time

# Publication-grade aesthetic style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.6,
    "grid.alpha": 0.35
})

# ==============================================================================
# 1. PHYSICAL CHATTER CAUSAL SYSTEM & BOUNDARY PERTURBATION GENERATOR
# ==============================================================================
class ChatterCausalSystem:
    """
    Physical Machining Causal Model with Altintas Regenerative Dynamics.
    Graph Nodes (14 Telemetry Metrics):
      0: Stiffness_k (N/m)
      1: Damping_zeta (ratio)
      2: Tool_Wear_Kt (N/m^2)
      3: Cut_Depth_a (mm)
      4: Spindle_RPM (rev/min)
      5: Stability_Limit_alim (mm)
      6: Chatter_Severity (dimensionless regenerative amplitude)
      7: Vibration_RMS (um)
      8: Dominant_Freq (Hz)
      9: Spectral_Ratio (chatter energy ratio)
      10: Force_Mean (N)
      11: Force_Peak (N)
      12: Spindle_Power (W)
      13: Surface_Roughness (um Ra)
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
        
        self.candidate_root_causes = [
            "Stiffness_k",
            "Damping_zeta",
            "Tool_Wear_Kt",
            "Cut_Depth_a",
            "Spindle_RPM"
        ]
        self.candidate_indices = [self.name_to_idx[n] for n in self.candidate_root_causes]
        
        # Ground-Truth Causal DAG
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(range(self.d))
        
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

    def forward_physics(self, k, zeta, kt, a, rpm, noise_scale=1.0):
        N = len(k)
        data = np.zeros((N, self.d))
        data[:, 0] = k
        data[:, 1] = zeta
        data[:, 2] = kt
        data[:, 3] = a
        data[:, 4] = rpm
        
        # Physical Stability Limit: a_lim = (2 * zeta * k / kt) * lobe_geometry(RPM)
        lobe_factor = 1.0 - 0.45 * np.sin(4 * np.pi * rpm)
        a_lim = (0.85 * k * zeta / np.maximum(0.1, kt)) * lobe_factor + np.random.normal(0, 0.015 * noise_scale, N)
        data[:, 5] = a_lim
        
        # Regenerative chatter onset with non-linear saturation
        diff = a - a_lim
        chatter_sev = np.log1p(np.exp(np.clip(diff * 14.0, -12, 16))) * 0.28 + np.random.normal(0, 0.01 * noise_scale, N)
        chatter_sev = np.maximum(0.0, chatter_sev)
        data[:, 6] = chatter_sev
        
        # Multi-sensor telemetry
        data[:, 7] = 0.12 + 1.8 * chatter_sev + np.random.normal(0, 0.02 * noise_scale, N) # Vib RMS (um)
        data[:, 8] = 250.0 * np.sqrt(np.maximum(0.1, k)) + 8.0 * chatter_sev + np.random.normal(0, 1.5 * noise_scale, N) # Dom Freq (Hz)
        data[:, 9] = 0.08 + 0.80 / (1.0 + np.exp(-6.0 * (chatter_sev - 0.15))) + np.random.normal(0, 0.02 * noise_scale, N) # Spectral ratio
        
        data[:, 10] = 300.0 * (kt * a) + np.random.normal(0, 8.0 * noise_scale, N) # Force Mean (N)
        data[:, 11] = data[:, 10] * (1.15 + 1.9 * chatter_sev) + np.random.normal(0, 15.0 * noise_scale, N) # Force Peak (N)
        data[:, 12] = 0.75 * data[:, 10] * rpm + 0.12 * data[:, 11] + np.random.normal(0, 10.0 * noise_scale, N) # Power (W)
        data[:, 13] = 0.35 + 2.5 * data[:, 7] + 0.0008 * data[:, 11] + np.random.normal(0, 0.03 * noise_scale, N) # Surface Ra (um)
        
        return data

    def sample_observational_data(self, n_samples=5000, seed=42):
        np.random.seed(seed)
        k = np.random.normal(1.0, 0.025, n_samples)
        zeta = np.random.normal(1.0, 0.025, n_samples)
        kt = np.random.normal(1.0, 0.03, n_samples)
        a = np.random.normal(0.50, 0.02, n_samples)
        rpm = np.random.normal(0.50, 0.02, n_samples)
        return self.forward_physics(k, zeta, kt, a, rpm, noise_scale=1.0)

    def sample_subtle_interventional_data(self, n_samples=20, root_cause="Stiffness_k", seed=None):
        """
        Physics-Informed Subtle Boundary Sampling:
        Injects realistic, borderline physical parameter perturbations without unrealistic artifacts.
        """
        if seed is not None:
            np.random.seed(seed)
            
        k = np.random.normal(1.0, 0.025, n_samples)
        zeta = np.random.normal(1.0, 0.025, n_samples)
        kt = np.random.normal(1.0, 0.03, n_samples)
        a = np.random.normal(0.50, 0.02, n_samples)
        rpm = np.random.normal(0.50, 0.02, n_samples)
        
        # Subtle physical parameter shifts (15% - 25% change)
        if root_cause == "Stiffness_k":
            k = np.random.normal(0.82, 0.03, n_samples) # 18% stiffness softening (fixture play)
        elif root_cause == "Damping_zeta":
            zeta = np.random.normal(0.70, 0.03, n_samples) # 30% damping loss (slender tool overhang)
        elif root_cause == "Tool_Wear_Kt":
            kt = np.random.normal(1.28, 0.04, n_samples) # 28% force coeff increase (flank wear)
        elif root_cause == "Cut_Depth_a":
            a = np.random.normal(0.72, 0.03, n_samples) # Aggressive depth exceeding local a_lim
        elif root_cause == "Spindle_RPM":
            rpm = np.random.normal(0.75, 0.03, n_samples) # Shift to stability lobe trough
            
        return self.forward_physics(k, zeta, kt, a, rpm, noise_scale=1.1)

# ==============================================================================
# 2. STATE-OF-THE-ART RCA ALGORITHMS
# ==============================================================================

# 1. BRCD: Bayesian Root Cause Discovery (Lee et al., ICML 2026)
class BRCD:
    def __init__(self, system):
        self.system = system
        self.d = system.d
        self.candidate_indices = system.candidate_indices
        self.node_names = system.node_names

    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        m = len(D_int)
        
        obs_models = {}
        for i in range(self.d):
            parents = list(dag.predecessors(i))
            if len(parents) == 0:
                mu = np.mean(D_obs[:, i])
                var = max(1e-4, np.var(D_obs[:, i]))
                obs_models[i] = {"type": "root", "mu": mu, "var": var}
            else:
                X_par = np.hstack([np.ones((len(D_obs), 1)), D_obs[:, parents]])
                beta = np.linalg.lstsq(X_par, D_obs[:, i], rcond=None)[0]
                residuals = D_obs[:, i] - X_par @ beta
                var = max(1e-4, np.var(residuals))
                obs_models[i] = {"type": "child", "beta": beta, "var": var, "parents": parents}

        scores = {}
        for r_idx in self.candidate_indices:
            parents = list(dag.predecessors(r_idx))
            if len(parents) == 0:
                mu_obs = obs_models[r_idx]["mu"]
                var_obs = obs_models[r_idx]["var"]
                mu_int = np.mean(D_int[:, r_idx])
                direct_shift = ((mu_int - mu_obs) ** 2) / var_obs
            else:
                parents = obs_models[r_idx]["parents"]
                beta = obs_models[r_idx]["beta"]
                var_obs = obs_models[r_idx]["var"]
                X_par_int = np.hstack([np.ones((m, 1)), D_int[:, parents]])
                pred = X_par_int @ beta
                res = D_int[:, r_idx] - pred
                direct_shift = (np.mean(res) ** 2) / var_obs
                
            descendants = nx.descendants(dag, r_idx)
            causal_relevance = 1.0 + 0.1 * len(descendants)
            scores[r_idx] = float(direct_shift * causal_relevance)

        ranking = [(self.node_names[r], scores[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

# 2. RCD: PC-based Conditional Independence testing (Ikram et al., NeurIPS 2022)
class RCD:
    def __init__(self, system):
        self.system = system
        self.candidate_indices = system.candidate_indices
        self.node_names = system.node_names

    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        F = np.concatenate([np.zeros(len(D_obs)), np.ones(len(D_int))])
        D_all = np.vstack([D_obs, D_int])
        
        p_scores = {}
        for r_idx in self.candidate_indices:
            parents = list(dag.predecessors(r_idx))
            if len(parents) == 0:
                corr, p_val = stats.pearsonr(F, D_all[:, r_idx])
            else:
                X_par = np.hstack([np.ones((len(D_all), 1)), D_all[:, parents]])
                beta_f = np.linalg.lstsq(X_par, F, rcond=None)[0]
                res_f = F - X_par @ beta_f
                
                beta_x = np.linalg.lstsq(X_par, D_all[:, r_idx], rcond=None)[0]
                res_x = D_all[:, r_idx] - X_par @ beta_x
                corr, p_val = stats.pearsonr(res_f, res_x)
                
            p_scores[r_idx] = -np.log10(max(1e-15, p_val))

        ranking = [(self.node_names[r], p_scores[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

# 3. RCG: Conditional Mutual Information (Ikram et al., UAI 2025)
class RCG:
    def __init__(self, system):
        self.system = system
        self.candidate_indices = system.candidate_indices
        self.node_names = system.node_names

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

# 4. BARO: Multivariate Robust Deviation (Pham et al., FSE 2024)
class BARO:
    def __init__(self, system):
        self.system = system
        self.candidate_indices = system.candidate_indices
        self.node_names = system.node_names

    def fit_and_rank(self, D_obs, D_int):
        scores = {}
        for r_idx in self.candidate_indices:
            med_obs = np.median(D_obs[:, r_idx])
            iqr_obs = max(1e-4, stats.iqr(D_obs[:, r_idx]))
            anom_shifts = np.abs(D_int[:, r_idx] - med_obs) / iqr_obs
            scores[r_idx] = float(np.mean(anom_shifts))
            
        ranking = [(self.node_names[r], scores[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

# 5. SimpleRCA: 95th Percentile Difference (Fang et al., 2025)
class SimpleRCA:
    def __init__(self, system):
        self.system = system
        self.candidate_indices = system.candidate_indices
        self.node_names = system.node_names

    def fit_and_rank(self, D_obs, D_int):
        scores = {}
        for r_idx in self.candidate_indices:
            p95_obs = np.percentile(D_obs[:, r_idx], 95)
            p95_int = np.percentile(D_int[:, r_idx], 95)
            std_obs = max(1e-4, np.std(D_obs[:, r_idx]))
            scores[r_idx] = float(max(0.0, (p95_int - p95_obs) / std_obs))
            
        ranking = [(self.node_names[r], scores[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

# 6. SmoothTraversal: Causal Graph Traversal (Orchard et al., NeurIPS 2025)
class SmoothTraversal:
    def __init__(self, system):
        self.system = system
        self.candidate_indices = system.candidate_indices
        self.node_names = system.node_names

    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        z_scores = {}
        for i in range(self.system.d):
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
# 3. BENCHMARK EXECUTION & PLOTTING SUITE
# ==============================================================================
def run_full_benchmark():
    print("=" * 85)
    print("PHYSICS-INFORMED ROOT CAUSE ANALYSIS BENCHMARK ON MACHINING CHATTER")
    print("Comparing SOTA Methods across Sample Sizes (m = 5 to 100)")
    print("=" * 85)

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
    n_trials_per_rc = 25
    candidates = system.candidate_root_causes
    
    benchmark_results = {m: {s: {"Top-1": 0, "Top-3": 0, "Top-5": 0, "MRR": 0.0, "total": 0, "time": 0.0} for s in sample_sizes} for m in algorithms}

    # Generate normal baseline data
    D_obs = system.sample_observational_data(n_samples=5000, seed=42)

    for m in sample_sizes:
        print(f"Testing on m = {m} anomalous samples...")
        for rc_name in candidates:
            for trial in range(n_trials_per_rc):
                seed = 1000 * m + trial
                D_int = system.sample_subtle_interventional_data(n_samples=m, root_cause=rc_name, seed=seed)

                for alg_name, alg in algorithms.items():
                    t0 = time.time()
                    ranked = alg.fit_and_rank(D_obs, D_int)
                    elapsed = time.time() - t0
                    
                    ranked_names = [r[0] for r in ranked]
                    
                    is_top1 = 1 if (len(ranked_names) > 0 and ranked_names[0] == rc_name) else 0
                    is_top3 = 1 if rc_name in ranked_names[:3] else 0
                    is_top5 = 1 if rc_name in ranked_names[:5] else 0
                    
                    rank_pos = ranked_names.index(rc_name) + 1 if rc_name in ranked_names else len(candidates) + 1
                    mrr = 1.0 / rank_pos
                    
                    res_dict = benchmark_results[alg_name][m]
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
    print("\n" + "=" * 85)
    print("BENCHMARK RESULTS SUMMARY (Sample Size m = 5 to 100)")
    print("=" * 85)
    print(df_summary.to_string(index=False))

    # ==========================================================================
    # PLOT: ICML FORMAT ACCURACY COMPARISON CURVES
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=300, sharey=True)
    
    palette = {
        "BRCD (Ours)": "#D90429",       # Crimson
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
                linewidth=2.0 if "BRCD" in alg_name else 1.3,
                markersize=5.5 if "BRCD" in alg_name else 4.2,
                color=palette[alg_name],
                alpha=0.92
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
    print(f"\nSaved SOTA Comparison Plot to {output_png}")

    return df_summary

if __name__ == "__main__":
    run_full_benchmark()
