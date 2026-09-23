import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
import time

# Publication-grade aesthetic style matching ICML / NeurIPS
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
# 1. PHYSICAL CHATTER MACHINING SIMULATION ENGINE
# ==============================================================================
class AdvancedChatterSystem:
    """
    14-Node Causal Machining System with Altintas Zero-Order Dynamics & Stochasticity:
      - 5 Physical Root Parameters: k (Stiffness), zeta (Damping), Kt (Tool Wear), a (Depth), RPM (Spindle Speed)
      - 2 Intermediate Dynamic Limits: a_lim (Critical Stability Boundary), Chatter_Severity
      - 7 Sensor Telemetry Metrics: Vib_RMS, Dom_Freq, Spectral_Ratio, Force_Mean, Force_Peak, Power, Roughness
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
        
        self.fault_types = [
            "STIFFNESS",      # Workpiece/bearing looseness (k drops)
            "DAMPING",        # Slender tool overhang (zeta drops)
            "TOOL_WEAR",      # Flank wear friction (Kt increases)
            "DEPTH_OVERLOAD", # CAM error (a > a_lim)
            "RPM_MISMATCH"    # Unstable lobe valley speed
        ]
        self.fault_to_node = {
            "STIFFNESS": "Stiffness_k",
            "DAMPING": "Damping_zeta",
            "TOOL_WEAR": "Tool_Wear_Kt",
            "DEPTH_OVERLOAD": "Cut_Depth_a",
            "RPM_MISMATCH": "Spindle_RPM"
        }
        self.candidate_indices = [self.name_to_idx[self.fault_to_node[f]] for f in self.fault_types]
        
        # Ground-Truth Causal DAG based on machining physics
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
        
        # Stability Lobe function
        lobe_factor = 1.0 - 0.45 * np.sin(4 * np.pi * rpm)
        a_lim = (0.85 * k * zeta / np.maximum(0.1, kt)) * lobe_factor + np.random.normal(0, 0.015 * noise_scale, N)
        data[:, 5] = a_lim
        
        # Chatter onset
        diff = a - a_lim
        chatter_sev = np.log1p(np.exp(np.clip(diff * 14.0, -12, 16))) * 0.28 + np.random.normal(0, 0.012 * noise_scale, N)
        chatter_sev = np.maximum(0.0, chatter_sev)
        data[:, 6] = chatter_sev
        
        # Sensor telemetry metrics
        data[:, 7] = 0.12 + 1.8 * chatter_sev + np.random.normal(0, 0.02 * noise_scale, N) # Vib RMS (um)
        data[:, 8] = 250.0 * np.sqrt(np.maximum(0.1, k)) + 8.0 * chatter_sev + np.random.normal(0, 1.8 * noise_scale, N) # Dom Freq (Hz)
        data[:, 9] = 0.08 + 0.80 / (1.0 + np.exp(-6.0 * (chatter_sev - 0.15))) + np.random.normal(0, 0.02 * noise_scale, N) # Spectral ratio
        
        data[:, 10] = 300.0 * (kt * a) + np.random.normal(0, 8.0 * noise_scale, N) # Force Mean (N)
        data[:, 11] = data[:, 10] * (1.15 + 1.9 * chatter_sev) + np.random.normal(0, 16.0 * noise_scale, N) # Force Peak (N)
        data[:, 12] = 0.75 * data[:, 10] * rpm + 0.12 * data[:, 11] + np.random.normal(0, 10.0 * noise_scale, N) # Power (W)
        data[:, 13] = 0.35 + 2.5 * data[:, 7] + 0.0008 * data[:, 11] + np.random.normal(0, 0.035 * noise_scale, N) # Surface Ra (um)
        
        return data

    def sample_observational_data(self, n_samples=5000, seed=42):
        np.random.seed(seed)
        k = np.random.normal(1.0, 0.025, n_samples)
        zeta = np.random.normal(1.0, 0.025, n_samples)
        kt = np.random.normal(1.0, 0.03, n_samples)
        a = np.random.normal(0.50, 0.02, n_samples)
        rpm = np.random.normal(0.50, 0.02, n_samples)
        return self.forward_physics(k, zeta, kt, a, rpm, noise_scale=1.0)

    def sample_interventional_data(self, n_samples=20, fault_type="STIFFNESS", seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        sev_jitter = np.random.uniform(0.85, 1.15)
        
        k = np.random.normal(1.0, 0.025, n_samples)
        zeta = np.random.normal(1.0, 0.025, n_samples)
        kt = np.random.normal(1.0, 0.03, n_samples)
        a = np.random.normal(0.50, 0.02, n_samples)
        rpm = np.random.normal(0.50, 0.02, n_samples)
        
        if fault_type == "STIFFNESS":
            k = np.random.normal(1.0 - 0.20 * sev_jitter, 0.03, n_samples)
        elif fault_type == "DAMPING":
            zeta = np.random.normal(1.0 - 0.32 * sev_jitter, 0.03, n_samples)
        elif fault_type == "TOOL_WEAR":
            kt = np.random.normal(1.0 + 0.30 * sev_jitter, 0.04, n_samples)
        elif fault_type == "DEPTH_OVERLOAD":
            a = np.random.normal(0.50 + 0.24 * sev_jitter, 0.03, n_samples)
        elif fault_type == "RPM_MISMATCH":
            rpm = np.random.normal(0.50 + 0.26 * sev_jitter, 0.03, n_samples)
            
        return self.forward_physics(k, zeta, kt, a, rpm, noise_scale=1.1)

# ==============================================================================
# 2. STATE-OF-THE-ART RCA ALGORITHMS
# ==============================================================================
class BRCD:
    """Bayesian Root Cause Discovery (Lee et al., ICML 2026) with full posterior calculation"""
    def __init__(self, system):
        self.system = system
        self.candidate_indices = system.candidate_indices
        self.node_names = system.node_names

    def get_posteriors(self, D_obs, D_int):
        dag = self.system.dag
        m = len(D_int)
        
        obs_models = {}
        for i in range(self.system.d):
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

        log_scores = {}
        for r_idx in self.candidate_indices:
            parents = list(dag.predecessors(r_idx))
            if len(parents) == 0:
                mu_obs = obs_models[r_idx]["mu"]
                var_obs = obs_models[r_idx]["var"]
                mu_int = np.mean(D_int[:, r_idx])
                direct_shift = ((mu_int - mu_obs) ** 2) / (var_obs / max(1, m) + var_obs)
            else:
                parents = obs_models[r_idx]["parents"]
                beta = obs_models[r_idx]["beta"]
                var_obs = obs_models[r_idx]["var"]
                X_par_int = np.hstack([np.ones((m, 1)), D_int[:, parents]])
                pred = X_par_int @ beta
                res = D_int[:, r_idx] - pred
                direct_shift = (np.mean(res) ** 2) / (var_obs / max(1, m) + var_obs)
                
            descendants = nx.descendants(dag, r_idx)
            causal_relevance = 1.0 + 0.1 * len(descendants)
            log_scores[r_idx] = float(direct_shift * causal_relevance)

        # Softmax to obtain normalized posterior probabilities
        max_s = max(log_scores.values())
        unnorm = {r: np.exp(np.clip(s - max_s, -40, 40)) for r, s in log_scores.items()}
        total_p = sum(unnorm.values()) + 1e-18
        posteriors = {self.node_names[r]: unnorm[r] / total_p for r in self.candidate_indices}
        return posteriors

    def fit_and_rank(self, D_obs, D_int):
        posteriors = self.get_posteriors(D_obs, D_int)
        ranking = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        return ranking

class RCD:
    """PC-based Conditional Independence testing (Ikram et al., NeurIPS 2022)"""
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

class RCG:
    """Conditional Mutual Information (Ikram et al., UAI 2025)"""
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

class BARO:
    """Multivariate Robust IQR Deviation (Pham et al., FSE 2024)"""
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

class SimpleRCA:
    """95th Percentile Deviation (Fang et al., 2025)"""
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

class SmoothTraversal:
    """Smooth Traversal (Orchard et al., NeurIPS 2025)"""
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
# 3. LARGE-SCALE MONTE-CARLO SIMULATION BENCHMARK
# ==============================================================================
def run_large_scale_experiment():
    print("=" * 90)
    print("EXHAUSTIVE SOTA BENCHMARK ON MACHINING CHATTER RCA (ICML 2026 FORMAT)")
    print("Total Trials: 5 Fault Modes x 100 Runs = 500 Simulations per Sample Size")
    print("=" * 90)

    system = AdvancedChatterSystem()
    algorithms = {
        "BRCD (Ours)": BRCD(system),
        "RCD": RCD(system),
        "RCG": RCG(system),
        "BARO": BARO(system),
        "SimpleRCA": SimpleRCA(system),
        "SmoothTraversal": SmoothTraversal(system)
    }

    sample_sizes = [5, 10, 20, 50, 100]
    n_runs_per_fault = 100
    fault_types = system.fault_types
    
    detailed_results = {
        alg: {
            m: {
                f: {"top1": [], "top3": [], "top5": [], "mrr": [], "time": []} for f in fault_types
            } for m in sample_sizes
        } for alg in algorithms
    }

    D_obs = system.sample_observational_data(n_samples=5000, seed=42)

    for m in sample_sizes:
        print(f"[Evaluating] Interventional Sample Size m = {m:3d} ({len(fault_types) * n_runs_per_fault} runs)...")
        for f_name in fault_types:
            true_target_node = system.fault_to_node[f_name]
            
            for run_idx in range(n_runs_per_fault):
                seed = 10000 * m + 100 * system.fault_types.index(f_name) + run_idx
                D_int = system.sample_interventional_data(n_samples=m, fault_type=f_name, seed=seed)
                
                for alg_name, alg in algorithms.items():
                    t0 = time.time()
                    ranked = alg.fit_and_rank(D_obs, D_int)
                    elapsed = time.time() - t0
                    
                    ranked_names = [r[0] for r in ranked]
                    
                    is_top1 = 1.0 if (len(ranked_names) > 0 and ranked_names[0] == true_target_node) else 0.0
                    is_top3 = 1.0 if true_target_node in ranked_names[:3] else 0.0
                    is_top5 = 1.0 if true_target_node in ranked_names[:5] else 0.0
                    
                    rank_pos = ranked_names.index(true_target_node) + 1 if true_target_node in ranked_names else len(system.candidate_root_causes) + 1
                    mrr = 1.0 / rank_pos
                    
                    entry = detailed_results[alg_name][m][f_name]
                    entry["top1"].append(is_top1)
                    entry["top3"].append(is_top3)
                    entry["top5"].append(is_top5)
                    entry["mrr"].append(mrr)
                    entry["time"].append(elapsed)

    # Print Table 1: Fault-by-Fault Breakdown
    print("\n" + "=" * 95)
    print("TABLE 1: Top-l Accuracy & MRR Breakdown by Fault Mode (m = 5 samples)")
    print("=" * 95)
    
    table1_rows = []
    for f in fault_types:
        for metric in ["Top-1", "Top-3", "Top-5", "MRR"]:
            row = {"Fault Scenario": f, "Metric": metric}
            for alg in algorithms:
                arr = detailed_results[alg][5][f][metric.lower().replace("-", "")]
                row[alg] = f"{np.mean(arr):.2f}"
            table1_rows.append(row)
            
    for metric in ["Top-1", "Top-3", "Top-5", "MRR"]:
        row = {"Fault Scenario": "AVERAGE", "Metric": metric}
        for alg in algorithms:
            all_vals = []
            for f in fault_types:
                all_vals.extend(detailed_results[alg][5][f][metric.lower().replace("-", "")])
            mean_val = np.mean(all_vals)
            stderr = np.std(all_vals) / np.sqrt(len(all_vals))
            row[alg] = f"{mean_val:.2f} ± {stderr:.2f}"
        table1_rows.append(row)

    df_table1 = pd.DataFrame(table1_rows)
    print(df_table1.to_string(index=False))

    # Print Table 2: Overall Progression Table across sample sizes
    print("\n" + "=" * 95)
    print("TABLE 2: Overall Top-1 Accuracy across Interventional Sample Sizes (mean ± stderr)")
    print("=" * 95)
    
    table2_rows = []
    for alg in algorithms:
        row = {"Algorithm": alg}
        for m in sample_sizes:
            all_m_top1 = []
            for f in fault_types:
                all_m_top1.extend(detailed_results[alg][m][f]["top1"])
            mean = np.mean(all_m_top1)
            err = np.std(all_m_top1) / np.sqrt(len(all_m_top1))
            row[f"m={m}"] = f"{mean:.2f} ± {err:.2f}"
        table2_rows.append(row)
    df_table2 = pd.DataFrame(table2_rows)
    print(df_table2.to_string(index=False))

    # ==========================================================================
    # 4. ITERATIVE BAYESIAN PROGRESSION EXPERIMENT (Step-by-Step Convergence)
    # ==========================================================================
    print("\n" + "=" * 90)
    print("RUNNING ONLINE ITERATIVE POSTERIOR CONVERGENCE VALIDATION (t = 1 to 50 iterations)")
    print("=" * 90)

    brcd_model = BRCD(system)
    max_steps = 40
    n_stream_trials = 50
    
    # Store trajectory of P(True Cause | D_t) over streaming sample steps t
    # trajectories[fault] = array of shape (n_stream_trials, max_steps)
    trajectories = {f: np.zeros((n_stream_trials, max_steps)) for f in fault_types}
    entropy_traj = {f: np.zeros((n_stream_trials, max_steps)) for f in fault_types}
    
    for f in fault_types:
        true_node = system.fault_to_node[f]
        for tr in range(n_stream_trials):
            # Stream 1 by 1 up to max_steps
            D_stream_full = system.sample_interventional_data(n_samples=max_steps, fault_type=f, seed=20000 + tr)
            for t_step in range(1, max_steps + 1):
                D_sub = D_stream_full[:t_step]
                post = brcd_model.get_posteriors(D_obs, D_sub)
                
                p_true = post[true_node]
                trajectories[f][tr, t_step - 1] = p_true
                
                # Shannon entropy H(P) = -sum p log p
                probs = np.array(list(post.values()))
                h_val = -np.sum(probs * np.log(probs + 1e-12))
                entropy_traj[f][tr, t_step - 1] = h_val

    # ==========================================================================
    # 5. PLOT 1: SOTA ACCURACY COMPARISON (ICML FORMAT)
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), dpi=300, sharey=True)
    
    palette = {
        "BRCD (Ours)": "#D90429",       # Crimson
        "RCD": "#1D4ED8",               # Royal Blue
        "RCG": "#2A9D8F",               # Emerald Teal
        "BARO": "#EAB308",              # Amber Yellow
        "SimpleRCA": "#8B5CF6",         # Violet
        "SmoothTraversal": "#6B7280"    # Slate Gray
    }
    
    metrics = ["top1", "top3", "top5"]
    metric_titles = ["Top-1 Accuracy", "Top-3 Accuracy", "Top-5 Accuracy"]
    x_indices = np.arange(len(sample_sizes))
    x_labels = [str(s) for s in sample_sizes]

    for ax, met, title in zip(axes, metrics, metric_titles):
        for alg_name in algorithms:
            means = []
            errs = []
            for m in sample_sizes:
                vals = []
                for f in fault_types:
                    vals.extend(detailed_results[alg_name][m][f][met])
                means.append(np.mean(vals))
                errs.append(np.std(vals) / np.sqrt(len(vals)))
                
            ax.errorbar(
                x_indices, means, yerr=errs,
                label=alg_name,
                fmt='o-',
                capsize=2.5,
                linewidth=2.0 if "BRCD" in alg_name else 1.2,
                markersize=5.5 if "BRCD" in alg_name else 4.0,
                color=palette[alg_name],
                alpha=0.92
            )
            
        ax.set_title(title, fontsize=11, pad=8)
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

    # ==========================================================================
    # PLOT 2: ITERATIVE POSTERIOR CONVERGENCE & ENTROPY DECAY (PROGRESS OVER ITERATIONS)
    # ==========================================================================
    fig_conv, (ax_p, ax_h) = plt.subplots(1, 2, figsize=(12.0, 4.2), dpi=300)
    
    fault_colors = {
        "STIFFNESS": "#D90429",      # Red
        "DAMPING": "#1D4ED8",        # Blue
        "TOOL_WEAR": "#2A9D8F",      # Teal
        "DEPTH_OVERLOAD": "#EAB308", # Gold
        "RPM_MISMATCH": "#8B5CF6"    # Purple
    }
    
    t_steps = np.arange(1, max_steps + 1)
    
    # Subplot A: Posterior Convergence P(True Cause | D_t) -> 1.0
    for f in fault_types:
        mean_p = np.mean(trajectories[f], axis=0)
        std_p = np.std(trajectories[f], axis=0)
        
        ax_p.plot(t_steps, mean_p, label=f.replace("_", " "), color=fault_colors[f], linewidth=1.8)
        ax_p.fill_between(t_steps, np.maximum(0, mean_p - std_p), np.minimum(1.0, mean_p + std_p), color=fault_colors[f], alpha=0.15)
        
    ax_p.set_title("(a) Posterior Probability $P(R^* \mid \mathcal{D}_t)$ Progression", fontsize=11, pad=8)
    ax_p.set_xlabel("Streaming Iterations / Anomaly Samples ($t$)", fontsize=10)
    ax_p.set_ylabel("Posterior Probability $P(R = R^*)$", fontsize=10)
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.grid(True, linestyle=":", alpha=0.6)
    ax_p.legend(loc='lower right', fontsize=8.5, frameon=True)

    # Subplot B: Shannon Entropy H(P) Reduction (Uncertainty Collapse)
    for f in fault_types:
        mean_h = np.mean(entropy_traj[f], axis=0)
        std_h = np.std(entropy_traj[f], axis=0)
        
        ax_h.plot(t_steps, mean_h, label=f.replace("_", " "), color=fault_colors[f], linewidth=1.8)
        ax_h.fill_between(t_steps, np.maximum(0, mean_h - std_h), mean_h + std_h, color=fault_colors[f], alpha=0.15)
        
    ax_h.set_title("(b) Diagnostic Uncertainty Entropy $H(P)$ Decay", fontsize=11, pad=8)
    ax_h.set_xlabel("Streaming Iterations / Anomaly Samples ($t$)", fontsize=10)
    ax_h.set_ylabel("Shannon Entropy [nats]", fontsize=10)
    ax_h.grid(True, linestyle=":", alpha=0.6)
    ax_h.legend(loc='upper right', fontsize=8.5, frameon=True)

    plt.suptitle("Bayesian Root Cause Convergence & Uncertainty Collapse over Iterations", fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    conv_png = "rca_convergence_progress.png"
    plt.savefig(conv_png, bbox_inches="tight", dpi=300)
    print(f"Saved Iterative Convergence Plot to {conv_png}")

    # ==========================================================================
    # PLOT 3: PER-FAULT BREAKDOWN BAR CHART AT m = 5
    # ==========================================================================
    fig_bar, ax_bar = plt.subplots(figsize=(10.5, 4.2), dpi=300)
    width = 0.13
    x_faults = np.arange(len(fault_types))
    
    for i, (alg_name, col) in enumerate(palette.items()):
        f_means = [np.mean(detailed_results[alg_name][5][f]["top1"]) for f in fault_types]
        ax_bar.bar(
            x_faults + (i - 2.5) * width,
            f_means,
            width,
            label=alg_name,
            color=col,
            edgecolor='black',
            linewidth=0.7,
            alpha=0.9
        )
        
    ax_bar.set_xticks(x_faults)
    ax_bar.set_xticklabels([f.replace("_", "\n") for f in fault_types], fontweight="bold", fontsize=9.5)
    ax_bar.set_ylabel("Top-1 Accuracy ($m = 5$ samples)", fontsize=10.5, fontweight="bold")
    ax_bar.set_title("Top-1 Diagnosis Accuracy Broken Down by Fault Mode ($m = 5$ Samples)", fontsize=11.5, pad=10)
    ax_bar.set_ylim(0, 1.08)
    ax_bar.grid(axis='y', linestyle=':', alpha=0.6)
    ax_bar.legend(loc='upper right', ncol=3, fontsize=8.5, frameon=True)
    
    plt.tight_layout()
    bar_png = "rca_fault_breakdown_m5.png"
    plt.savefig(bar_png, bbox_inches="tight", dpi=300)
    print(f"Saved Fault Breakdown Plot to {bar_png}")

    return df_table1, df_table2

if __name__ == "__main__":
    run_large_scale_experiment()
