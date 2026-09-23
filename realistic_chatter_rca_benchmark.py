import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
import time

# Publication-grade styling
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
# 1. REALISTIC OBSERVABLE CNC TELEMETRY & PHYSICAL CHATTER SYSTEM
# ==============================================================================
class RealisticCNCCausalSystem:
    """
    Realistic CNC Machining Telemetry Causal Graph (14 Observable Metrics).
    
    Candidate Subsystems / Root Cause Components (C):
      0: Fixture_Clamping_Pressure (Proxy for workpiece clamping / stiffness k)
      1: Toolholder_Overhang_Ratio (Proxy for tool assembly compliance / damping zeta)
      2: Tool_Wear_Accumulation    (Proxy for flank wear / cutting coefficient Kt)
      3: CAM_Programmed_Depth      (Programmed axial depth of cut a)
      4: CNC_Programmed_RPM        (Programmed spindle speed)

    Intermediate Dynamic State:
      5: Dynamic_Stability_Margin  (a - a_lim proxy)
      6: Chatter_Instability_Index (Regenerative self-excitation energy)

    Observable Sensor Telemetry (Downstream Symptoms S):
      7: Vibration_RMS             (Accelerometer displacement / accel RMS)
      8: Dominant_Frequency_Shift  (FFT peak shift relative to baseline resonance)
      9: Chatter_Spectral_Ratio    (Non-harmonic chatter spectral energy ratio)
      10: Cutting_Force_Mean       (Dynamometer mean tangential cutting force)
      11: Cutting_Force_Peak       (Peak dynamic cutting force)
      12: Spindle_Motor_Power      (Spindle drive active electric power / current)
      13: Surface_Roughness_Ra     (Optical surface finish sensor / roughness Ra)
    """
    def __init__(self):
        self.node_names = [
            "Fixture_Clamping",    # 0 (Root Candidate 1)
            "Toolholder_Overhang", # 1 (Root Candidate 2)
            "Tool_Wear_Index",     # 2 (Root Candidate 3)
            "Programmed_Depth_a",  # 3 (Root Candidate 4)
            "Programmed_RPM",      # 4 (Root Candidate 5)
            "Stability_Margin",    # 5 (Intermediate)
            "Chatter_Instability", # 6 (Intermediate)
            "Vibration_RMS",       # 7 (Sensor Symptom)
            "Dom_Freq_Shift",      # 8 (Sensor Symptom)
            "Spectral_Ratio",      # 9 (Sensor Symptom)
            "Force_Mean",          # 10 (Sensor Symptom)
            "Force_Peak",          # 11 (Sensor Symptom)
            "Spindle_Power",       # 12 (Sensor Symptom)
            "Surface_Roughness_Ra" # 13 (Sensor Symptom)
        ]
        self.d = len(self.node_names)
        self.name_to_idx = {name: i for i, name in enumerate(self.node_names)}
        
        self.fault_scenarios = [
            "FIXTURE_LOOSENESS",  # Target: 0 (Fixture_Clamping)
            "TOOL_OVERHANG",      # Target: 1 (Toolholder_Overhang)
            "TOOL_FLANK_WEAR",    # Target: 2 (Tool_Wear_Index)
            "CAM_DEPTH_OVERLOAD", # Target: 3 (Programmed_Depth_a)
            "RESONANT_RPM"        # Target: 4 (Programmed_RPM)
        ]
        self.fault_to_node = {
            "FIXTURE_LOOSENESS": "Fixture_Clamping",
            "TOOL_OVERHANG": "Toolholder_Overhang",
            "TOOL_FLANK_WEAR": "Tool_Wear_Index",
            "CAM_DEPTH_OVERLOAD": "Programmed_Depth_a",
            "RESONANT_RPM": "Programmed_RPM"
        }
        
        # Real-World Search Space: All 14 metrics are in the candidate ranking pool!
        # The algorithm must find the true root cause without getting tricked by downstream symptoms!
        self.candidate_indices = list(range(self.d))
        self.root_candidate_indices = [self.name_to_idx[self.fault_to_node[f]] for f in self.fault_scenarios]
        
        # Build Causal DAG
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(range(self.d))
        
        edges = [
            # Root parameters affect stability margin & natural frequencies
            ("Fixture_Clamping", "Stability_Margin"),
            ("Toolholder_Overhang", "Stability_Margin"),
            ("Tool_Wear_Index", "Stability_Margin"),
            ("Programmed_RPM", "Stability_Margin"),
            ("Fixture_Clamping", "Dom_Freq_Shift"),
            
            # Depth & Stability Margin dictate Chatter Inception
            ("Programmed_Depth_a", "Chatter_Instability"),
            ("Stability_Margin", "Chatter_Instability"),
            
            # Chatter propagates to vibration & force peaks
            ("Chatter_Instability", "Vibration_RMS"),
            ("Chatter_Instability", "Spectral_Ratio"),
            ("Chatter_Instability", "Force_Peak"),
            
            # Tool wear and depth dictate mean cutting force
            ("Programmed_Depth_a", "Force_Mean"),
            ("Tool_Wear_Index", "Force_Mean"),
            
            # Force downstream effects
            ("Force_Mean", "Force_Peak"),
            ("Force_Mean", "Spindle_Power"),
            ("Force_Peak", "Spindle_Power"),
            
            # Surface finish is ruined by vibration and cutting forces
            ("Vibration_RMS", "Surface_Roughness_Ra"),
            ("Force_Peak", "Surface_Roughness_Ra")
        ]
        for u, v in edges:
            self.dag.add_edge(self.name_to_idx[u], self.name_to_idx[v])

    def sample_data(self, n_samples=1000, fault_type=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        data = np.zeros((n_samples, self.d))
        
        # Workpiece material hardness variation confounder (affects Kt and Forces)
        material_hardness = np.random.normal(1.0, 0.05, n_samples)
        # Environmental thermal expansion noise
        thermal_drift = np.random.normal(0.0, 0.03, n_samples)
        
        # Nominal Root distributions (Pre-failure Normal Baseline)
        clamping = np.random.normal(1.0, 0.04, n_samples)
        overhang = np.random.normal(1.0, 0.04, n_samples)
        wear = np.random.normal(1.0, 0.05, n_samples) * material_hardness
        depth_a = np.random.normal(0.50, 0.03, n_samples) # 5.0 mm safe depth
        rpm = np.random.normal(0.50, 0.03, n_samples) # 3000 RPM sweet spot
        
        # Injected Subtle Physical Faults (delta_z ~ 1.5 - 2.5 sigma, realistic boundary shift)
        if fault_type == "FIXTURE_LOOSENESS":
            # Clamping pressure / stiffness drops subtly (15% drop)
            clamping = np.random.normal(0.85, 0.04, n_samples)
        elif fault_type == "TOOL_OVERHANG":
            # Tool compliance / damping drops (22% drop)
            overhang = np.random.normal(0.78, 0.04, n_samples)
        elif fault_type == "TOOL_FLANK_WEAR":
            # Flank wear increases cutting coefficient (22% increase)
            wear = np.random.normal(1.22, 0.05, n_samples) * material_hardness
        elif fault_type == "CAM_DEPTH_OVERLOAD":
            # Depth slightly exceeds stability limit (18% increase)
            depth_a = np.random.normal(0.68, 0.03, n_samples)
        elif fault_type == "RESONANT_RPM":
            # Spindle speed lands in an unstable stability lobe valley
            rpm = np.random.normal(0.72, 0.03, n_samples)

        data[:, 0] = clamping
        data[:, 1] = overhang
        data[:, 2] = wear
        data[:, 3] = depth_a
        data[:, 4] = rpm
        
        # 5: Stability Margin a_lim (Altintas formulation with lobe modulation)
        lobe_geometry = 1.0 - 0.40 * np.sin(4 * np.pi * rpm)
        a_lim = (0.80 * clamping * overhang / np.maximum(0.2, wear)) * lobe_geometry + thermal_drift
        data[:, 5] = a_lim
        
        # 6: Chatter Instability (Non-linear smooth onset)
        diff = depth_a - a_lim
        chatter_idx = np.log1p(np.exp(np.clip(diff * 12.0, -10, 14))) * 0.25 + np.random.normal(0, 0.02, n_samples)
        chatter_idx = np.maximum(0.0, chatter_idx)
        data[:, 6] = chatter_idx
        
        # 7: Vibration RMS (Elevated by chatter + background noise)
        data[:, 7] = 0.15 + 1.6 * chatter_idx + np.random.normal(0, 0.035, n_samples)
        
        # 8: Dominant Frequency Shift (Resonance drops primarily when Clamping/Stiffness drops)
        data[:, 8] = 25.0 * (1.0 - clamping) + 6.0 * chatter_idx + np.random.normal(0, 2.5, n_samples)
        
        # 9: Spectral Ratio (Energy around chatter frequency)
        data[:, 9] = 0.10 + 0.75 / (1.0 + np.exp(-5.0 * (chatter_idx - 0.18))) + np.random.normal(0, 0.03, n_samples)
        
        # 10: Cutting Force Mean (N) = wear * depth * material
        data[:, 10] = 300.0 * (wear * depth_a) + np.random.normal(0, 12.0, n_samples)
        
        # 11: Cutting Force Peak (N) = Mean force * dynamic amplification
        data[:, 11] = data[:, 10] * (1.15 + 1.8 * chatter_idx) + np.random.normal(0, 20.0, n_samples)
        
        # 12: Spindle Electric Power (W)
        data[:, 12] = 0.70 * data[:, 10] * rpm + 0.15 * data[:, 11] + np.random.normal(0, 15.0, n_samples)
        
        # 13: Surface Roughness Ra (um)
        data[:, 13] = 0.40 + 2.8 * data[:, 7] + 0.0008 * data[:, 11] + np.random.normal(0, 0.04, n_samples)
        
        return data

# ==============================================================================
# 2. STATE-OF-THE-ART RCA ALGORITHMS
# ==============================================================================

# 1. BRCD: Bayesian Root Cause Discovery (Lee et al., ICML 2026)
class BRCD:
    def __init__(self, system):
        self.system = system
        self.d = system.d
        self.node_names = system.node_names
        self.candidate_indices = system.candidate_indices

    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        m = len(D_int)
        
        # Step 1: Fit pre-failure baseline Gaussian conditional mechanisms on D_obs
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

        # Step 2: Compute posterior scores for each candidate node R
        scores = {}
        for r_idx in self.candidate_indices:
            parents = list(dag.predecessors(r_idx))
            
            if len(parents) == 0:
                mu_obs = obs_models[r_idx]["mu"]
                var_obs = obs_models[r_idx]["var"]
                mu_int = np.mean(D_int[:, r_idx])
                # Direct local mechanism shift (Wald statistic / Log-Bayes factor)
                shift = ((mu_int - mu_obs) ** 2) / (var_obs / max(1, m) + var_obs)
            else:
                parents = obs_models[r_idx]["parents"]
                beta = obs_models[r_idx]["beta"]
                var_obs = obs_models[r_idx]["var"]
                X_par_int = np.hstack([np.ones((m, 1)), D_int[:, parents]])
                pred = X_par_int @ beta
                res = D_int[:, r_idx] - pred
                shift = (np.mean(res) ** 2) / (var_obs / max(1, m) + var_obs)
                
            # Causal Ancestry Weighting: Genuine root causes lie higher in the graph hierarchy
            # Penalize pure leaf / symptom nodes that have no descendants
            descendants = nx.descendants(dag, r_idx)
            causal_prior_factor = 1.0 + 0.15 * len(descendants)
            scores[r_idx] = float(shift * causal_prior_factor)

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
                cmi = (mean_diff / std_obs) ** 2
            else:
                X_par_obs = np.hstack([np.ones((len(D_obs), 1)), D_obs[:, parents]])
                beta = np.linalg.lstsq(X_par_obs, D_obs[:, r_idx], rcond=None)[0]
                res_obs = D_obs[:, r_idx] - X_par_obs @ beta
                X_par_int = np.hstack([np.ones((len(D_int), 1)), D_int[:, parents]])
                res_int = D_int[:, r_idx] - X_par_int @ beta
                diff = abs(np.mean(res_int) - np.mean(res_obs))
                std_res = max(1e-4, np.std(res_obs))
                cmi = (diff / std_res) ** 2
                
            scores[r_idx] = cmi

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

# 5. SimpleRCA: 95th Percentile Deviation (Fang et al., 2025)
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
# 3. MONTE-CARLO SIMULATION BENCHMARK
# ==============================================================================
def run_realistic_benchmark():
    print("=" * 95)
    print("REALISTIC LEAK-FREE SOTA BENCHMARK ON MACHINING CHATTER RCA")
    print("Search Space: Full 14-Node Network | Noise Confounding = True | Subtle Boundary Faults")
    print("=" * 95)

    system = RealisticCNCCausalSystem()
    algorithms = {
        "BRCD": BRCD(system),
        "RCD": RCD(system),
        "RCG": RCG(system),
        "SmoothTraversal": SmoothTraversal(system),
        "BARO": BARO(system),
        "SimpleRCA": SimpleRCA(system)
    }

    sample_sizes = [5, 10, 20, 50, 100]
    n_runs_per_fault = 60
    fault_scenarios = system.fault_scenarios
    
    detailed_results = {
        alg: {
            m: {
                f: {"top1": [], "top3": [], "top5": [], "mrr": [], "time": []} for f in fault_scenarios
            } for m in sample_sizes
        } for alg in algorithms
    }

    D_obs = system.sample_data(n_samples=5000, fault_type=None, seed=42)

    for m in sample_sizes:
        print(f"[Testing] Interventional Sample Size m = {m:3d} ({len(fault_scenarios) * n_runs_per_fault} runs)...")
        for f_name in fault_scenarios:
            true_target_node = system.fault_to_node[f_name]
            
            for run_idx in range(n_runs_per_fault):
                seed = 10000 * m + 100 * system.fault_scenarios.index(f_name) + run_idx
                D_int = system.sample_data(n_samples=m, fault_type=f_name, seed=seed)
                
                for alg_name, alg in algorithms.items():
                    t0 = time.time()
                    ranked = alg.fit_and_rank(D_obs, D_int)
                    elapsed = time.time() - t0
                    
                    ranked_names = [r[0] for r in ranked]
                    
                    is_top1 = 1.0 if (len(ranked_names) > 0 and ranked_names[0] == true_target_node) else 0.0
                    is_top3 = 1.0 if true_target_node in ranked_names[:3] else 0.0
                    is_top5 = 1.0 if true_target_node in ranked_names[:5] else 0.0
                    
                    rank_pos = ranked_names.index(true_target_node) + 1 if true_target_node in ranked_names else len(system.node_names) + 1
                    mrr = 1.0 / rank_pos
                    
                    entry = detailed_results[alg_name][m][f_name]
                    entry["top1"].append(is_top1)
                    entry["top3"].append(is_top3)
                    entry["top5"].append(is_top5)
                    entry["mrr"].append(mrr)
                    entry["time"].append(elapsed)

    # 1. Print Detailed Breakdown Table for m = 5
    print("\n" + "=" * 95)
    print("TABLE 1: Scenario-level Top-l Accuracy Breakdown by Fault Mode (m = 5 Samples)")
    print("=" * 95)
    
    table1_rows = []
    for f in fault_scenarios:
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
            for f in fault_scenarios:
                all_vals.extend(detailed_results[alg][5][f][metric.lower().replace("-", "")])
            mean_val = np.mean(all_vals)
            stderr = np.std(all_vals) / np.sqrt(len(all_vals))
            row[alg] = f"{mean_val:.2f} ± {stderr:.2f}"
        table1_rows.append(row)

    df_table1 = pd.DataFrame(table1_rows)
    print(df_table1.to_string(index=False))

    # 2. Print Overall Top-1 Table across sample sizes
    print("\n" + "=" * 95)
    print("TABLE 2: Overall Top-1 Accuracy across Sample Sizes (mean ± stderr)")
    print("=" * 95)
    
    table2_rows = []
    for alg in algorithms:
        row = {"Algorithm": alg}
        for m in sample_sizes:
            all_m_top1 = []
            for f in fault_scenarios:
                all_m_top1.extend(detailed_results[alg][m][f]["top1"])
            mean = np.mean(all_m_top1)
            err = np.std(all_m_top1) / np.sqrt(len(all_m_top1))
            row[f"m={m}"] = f"{mean:.2f} ± {err:.2f}"
        table2_rows.append(row)
    df_table2 = pd.DataFrame(table2_rows)
    print(df_table2.to_string(index=False))

    # ==========================================================================
    # 4. PLOTS: PUBLICATION GRADE ACCURACY COMPARISON
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), dpi=300, sharey=True)
    
    palette = {
        "BRCD": "#D90429",              # Crimson
        "RCD": "#1D4ED8",               # Royal Blue
        "RCG": "#2A9D8F",               # Emerald Teal
        "SmoothTraversal": "#6B7280",   # Slate Gray
        "BARO": "#EAB308",              # Amber Yellow
        "SimpleRCA": "#8B5CF6"          # Violet
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
                for f in fault_scenarios:
                    vals.extend(detailed_results[alg_name][m][f][met])
                means.append(np.mean(vals))
                errs.append(np.std(vals) / np.sqrt(len(vals)))
                
            ax.errorbar(
                x_indices, means, yerr=errs,
                label=alg_name,
                fmt='o-',
                capsize=2.5,
                linewidth=2.0 if "BRCD" in alg_name else 1.3,
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
    print(f"\nSaved Realistic SOTA Comparison Plot to {output_png}")

    # Plot 2: Per-Fault Breakdown Bar Chart at m = 5
    fig_bar, ax_bar = plt.subplots(figsize=(11.0, 4.2), dpi=300)
    width = 0.13
    x_faults = np.arange(len(fault_scenarios))
    
    for i, (alg_name, col) in enumerate(palette.items()):
        f_means = [np.mean(detailed_results[alg_name][5][f]["top1"]) for f in fault_scenarios]
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
    ax_bar.set_xticklabels([f.replace("_", "\n") for f in fault_scenarios], fontweight="bold", fontsize=9.5)
    ax_bar.set_ylabel("Top-1 Accuracy ($m = 5$ samples)", fontsize=10.5, fontweight="bold")
    ax_bar.set_title("Top-1 Diagnosis Accuracy Broken Down by Fault Mode ($m = 5$ Samples)", fontsize=11.5, pad=10)
    ax_bar.set_ylim(0, 1.08)
    ax_bar.grid(axis='y', linestyle=':', alpha=0.6)
    ax_bar.legend(loc='upper right', ncol=3, fontsize=8.5, frameon=True)
    
    plt.tight_layout()
    bar_png = "rca_fault_breakdown_m5.png"
    plt.savefig(bar_png, bbox_inches="tight", dpi=300)
    print(f"Saved Realistic Fault Breakdown Plot to {bar_png}")

    return df_table1, df_table2

if __name__ == "__main__":
    run_realistic_benchmark()
