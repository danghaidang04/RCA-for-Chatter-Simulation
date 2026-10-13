import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
from brcd_causal import BRCD_Causal
from benchmark_rca_sota import RCD, RCG, BARO, SimpleRCA, SmoothTraversal

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
    "lines.linewidth": 1.5,
    "grid.alpha": 0.35
})

# ==============================================================================
# 1. PHYSICAL CHATTER CAUSAL SIMULATOR
# ==============================================================================
class ChatterCausalSystem:
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
        
        lobe_factor = 1.0 - 0.45 * np.sin(4 * np.pi * rpm)
        a_lim = (0.85 * k * zeta / np.maximum(0.1, kt)) * lobe_factor + np.random.normal(0, 0.015 * noise_scale, N)
        data[:, 5] = a_lim
        
        diff = a - a_lim
        chatter_sev = np.log1p(np.exp(np.clip(diff * 12.0, -10, 15))) * 0.28 + np.random.normal(0, 0.01 * noise_scale, N)
        chatter_sev = np.maximum(0.0, chatter_sev)
        data[:, 6] = chatter_sev
        
        data[:, 7] = 0.12 + 1.8 * chatter_sev + np.random.normal(0, 0.02 * noise_scale, N)
        data[:, 8] = 250.0 * np.sqrt(np.maximum(0.1, k)) + 8.0 * chatter_sev + np.random.normal(0, 1.5 * noise_scale, N)
        data[:, 9] = 0.08 + 0.80 / (1.0 + np.exp(-6.0 * (chatter_sev - 0.15))) + np.random.normal(0, 0.02 * noise_scale, N)
        
        data[:, 10] = 300.0 * (kt * a) + np.random.normal(0, 8.0 * noise_scale, N)
        data[:, 11] = data[:, 10] * (1.15 + 1.9 * chatter_sev) + np.random.normal(0, 15.0 * noise_scale, N)
        data[:, 12] = 0.75 * data[:, 10] * rpm + 0.12 * data[:, 11] + np.random.normal(0, 10.0 * noise_scale, N)
        data[:, 13] = 0.35 + 2.5 * data[:, 7] + 0.0008 * data[:, 11] + np.random.normal(0, 0.03 * noise_scale, N)
        
        return data

    def sample_observational_data(self, n_samples=5000, seed=42):
        np.random.seed(seed)
        k = np.random.normal(1.0, 0.025, n_samples)
        zeta = np.random.normal(1.0, 0.025, n_samples)
        kt = np.random.normal(1.0, 0.03, n_samples)
        a = np.random.normal(0.50, 0.02, n_samples)
        rpm = np.random.normal(0.50, 0.02, n_samples)
        return self.forward_physics(k, zeta, kt, a, rpm, noise_scale=1.0)

# ==============================================================================
# 2. GAN ADVERSARIAL ANOMALY GENERATOR
# ==============================================================================
class AnomalyGenerator(nn.Module):
    def __init__(self, latent_dim=8, n_causes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_causes, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, n_causes),
            nn.Tanh()
        )
        self.register_buffer("delta_scales", torch.tensor([0.22, 0.35, 0.28, 0.25, 0.25]))

    def forward(self, z, rc_onehot):
        inp = torch.cat([z, rc_onehot], dim=-1)
        raw_deltas = self.net(inp)
        masked_deltas = raw_deltas * rc_onehot * self.delta_scales
        return masked_deltas

class TelemetryDiscriminator(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_adversarial_anomaly_generator(system, n_epochs=120, batch_size=64):
    print("Training Physics-Constrained Adversarial Anomaly Generator...")
    D_obs = system.sample_observational_data(n_samples=4000)
    obs_tensor = torch.tensor(D_obs, dtype=torch.float32)
    mean_vec = obs_tensor.mean(dim=0, keepdim=True)
    std_vec = obs_tensor.std(dim=0, keepdim=True) + 1e-6
    obs_norm = (obs_tensor - mean_vec) / std_vec

    generator = AnomalyGenerator(latent_dim=8, n_causes=5)
    discriminator = TelemetryDiscriminator(input_dim=14)
    
    opt_G = optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    bce_loss = nn.BCELoss()

    for epoch in range(n_epochs):
        idx = np.random.choice(len(D_obs), batch_size, replace=False)
        real_x = obs_norm[idx]
        
        z = torch.randn(batch_size, 8)
        rc_target = np.random.randint(0, 5, batch_size)
        rc_onehot = torch.zeros(batch_size, 5)
        rc_onehot.scatter_(1, torch.tensor(rc_target).unsqueeze(1), 1.0)
        
        deltas = generator(z, rc_onehot).detach().numpy()
        fake_data = system.forward_physics(1.0 - deltas[:, 0], 1.0 - deltas[:, 1], 1.0 + deltas[:, 2], 0.50 + deltas[:, 3], 0.50 + deltas[:, 4])
        fake_tensor = torch.tensor(fake_data, dtype=torch.float32)
        fake_norm = (fake_tensor - mean_vec) / std_vec

        opt_D.zero_grad()
        out_real = discriminator(real_x)
        out_fake = discriminator(fake_norm.detach())
        loss_d = bce_loss(out_real, torch.ones_like(out_real)) + bce_loss(out_fake, torch.zeros_like(out_fake))
        loss_d.backward()
        opt_D.step()

        opt_G.zero_grad()
        deltas_grad = generator(z, rc_onehot)
        deltas_np = deltas_grad.detach().numpy()
        fake_data_g = system.forward_physics(1.0 - deltas_np[:, 0], 1.0 - deltas_np[:, 1], 1.0 + deltas_np[:, 2], 0.50 + deltas_np[:, 3], 0.50 + deltas_np[:, 4])
        fake_tensor_g = torch.tensor(fake_data_g, dtype=torch.float32)
        fake_norm_g = (fake_tensor_g - mean_vec) / std_vec
        
        out_g = discriminator(fake_norm_g)
        loss_adv = bce_loss(out_g, torch.ones_like(out_g))
        loss_reg = torch.mean(deltas_grad ** 2) * 2.0
        loss_g = loss_adv + loss_reg
        loss_g.backward()
        opt_G.step()

    print("GAN Training completed successfully.")
    return generator

# ==============================================================================
# 3. BENCHMARK EXECUTION ACROSS SAMPLES
# ==============================================================================
def run_benchmark():
    print("=" * 90)
    print("SOTA RCA BENCHMARK ON REALISTIC ADVERSARIAL ANOMALIES (GAN-GENERATED)")
    print("Comparing: BRCD (Bayesian), RCD (PC/CI), RCG (CMI), BARO, SimpleRCA, SmoothTraversal")
    print("=" * 90)

    system = ChatterCausalSystem()
    generator = train_adversarial_anomaly_generator(system, n_epochs=120)

    algorithms = {
        "BRCD (Ours)": BRCD_Causal(system),
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

    D_obs = system.sample_observational_data(n_samples=5000, seed=42)

    for m in sample_sizes:
        print(f"Testing on m = {m} anomalous samples...")
        for rc_idx, rc_name in enumerate(candidates):
            for trial in range(n_trials_per_rc):
                with torch.no_grad():
                    z = torch.randn(m, 8)
                    rc_onehot = torch.zeros(m, 5)
                    rc_onehot[:, rc_idx] = 1.0
                    deltas = generator(z, rc_onehot).numpy()
                    
                k = 1.0 - deltas[:, 0]
                zeta = 1.0 - deltas[:, 1]
                kt = 1.0 + deltas[:, 2]
                a = 0.50 + deltas[:, 3]
                rpm = 0.50 + deltas[:, 4]
                
                D_int = system.forward_physics(k, zeta, kt, a, rpm, noise_scale=1.1)

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
    print("\n" + "=" * 90)
    print("BENCHMARK COMPARISON TABLE ON REALISTIC ADVERSARIAL ANOMALIES (m = 5 to 100)")
    print("=" * 90)
    print(df_summary.to_string(index=False))

    # ==========================================================================
    # PLOT: ICML FORMAT
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
                linewidth=2.0 if "BRCD" in alg_name else 1.2,
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
    output_png = "rca_adversarial_accuracy_comparison.png"
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    print(f"\nSaved Adversarial Benchmark Figure to {output_png}")

    return df_summary

if __name__ == "__main__":
    run_benchmark()
