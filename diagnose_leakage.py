import numpy as np
import scipy.stats as stats
import networkx as nx
import pandas as pd

from run_full_paper_benchmark import AdvancedChatterSystem, BRCD, RCD, RCG, BARO, SimpleRCA, SmoothTraversal

system = AdvancedChatterSystem()
# Evaluate ranking across ALL 14 nodes in the graph
system.candidate_indices = list(range(system.d))

D_obs = system.sample_observational_data(n_samples=3000, seed=42)

# Generate 5 interventional samples for STIFFNESS
D_int = system.sample_interventional_data(n_samples=5, fault_type="STIFFNESS", seed=101)

print("=== DATA SUMMARY (Mean D_obs vs Mean D_int) ===")
for i, name in enumerate(system.node_names):
    m_obs = np.mean(D_obs[:, i])
    m_int = np.mean(D_int[:, i])
    s_obs = np.std(D_obs[:, i])
    z_shift = abs(m_int - m_obs) / max(1e-4, s_obs)
    print(f"[{i:2d}] {name:22s}: ObsMean={m_obs:7.2f}, IntMean={m_int:7.2f}, Z-Shift={z_shift:6.2f}")

print("\n=== RANKING RESULTS ACROSS ALL 14 CANDIDATES ===")
algs = {
    "BRCD": BRCD(system),
    "RCD": RCD(system),
    "RCG": RCG(system),
    "BARO": BARO(system),
    "SimpleRCA": SimpleRCA(system),
    "SmoothTraversal": SmoothTraversal(system)
}

for name, alg in algs.items():
    ranked = alg.fit_and_rank(D_obs, D_int)
    print(f"\n[{name}] Top 5 Diagnosed Nodes:")
    for rank, (node, score) in enumerate(ranked[:5], 1):
        print(f"  Rank {rank}: {node:22s} (Score = {score:.4f})")
