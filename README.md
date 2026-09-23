# Physics-Informed Root Cause Analysis (RCA) of Machining Chatter via Causal Bayesian Discovery

An open-source Python research framework for **Machining Chatter Dynamics Simulation** (reproducing Yusuf Altintas's regenerative chatter models without MATLAB) and **Self-Supervised Root Cause Analysis (RCA)** benchmarking using state-of-the-art causal discovery methods (**BRCD**, **RCD**, **RCG**, **BARO**, **SimpleRCA**, **Smooth Traversal**).

---

## 📌 1. Research Motivation & Problem Formulation

In CNC milling and high-speed machining, **regenerative chatter** is an unstable self-excited vibration between the cutting tool and workpiece. Identifying the root cause of chatter in real-time is critical for zero-defect manufacturing:

```
                           [Root Physical Faults: R]
  ┌───────────────────────┬────────────────────────┬──────────────────────┐
  │ 1. STIFFNESS (k)      │ 2. DAMPING (zeta)      │ 3. TOOL WEAR (Kt)    │
  │ (Fixture/bearing play)│ (Slender tool overhang)│ (Flank wear friction)│
  └───────────────────────┴────────────────────────┴──────────────────────┘
  ┌────────────────────────────────────────────────┬──────────────────────┐
  │ 4. DEPTH OVERLOAD (a >> a_lim)                 │ 5. RPM MISMATCH      │
  │ (CAM programming error)                        │ (Lobe pocket valley) │
  └────────────────────────────────────────────────┴──────────────────────┘
                                     │
                                     ▼
                    [Regenerative Stability Boundary]
                   a > a_lim(k, zeta, Kt, RPM) ?
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼ (Yes: Unstable)                       ▼ (No: Stable)
        [Regenerative Chatter]                  [Normal Safe Cutting]
    - Severe limit-cycle vibration          - Low steady-state vibration
    - Non-tooth-passing chatter peaks       - Dominant tooth-passing harmonics
    - Massive cutting force surges          - Stable, predictable forces
```

### The Label Scarcity Problem:
In industrial production, collecting **annotated ground-truth failure datasets** is practically impossible: when chatter occurs, sensors only detect elevated vibration without knowing whether the primary root cause was fixture loosening, tool wear, tool overhang, or improper spindle speed.

### Our Solution:
1. **Open-Source Physical Simulator**: Re-implements Yusuf Altintas's analytical Stability Lobe Diagrams (SLDs) and time-domain Delay Differential Equation (DDE) dynamics in pure Python (`numpy`, `scipy`).
2. **Physics-Informed Self-Supervised Fault Synthesis**: Injects realistic physical parameter degradations ($k, \zeta, K_t, a, \text{RPM}$) near the stability boundary to generate multi-sensor telemetry (vibrations, forces, power, roughness).
3. **Causal RCA Benchmarking**: Formulates the CNC machining process as a **Causal DAG** and evaluates state-of-the-art causal discovery and Bayesian RCA algorithms (**BRCD**, **RCD**, **RCG**, **BARO**, **SimpleRCA**, **Smooth Traversal**).

---

## 🔬 2. Causal Graph Architecture

The 14-node Causal Directed Acyclic Graph (DAG) is constructed strictly from the governing equations of metal cutting mechanics (*Altintas, 2012*):

```
                   +-------------------------------------------------------+
                   |               Root Physical Parameters (R)            |
                   |   Stiffness (k)   Damping (zeta)   Tool Wear (Kt)     |
                   |   Cut Depth (a)   Spindle Speed (RPM)                 |
                   +-------------------------------------------------------+
                                   |                       |
                                   v                       v
                   +-------------------------------+  +--------------------+
                   |     Stability Limit (a_lim)   |  | Dominant Freq (fn) |
                   +-------------------------------+  +--------------------+
                                   |                             |
                                   v                             |
                   +-------------------------------+             |
                   |   Chatter Severity (a > a_lim)|             |
                   +-------------------------------+             |
                             /             \                     |
                            v               v                    v
                   +-----------------+   +---------------------------------+
                   | Force Mean/Peak |   | Vib RMS & Chatter Spectral Peak |
                   +-----------------+   +---------------------------------+
                            \               /
                             v             v
                   +-----------------------------------------------+
                   |     Telemetry: Spindle Power & Roughness      |
                   +-----------------------------------------------+
```

---

## 📊 3. SOTA Benchmark Experimental Results

We evaluate 6 leading Root Cause Analysis algorithms across $N = 2,500$ independent Monte-Carlo simulations (500 runs per sample size $m \in [5, 10, 20, 50, 100]$):

### Table 1: Detailed Breakdown by Fault Mode ($m = 5$ Samples)
*(Format matching Table 4 & Table 5 in the BRCD ICML 2026 Paper)*

| Fault Scenario | Metric | BRCD (Ours) | RCD | RCG | BARO | SimpleRCA | SmoothTraversal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **STIFFNESS** | Top-1 | **1.00** | **1.00** | **1.00** | **1.00** | 0.53 | **1.00** |
| | Top-3 | **1.00** | **1.00** | **1.00** | **1.00** | 0.98 | **1.00** |
| | Top-5 | **1.00** | **1.00** | **1.00** | **1.00** | 1.00 | **1.00** |
| | MRR | **1.00** | **1.00** | **1.00** | **1.00** | 0.74 | **1.00** |
| **DAMPING** | Top-1 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| | Top-3 | **1.00** | **1.00** | **1.00** | **1.00** | 0.93 | **1.00** |
| | Top-5 | **1.00** | **1.00** | **1.00** | **1.00** | 1.00 | **1.00** |
| | MRR | **1.00** | **1.00** | **1.00** | **1.00** | 0.44 | **1.00** |
| **TOOL_WEAR** | Top-1 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| | Top-3 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| | MRR | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **DEPTH_OVERLOAD** | Top-1 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| | Top-3 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| | MRR | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **RPM_MISMATCH** | Top-1 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| | Top-3 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| | MRR | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **AVERAGE** | **Top-1** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **0.71 ± 0.02** | **1.00 ± 0.00** |
| | **Top-3** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **0.98 ± 0.01** | **1.00 ± 0.00** |
| | **Top-5** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** |
| | **MRR** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **0.84 ± 0.01** | **1.00 ± 0.00** |

---

### Table 2: Overall Top-1 Accuracy across Interventional Sample Sizes ($\text{mean} \pm \text{stderr}$)
*(Format matching Table 2 & Table 3 in the BRCD ICML 2026 Paper)*

| Algorithm | $m = 5$ | $m = 10$ | $m = 20$ | $m = 50$ | $m = 100$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BRCD (Ours)** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** |
| **RCD** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** |
| **RCG** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** |
| **BARO** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** |
| **SimpleRCA** | 0.71 ± 0.02 | 0.68 ± 0.02 | 0.64 ± 0.02 | 0.63 ± 0.02 | 0.64 ± 0.02 |
| **SmoothTraversal** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** | **1.00 ± 0.00** |

---

## 📈 4. Visualizations & Convergence Validation

### (a) Online Iterative Posterior Convergence & Shannon Entropy Decay
Demonstrating the anytime Bayesian updating property: as new streaming anomaly samples $t = 1 \dots 40$ arrive, posterior probability $P(R^* \mid \mathcal{D}_t)$ rapidly concentrates to $1.0$, while diagnostic uncertainty entropy $H(P)$ collapses to $0.0$:
![Iterative Convergence](rca_convergence_progress.png)

### (b) Per-Fault Mode Diagnosis Breakdown ($m = 5$ Samples)
Demonstrates why heuristic methods (like SimpleRCA) fail on subtle damping and stiffness faults, while Causal Bayesian methods (**BRCD**) achieve perfect diagnosis:
![Fault Breakdown](rca_fault_breakdown_m5.png)

### (c) SOTA Accuracy Comparison across Sample Sizes
Accuracy@1, Accuracy@3, and Accuracy@5 curves with standard error bars:
![SOTA Accuracy Comparison](rca_sota_accuracy_comparison.png)

### (d) Analytical Stability Lobe Diagrams (Altintas Reproduction)
Open-source Python reproduction of Example #1 (2-DOF Shaping) and Example #2 (Multi-DOF Milling):
![Stability Lobes](stability_lobes_reproduced.png)

### (e) Physical Signatures Under Distinct Fault Modes
Time-domain vibration waveforms and FFT spectra showing distinct physical fingerprints:
![Physical Signatures](chatter_physical_signatures.png)

---

## 💡 5. Scientific Validation & Discussion

1. **Theoretical Consistency**:
   - The observed exponential decay of entropy $H(P)$ and fast concentration of $P(R^* \mid \mathcal{D}_t)$ validates **Theorem 4.3 & 4.4 (Posterior Consistency & Exponential Concentration Bound)** from the BRCD paper.
2. **Failure Modes of Heuristic Baselines**:
   - `SimpleRCA` achieves $0\%$ Top-1 accuracy on `DAMPING` degradation because damping loss does not produce massive static mean shifts, but rather subtle regenerative amplification. Non-causal percentile methods confuse downstream symptoms with root causes.
3. **Physical Identifiability**:
   - Machine dynamics inherently satisfy **Interventional Faithfulness**: stiffness $k$ shifts structural resonance $f_n = \frac{1}{2\pi}\sqrt{k/m}$; tool wear $K_t$ scales cutting forces; depth overload $a$ triggers large limit cycles.

---

## 🛠️ 6. Code Structure & Reproducibility

### Files:
* [`run_full_paper_benchmark.py`](file:///Users/danghaidang04/CodeSpace/RCA/run_full_paper_benchmark.py): Large-scale Monte-Carlo benchmark ($2,500$ runs) + online streaming convergence experiment.
* [`benchmark_rca_chatter.py`](file:///Users/danghaidang04/CodeSpace/RCA/benchmark_rca_chatter.py): Core benchmark script for SOTA algorithms.
* [`chatter_simulation.py`](file:///Users/danghaidang04/CodeSpace/RCA/chatter_simulation.py): Analytical Stability Lobe Diagram solver (Example 1 & Example 2).
* [`rca_chatter_bayesian.py`](file:///Users/danghaidang04/CodeSpace/RCA/rca_chatter_bayesian.py): Time-domain DDE solver + initial Bayesian Network demonstration.

### Run All Experiments:
```bash
# 1. Install dependencies
pip install numpy scipy matplotlib pandas networkx

# 2. Run the Full Large-Scale Benchmark & Convergence Experiments
python run_full_paper_benchmark.py
```

---

## 📚 7. References
1. Altintas, Y. (2012). *Manufacturing Automation: Metal Cutting Mechanics, Machine Tool Vibrations, and CNC Design* (2nd ed.). Cambridge University Press.
2. Cristello, J. (2022). *Machine Tool Vibrations - Assignment 4 (ENME 619L01)*.
3. Lee, K., Zhou, Z., & Kocaoglu, M. (2026). *Root Cause Analysis of Failures via Bayesian Root Cause Discovery (BRCD)*. In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*.
4. Ikram, A., Chakraborty, S., Mitra, S., Saini, S., Bagchi, S., & Kocaoglu, M. (2022). *Root cause analysis of failures in microservices through causal discovery (RCD)*. *NeurIPS*, 35, 31158-31170.
5. Pham, L., Ha, H., & Zhang, H. (2024). *BARO: Robust root cause analysis for microservices via multivariate Bayesian online change point detection*. *ACM FSE*.
