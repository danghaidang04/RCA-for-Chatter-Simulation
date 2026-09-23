# Physics-Informed Root Cause Analysis (RCA) of Machining Chatter via Causal Bayesian Discovery

An open-source Python research framework for **Machining Chatter Dynamics Simulation** (reproducing Yusuf Altintas's regenerative chatter models without MATLAB) and **Self-Supervised Root Cause Analysis (RCA)** benchmarking using state-of-the-art causal discovery methods (**BRCD**, **RCD**, **RCG**, **BARO**, **SimpleRCA**, **Smooth Traversal**).

---

## 📌 1. Research Motivation & Problem Statement

In CNC machining and advanced manufacturing, **regenerative chatter** is an unstable, self-excited vibration between the cutting tool and workpiece. It causes poor surface finish, dimensional errors, accelerated tool wear, and potential damage to spindle bearings.

```
                           [Root Causes: R]
  ┌───────────────────────┬────────────────────────┬──────────────────────┐
  │ Stiffness Loss (k)    │ Damping Loss (zeta)    │ Tool Flank Wear (Kt) │
  │ (Fixture/bearing play)│ (Slender tool overhang)│ (Excessive friction) │
  └───────────────────────┴────────────────────────┴──────────────────────┘
  ┌────────────────────────────────────────────────┬──────────────────────┐
  │ Excessive Cut Depth (a >> a_lim)               │ Unstable Spindle RPM │
  │ (CAM programming overload)                     │ (Lobe pocket valley) │
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

## 🔬 2. Causal Architecture of the CNC Machining System

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

We evaluate 6 leading Root Cause Analysis algorithms across different anomalous sample sizes ($m \in [5, 10, 20, 50, 100]$):

### Quantitative Comparison Table:

| Algorithm | Method Class | Top-1 ($m=5$) | MRR ($m=5$) | Top-1 ($m=20$) | MRR ($m=20$) | Top-1 ($m=100$) | MRR ($m=100$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BRCD (Ours)** | **Bayesian Causal (ICML '26)** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **RCD** | Constraint-based PC / CI (NeurIPS '22) | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **RCG** | Conditional Mutual Info (UAI '25) | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **SmoothTraversal** | Causal Graph Traversal (NeurIPS '25) | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **BARO** | Multivariate Robust IQR (FSE '24) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **SimpleRCA** | Marginal 95th Percentile (2025) | 0.69 | 0.83 | 0.67 | 0.80 | 0.61 | 0.75 |

---

## 📈 4. Visualizations & Validation

### (a) SOTA Accuracy Comparison (ICML Format)
Accuracy@1, Accuracy@3, and Accuracy@5 as a function of the interventional sample size $m$:
![SOTA Accuracy Comparison](rca_sota_accuracy_comparison.png)

### (b) Analytical Stability Lobe Diagrams (Altintas / Assignment 4 Reproduction)
Open-source Python reproduction of Example #1 (2-DOF Shaping) and Example #2 (Multi-DOF Milling):
![Stability Lobes](stability_lobes_reproduced.png)

### (c) Physical Signatures Under Distinct Fault Modes
Time-domain vibration waveforms and FFT spectra showing distinct physical fingerprints:
* **Stiffness Degradation ($k$ drops 50%)**: Natural resonance shifts from $250\text{ Hz} \to 177\text{ Hz}$.
* **Excessive Depth ($a \gg a_{\lim}$)**: Violent regenerative growth into non-linear tool fly-out limit cycles.
![Physical Signatures](chatter_physical_signatures.png)

---

## 💡 5. Scientific Insights & Feasibility Assessment

1. **Why Causal Bayesian RCA Excels in Machining**:
   - Machine tool dynamics follow strict physical conservation laws. Under soft interventions, only the true root cause mechanism shifts ($F \to R^*$), while all non-intervened mechanisms $p(X_j \mid Pa(X_j))$ remain invariant.
   - **BRCD** leverages this modularity property, achieving high Top-1 accuracy even with scarce failure samples ($m = 5$).
2. **Physical Distinguishability (Identifiability)**:
   - **Fixture/bearing looseness ($k$)**: The only fault that shifts structural resonance frequency $f_n = \frac{1}{2\pi}\sqrt{k/m}$.
   - **Tool flank wear ($K_t$)**: Directly scales cutting forces without shifting natural frequencies.
   - **CAM depth overload ($a$)**: Produces excessive mean force and violent chatter limit cycles simultaneously.
3. **Pure Physics vs. Black-Box GANs**:
   - Rather than using black-box neural networks (which can generate physically inconsistent telemetry), **Physics-Informed Boundary Sampling** provides explainable, reproducible, and physically faithful synthetic datasets.

---

## 🛠️ 6. Code Structure & Usage

### File Structure:
* [`chatter_simulation.py`](file:///Users/danghaidang04/CodeSpace/RCA/chatter_simulation.py): Analytical stability lobe solver (Example 1 & Example 2).
* [`rca_chatter_bayesian.py`](file:///Users/danghaidang04/CodeSpace/RCA/rca_chatter_bayesian.py): Time-domain DDE solver + Bayesian Network RCA diagnostic engine.
* [`benchmark_rca_chatter.py`](file:///Users/danghaidang04/CodeSpace/RCA/benchmark_rca_chatter.py): Complete benchmark comparing BRCD, RCD, RCG, BARO, SimpleRCA, SmoothTraversal across sample sizes.
* [`brcd_causal.py`](file:///Users/danghaidang04/CodeSpace/RCA/brcd_causal.py): Causal mechanism invariance implementation of BRCD (ICML 2026).

### Quick Start:
```bash
# 1. Install dependencies
pip install numpy scipy matplotlib pandas networkx

# 2. Run SOTA RCA Benchmark
python benchmark_rca_chatter.py

# 3. Run Chatter Physical Simulation & Stability Lobes
python rca_chatter_bayesian.py
```

---

## 📚 7. References
1. Altintas, Y. (2012). *Manufacturing Automation: Metal Cutting Mechanics, Machine Tool Vibrations, and CNC Design* (2nd ed.). Cambridge University Press.
2. Cristello, J. (2022). *Machine Tool Vibrations - Assignment 4 (ENME 619L01)*.
3. Lee, K., Zhou, Z., & Kocaoglu, M. (2026). *Root Cause Analysis of Failures via Bayesian Root Cause Discovery (BRCD)*. In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*.
4. Ikram, A., Chakraborty, S., Mitra, S., Saini, S., Bagchi, S., & Kocaoglu, M. (2022). *Root cause analysis of failures in microservices through causal discovery (RCD)*. *NeurIPS*, 35, 31158-31170.
5. Pham, L., Ha, H., & Zhang, H. (2024). *BARO: Robust root cause analysis for microservices via multivariate Bayesian online change point detection*. *ACM FSE*.
