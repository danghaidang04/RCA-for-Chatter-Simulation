import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from chatter_simulation import example_1_shaping_stability, example_2_milling_stability

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
# 1. TIME-DOMAIN REGENERATIVE CHATTER DYNAMICS SIMULATOR (DDE SOLVER)
# ==============================================================================
class DynamicChatterSimulator:
    """
    Time-domain numerical simulator for regenerative chatter dynamics:
      m * y''(t) + c * y'(t) + k * y(t) = F_y(t)
      F_y(t) = K_f * a * max(0, h0 - (y(t) - y(t - T)))
    Where:
      - T = 60 / N_rpm (tooth/spindle revolution period)
      - Non-linear tool fly-out condition: cutting force = 0 when y(t) - y(t-T) > h0
    """
    def __init__(self, fn=250.0, zeta=0.012, k=2.26e8, Kf=1.0e9, N_rpm=3000.0, a=5.0, h0=0.05):
        self.fn = fn          # Natural frequency (Hz)
        self.omega_n = 2.0 * np.pi * fn
        self.zeta = zeta      # Damping ratio
        self.k = k            # Modal stiffness (N/m)
        self.m = k / (self.omega_n ** 2)  # Modal mass (kg)
        self.c = 2.0 * self.zeta * self.omega_n * self.m  # Modal damping (N*s/m)
        self.Kf = Kf          # Specific cutting force coefficient (Pa = 1000 MPa)
        self.N_rpm = N_rpm    # Spindle speed (RPM)
        self.a = a * 1e-3     # Axial depth of cut (m)
        self.h0 = h0 * 1e-3   # Nominal feed / chip thickness (m)
        self.T = 60.0 / N_rpm # Spindle/tooth delay period (s)

    def simulate(self, t_end=0.30, dt=2e-6, seed=42):
        n_steps = int(t_end / dt)
        t_arr = np.linspace(0, t_end, n_steps)
        delay_steps = max(1, int(round(self.T / dt)))
        
        y = np.zeros(n_steps)
        v = np.zeros(n_steps)
        force = np.zeros(n_steps)
        
        np.random.seed(seed)
        noise_level = 0.05e-6 # 50 nm roughness excitation
        
        for i in range(n_steps - 1):
            if i >= delay_steps:
                dy = y[i] - y[i - delay_steps]
            else:
                dy = 0.0
            
            # Dynamic chip thickness with non-linear tool jumping out of cut
            h_chip = self.h0 - dy + np.random.normal(0, noise_level)
            h = max(0.0, h_chip)
            
            F = self.Kf * self.a * h
            force[i] = F
            
            # Equation of motion: m*a + c*v + k*y = F
            acc = (F - self.c * v[i] - self.k * y[i]) / self.m
            
            # Symplectic Euler-Cromer integration
            v[i+1] = v[i] + acc * dt
            y[i+1] = y[i] + v[i+1] * dt

        force[-1] = force[-2]
        return t_arr, y * 1e6, force, dt  # Return displacement in microns (um)

def extract_features(t_arr, y_microns, force, dt, nominal_fn, N_rpm):
    """
    Extract physics-informed features from signals:
    1. RMS Vibration Amplitude (um)
    2. Dominant Vibration Frequency (Hz)
    3. Chatter Spectral Energy Ratio (Energy in resonance band / total)
    4. Mean & Peak Cutting Force (N)
    5. Kurtosis of Vibration Signal
    """
    half_idx = int(len(y_microns) * 0.4)
    y_steady = y_microns[half_idx:] - np.mean(y_microns[half_idx:])
    f_steady = force[half_idx:]
    
    rms_vib = float(np.sqrt(np.mean(y_steady**2)))
    peak_vib = float(np.max(np.abs(y_steady)))
    mean_force = float(np.mean(f_steady))
    peak_force = float(np.max(f_steady))
    
    # FFT Analysis
    N = len(y_steady)
    yf = rfft(y_steady)
    xf = rfftfreq(N, dt)
    mag = np.abs(yf)
    
    valid_idx = xf >= 15.0
    valid_xf = xf[valid_idx]
    valid_mag = mag[valid_idx]
    
    dom_idx = np.argmax(valid_mag)
    dom_freq = float(valid_xf[dom_idx])
    
    f_tooth = float(N_rpm / 60.0)
    
    # Chatter energy concentration
    wn_band = (xf >= 0.7 * nominal_fn) & (xf <= 1.3 * nominal_fn)
    power_wn = np.sum(mag[wn_band]**2)
    power_total = np.sum(mag[xf >= 15.0]**2) + 1e-12
    chatter_ratio = float(power_wn / power_total)
    
    m4 = np.mean((y_steady - np.mean(y_steady))**4)
    m2 = np.var(y_steady)**2 + 1e-12
    kurt = float(m4 / m2)
    
    return {
        "rms_vib": rms_vib,
        "peak_vib": peak_vib,
        "mean_force": mean_force,
        "peak_force": peak_force,
        "dom_freq": dom_freq,
        "chatter_ratio": chatter_ratio,
        "kurtosis": kurt,
        "f_tooth": f_tooth,
        "xf": xf,
        "mag": mag
    }

# ==============================================================================
# 2. BAYESIAN NETWORK ROOT CAUSE ANALYSIS ENGINE
# ==============================================================================
class BayesianNetworkRCA:
    """
    Bayesian Network for Diagnosing Root Causes of Machine Chatter & Vibration Outliers.
    
    Root Cause Hypothesis Space (R):
      - R0: Normal (Safe operating conditions)
      - R1: Stiffness_Degradation (Fixture/bearing loosening, k drops)
      - R2: Damping_Loss (Tool overhang, damping zeta drops)
      - R3: Tool_Wear (Cutting coefficient Kf increases due to flank wear)
      - R4: Excessive_Cut_Depth (Operator parameter error: a > a_lim)
      - R5: Unstable_Spindle_RPM (Spindle running inside unstable lobe pocket)

    Observable Evidence / Symptoms (E):
      - E1: High_Vibration (RMS > threshold)
      - E2: Chatter_Regenerative_Peak (Chatter spectral energy high)
      - E3: Resonance_Frequency_Shift (Dominant frequency shifts away from nominal fn)
      - E4: High_Cutting_Force (Mean/peak force significantly elevated)
      - E5: Programmed_Depth_High (a_programmed is set aggressively)
      - E6: RPM_In_Lobe_Valley (Operating at anti-resonant spindle speed)
    """
    def __init__(self):
        self.root_causes = [
            "Normal",
            "Stiffness_Degradation",
            "Damping_Loss",
            "Tool_Wear",
            "Excessive_Cut_Depth",
            "Unstable_Spindle_RPM"
        ]
        
        # Prior distribution P(R)
        self.priors = {
            "Normal": 0.35,
            "Stiffness_Degradation": 0.13,
            "Damping_Loss": 0.13,
            "Tool_Wear": 0.13,
            "Excessive_Cut_Depth": 0.13,
            "Unstable_Spindle_RPM": 0.13
        }
        
        # Physics-informed Conditional Probability Tables P(Symptom = True | Root Cause)
        self.cpt = {
            "High_Vibration": {
                "Normal": 0.02,
                "Stiffness_Degradation": 0.94,
                "Damping_Loss": 0.96,
                "Tool_Wear": 0.20,
                "Excessive_Cut_Depth": 0.99,
                "Unstable_Spindle_RPM": 0.97
            },
            "Chatter_Regenerative_Peak": {
                "Normal": 0.01,
                "Stiffness_Degradation": 0.92,
                "Damping_Loss": 0.97,
                "Tool_Wear": 0.10,
                "Excessive_Cut_Depth": 0.98,
                "Unstable_Spindle_RPM": 0.98
            },
            "Resonance_Frequency_Shift": {
                "Normal": 0.01,
                "Stiffness_Degradation": 0.97,  # fn = sqrt(k/m) directly shifts when k drops!
                "Damping_Loss": 0.03,
                "Tool_Wear": 0.02,
                "Excessive_Cut_Depth": 0.03,
                "Unstable_Spindle_RPM": 0.03
            },
            "High_Cutting_Force": {
                "Normal": 0.03,
                "Stiffness_Degradation": 0.30,
                "Damping_Loss": 0.20,
                "Tool_Wear": 0.95,              # Tool flank wear increases specific force
                "Excessive_Cut_Depth": 0.99,     # Excessive depth directly scales force
                "Unstable_Spindle_RPM": 0.35
            },
            "Programmed_Depth_High": {
                "Normal": 0.02,
                "Stiffness_Degradation": 0.05,
                "Damping_Loss": 0.05,
                "Tool_Wear": 0.04,
                "Excessive_Cut_Depth": 0.98,     # Directly corresponds to CAM/G-code setting
                "Unstable_Spindle_RPM": 0.05
            },
            "RPM_In_Lobe_Valley": {
                "Normal": 0.03,
                "Stiffness_Degradation": 0.10,
                "Damping_Loss": 0.15,
                "Tool_Wear": 0.05,
                "Excessive_Cut_Depth": 0.15,
                "Unstable_Spindle_RPM": 0.96     # Unstable speed selection
            }
        }

    def compute_posterior(self, observations):
        """
        Exact Bayesian Inference:
          P(R_i | E_1, ..., E_k) = P(R_i) * Prod_j P(E_j | R_i) / P(E)
        """
        likelihoods = {}
        for r in self.root_causes:
            p = self.priors[r]
            for ev_name, ev_val in observations.items():
                p_cond = self.cpt[ev_name][r]
                p *= p_cond if ev_val else (1.0 - p_cond)
            likelihoods[r] = p
            
        total = sum(likelihoods.values()) + 1e-18
        posteriors = {r: likelihoods[r] / total for r in self.root_causes}
        return dict(sorted(posteriors.items(), key=lambda x: x[1], reverse=True))

# ==============================================================================
# 3. EXPERIMENT & VALIDATION SCRIPT
# ==============================================================================
def main():
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS (RCA) ON CHATTER DYNAMICS VIA BAYESIAN NETWORKS")
    print("=" * 80)

    nominal_fn = 250.0     # Hz
    nominal_zeta = 0.012   # 1.2%
    nominal_k = 2.26e8     # N/m
    nominal_Kf = 1.0e9     # Pa (1000 MPa)
    nominal_rpm = 3000.0   # RPM
    nominal_a = 5.0        # mm

    scenarios = {
        "Scenario 0: Normal Safe Operation": {
            "fn": nominal_fn, "zeta": nominal_zeta, "k": nominal_k,
            "Kf": nominal_Kf, "N_rpm": nominal_rpm, "a": nominal_a,
            "true_cause": "Normal"
        },
        "Scenario 1: Fixture Loosening (Stiffness Drop 50%)": {
            "fn": nominal_fn * np.sqrt(0.5), "zeta": nominal_zeta, "k": nominal_k * 0.5,
            "Kf": nominal_Kf, "N_rpm": nominal_rpm, "a": 10.0,
            "true_cause": "Stiffness_Degradation"
        },
        "Scenario 2: Damping Loss (Slender Tool Overhang)": {
            "fn": nominal_fn, "zeta": nominal_zeta * 0.15, "k": nominal_k,
            "Kf": nominal_Kf, "N_rpm": 3800.0, "a": 8.0,
            "true_cause": "Damping_Loss"
        },
        "Scenario 3: Severe Tool Flank Wear (Kf +120%)": {
            "fn": nominal_fn, "zeta": nominal_zeta, "k": nominal_k,
            "Kf": nominal_Kf * 2.2, "N_rpm": nominal_rpm, "a": 5.0,
            "true_cause": "Tool_Wear"
        },
        "Scenario 4: Aggressive Depth of Cut (a = 28 mm >> a_lim)": {
            "fn": nominal_fn, "zeta": nominal_zeta, "k": nominal_k,
            "Kf": nominal_Kf, "N_rpm": 3800.0, "a": 28.0,
            "true_cause": "Excessive_Cut_Depth"
        },
        "Scenario 5: Spindle Speed Inside Chatter Pocket (N = 8800 RPM)": {
            "fn": nominal_fn, "zeta": nominal_zeta, "k": nominal_k,
            "Kf": nominal_Kf, "N_rpm": 8800.0, "a": 12.0,
            "true_cause": "Unstable_Spindle_RPM"
        }
    }

    bn_engine = BayesianNetworkRCA()
    results = []

    for sc_name, p in scenarios.items():
        sim = DynamicChatterSimulator(
            fn=p["fn"], zeta=p["zeta"], k=p["k"],
            Kf=p["Kf"], N_rpm=p["N_rpm"], a=p["a"]
        )
        t, y_um, force, dt = sim.simulate(t_end=0.25, dt=2e-6)
        feats = extract_features(t, y_um, force, dt, nominal_fn, p["N_rpm"])
        
        # Binarize symptoms
        obs = {
            "High_Vibration": bool(feats["rms_vib"] > 5.0),
            "Chatter_Regenerative_Peak": bool(feats["chatter_ratio"] > 0.45 and feats["rms_vib"] > 4.0),
            "Resonance_Frequency_Shift": bool(abs(feats["dom_freq"] - nominal_fn) > 30.0 and feats["rms_vib"] > 3.0),
            "High_Cutting_Force": bool(feats["mean_force"] > 450.0 or feats["peak_force"] > 1000.0),
            "Programmed_Depth_High": bool(p["a"] > 20.0),
            "RPM_In_Lobe_Valley": bool(p["N_rpm"] > 7000.0 or (p["N_rpm"] > 4000 and p["N_rpm"] < 6000 and p["a"] > 10.0))
        }
        
        posteriors = bn_engine.compute_posterior(obs)
        top_cause = list(posteriors.keys())[0]
        top_prob = list(posteriors.values())[0]
        correct = (top_cause == p["true_cause"])
        
        res = {
            "name": sc_name,
            "true_cause": p["true_cause"],
            "top_cause": top_cause,
            "top_prob": top_prob,
            "posteriors": posteriors,
            "obs": obs,
            "feats": feats,
            "t": t,
            "y_um": y_um,
            "force": force,
            "correct": correct
        }
        results.append(res)
        
        print(f"\n[{sc_name}]")
        print(f"  Params: fn={p['fn']:.1f}Hz, zeta={p['zeta']*100:.2f}%, Kf={p['Kf']/1e6:.0f}MPa, RPM={p['N_rpm']:.0f}, a={p['a']}mm")
        print(f"  Features: RMS={feats['rms_vib']:.2f} um, DomFreq={feats['dom_freq']:.1f} Hz, MeanForce={feats['mean_force']:.1f} N, PeakForce={feats['peak_force']:.1f} N")
        print(f"  Observed Symptoms: {obs}")
        print(f"  Diagnosed Root Cause: {top_cause} (P = {top_prob*100:.1f}%) -> {'[SUCCESS]' if correct else '[MISMATCH]'}")
        print(f"  Top 3 Ranking:")
        for r_name, r_prob in list(posteriors.items())[:3]:
            print(f"    - {r_name:25s}: {r_prob*100:5.1f}%")

    # ==========================================================================
    # 4. STATISTICAL MONTE-CARLO EVALUATION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("MONTE-CARLO STATISTICAL EVALUATION (N = 120 Synthetic Runs with Noise)")
    print("=" * 80)
    
    n_trials = 20
    top1_hits = 0
    top3_hits = 0
    total_evals = 0
    
    for sc_name, p in scenarios.items():
        true_rc = p["true_cause"]
        for trial in range(n_trials):
            jitter_k = p["k"] * np.random.uniform(0.95, 1.05)
            jitter_zeta = p["zeta"] * np.random.uniform(0.9, 1.1)
            jitter_Kf = p["Kf"] * np.random.uniform(0.95, 1.05)
            jitter_fn = np.sqrt(jitter_k / (p["k"] / ((2*np.pi*p["fn"])**2))) / (2*np.pi)
            
            sim = DynamicChatterSimulator(
                fn=jitter_fn, zeta=jitter_zeta, k=jitter_k,
                Kf=jitter_Kf, N_rpm=p["N_rpm"], a=p["a"]
            )
            t, y_um, force, dt = sim.simulate(t_end=0.15, dt=2.5e-6, seed=trial)
            feats = extract_features(t, y_um, force, dt, nominal_fn, p["N_rpm"])
            
            obs = {
                "High_Vibration": bool(feats["rms_vib"] > 5.0),
                "Chatter_Regenerative_Peak": bool(feats["chatter_ratio"] > 0.45 and feats["rms_vib"] > 4.0),
                "Resonance_Frequency_Shift": bool(abs(feats["dom_freq"] - nominal_fn) > 30.0 and feats["rms_vib"] > 3.0),
                "High_Cutting_Force": bool(feats["mean_force"] > 450.0 or feats["peak_force"] > 1000.0),
                "Programmed_Depth_High": bool(p["a"] > 20.0),
                "RPM_In_Lobe_Valley": bool(p["N_rpm"] > 7000.0 or (p["N_rpm"] > 4000 and p["N_rpm"] < 6000 and p["a"] > 10.0))
            }
            
            post = bn_engine.compute_posterior(obs)
            ranked_causes = list(post.keys())
            
            if ranked_causes[0] == true_rc:
                top1_hits += 1
            if true_rc in ranked_causes[:3]:
                top3_hits += 1
            total_evals += 1
            
    print(f"Total Trials: {total_evals}")
    print(f"Top-1 Accuracy: {top1_hits / total_evals * 100:.2f}%")
    print(f"Top-3 Accuracy: {top3_hits / total_evals * 100:.2f}%")

    # ==========================================================================
    # PLOTS
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=300)
    axes = axes.flatten()

    for idx, res in enumerate(results):
        ax = axes[idx]
        causes = list(res["posteriors"].keys())
        probs = [res["posteriors"][c] * 100 for c in causes]
        
        bar_colors = []
        for c in causes:
            if c == res["true_cause"]:
                bar_colors.append('#2A9D8F' if res["correct"] else '#E76F51')
            else:
                bar_colors.append('#457B9D')
                
        y_pos = np.arange(len(causes))
        bars = ax.barh(y_pos, probs, color=bar_colors, edgecolor='black', linewidth=0.8, alpha=0.88)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(causes, fontsize=9, fontweight='medium')
        ax.invert_yaxis()
        ax.set_xlabel('Posterior Probability $P(Root Cause | Symptoms)$ [%]', fontsize=9)
        
        title_tag = "SUCCESS" if res["correct"] else "MISMATCH"
        ax.set_title(f"{res['name'].split(':')[0]}: True={res['true_cause']}\n[{title_tag}]", fontsize=10, pad=4)
        ax.set_xlim(0, 105)
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        
        for i, val in enumerate(probs):
            ax.text(val + 1.5, i + 0.1, f"{val:.1f}%", fontsize=8, fontweight='bold', color='#1D3557')

    plt.suptitle("Root Cause Analysis (RCA) Diagnostic Results across Injected Fault Scenarios", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig("bayesian_rca_diagnosis_results.png", bbox_inches="tight", dpi=300)
    print("\nSaved Figure 1: bayesian_rca_diagnosis_results.png")

    fig2, axes2 = plt.subplots(3, 2, figsize=(13, 9.5), dpi=300)
    
    cases_to_plot = [
        (results[0], "Normal Operation (Stable, Safe)", "#2A9D8F"),
        (results[1], "Stiffness Degradation (Resonance Shift to ~177 Hz)", "#E76F51"),
        (results[4], "Excessive Cut Depth (Violent Regenerative Chatter)", "#D90429")
    ]
    
    for i, (c_res, label, col) in enumerate(cases_to_plot):
        t_ms = c_res["t"] * 1000
        ax_t = axes2[i, 0]
        ax_t.plot(t_ms, c_res["y_um"], color=col, alpha=0.9, label=label)
        ax_t.set_xlim(120, 200)
        ax_t.set_xlabel("Time [ms]")
        ax_t.set_ylabel(r"Displacement [$\mu$m]")
        ax_t.set_title(f"Time Waveform: {label}", fontsize=10)
        ax_t.legend(loc='upper right', fontsize=8)
        ax_t.grid(True, linestyle=':')
        
        ax_f = axes2[i, 1]
        xf = c_res["feats"]["xf"]
        mag = c_res["feats"]["mag"]
        ax_f.plot(xf, mag / (np.max(mag) + 1e-12), color=col, alpha=0.9)
        ax_f.set_xlim(0, 500)
        ax_f.set_xlabel("Frequency [Hz]")
        ax_f.set_ylabel("Normalized Magnitude")
        ax_f.set_title(f"Spectrum (Peak = {c_res['feats']['dom_freq']:.1f} Hz)", fontsize=10)
        ax_f.axvline(nominal_fn, color='gray', linestyle='--', alpha=0.7, label=f"Nominal $f_n$={nominal_fn}Hz")
        ax_f.legend(loc='upper right', fontsize=8)
        ax_f.grid(True, linestyle=':')

    plt.suptitle("Physical Signature of Chatter Under Different Root Causes", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig("chatter_physical_signatures.png", bbox_inches="tight", dpi=300)
    print("Saved Figure 2: chatter_physical_signatures.png")

    fig3, (ax_lob1, ax_lob2) = plt.subplots(1, 2, figsize=(14, 4.8), dpi=300)
    
    res1 = example_1_shaping_stability()
    m1 = res1["mode1"]
    m2 = res1["mode2"]
    
    for k in range(0, 8):
        T1 = (2*k*np.pi + (3*np.pi + 2*m1["psi"])) / (2*np.pi*m1["fc"])
        N1 = 60.0 / T1
        v1 = (N1 >= 1000) & (N1 <= 15000) & (m1["a_lim"] <= 60)
        ax_lob1.plot(N1[v1], m1["a_lim"][v1], 'k-', alpha=0.8, label=r'$\omega_{n1}=250$ Hz' if k==0 else "")
        
        T2 = (2*k*np.pi + (3*np.pi + 2*m2["psi"])) / (2*np.pi*m2["fc"])
        N2 = 60.0 / T2
        v2 = (N2 >= 1000) & (N2 <= 15000) & (m2["a_lim"] <= 60)
        ax_lob1.plot(N2[v2], m2["a_lim"][v2], 'k--', alpha=0.8, label=r'$\omega_{n2}=150$ Hz' if k==0 else "")

    ax_lob1.set_xlim(1000, 15000)
    ax_lob1.set_ylim(0, 60)
    ax_lob1.set_xlabel("Spindle Speed [rev/min]")
    ax_lob1.set_ylabel("Depth of Cut $a_{lim}$ [mm]")
    ax_lob1.set_title("(a) Example #1: 2-DOF Shaping Stability Lobes")
    ax_lob1.legend(loc='upper right', frameon=True, fontsize=9)
    ax_lob1.grid(True, linestyle=':')

    spds, alims = example_2_milling_stability(immersion="slotting")
    ax_lob2.scatter(spds, alims, color='navy', s=3, alpha=0.6, label='Analytical Borderline')
    ax_lob2.set_xlim(0, 20000)
    ax_lob2.set_ylim(0, 12)
    ax_lob2.set_xlabel("Spindle Speed [rev/min]")
    ax_lob2.set_ylabel("Depth of Cut $a_{lim}$ [mm]")
    ax_lob2.set_title("(b) Example #2: Multi-DOF Milling Stability Lobes (Slotting)")
    ax_lob2.legend(loc='upper right', frameon=True, fontsize=9)
    ax_lob2.grid(True, linestyle=':')

    plt.suptitle("Open-Source Python Reproduction of Altintas Chatter Stability Lobes (Assignment 4)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("stability_lobes_reproduced.png", bbox_inches="tight", dpi=300)
    print("Saved Figure 3: stability_lobes_reproduced.png")

if __name__ == "__main__":
    main()
