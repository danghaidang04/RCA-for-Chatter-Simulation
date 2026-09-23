import numpy as np
import matplotlib.pyplot as plt
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
    "lines.linewidth": 1.6,
    "grid.alpha": 0.35
})

def plot_stability_lobes_perfect():
    """
    Plots smooth, continuous Stability Lobe Diagrams matching Josmar Cristello Assignment 4 / Altintas:
      - Subplot A: Example #1 (2-DOF Shaping Machine)
      - Subplot B: Example #2 (Multi-DOF End Milling - Slotting & Down Milling)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)
    
    # -------------------------------------------------------------------------
    # Example 1: 2-DOF Shaping System
    # -------------------------------------------------------------------------
    res1 = example_1_shaping_stability()
    m1 = res1["mode1"] # around wn1 = 250 Hz
    m2 = res1["mode2"] # around wn2 = 150 Hz
    
    # Generate continuous lobes for mode 1 (k = 0 to 7)
    for k in range(0, 8):
        T1 = (2 * k * np.pi + (3 * np.pi + 2 * m1["psi"])) / (2 * np.pi * m1["fc"])
        N1 = 60.0 / T1
        # Sort by speed to ensure clean continuous lines
        idx1 = np.argsort(N1)
        n_sorted = N1[idx1]
        a_sorted = m1["a_lim"][idx1]
        
        valid = (n_sorted >= 1000) & (n_sorted <= 15000) & (a_sorted <= 60) & (a_sorted >= 0)
        if np.any(valid):
            ax1.plot(n_sorted[valid], a_sorted[valid], 'k-', linewidth=1.5,
                     label=r'$\omega_{n1} = 250$ Hz' if k == 0 else "")

    # Generate continuous lobes for mode 2 (k = 0 to 6)
    for k in range(0, 7):
        T2 = (2 * k * np.pi + (3 * np.pi + 2 * m2["psi"])) / (2 * np.pi * m2["fc"])
        N2 = 60.0 / T2
        idx2 = np.argsort(N2)
        n_sorted2 = N2[idx2]
        a_sorted2 = m2["a_lim"][idx2]
        
        valid2 = (n_sorted2 >= 1000) & (n_sorted2 <= 15000) & (a_sorted2 <= 60) & (a_sorted2 >= 0)
        if np.any(valid2):
            ax1.plot(n_sorted2[valid2], a_sorted2[valid2], 'k--', linewidth=1.5,
                     label=r'$\omega_{n2} = 150$ Hz' if k == 0 else "")

    ax1.set_xlim(1000, 15000)
    ax1.set_ylim(0, 60)
    ax1.set_xlabel("Spindle Speed [rev/min]")
    ax1.set_ylabel("Axial Depth of Cut $a_{lim}$ [mm]")
    ax1.set_title("(a) Example #1: 2-DOF Shaping Stability Lobes")
    ax1.legend(loc='upper right', frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Example 2: Multi-DOF Milling (Smooth lower envelope reconstruction)
    # -------------------------------------------------------------------------
    # Slotting
    spds_slot, alims_slot = example_2_milling_stability(immersion="slotting")
    # Down Milling
    spds_down, alims_down = example_2_milling_stability(immersion="down_milling")

    # Reconstruct smooth minimum envelope using binned lower boundary (matching Cristello's MATLAB binning)
    def compute_clean_envelope(speeds, alims, n_bins=120):
        s_min, s_max = 500, 20000
        bin_edges = np.linspace(s_min, s_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        env_alim = []
        env_speed = []
        
        for i in range(n_bins):
            mask = (speeds >= bin_edges[i]) & (speeds < bin_edges[i+1])
            if np.any(mask):
                min_a = np.min(alims[mask])
                env_alim.append(min_a)
                env_speed.append(bin_centers[i])
                
        return np.array(env_speed), np.array(env_alim)

    env_s_slot, env_a_slot = compute_clean_envelope(spds_slot, alims_slot, n_bins=140)
    env_s_down, env_a_down = compute_clean_envelope(spds_down, alims_down, n_bins=140)

    # Plot continuous solid line for slotting and dashed for half immersion down milling
    ax2.plot(env_s_slot, env_a_slot, 'k-', linewidth=1.6, label='Slotting (Full Immersion)')
    ax2.plot(env_s_down, env_a_down, color='#D90429', linestyle='--', linewidth=1.6, label='Half Immersion Down-Milling')
    
    ax2.set_xlim(0, 20000)
    ax2.set_ylim(0, 30)
    ax2.set_xlabel("Spindle Speed [rev/min]")
    ax2.set_ylabel("Critical Depth of Cut $a_{lim}$ [mm]")
    ax2.set_title("(b) Example #2: Multi-DOF Milling Stability Lobes (Altintas)")
    ax2.legend(loc='upper right', frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Analytical Machining Chatter Stability Lobes (Python Open-Source Solver)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("stability_lobes_reproduced.png", bbox_inches="tight", dpi=300)
    print("Saved perfect continuous stability lobes to stability_lobes_reproduced.png")

if __name__ == "__main__":
    plot_stability_lobes_perfect()
