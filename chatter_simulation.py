import numpy as np
import matplotlib.pyplot as plt

def example_1_shaping_stability():
    """
    Python implementation of Example #1 (Josmar Cristello Assignment 4 / Altintas Page 131)
    2-DOF Shaping Machine Chatter Stability Lobes
    """
    # System Parameters
    f_n1 = 250.0  # Hz
    f_n2 = 150.0  # Hz
    omega_n1 = f_n1 * 2 * np.pi
    omega_n2 = f_n2 * 2 * np.pi
    zeta1 = 0.012
    zeta2 = 0.010
    k1 = 2.26e8   # N/m
    k2 = 2.13e8   # N/m
    K_f = 1.0e9   # 1,000 MPa = 10^9 N/m^2 (Altintas Book Page 131)
    theta1 = np.deg2rad(30.0)
    theta2 = np.deg2rad(-45.0)

    # Frequency range for transfer function
    freqs = np.linspace(0.1, 2000.0, 50000)  # rad/s
    s = 1j * freqs

    # Transfer functions for mode 1 and mode 2 projected onto y-direction
    Phi1 = (omega_n1**2 * np.cos(theta1)**2) / (k1 * (s**2 + 2*zeta1*omega_n1*s + omega_n1**2))
    Phi2 = (omega_n2**2 * np.cos(theta2)**2) / (k2 * (s**2 + 2*zeta2*omega_n2*s + omega_n2**2))
    Phi = Phi1 + Phi2

    # Mode 1 scan (around 250 Hz -> 1570.7 to 2000 rad/s)
    pt1_w1 = 1570.7
    pt2_w1 = 2000.0
    omega_c_w1 = np.linspace(pt1_w1, pt2_w1, 10000)
    s_w1 = 1j * omega_c_w1
    Phi_w1 = (omega_n1**2 * np.cos(theta1)**2) / (k1 * (s_w1**2 + 2*zeta1*omega_n1*s_w1 + omega_n1**2)) + \
             (omega_n2**2 * np.cos(theta2)**2) / (k2 * (s_w1**2 + 2*zeta2*omega_n2*s_w1 + omega_n2**2))
    G_w1 = np.real(Phi_w1)
    H_w1 = np.imag(Phi_w1)

    neg_mask1 = G_w1 < -1e-15
    omega_c_w1 = omega_c_w1[neg_mask1]
    G_w1 = G_w1[neg_mask1]
    H_w1 = H_w1[neg_mask1]
    fc_w1 = omega_c_w1 / (2 * np.pi)
    psi_w1 = np.arctan2(H_w1, G_w1)
    a_lim_w1 = -1.0 / (2.0 * K_f * G_w1) * 1000.0  # in mm

    # Mode 2 scan (around 150 Hz -> 942.9 to 1078.6 rad/s)
    pt1_w2 = 942.9
    pt2_w2 = 1078.6
    omega_c_w2 = np.linspace(pt1_w2, pt2_w2, 10000)
    s_w2 = 1j * omega_c_w2
    Phi_w2 = (omega_n1**2 * np.cos(theta1)**2) / (k1 * (s_w2**2 + 2*zeta1*omega_n1*s_w2 + omega_n1**2)) + \
             (omega_n2**2 * np.cos(theta2)**2) / (k2 * (s_w2**2 + 2*zeta2*omega_n2*s_w2 + omega_n2**2))
    G_w2 = np.real(Phi_w2)
    H_w2 = np.imag(Phi_w2)

    neg_mask2 = G_w2 < -1e-15
    omega_c_w2 = omega_c_w2[neg_mask2]
    G_w2 = G_w2[neg_mask2]
    H_w2 = H_w2[neg_mask2]
    fc_w2 = omega_c_w2 / (2 * np.pi)
    psi_w2 = np.arctan2(H_w2, G_w2)
    a_lim_w2 = -1.0 / (2.0 * K_f * G_w2) * 1000.0  # in mm

    return {
        "freqs": freqs,
        "Phi": Phi,
        "mode1": {"omega_c": omega_c_w1, "fc": fc_w1, "psi": psi_w1, "a_lim": a_lim_w1},
        "mode2": {"omega_c": omega_c_w2, "fc": fc_w2, "psi": psi_w2, "a_lim": a_lim_w2}
    }

def example_2_milling_stability(immersion="slotting"):
    """
    Python implementation of Example #2 (Josmar Cristello Assignment 4 / Altintas Page 160)
    Multi-DOF Milling Stability Lobes (Zero-Order Analytical Method)
    """
    N = 4             # Number of flutes
    Kt = 796e6        # N/m^2 (796 N/mm^2)
    Kr = 0.212        # Radial cutting constant
    
    if immersion == "slotting":
        phi_st = 0.0
        phi_ex = np.pi
    else: # half-immersion down milling
        phi_st = np.pi / 2.0
        phi_ex = np.pi

    def alpha_calc(pst, pex, kr):
        axx = 0.5 * (np.cos(2*pex) - 2*kr*pex + kr*np.sin(2*pex)) - 0.5 * (np.cos(2*pst) - 2*kr*pst + kr*np.sin(2*pst))
        axy = 0.5 * (-np.sin(2*pex) - 2*pex + kr*np.cos(2*pex)) - 0.5 * (-np.sin(2*pst) - 2*pst + kr*np.cos(2*pst))
        ayx = 0.5 * (-np.sin(2*pex) + 2*pex + kr*np.cos(2*pex)) - 0.5 * (-np.sin(2*pst) + 2*pst + kr*np.cos(2*pst))
        ayy = 0.5 * (-np.cos(2*pex) - 2*kr*pex - kr*np.sin(2*pex)) - 0.5 * (-np.cos(2*pst) - 2*kr*pst - kr*np.sin(2*pst))
        return axx, axy, ayx, ayy

    alpha_xx, alpha_xy, alpha_yx, alpha_yy = alpha_calc(phi_st, phi_ex, Kr)

    freq_n_x = np.array([262.16, 503.60, 667.92, 886.78, 2201.5, 2799.4, 3837.7, 4923.6, 6038.7])
    omega_n_x = freq_n_x * 2 * np.pi
    zeta_x = np.array([0.048, 0.096, 0.057, 0.074, 0.015, 0.027, 0.007, 0.019, 0.025])
    sig_x = np.array([1.994606, 1.932445, 13.64547, 0.8632813, -5.011467, -13.94105, -6.874543, -8.824935, 2.564520]) * 1e-6
    nu_x = np.array([0.3779091, 2.44616, 4.128093, 7.195716, 4.623555, 8.317728, 6.560925, 6.54481, 1.986276]) * 1e-5

    freq_n_y = np.array([285.53, 587.8, 749.6, 804.9, 1573.7, 2038.1, 2303.3, 2681.0, 2870.5, 3838.8, 4928.9, 6073.4])
    omega_n_y = freq_n_y * 2 * np.pi
    zeta_y = np.array([0.021, 0.089, 0.027, 0.075, 0.027, 0.016, 0.220, 0.019, 0.014, 0.006, 0.017, 0.016])
    sig_y = np.array([0.9797268, 13.49004, -31.06831, 30.33849, 1.283945, 1.298625, 0.9518897, 11.90834, -19.12932, -11.19589, -18.59880, -7.608392]) * 1e-6
    nu_y = np.array([0.05959362, 0.4544067, 3.816475, 8.813244, 0.8678961, 1.270846, 3.750897, 2.825781, 4.051860, 8.475725, 8.993226, 2.022991]) * 1e-5

    omega_d_x = omega_n_x * np.sqrt(np.maximum(0.0, 1.0 - zeta_x**2))
    omega_d_y = omega_n_y * np.sqrt(np.maximum(0.0, 1.0 - zeta_y**2))

    alpha_res_x = 2.0 * (zeta_x * omega_n_x * sig_x - omega_d_x * nu_x)
    alpha_res_y = 2.0 * (zeta_y * omega_n_y * sig_y - omega_d_y * nu_y)
    beta_res_x = 2.0 * sig_x
    beta_res_y = 2.0 * sig_y

    def get_transfer_func(omega):
        s = 1j * omega
        Hxx = 0.0 + 0.0j
        for k in range(len(freq_n_x)):
            Hxx += (beta_res_x[k]*s + alpha_res_x[k]) / (s**2 + 2*zeta_x[k]*omega_n_x[k]*s + omega_n_x[k]**2)
        Hyy = 0.0 + 0.0j
        for k in range(len(freq_n_y)):
            Hyy += (beta_res_y[k]*s + alpha_res_y[k]) / (s**2 + 2*zeta_y[k]*omega_n_y[k]*s + omega_n_y[k]**2)
        return Hxx, Hyy

    scan_freqs = np.concatenate([
        np.linspace(600, 900, 400),
        np.linspace(3600, 3900, 400),
        np.linspace(4600, 4900, 400)
    ])

    speeds = []
    a_lims = []

    for fc in scan_freqs:
        omega_c = fc * 2 * np.pi
        Ixx, Iyy = get_transfer_func(omega_c)
        
        a0 = Ixx * Iyy * (alpha_xx * alpha_yy - alpha_xy * alpha_yx)
        a1 = alpha_xx * Ixx + alpha_yy * Iyy
        
        disc = a1**2 - 4.0 * a0
        lam = -1.0 / (2.0 * a0) * (a1 - np.sqrt(disc))
        
        lam_R = np.real(lam)
        lam_I = np.imag(lam)
        
        if lam_R < 0:
            kappa = lam_I / lam_R
            psi = np.arctan2(lam_I, lam_R)
            
            alim_val = -2.0 * np.pi * lam_R / (N * Kt) * (1.0 + kappa**2) * 1000.0
            
            for k_lobe in range(1, 21):
                T = (1.0 / omega_c) * ((np.pi - 2.0 * psi) + 2.0 * k_lobe * np.pi)
                if T > 0:
                    n_rpm = 60.0 / (N * T)
                    if 0 <= n_rpm <= 20000 and 0 <= alim_val <= 60:
                        speeds.append(n_rpm)
                        a_lims.append(alim_val)

    speeds = np.array(speeds)
    a_lims = np.array(a_lims)
    sort_idx = np.argsort(speeds)
    return speeds[sort_idx], a_lims[sort_idx]
