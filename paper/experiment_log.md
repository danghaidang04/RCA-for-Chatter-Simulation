# Experiment log (bridge between results and manuscript)

## Contribution (one sentence)
A physics-based milling simulator grounded in real multi-sensor data shows that causal RCA accuracy on simulated machining faults is dominated by simulator assumptions: accuracy near 1.0 when training and test share a generator drops to 0.35–0.71 (m=10) under moderate parameter mismatch, while the simulated "healthy" baseline remains distinguishable from real data (C2ST ≈ 0.82).

## Experiments (claim -> evidence)
| # | Claim | Setup | Key result | Files |
|---|---|---|---|---|
| E1 | Simulator reproduces ZOA stability lobes | 4-flute Altintas FRF, DDE grid 16x13 vs ZOA | ZOA reproduces repo code to 1e-16; 79% agreement with time-domain grid | output/exp1_summary.json, exp1_lobes_validation.png |
| E2 | Real MSM subset matches simulator inputs | dataset 5,6, tool 2SE0250IX150A, 73 cuts, 219 windows | N and F equal mtc Spindle_Speed/Feed_Rate (ratio 1.000); ap=38.1 mm = flute length (catalog) | label.csv, mtc.csv |
| E3 | Real healthy baseline can be modelled by spectral envelope + harmonics | held-out cuts, stratified by ae, balanced-acc C2ST | 0.82 ± 0.04 (3 splits), not 0.5 | output/noise_model_gate.json (LAST RUN ONLY); other values from console logs (see below) |
| E4 | Fault gain can be set by effect size, not distribution matching | grid MMD flat (0.651-0.652 for g_acc<=1e-5.5); effect-size match: +0.375 dex acc, +4.86 dB mic | g_acc=7.7e-6, g_mic=0.0116 | output/rca_gain.json, rca_gain_effect.json |
| E5 | Classifiers separate DDE-fault classes on simulated data | 16 held-out cuts, 6 classes | GB 0.886 top-1; NB 0.579; chance 0.167 | output/rca_results.json |
| E6 | Causal RCA algorithms rank true root-cause node | 14 nodes, m in {5,10,20,50}, 100 trials x 5 faults; in-distribution vs shifted DDE parameters | m=10 top-1: ST 0.98->0.71, BRCD 0.90->0.58, RCG 0.90->0.59, BARO 0.94->0.35, RCD 0.80->0.39, SimpleRCA 0.52->0.44 | output/rca_causal.json |
| E7 | Difficulty depends on assumed gain | gain x0.3/1/3 | GB 0.698/0.882/0.943 | output/rca_sweep.json |

## Completed after the first draft
- E8 Ablation (3 seeds, m=10): shuffled/empty graph gives the same accuracy as the physical graph (BRCD 0.92/0.92/0.92 in-dist; 0.60/0.60/0.58 shifted); label-free nodes give 0.07-0.29 (in-dist) and 0.07-0.37 (shifted); gain x3 under shifted parameters lowers BRCD 0.60->0.27. Files: output/rca_ablation.json, rca_ablation.log.
- E9 Real defective tool D* (dataset 3, 12 cuts) vs healthy D (dataset 14, 64 cuts), AISI 4140: real shifts of low-frequency bands (+3.9, +4.0 sd) and non-synchronous indices (-3.7, -1.2 sd) are not reproduced by any simulated fault (sim low-band shifts within +-0.1; non-sync -0.3..+0.5); cosine 0.55-0.62 for all causes; TOOL_WEAR most similar in 89% of bootstrap resamples. Files: output/real_defect_check.json.

## Failed / negative experiments (report in paper)
- Direct DDE parameter calibration with CMA-ES (v2, v3, v4): loss plateaued ~3 (standardised units); stopped by the stop-on-nonconvergence rule. Earlier low MMD values (0.013-0.022) coexisted with large mismatch in condition dependence (mic vs rpm correlation +0.85 real, 0.00 simulated).
- Feature-window mismatch (real 4 s vs simulated 0.18 s) inflated apparent noise mismatch; fixed by 0.3 s windows in both.
- C2ST was initially read against the wrong chance level (5:1 synthetic:real); corrected to balanced accuracy. All earlier C2ST numbers are discarded.
- White+pink noise floor mis-fit real spectra (real accelerometer: flat floor + brick-wall at ~23 kHz + 1.5-2.3 kHz cluster; mic: broad hump), replaced by free spectral envelope.
- Classifier results are inflated by generator sharing (see E6 shifted column).

## Values taken from console logs (NOT archived as files; verify before submission)
- C2ST linear mean-only 0.986; linear+variation 0.876.
- Stratified 3-split C2ST: kernel ridge mean-only 0.851/0.846/0.895; hierarchical 0.817/0.778/0.866; with harmonic-proximity feature: mean-only 0.794/0.796/0.865, hierarchical 0.823/0.811/0.865 (no clear benefit within split noise).
- Per-cause top-1 at m=10 (in rca_causal.log).

## Not done (do not claim in paper)
- Validation of the RCA RANKING on real root causes (only the signature of one cause was compared; clamp failure and depth cases are single cuts on other tools).
- Verification of BRCD/RCG/SimpleRCA references by the author (user states sources are correct; not independently verified here).
- Mode-harmonic hypothesis (1.5-2.3 kHz cluster) is unproven.
