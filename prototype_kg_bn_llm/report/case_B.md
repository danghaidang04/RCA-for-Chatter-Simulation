# Case B: simulated cut, shifted generator

Model variant `kg_validated`, m=10 windows, temperature T=1.5. Truth: TOOL_WEAR (simulated, physical parameters shifted).

## Ranking (from the Bayesian network; final)

| rank | state | posterior |
|---|---|---|
| 1 | TOOL_WEAR | 0.512 |
| 2 | DEPTH_OVERLOAD | 0.231 |
| 3 | DAMPING | 0.151 |
| 4 | RPM_MISMATCH | 0.107 |
| 5 | STIFFNESS | 0.000 |
| 6 | HEALTHY | 0.000 |

Out-of-model check: d²=14.3 vs threshold 44.3 -> within the reference range.

## Explanation (backend: template; validated: True)

The model ranks TOOL_WEAR first with posterior 0.51 (temperature-calibrated, m=10 windows). The runner-up is DEPTH_OVERLOAD with 0.23.

**Supporting evidence**

- log10_acc: observed shift +2.78 sd, contribution +31.03; edge weight +1.24 [+1.01, +1.49]; the observed sign agrees with the knowledge graph (Tool wear -> Higher cutting-force coefficients (Wear raises cutting-force coefficients.; tier ASSUMED, ref: none (no verified source for this tool)) | Higher cutting-force coefficients -> log10 accelerometer rms (Higher cutting force raises vibration.; tier ASSUMED, ref: physics reasoning); tier ASSUMED).
- acc_res: observed shift +1.52 sd, contribution +29.38; edge weight +1.17 [+1.00, +1.36]; this edge is not in the knowledge graph (suggested by the data) (not in the knowledge graph (data-suggested edge; expert review needed); tier ASSUMED).
- mic_dB: observed shift +2.53 sd, contribution -26.84; edge weight +1.06 [+0.83, +1.29]; the observed sign agrees with the knowledge graph (Tool wear -> Higher cutting-force coefficients (Wear raises cutting-force coefficients.; tier ASSUMED, ref: none (no verified source for this tool)) | Higher cutting-force coefficients -> Microphone level (dB) (Rubbing raises the sound level.; tier ASSUMED, ref: physics reasoning); tier ASSUMED).
- acc_high: observed shift +2.11 sd, contribution +15.11; edge weight +1.18 [+0.93, +1.45]; this edge is not in the knowledge graph (suggested by the data) (not in the knowledge graph (data-suggested edge; expert review needed); tier ASSUMED).

**Conflicts or unexplained**

- No node is flagged as unexplained by the top hypothesis.

**Caveats**

- Edge weights and the covariance are learned from simulated windows, so they inherit the simulator's assumptions (nominal parameters above the stability limit, assumed fault severities and gains).
- Knowledge-graph edges and signs are physics reasoning with provenance tier ASSUMED, tested only against simulated data.
- The posterior is temperature-calibrated on simulated validation draws; calibration on real data has not been tested.
- No real multi-cause labelled data exist, so the ranking has not been validated on real cuts.

**Suggested checks (from the KG)**

- Measure flank wear VB under a microscope
