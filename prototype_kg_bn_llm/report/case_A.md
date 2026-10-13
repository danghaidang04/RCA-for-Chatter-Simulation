# Case A: simulated cut, nominal generator

Model variant `kg_validated`, m=10 windows, temperature T=1.5. Truth: DAMPING (simulated).

## Ranking (from the Bayesian network; final)

| rank | state | posterior |
|---|---|---|
| 1 | DAMPING | 0.999 |
| 2 | DEPTH_OVERLOAD | 0.001 |
| 3 | TOOL_WEAR | 0.000 |
| 4 | STIFFNESS | 0.000 |
| 5 | RPM_MISMATCH | 0.000 |
| 6 | HEALTHY | 0.000 |

Out-of-model check: d²=8.1 vs threshold 44.3 -> within the reference range.

## Explanation (backend: template; validated: True)

The model ranks DAMPING first with posterior 1.00 (temperature-calibrated, m=10 windows).

**Supporting evidence**

- acc_res: observed shift +1.50 sd, contribution +27.71; edge weight +1.47 [+1.19, +1.70]; the observed sign agrees with the knowledge graph (Lower damping -> Less damped tool-tip mode (Reduced damping ratio.; tier ASSUMED, ref: altintas2020 (general theory)) | Less damped tool-tip mode -> Accelerometer 1-3.2 kHz (A less damped mode rings more.; tier ASSUMED, ref: physics reasoning); tier ASSUMED).
- log10_acc: observed shift +3.14 sd, contribution +27.45; edge weight +1.88 [+1.50, +2.20]; this edge is not in the knowledge graph (suggested by the data) (not in the knowledge graph (data-suggested edge; expert review needed); tier ASSUMED).
- mic_high: observed shift +0.20 sd, contribution -5.04; edge weight +0.17 [+0.12, +0.23]; this edge is not in the knowledge graph (suggested by the data) (not in the knowledge graph (data-suggested edge; expert review needed); tier ASSUMED).
- mic_dB: observed shift +2.78 sd, contribution +2.97; edge weight +1.74 [+1.37, +2.09]; this edge is not in the knowledge graph (suggested by the data) (not in the knowledge graph (data-suggested edge; expert review needed); tier ASSUMED).

**Conflicts or unexplained**

- No node is flagged as unexplained by the top hypothesis.

**Caveats**

- Edge weights and the covariance are learned from simulated windows, so they inherit the simulator's assumptions (nominal parameters above the stability limit, assumed fault severities and gains).
- Knowledge-graph edges and signs are physics reasoning with provenance tier ASSUMED, tested only against simulated data.
- The posterior is temperature-calibrated on simulated validation draws; calibration on real data has not been tested.
- No real multi-cause labelled data exist, so the ranking has not been validated on real cuts.

**Suggested checks (from the KG)**

- Repeat the hammer test with the current stick-out
