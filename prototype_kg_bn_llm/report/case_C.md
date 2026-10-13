# Case C: real intentionally defective tool D*

Model variant `kg_validated`, m=10 windows, temperature T=1.5. Truth: defective tool (cause not in the model's label set).

## Ranking (from the Bayesian network; final)

| rank | state | posterior |
|---|---|---|
| 1 | STIFFNESS | 1.000 |
| 2 | RPM_MISMATCH | 0.000 |
| 3 | DAMPING | 0.000 |
| 4 | TOOL_WEAR | 0.000 |
| 5 | DEPTH_OVERLOAD | 0.000 |
| 6 | HEALTHY | 0.000 |

Out-of-model check: d²=3472.0 vs threshold 44.3 -> FLAGGED.

## Explanation (backend: template; validated: True)

The model ranks STIFFNESS first with posterior 1.00 (temperature-calibrated, m=10 windows). The out-of-model check flags this input, so the ranking should be treated as unreliable here.

**Supporting evidence**

- acc_res: observed shift +6.35 sd, contribution +481.42; edge weight +2.06 [+1.82, +2.28]; the observed sign agrees with the knowledge graph (Loose clamping / lower modal stiffness -> Lower natural frequency and higher compliance (A looser clamp lowers modal stiffness.; tier ASSUMED, ref: altintas2020 (general theory)) | Lower natural frequency and higher compliance -> Accelerometer 1-3.2 kHz (The structural modes sit in the 1-3.2 kHz band and are excited more.; tier ASSUMED, ref: physics reasoning); tier ASSUMED).
- chatter_acc: observed shift -3.47 sd, contribution -69.97; edge weight +0.14 [-0.25, +0.49]; the observed sign conflicts with the knowledge graph (Loose clamping / lower modal stiffness -> Lower natural frequency and higher compliance (A looser clamp lowers modal stiffness.; tier ASSUMED, ref: altintas2020 (general theory)) | Lower natural frequency and higher compliance -> Non-synchronous index (accelerometer) (A lower stability limit favours non-synchronous (chatter) content.; tier ASSUMED, ref: physics reasoning); tier ASSUMED).
- mic_high: observed shift +1.73 sd, contribution -61.64; edge weight +0.18 [+0.14, +0.22]; this edge is not in the knowledge graph (suggested by the data) (not in the knowledge graph (data-suggested edge; expert review needed); tier ASSUMED).
- mic_dB: observed shift +3.77 sd, contribution -43.95; edge weight +1.55 [+1.35, +1.73]; this edge is not in the knowledge graph (suggested by the data) (not in the knowledge graph (data-suggested edge; expert review needed); tier ASSUMED).

**Conflicts or unexplained**

- Out-of-model check: the Mahalanobis residual 3472.0 exceeds the reference threshold 44.3; the data are not well explained by any hypothesis.
- Unexplained node acc_low: residual +12.42 reference s.d. after the best hypothesis.
- Unexplained node mic_low: residual +11.43 reference s.d. after the best hypothesis.
- Unexplained node chatter_acc: residual -11.06 reference s.d. after the best hypothesis.
- Unexplained node acc_res: residual +10.13 reference s.d. after the best hypothesis.

**Caveats**

- Edge weights and the covariance are learned from simulated windows, so they inherit the simulator's assumptions (nominal parameters above the stability limit, assumed fault severities and gains).
- Knowledge-graph edges and signs are physics reasoning with provenance tier ASSUMED, tested only against simulated data.
- The posterior is temperature-calibrated on simulated validation draws; calibration on real data has not been tested.
- No real multi-cause labelled data exist, so the ranking has not been validated on real cuts.

**Suggested checks (from the KG)**

- Re-torque the clamp and re-measure the tool-tip FRF
