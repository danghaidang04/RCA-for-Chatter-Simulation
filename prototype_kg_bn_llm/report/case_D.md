# Case D: real healthy tool D

Model variant `kg_validated`, m=10 windows, temperature T=1.5. Truth: healthy.

## Ranking (from the Bayesian network; final)

| rank | state | posterior |
|---|---|---|
| 1 | HEALTHY | 1.000 |
| 2 | TOOL_WEAR | 0.000 |
| 3 | RPM_MISMATCH | 0.000 |
| 4 | DEPTH_OVERLOAD | 0.000 |
| 5 | STIFFNESS | 0.000 |
| 6 | DAMPING | 0.000 |

Out-of-model check: d²=72.1 vs threshold 44.3 -> FLAGGED.

## Explanation (backend: template; validated: True)

The model ranks HEALTHY first with posterior 1.00 (temperature-calibrated, m=10 windows). The out-of-model check flags this input, so the ranking should be treated as unreliable here.

**Supporting evidence**

- No knowledge-graph edge of the top cause carries a large contribution.

**Conflicts or unexplained**

- Out-of-model check: the Mahalanobis residual 72.1 exceeds the reference threshold 44.3; the data are not well explained by any hypothesis.

**Caveats**

- Edge weights and the covariance are learned from simulated windows, so they inherit the simulator's assumptions (nominal parameters above the stability limit, assumed fault severities and gains).
- Knowledge-graph edges and signs are physics reasoning with provenance tier ASSUMED, tested only against simulated data.
- The posterior is temperature-calibrated on simulated validation draws; calibration on real data has not been tested.
- No real multi-cause labelled data exist, so the ranking has not been validated on real cuts.

**Suggested checks (from the KG)**

