| case | description | top-1 (posterior) | out-of-model check | explanation backend | validated |
|---|---|---|---|---|---|
| A | simulated cut, nominal generator | DAMPING (1.00) | ok (d²=8) | template | True |
| B | simulated cut, shifted generator | TOOL_WEAR (0.51) | ok (d²=14) | template | True |
| C | real intentionally defective tool D* | STIFFNESS (1.00) | FLAGGED (d²=3472) | template | True |
| D | real healthy tool D | HEALTHY (1.00) | FLAGGED (d²=72) | template | True |
