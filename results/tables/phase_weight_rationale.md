# Phase Weight Rationale

Weight order: `(corr, fmt, parse, finish, tol, brev)`.

| Phase | Goal | Stage Mix | Initial Weights | Enabled Components | Trainer Overrides | Rationale |
|---|---|---|---|---|---|---|
| phase_a | Structure stabilization on Stage 1. | stage1_easy_numeric: 100% | (2, 1, 1, 1.5, 0, 0.25) | corr=on, fmt=on, parse=on, finish=on, tol=off, brev=on | {} | Structure-first start: formatting, parseability, and completion weights are high enough to make tagged, parseable, completed answers the early objective before stronger correctness pressure. |
| phase_b | Correctness strengthening with Stage 1/2 mix. | stage1_easy_numeric: 70%, stage2_float_numeric: 30% | (4, 0.75, 0.75, 1, 0, 0.2) | corr=on, fmt=on, parse=on, finish=on, tol=off, brev=on | {} | Moves pressure toward correctness after Phase A; formatting and parseability remain active as guard rails while Stage 2 enters the mix. |
| phase_c | Precision and harder reasoning with Stage 2/3 mix. | stage2_float_numeric: 60%, stage3_hard_numeric: 40% | (5, 0.5, 0.5, 0.75, 1, 0.2) | corr=on, fmt=on, parse=on, finish=on, tol=on, brev=on | {} | Raises correctness for harder numeric reasoning and enables tolerance reward for near-miss numeric answers; structure weights are lower because controller guards maintain formatting. |
