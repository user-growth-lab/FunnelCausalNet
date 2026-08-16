# Reproduction matrix

| Paper component | Main entry point in v1.0.0 | Data requirement | External status |
|---|---|---|---|
| E1 uplift estimation | Not included | Criteo-derived MT7 and Hillstrom | Not reproducible from this release |
| E2 funnel ablation | `python -m experiments.run_e2_ablation` | Criteo-derived MT7 | Included; requires user-supplied public data |
| E2b zero-inflation stress | `python -m experiments.run_e2b_prop2_stress` | Criteo features; outcomes and treatments are semi-synthetic | Included; requires user-supplied public data |
| E3 conflict diagnostic | `python -m experiments.run_e3_conflict` | Criteo-derived MT7 | Included; requires user-supplied public data |
| E4 budgeted allocation | `python -m experiments.run_e4_pareto` | Criteo-derived MT7 | Included; requires user-supplied public data |
| E5 conformal audit | Core methods and smoke test only | Criteo-derived MT7 | Full experiment driver is not included in v1.0.0 |
| E6 scaling | `python -m experiments.run_e6_complexity` | Criteo features; outcomes and treatments are semi-synthetic | Included; timing is environment-dependent |
| E7 industrial policy frontier | Not included | Restricted industrial coupon RCT | Not independently reproducible externally |

Fixed seeds and unified configurations support repeatability for the included public and semi-synthetic paths. Hardware, library versions, public-dataset preprocessing, and floating-point nondeterminism may produce small differences. This repository does not claim reproduction of industrial aggregates without the restricted data.
