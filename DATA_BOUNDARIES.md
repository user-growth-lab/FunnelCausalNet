# Data availability and boundaries

This repository intentionally contains no datasets or experiment outputs.

| Data source | Included | Intended use | Boundary |
|---|---:|---|---|
| Criteo Uplift v2.x | No | Public benchmark and Criteo-MT7 feature base | Obtain from the original source and follow its terms. |
| Criteo-MT7 | Generated at run time | Semi-synthetic multi-tier experiments | Generated from user-supplied Criteo features; no generated rows are bundled. |
| Hillstrom | No | Public binary-RCT scope-boundary experiment | Obtain from the original source and follow its terms. |
| Restricted industrial coupon RCT | No | Industrial evaluation | Restricted by platform agreement; must never be committed or redistributed. |

The following are excluded and must remain outside the public repository:

- raw, sampled, cached, or transformed industrial records;
- user-level predictions, matched-policy rows, or per-user diagnostics;
- experiment `results/`, run manifests containing private paths, and generated figures derived from restricted micro-data;
- model checkpoints, local configuration files, credentials, and industrial-data path markers;
- internal online-evaluation traces and non-public business metrics.

Public release of the code does not make the industrial experiment independently reproducible. Only aggregate industrial evidence approved for disclosure is reported in the paper.
