# FunnelCausalNet

Author-maintained research artifact for **FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation**, accepted at the 35th ACM International Conference on Information and Knowledge Management (CIKM 2026).

- Paper: [arXiv:2608.11675](https://arxiv.org/abs/2608.11675)
- Repository: [user-growth-lab/FunnelCausalNet](https://github.com/user-growth-lab/FunnelCausalNet)
- Release: `v1.0.0`

This repository contains the core FunnelCausalNet model, selected comparison methods, public and semi-synthetic experiment drivers, allocation and conformal-audit utilities, selected analysis scripts, and smoke tests.

No dataset, user-level record, experiment result, trained checkpoint, credential, or private path is distributed here. The restricted industrial experiment reported in the paper cannot be independently reproduced from this repository; see [DATA_BOUNDARIES.md](DATA_BOUNDARIES.md) and [REPRODUCTION_MATRIX.md](REPRODUCTION_MATRIX.md).

## Environment

- Python 3.9--3.11
- macOS or Linux
- CPU execution is supported; a GPU is not required for the smoke tests

```bash
git clone https://github.com/user-growth-lab/FunnelCausalNet.git
cd FunnelCausalNet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$(pwd)/code"
```

The dependency file records tested compatibility ranges rather than an exact cross-platform lock. See [ENVIRONMENT.md](ENVIRONMENT.md) for the release-validation environment.

## Data setup

Datasets are not bundled. Obtain the public benchmark data from its original source and place it at the path documented in [data/README.md](data/README.md).

## Quick checks

The allocator check uses generated arrays and requires no dataset:

```bash
PYTHONPATH=code python -m tests.smoke_pareto_ip
```

The FunnelCausalNet and joint-conformal smoke tests require the public Criteo file:

```bash
PYTHONPATH=code python -m tests.smoke_funnel_causal_net
PYTHONPATH=code python -m tests.smoke_joint_conformal
```

## Public and semi-synthetic experiment entry points

```bash
PYTHONPATH=code python -m experiments.run_e2_ablation
PYTHONPATH=code python -m experiments.run_e2b_prop2_stress
PYTHONPATH=code python -m experiments.run_e3_conflict
PYTHONPATH=code python -m experiments.run_e4_pareto
PYTHONPATH=code python -m experiments.run_e6_complexity
```

These experiments require user-supplied public data and can be computationally intensive. The exact inclusion and data boundary for each paper component is listed in [REPRODUCTION_MATRIX.md](REPRODUCTION_MATRIX.md).

In the approved v1.0.0 source snapshot, `python -m experiments.run_e3_conflict --help` has an `argparse` help-formatting issue caused by a literal percent sign in its help text. This affects only help rendering; the runner accepts the documented arguments and its experiment logic is unchanged.

## Repository layout

```text
code/
  methods/       FunnelCausalNet, baselines, conformal audit, and allocation
  semisynth/     Criteo-MT7 semi-synthetic generator
  experiments/   Included public and semi-synthetic experiment drivers
  analysis/      Selected aggregation and plotting utilities
  tests/         Smoke tests
data/README.md   Dataset sources, terms, and expected local paths
```

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{zhang2026funnelcausalnet,
  title         = {FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation},
  author        = {Zhang, Yu and Wang, Zhihan and Chen, Guanlin and Jiang, Min and Li, Shuai},
  year          = {2026},
  eprint        = {2608.11675},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.11675}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Maintainer statement

This repository is maintained by the paper authors as an approved research artifact. The `user-growth-lab` organization is an author-maintained research namespace and is not an official GitHub presence of any contributor's employer.
