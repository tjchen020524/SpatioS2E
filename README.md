# SpatioS2E

[![Tests](https://github.com/tjchen020524/SpatioS2E/actions/workflows/tests.yml/badge.svg)](https://github.com/tjchen020524/SpatioS2E/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-2F855A.svg)](LICENSE)

Code for the study:

**Pretrained gene representations transfer mean expression more broadly than
spatial patterns in virtual spatial transcriptomics**

The analyses distinguish gene-mean prediction from recovery of within-gene
spatial variation across four brain and breast cancer cohorts. This repository
provides evaluation tools, gene-conditioned predictors and workflows for
reproducing the analyses.

![Image features and fitted- and held-out-gene prediction workflows](docs/assets/figure_1_cde.png)

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/tjchen020524/SpatioS2E.git
cd SpatioS2E
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Quick start

Run the synthetic example:

```bash
python examples/synthetic_components.py
```

The example reports full-matrix, gene-mean and within-gene metrics for mean-only
and spatial predictions. No external data or model weights are required.

## Documentation

- [Usage guide](docs/usage.md) — evaluate predictions and train gene-conditioned models.
- [Reproducing the paper](docs/reproduction.md) — prepare inputs and run the study workflows.
- [Analysis index](docs/paper_code_map.md) — locate the code for each manuscript analysis.
- [Data and pretrained models](docs/data_availability.md) — find cohort accessions and required inputs.

## Citation and support

See [CITATION.cff](CITATION.cff) for software citation details. For questions or
bug reports, please use
[GitHub Issues](https://github.com/tjchen020524/SpatioS2E/issues).

The code is released under the [MIT License](LICENSE).
