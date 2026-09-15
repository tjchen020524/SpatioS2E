# SpatioS2E

Code and analyses accompanying:

**Pretrained gene representations transfer mean expression more broadly than
spatial patterns in virtual spatial transcriptomics**

Tingjun Chen and Stephanie C. Hicks

[![Tests](https://github.com/tjchen020524/SpatioS2E/actions/workflows/tests.yml/badge.svg)](https://github.com/tjchen020524/SpatioS2E/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-2F855A.svg)](LICENSE)

We distinguish prediction of a gene's mean expression from recovery of its
spatial variation. This repository includes the evaluation tools,
gene-conditioned predictors and analysis workflows used in the study.

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

Run a small synthetic example without downloading data or model weights:

```bash
python examples/synthetic_components.py
```

The example compares mean-only and spatial predictions, reporting overall
agreement and within-gene spatial accuracy.

## Documentation

- [Using SpatioS2E](docs/usage.md) — evaluate predictions and use the training interfaces.
- [Reproducing the paper](docs/reproduction.md) — prepare inputs and run the study workflows.
- [Manuscript-to-code map](docs/paper_code_map.md) — find the code behind each analysis.
- [Data and model artifacts](docs/data_availability.md) — cohort accessions and required external inputs.

Full study workflows require separately obtained cohort data and pretrained
model artifacts.

## Citation

Software citation metadata are available in [CITATION.cff](CITATION.cff).

## License and contact

The code is available under the [MIT License](LICENSE).
For questions or bug reports, open a
[GitHub issue](https://github.com/tjchen020524/SpatioS2E/issues).
