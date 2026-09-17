# SpatioS2E

[![Tests](https://github.com/tjchen020524/SpatioS2E/actions/workflows/tests.yml/badge.svg)](https://github.com/tjchen020524/SpatioS2E/actions/workflows/tests.yml)

Code for the study:

**Pretrained gene representations transfer mean expression more broadly than
spatial patterns in virtual spatial transcriptomics**

We ask whether pretrained gene representations help spatial predictors estimate
gene means, recover spatial variation, or both. To test this, we train spatial
predictors using tissue-image features and fixed gene vectors extracted from
Decima or scGPT, alongside matched control-vector models. We evaluate the
predictors on genes excluded from fitting and model selection, in new donors
or patients across three human brain regions and HER2-positive breast cancer.

## Installation

Requires Python 3.10 or newer; tested on CPython 3.10/Linux.

```bash
git clone https://github.com/tjchen020524/SpatioS2E.git
cd SpatioS2E
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Try the evaluation

This synthetic example compares a prediction that assigns one value to each
gene with one that also captures spatial variation. It needs no external data
or model weights.

```bash
python examples/synthetic_components.py
```

To evaluate your own prediction matrices, see [Evaluation and training](docs/usage.md).

## Reproduce the paper

- [Get the data and pretrained models](docs/data_availability.md).
- [Prepare inputs and run the experiments](docs/reproduction.md).
- [Find analysis code by figure or table](docs/paper_code_map.md).

## Citation and support

Citation details are in [CITATION.cff](CITATION.cff). The code is available under
the [MIT License](LICENSE). Questions and bug reports can be posted in
[Issues](https://github.com/tjchen020524/SpatioS2E/issues).
