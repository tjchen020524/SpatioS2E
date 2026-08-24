# Manuscript-to-code map

This document separates reusable public implementations from cohort-specific
data and generated evidence. Figure numbers refer to the current manuscript.

| Analysis | Public implementation | External inputs |
| --- | --- | --- |
| Fig. 1 fitted-gene framework | `spatios2e.models.SpatioS2EModel`, `MorphologyGraphModel` and the model factory | UNI2-h features, spatial coordinates/graphs, Decima vectors and training-derived abundance artifacts |
| Fig. 1 component endpoints | `spatios2e.evaluation.component_metrics` | Observed and predicted `[spot, gene]` matrices |
| Fig. 2 fitted-gene contrasts | fitted-gene models plus YAML configuration | Cohort-specific biological splits and variant configs |
| Held-out-gene assay | `FactorizedDotProductDecoder` and `spatios2e.models.gene_vectors` | Fixed training/held-out gene split, frozen spot features, and pretrained, random or constant gene vectors |
| No-image gene-mean counterfactual | `fit_gene_mean_counterfactual` and `broadcast_gene_means` | Frozen gene vectors and training-individual gene means; no spot-level input |
| Fig. 3 section-centred sensitivity | `center_within_section` and `component_metrics` | Tissue-section identifiers |
| Fig. 4 bias-free audit | `BiasFreeFactorizedDecoder` | Same target-disjoint inputs and controls as Fig. 3 |
| Fig. 4 decoder-family audit | `ConcatenationMLPDecoder` | Same target-disjoint inputs and controls as Fig. 3 |
| Fig. 4 residual-only assay | `SectionCenteredResidualDecoder` and `center_within_section` | Exactly section-centred training targets |
| Fig. 4 branch intervention | `FactorizedDotProductDecoder.forward_branches` | Frozen checkpoint and a reproducible permutation of held-out-gene identity |

The exact section and primary downstream gene partitions for all four cohorts
are versioned in `configs/manuscript/`. `heldout_assay.yaml` records the matched
decoder, vector controls, optimization and endpoint policy; `external_models.yaml`
records upstream revisions and hashes without redistributing checkpoints.

The supplied fitted-gene trainer implements the manuscript schedule's balanced
gene chunks, full-panel coverage assertion, minimum-epoch rule and
validation-based checkpoint selection. The checkpoint evaluator applies the
reported gene-variance eligibility and constant-prediction rules and can export
training-derived top-HVG summaries.

At the manuscript dimensions (1,536 spot features, 1,920 gene features, a
512-unit hidden layer and 96-dimensional programs), the public implementations
have 2,371,777 parameters for the full factorized decoder, 1,875,905 for the
bias-free decoder, 2,441,345 for the 704-unit concatenation MLP and 1,875,904
for the residual-only decoder. These counts are locked by the test suite.

## Interpretation of model names

SpatioS2E is the fitted-gene experimental framework in this repository. Decima
is an external pretrained reference-sequence model used to produce frozen gene
vectors; it is not a predictor introduced by this package. The held-out-gene
models are separately trained decoders and are not zero-shot deployments of the
fitted-gene architecture.

## Numerical evidence

The public software repository deliberately does not contain raw cohort data,
third-party weights, trained checkpoints or manuscript results. Numerical
source data, checksums and evidence manifests are maintained with the
manuscript submission package. Users can apply the public component evaluator
to exported prediction matrices to reproduce the definitions and exact MSE
identity independently of the original cohorts.
