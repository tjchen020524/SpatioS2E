# Manuscript-to-code map

This document separates reusable public implementations from cohort-specific
data and generated evidence. Figure numbers refer to the current manuscript.

| Analysis | Public implementation | External inputs |
| --- | --- | --- |
| Fig. 1 evaluation framework | `component_metrics`, `center_within_section` and frozen split records | Observed and predicted `[spot, gene]` matrices plus biological and target identities |
| Fig. 1 fitted-target assay | `SpatioS2EModel`, `MorphologyGraphModel` and the model factory | UNI2-h features, spatial coordinates/graphs, Decima vectors and training-tissue gene means |
| Fig. 1 held-out-target assay | `FactorizedDotProductDecoder`, `fit_heldout_decoder` and gene-vector controls | Fixed target partition, frozen spot features and fixed gene vectors |
| Fig. 2 fitted-target calibration | fitted-target models plus YAML configuration | Cohort-specific biological splits and matched variant configurations |
| Fig. 3 component-resolved held-out transfer | `evaluate_heldout_decoder` and `component_metrics` | Target-disjoint prediction matrices from fixed biological partitions |
| Fig. 3 no-image gene-mean model | `fit_gene_mean_counterfactual` and `broadcast_gene_means` | Frozen gene vectors and training-individual gene means; no spot-level input |
| Fig. 4 signal-dependent spatial transfer | `center_within_section` and observed-defined gene-PCC policy | Training-only variance rankings and section identities |
| Fig. 5 decoder robustness | `BiasFreeFactorizedDecoder` and `ConcatenationMLPDecoder` | Matched Decima target-disjoint inputs and controls |
| Fig. 5 representation replication | `extract_scgpt_gene_tokens`, `gene_vectors` and the shared held-out fitting loop | Whole-human scGPT checkpoint, matched covered targets and within-representation controls |
| Supplementary residual-only assay | `SectionCenteredResidualDecoder` and `center_within_section` | Exactly section-centred training targets |
| Supplementary branch interventions | `FactorizedDotProductDecoder.forward_branches` | Frozen checkpoint and reproducible within-partition identity permutations |

The exact section and primary downstream gene partitions for all four cohorts
are versioned in `configs/manuscript/`. `heldout_assay.yaml` records the matched
decoder, vector controls, optimization and endpoint policy; `external_models.yaml`
records upstream revisions and hashes without redistributing checkpoints.

The supplied fitted-gene trainer implements the manuscript schedule's balanced
gene chunks, full-panel coverage assertion, minimum-epoch rule and
validation-based checkpoint selection. The checkpoint evaluator applies the
reported gene-variance eligibility and constant-prediction rules and can export
training-derived top-HVG summaries.

At the Decima assay dimensions (1,536 spot features, 1,920 gene features, a
512-unit hidden layer and 96-dimensional programs), the public implementations
have 2,371,777 parameters for the full factorized decoder, 1,875,905 for the
bias-free decoder, 2,441,345 for the 704-unit concatenation MLP and 1,875,904
for the residual-only decoder. These counts are locked by the test suite.

## Interpretation of names

SpatioS2E names the software repository and Python package, not one model in
the manuscript. The repository includes both fitted-target experimental models
and separately trained held-out-target decoders. Decima and scGPT are external
pretrained models used only to produce frozen gene vectors; neither is a
spatial predictor introduced by this package. The held-out-target models are
not zero-shot deployments of the fitted-target architecture. Decima and scGPT
results are parallel within-representation contrasts, not a representation
leaderboard.

## Numerical evidence

The public software repository deliberately does not contain raw cohort data,
third-party weights, trained checkpoints or manuscript results. Numerical
source data, checksums and evidence manifests are maintained with the
manuscript submission package. Users can apply the public component evaluator
to exported prediction matrices to reproduce the definitions and exact MSE
identity independently of the original cohorts.
