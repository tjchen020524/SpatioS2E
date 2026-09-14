# Manuscript-to-code map

The [reproduction guide](reproduction.md) maps these components to actual
archived drivers, their upstream inputs and output tables. The reusable APIs
in the table below are not the complete manuscript workflow by themselves.

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
| Extended Data Fig. 8 decoder robustness | `BiasFreeFactorizedDecoder` and `ConcatenationMLPDecoder` | Matched Decima target-disjoint inputs and controls |
| Fig. 5 representation replication | `extract_scgpt_gene_tokens`, `gene_vectors` and the shared held-out fitting loop | Whole-human scGPT checkpoint, matched covered targets and within-representation controls |
| Fig. 5c and Extended Data Fig. 10 external audits | `experiments/external_genequery_component_audit/` | HER2ST inputs, official GeneQuery vectors and ResNet weights; DeepSpot-M source and released checkpoint |
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

## Model roles

The repository includes fitted-target models and separately trained
held-out-target decoders. Decima and scGPT provide frozen gene vectors; all
reported comparisons are made against matched controls within each
representation family.

## Numerical evidence

Numerical Source Data accompany the manuscript submission; their public
deposit link remains pending author approval. Raw cohort inputs, third-party
weights, downstream checkpoints and dense predictions are not bundled here.
The component evaluator also works with independently supplied compatible
prediction matrices.
