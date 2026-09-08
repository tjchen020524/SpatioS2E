# Changelog

## 1.1.0 — submission candidate, 2026-09-07

- Add the HER2ST GeneQuery frozen-feature and end-to-end audit workflows,
  DeepSpot-M released-checkpoint adapters, no-image baseline and shared evaluator.
- Support three-seed paired summaries, including signed MSE component changes.
- Replace research-cluster paths with configurable local inputs.
- Update the manuscript-to-code map to the five main and ten Extended Data figures.
- Preserve the v1.0.0 tag and historical validation records unchanged.

## 1.0.0 — 2026-08-31

- Mark the first stable, citable manuscript software release.
- Define SpatioS2E as the repository and Python package rather than the name of
  a single predictive model.
- Document the supported uses and the boundary between reusable evaluation,
  trainable reference implementations and externally supplied data or model
  weights.
- Adopt the collective MIT copyright holder `SpatioS2E authors`.
- Add release-metadata consistency tests and include citation, license and
  changelog files in the validated source snapshot.

## 0.2.0 — 2026-08-28

- Align the public hippocampus example with the manuscript's donor-disjoint
  22/8/4-section partition, training-tissue gene-mean anchor and balanced
  full-gene training schedule.
- Add the primary held-out-target factorized dot-product decoder and the three
  architecture-audit decoders.
- Add matched gene-vector control utilities.
- Add a reusable individual- and target-disjoint decoder fitting loop.
- Add a checkpoint-hashed extractor for static scGPT gene-token vectors and
  freeze the whole-human representation contract used in the manuscript.
- Add the matched no-image gene-mean counterfactual and command-line entry point.
- Add component-resolved endpoints, section-wise centring and exact MSE
  decomposition.
- Add explicit gene-PCC coverage rules and training-derived top-HVG summaries.
- Replace ambiguous pooled-correlation terminology with full-matrix PCC while
  retaining the original evaluation JSON keys for compatibility. Add explicit
  `gene_mean_pcc` and `gene_mean_rmse` keys while preserving their legacy
  `abundance_*` aliases.
- Add decoder, metric, vector-control and training-prior tests.
- Freeze exact four-cohort biological section splits, primary downstream gene
  partitions, held-out-assay settings and external-model checksums.
- Add tests for split counts, disjointness, immutable gene-list checksums,
  held-out-target fitting and scGPT token extraction.

## 0.1.0

- Initial package with fitted-gene models, preprocessing, training and
  hippocampus example code.
