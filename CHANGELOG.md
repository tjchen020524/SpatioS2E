# Changelog

## 0.2.0 — unreleased

- Align the public hippocampus example with the manuscript's donor-disjoint
  22/8/4-section partition, training-tissue abundance anchor and balanced
  full-gene training schedule.
- Add the primary held-out-gene factorized dot-product decoder and the three
  architecture-audit decoders.
- Add matched gene-vector control utilities.
- Add the matched no-image gene-mean counterfactual and command-line entry point.
- Add component-resolved endpoints, section-wise centring and exact MSE
  decomposition.
- Add explicit gene-PCC coverage rules and training-derived top-HVG summaries.
- Replace ambiguous pooled-correlation terminology with full-matrix PCC while
  retaining the original evaluation JSON keys for compatibility.
- Add decoder, metric, vector-control and training-prior tests.
- Freeze exact four-cohort biological section splits, primary downstream gene
  partitions, held-out-assay settings and external-model checksums.
- Add tests for split counts, disjointness and immutable gene-list checksums.

## 0.1.0

- Initial package with fitted-gene models, preprocessing, training and
  hippocampus example code.
