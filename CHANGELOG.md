# Changelog

## 1.1.0

- Add UNI2-h and Decima input preparation, fitted-feature assembly and a
  prepared-input workflow for the primary held-out-gene analysis.
- Include manuscript analysis scripts, cohort configurations and source hashes.
- Add GeneQuery and DeepSpot-M component analyses with recorded run settings,
  matched controls and three-seed GeneQuery summaries.
- Make external BLEEP and ST-Net source paths configurable.
- Add a synthetic quick start and expand the reproduction documentation.
- Test the archived primary workflow and an independently installed wheel.

## 1.0.0 — 2026-08-31

- Document the evaluation APIs, reference models and required external inputs.
- Add citation metadata, license information and packaging consistency tests.

## 0.2.0 — 2026-08-28

- Align the hippocampus example with the donor-disjoint 22/8/4-section partition,
  training-tissue gene-mean anchor and balanced full-gene training schedule.
- Add held-out-gene decoders, matched gene-vector controls and a reusable
  individual- and target-disjoint fitting loop.
- Add static scGPT gene-token extraction and a no-image gene-mean ridge model.
- Add component-resolved endpoints, section-wise centring, gene-PCC eligibility
  rules and training-derived high-variance-gene summaries.
- Add explicit `gene_mean_pcc` and `gene_mean_rmse` keys while preserving the
  legacy `abundance_*` aliases.
- Include four-cohort biological splits, gene partitions, assay settings and
  external-model checksums, with tests for split integrity and model behaviour.

## 0.1.0

- Initial package with fitted-gene models, preprocessing, training and
  hippocampus example code.
