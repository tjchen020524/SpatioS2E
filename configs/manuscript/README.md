# Manuscript configuration record

This directory freezes the non-sensitive design inputs used in the manuscript.

- `cohorts.yaml` records accession, image modality, biological unit and section,
  individual and gene-universe counts.
- `biological_splits/*.json` contains the exact section identifiers used for
  training, validation and testing.
- `gene_splits/*/{train,heldout}_genes.txt` contains the exact primary
  downstream gene partition used in Figures 3 and 4.
- `heldout_assay.yaml` records the matched decoder and optimization protocol.
- `external_models.yaml` records upstream model versions and checksums without
  redistributing licensed checkpoints.

These files do not contain expression matrices, tissue images, frozen image or
gene vectors, trained weights or predictions. Data must be obtained from the
listed source studies and processed under their access and redistribution
terms. The split and checksum tests in `tests/test_manuscript_splits.py` detect
accidental drift in this record.
