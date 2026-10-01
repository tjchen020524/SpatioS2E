# Recomputing no-image retention

An independently fitted ridge predicts one value per gene. For each
representation, cohort and seed, define the fraction of full-matrix PCC
improvement retained without images using a common reference:

```text
R_seed = (PCC_ridge_pretrained - PCC_decoder_random_seed)
         / (PCC_decoder_pretrained_seed - PCC_decoder_random_seed)
retained_percent = 100 * mean(R_seed)
```

The same definition applies to Decima and scGPT. Use the arithmetic mean of
seed-paired ratios, not the ratio of mean improvements. Each representation
and cohort uses one fixed pretrained-vector ridge fit with decoder seeds
42, 123 and 456. The denominator must be positive. The ratio is not bounded
by 100% and does not measure a fraction of spatial information. Different
gene panels do not become a matched-target benchmark through this definition.

For MSE, reverse the differences: `(MSE_decoder_random - MSE_ridge_pretrained)`
divided by `(MSE_decoder_random - MSE_decoder_pretrained)`.

## Inputs and command

Extract the numerical Source Data archive supplied with the manuscript into
a working directory. Record its version and checksums. The script reads:

- `figure_3_absolute.tsv`
- `supp_stage2_mean_only_absolute.tsv`
- `supp_stage2_mean_only_matched_controls.tsv`
- `representation_scgpt_decoder_absolute_per_run.tsv`
- `representation_scgpt_no_image_absolute.tsv`

From the repository root, run:

```bash
python scripts/recompute_no_image_gains.py /path/to/source_data
```

Only the Python standard library is required. The command writes or replaces
five derived TSV files in that directory; the absolute-score inputs are unchanged:

| Outputs | Definition |
| --- | --- |
| `no_image_common_reference_per_run.tsv`, `no_image_common_reference_summary.tsv` | Common decoder-random reference for both representations |
| `no_image_displayed_summary.tsv` | Mean seed-paired percentages and associated figure panels |
| `no_image_matched_gain_per_run.tsv`, `no_image_matched_gain_summary.tsv` | Auxiliary comparison using the ridge's own random control in its numerator |

The auxiliary own-control calculation is a different statistic. It does not
supply the common-reference retention percentages. The
`ratio_of_mean_improvements_percent` column also records a distinct aggregation
and should not be substituted for the mean seed-paired ratio.

This script summarizes the supplied inputs; it does not assume that all
manuscript versions use identical target panels or summary conventions.
