#!/usr/bin/env python3
"""Audit the five-figure manuscript and freeze its current evidence manifest."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import base64
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image


ROOT = DATA_ROOT
SUBMISSION = ROOT / "paper/submission"
FIGURES = SUBMISSION / "figures"
SOURCE = SUBMISSION / "source_data"
SECTIONS = SUBMISSION / "sections"
STAGE = ROOT / "experiments/heldout_gene_second_stage"

VALIDATION = SOURCE / "results_evidence_validation.tsv"
NUMERICAL = SOURCE / "results_numerical_audit.tsv"
MANIFEST = SOURCE / "evidence_manifest.json"
REPORT = SOURCE / "results_evidence_audit.md"

COHORTS_INTERNAL = ["hippocampus_donor_disjoint", "dlpfc", "nac", "her2st"]
COHORTS_DISPLAY = ["Hippocampus", "DLPFC", "NAc", "HER2ST"]
SEEDS = [42, 123, 456]
EXPECTED_PANEL_GROUPS = {1: 4, 2: 3, 3: 5, 4: 5, 5: 3}
EXPECTED_ARTBOARD_HEIGHTS = {1: {"150mm"}, 2: {"146mm"}, 3: {"226mm", "226.0mm"}, 4: {"230mm", "230.0mm"}, 5: {"160mm"}}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def pptx_objects(path: Path) -> dict[str, int]:
    with zipfile.ZipFile(path) as archive:
        xml = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
    namespace = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
    return {
        "groups": len(list(xml.iter(namespace + "grpSp"))),
        "native_shapes": len(list(xml.iter(namespace + "sp"))),
        "pictures": len(list(xml.iter(namespace + "pic"))),
    }


def main() -> None:
    checks: list[dict[str, object]] = []
    numbers: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    def number(claim: str, cohort: str, value: float, unit: str, source: Path) -> None:
        numbers.append(
            {
                "claim": claim,
                "cohort": cohort,
                "value": float(value),
                "unit": unit,
                "source": str(source.relative_to(ROOT)),
            }
        )

    # Final artwork and native-editability checks.
    figure_sources: dict[int, list[Path]] = {
        1: [SOURCE / "figure_1_cohort_design.tsv", SOURCE / "figure_1_decomposition.tsv.gz"],
        2: [
            SOURCE / "figure_2_component_strategy_effects.tsv",
            SOURCE / "figure_2_gene_conditioning_maps.tsv.gz",
            SOURCE / "figure_2_gene_conditioning_map_selection.tsv",
            SOURCE / "figure_2_gene_conditioning_gene_audit.tsv",
            SOURCE / "figure_2_gene_conditioning_map_manifest.json",
        ],
        3: [
            SOURCE / "figure_3_absolute.tsv",
            SOURCE / "figure_3_mse_decomposition.tsv",
            SOURCE / "figure_3_paired_effects.tsv",
            SOURCE / "figure_3_spatial_example.tsv.gz",
            SOURCE / "figure_3_spatial_example_metrics.tsv",
            SOURCE / "figure_3_spatial_example_selection_audit.tsv",
            SOURCE / "figure_3_spatial_example_selection_manifest.json",
            SOURCE / "figure_3_scale_calibration.json",
            SOURCE / "figure_3_spatial_display_metrics.tsv",
            SOURCE / "figure_3_approved_artwork_manifest.json",
            SOURCE / "supp_stage2_mean_only_matched_controls.tsv",
            SOURCE / "supp_stage2_mean_only_full_increment.tsv",
        ],
        4: [
            SOURCE / "figure_4_approved_artwork_manifest.json",
            SOURCE / "supp_training_defined_top50_hvg_per_gene.tsv.gz",
            SOURCE / "supp_training_defined_top50_hvg_matched_effects.tsv",
            SOURCE / "figure_4_variance_decile_summary.tsv",
            SOURCE / "figure_5_target_subset_summary.tsv",
            SOURCE / "supp_training_defined_top50_hvg_per_gene_technical_mean.tsv.gz",
            SOURCE / "supp_training_defined_top50_hvg_section_centered_per_gene.tsv.gz",
            SOURCE / "figure_3_identity_effects.tsv",
            SOURCE / "figure_3_fitted_target_capacity.tsv",
            SOURCE / "figure_3_biological_individuals.tsv",
            SOURCE / "figure_3_biological_bootstrap.tsv",
        ],
        5: [
            SOURCE / "representation_scgpt_decoder_absolute_per_run.tsv",
            SOURCE / "representation_scgpt_decoder_effects_per_run.tsv",
            SOURCE / "figure_5_common_heldout_targets.tsv.gz",
            SOURCE / "figure_5_whole_human_gene_effects_technical_mean.tsv.gz",
            SOURCE / "figure_5_gene_effect_concordance.tsv",
            SOURCE / "representation_scgpt_provenance.json",
            SOURCE / "extended_data_figure_10_design.tsv",
            SOURCE / "extended_data_figure_10_component_attribution.tsv",
        ],
    }
    figure_claims = {
        1: "component-resolved evaluation, generalization design and two downstream assays",
        2: "matched fitted-gene perturbations show cohort-dependent effects of mean anchors, neighbourhood context and gene conditioning",
        3: "held-out-gene matrix gains are dominated by recovery of gene-mean log-expression",
        4: "within-gene spatial transfer is signal dependent and concentrated among high-variance targets",
        5: "scGPT mean-dominant transfer, same-target spatial concordance and comparator-dependent external component effects",
    }
    extended_data_sources: dict[int, list[Path]] = {
        1: [
            SOURCE / "extended_data_figure_1_absolute.tsv",
            SOURCE / "extended_data_figure_1_effects.tsv",
        ],
        4: [
            SOURCE / "supp_section_centered_per_run.tsv",
            SOURCE / "supp_section_centered_paired_effects.tsv",
            SOURCE / "supp_section_centered_paired_effects_summary.tsv",
            SOURCE / "supp_section_centered_per_individual.tsv",
            SOURCE / "supp_section_centered_per_section.tsv.gz",
            SOURCE / "supp_section_centered_manifest.json",
        ],
        7: [
            SOURCE / "supp_training_only_partition_assignments.tsv.gz",
            SOURCE / "supp_training_only_partition_cohorts.tsv",
            SOURCE / "supp_training_only_partition_paired_effects.tsv",
            SOURCE / "supp_stage2_partition_effects.tsv",
            SOURCE / "supp_stage2_partition_effects_per_run.tsv",
            SOURCE / "supp_stage2_partition_absolute.tsv",
            SOURCE / "supp_stage2_partition_similarity.tsv",
        ],
        8: [
            SOURCE / "figure_4_architecture_absolute_per_run.tsv",
            SOURCE / "figure_4_architecture_effects.tsv",
            SOURCE / "extended_data_figure_8_primary_readouts.tsv",
            SOURCE / "extended_data_figure_8_primary_manifest.json",
        ],
        2: [
            SOURCE / "figure_2_per_gene_graph_pcc.tsv.gz",
            SOURCE / "figure_2_per_individual_graph_pcc.tsv",
            SOURCE / "figure_2_hippocampus_map.tsv.gz",
            SOURCE / "figure_2_dlpfc_map.tsv.gz",
        ],
        3: [
            SOURCE / "supp_stage2_mean_only_absolute.tsv",
            SOURCE / "supp_stage2_mean_only_full_increment.tsv",
            SOURCE / "supp_stage2_mean_only_matched_controls.tsv",
            SOURCE / "supp_stage2_mean_only_cross_cohort.tsv",
            SOURCE / "supp_stage2_detection_sensitivity.tsv",
            SOURCE / "supp_stage2_alternative_abundance.tsv",
            SOURCE / "supp_stage2_individual_bootstrap.tsv",
        ],
        6: [
            SOURCE / "supp_stage2_decoder_control_effects.tsv",
            SOURCE / "supp_stage2_decoder_control_effects_per_run.tsv",
            SOURCE / "supp_stage2_decoder_control_absolute.tsv",
            SOURCE / "supp_stage2_fitted_target_capacity.tsv",
            SOURCE / "supp_stage2_fitted_target_capacity_per_run.tsv",
        ],
        5: [
            SOURCE / "supp_training_defined_top50_hvg_per_gene_technical_mean.tsv.gz",
            SOURCE / "supp_training_defined_top50_hvg_section_centered_absolute.tsv",
            SOURCE / "supp_training_defined_top50_hvg_matched_summary.tsv",
            SOURCE / "supp_training_defined_top50_hvg_manifest.json",
        ],
        9: [
            SOURCE / "representation_scgpt_provenance.json",
            SOURCE / "representation_scgpt_coverage.tsv",
            SOURCE / "representation_scgpt_decoder_absolute_per_run.tsv",
            SOURCE / "representation_scgpt_decoder_effects_per_run.tsv",
            SOURCE / "representation_scgpt_per_individual_effects.tsv",
            SOURCE / "representation_scgpt_constant_pcc_audit.tsv",
            SOURCE / "representation_scgpt_no_image_effects_summary.tsv",
            SOURCE / "figure_5_variance_decile_summary.tsv",
            SOURCE / "supp_table_16_scgpt_replication.tsv",
            SOURCE / "supp_table_17_scgpt_checkpoint_sensitivity.tsv",
        ],
        10: [
            SOURCE / "extended_data_figure_10_design.tsv",
            SOURCE / "extended_data_figure_10_absolute.tsv",
            SOURCE / "extended_data_figure_10_component_attribution.tsv",
            SOURCE / "extended_data_figure_10_individual_attribution.tsv",
            SOURCE / "extended_data_figure_10_matched_pcc_effects.tsv",
        ],
    }
    extended_data_claims = {
        1: "complete fitted-gene calibration and matched intervention effects",
        4: "section-wise centring and aggregation sensitivities",
        7: "training-only held-out-gene partition robustness",
        8: "primary-split decoder architecture robustness and frozen-pathway diagnostics",
        2: "fitted-gene neighbourhood effects across genes and individuals",
        3: "no-image gene-mean prediction and sensitivities",
        6: "gene-identity and target-fitted capacity controls",
        5: "training-defined high-variance held-out-gene effects",
        9: "scGPT controls, biological units and common-target variance-decile profiles",
        10: "external gene-query implementations reproduce calibration-dominant transfer under distinct evidence tiers",
    }
    figure_records: dict[str, object] = {}
    for figure in range(1, 6):
        outputs = [FIGURES / f"figure_{figure}.{suffix}" for suffix in ("png", "pdf", "pptx", "svg")]
        missing = [str(path) for path in outputs if not path.exists()]
        check(f"figure_{figure}:all_formats", not missing, "all PNG/PDF/PPTX/SVG outputs present" if not missing else f"missing={missing}")
        objects = pptx_objects(outputs[2]) if outputs[2].exists() else {"groups": 0, "native_shapes": 0, "pictures": 0}
        check(
            f"figure_{figure}:native_editability",
            objects["groups"] >= EXPECTED_PANEL_GROUPS[figure] and objects["native_shapes"] >= 40,
            (
                f"groups={objects['groups']} (expected at least {EXPECTED_PANEL_GROUPS[figure]}), "
                f"native_shapes={objects['native_shapes']}, pictures={objects['pictures']}"
            ),
        )
        artwork: dict[str, object] = {}
        if outputs[3].exists():
            svg_root = ET.parse(outputs[3]).getroot()
            font_sizes = []
            def collect_fonts(element, inherited_scale=1.0):
                scale = inherited_scale
                for match in re.finditer(r"scale\(\s*([\d.eE+-]+)", element.get("transform", "")):
                    scale *= float(match.group(1))
                if element.tag.endswith("text") and "font-size" in element.attrib:
                    font_sizes.append(float(element.attrib["font-size"]) * scale)
                for child in element:
                    collect_fonts(child, scale)
            collect_fonts(svg_root)
            # Honour both the root viewBox and nested group scaling.
            viewbox_width = float(svg_root.attrib["viewBox"].split()[2])
            svg_units_to_pt = 178.0 / viewbox_width / 25.4 * 72.0
            minimum_font_pt = min(font_sizes) * svg_units_to_pt if font_sizes else 0.0
            artwork.update(
                {
                    "width": svg_root.attrib.get("width"),
                    "height": svg_root.attrib.get("height"),
                    "minimum_font_pt": round(minimum_font_pt, 3),
                }
            )
            check(
                f"figure_{figure}:editorial_artboard",
                svg_root.attrib.get("width") in {"178mm", "178.0mm"}
                and svg_root.attrib.get("height") in EXPECTED_ARTBOARD_HEIGHTS[figure],
                f"SVG artboard={svg_root.attrib.get('width')} × {svg_root.attrib.get('height')}",
            )
            check(
                f"figure_{figure}:minimum_final_size_font",
                minimum_font_pt >= 4.99,
                f"minimum active SVG text={minimum_font_pt:.2f} pt",
            )
        if outputs[0].exists():
            with Image.open(outputs[0]) as png:
                artwork["png_pixels"] = [png.width, png.height]
                check(
                    f"figure_{figure}:publication_raster",
                    png.width >= 3154 and png.height >= 1984,
                    f"PNG dimensions={png.width} × {png.height} at 450 dpi",
                )
        missing_sources = [str(path) for path in figure_sources[figure] if not path.exists()]
        check(f"figure_{figure}:source_files", not missing_sources, "all curated source files present" if not missing_sources else f"missing={missing_sources}")
        figure_records[str(figure)] = {
            "claim": figure_claims[figure],
            "outputs": [record(path) for path in outputs if path.exists()],
            "powerpoint": objects,
            "artwork": artwork,
            "source_data": [record(path) for path in figure_sources[figure] if path.exists()],
        }
        if figure == 1:
            manifest_path = SOURCE / "figure_1_editable_abc_manifest.json"
            author = json.loads(manifest_path.read_text())
            figure_records["1"]["author_artwork"] = author
            check("figure_1:author_export_checksums", all(
                (ROOT / item["path"]).exists() and sha256(ROOT / item["path"]) == item["sha256"]
                for item in [*author["outputs"], *author["panels"], *author["raster_assets"]]
            ), "Reviewed native artwork, panels and raster assets match the promotion manifest")
            embedded = [base64.b64decode(href.split(",", 1)[1])
                        for element in svg_root.iter() if element.tag.endswith("image")
                        for href in [element.get("{http://www.w3.org/1999/xlink}href", element.get("href", ""))]
                        if href.startswith("data:image/png;base64,")]
            with zipfile.ZipFile(outputs[2]) as archive:
                ppt_images = [archive.read(name) for name in archive.namelist() if name.startswith("ppt/media/")]
            check("figure_1:raster_assets_preserved", all(
                (ROOT / item["path"]).read_bytes() in embedded and (ROOT / item["path"]).read_bytes() in ppt_images
                for item in author["raster_assets"]
            ), "Tissue and expression raster assets preserved in SVG and PPTX")
            check("figure_1:six_panels_and_connected_schematic", all(
                (FIGURES / f"figure_1_panels/figure_1_panel_{letter}.png").exists()
                for letter in ("a", "b", "c", "d", "e", "f", "cde")
            ) and (FIGURES / "figure_1_panels/figure_1_panel_cde.pptx").exists(),
                "Individual panels and standalone editable c–e deck present")

    extended_data_records: dict[str, object] = {}
    for figure in range(1, 11):
        outputs = [FIGURES / f"extended_data_figure_{figure}.{suffix}" for suffix in ("png", "pdf", "svg")]
        missing_outputs = [str(path) for path in outputs if not path.exists()]
        missing_sources = [str(path) for path in extended_data_sources[figure] if not path.exists()]
        check(
            f"extended_data_figure_{figure}:all_formats",
            not missing_outputs,
            "all PNG/PDF/SVG outputs present" if not missing_outputs else f"missing={missing_outputs}",
        )
        check(
            f"extended_data_figure_{figure}:source_files",
            not missing_sources,
            "all curated source files present" if not missing_sources else f"missing={missing_sources}",
        )
        extended_data_records[str(figure)] = {
            "claim": extended_data_claims[figure],
            "outputs": [record(path) for path in outputs if path.exists()],
            "source_data": [record(path) for path in extended_data_sources[figure] if path.exists()],
        }

    # Figure 1.
    cohort_design = pd.read_csv(SOURCE / "figure_1_cohort_design.tsv", sep="\t")
    decomposition = pd.read_csv(SOURCE / "figure_1_decomposition.tsv.gz", sep="\t")
    decomposition_error = float(
        np.abs(
            decomposition["observed"]
            - decomposition["observed_gene_mean"]
            - decomposition["centered_observed"]
        ).max()
    )
    check("figure_1:four_cohorts", cohort_design["cohort"].tolist() == COHORTS_DISPLAY, f"cohorts={cohort_design['cohort'].tolist()}")
    check("figure_1:exact_expression_decomposition", decomposition_error < 1e-10, f"maximum absolute residual={decomposition_error:.3e}")

    # Figure 2.
    fitted = pd.read_csv(SOURCE / "figure_2_absolute.tsv", sep="\t")
    fitted_effects = pd.read_csv(SOURCE / "figure_2_component_strategy_effects.tsv", sep="\t")
    check("figure_2:absolute_complete", len(fitted) == 96 and set(fitted["seed"]) == set(SEEDS), f"rows={len(fitted)}")
    main_contrasts = {"anchor_no_graph", "graph_tissue", "sequence_tissue"}
    main_effect_rows = fitted_effects.loc[fitted_effects["contrast"].isin(main_contrasts)]
    check("figure_2:matched_main_contrasts", len(main_effect_rows) == 36, f"rows={len(main_effect_rows)} expected=36")

    # Figure 3: primary Decima assay and exact MSE decomposition.
    heldout = pd.read_csv(SOURCE / "figure_3_absolute.tsv", sep="\t")
    decima_decomp = pd.read_csv(SOURCE / "figure_3_mse_decomposition.tsv", sep="\t")
    check("figure_3:absolute_complete", len(heldout) == 36, f"rows={len(heldout)} expected=36")
    primary_pathways = pd.read_csv(SOURCE / "extended_data_figure_8_primary_readouts.tsv", sep="\t")
    primary_manifest = json.loads((SOURCE / "extended_data_figure_8_primary_manifest.json").read_text())
    check("extended_data_figure_8:primary_readouts", len(primary_pathways)==108 and primary_manifest["n_checkpoints"]==36 and primary_manifest["partition"]=="primary training-only", "108 readouts from 36 primary-split checkpoints")
    check("extended_data_figure_8:intact_validation", max(primary_manifest["complete_prediction_max_errors"].values())<1e-6 and sha256(SOURCE / "extended_data_figure_8_primary_readouts.tsv")==primary_manifest["output_sha256"], f"maximum endpoint discrepancy={max(primary_manifest['complete_prediction_max_errors'].values()):.3g}")
    max_decima_identity = float(decima_decomp["identity_error"].abs().max())
    check("figure_3:exact_mse_identity", max_decima_identity < 1e-12, f"maximum absolute error={max_decima_identity:.3e}")
    decima_shares = decima_decomp.groupby("cohort")["abundance_fraction_of_total_reduction"].mean()
    check(
        "figure_3:gene_mean_dominance",
        bool((decima_shares > 0.90).all()),
        f"cohort means={decima_shares.round(4).to_dict()}",
    )
    for cohort, value in decima_shares.items():
        number("Decima gene-mean share of MSE reduction", cohort, value, "fraction", SOURCE / "figure_3_mse_decomposition.tsv")

    # Figure 4: variance continuum and high-variance targets.
    deciles = pd.read_csv(SOURCE / "figure_4_variance_decile_summary.tsv", sep="\t")
    check(
        "figure_4:variance_deciles_complete",
        len(deciles) == 160 and set(deciles["training_variance_decile"]) == set(range(1, 11)),
        f"rows={len(deciles)}, deciles={sorted(deciles['training_variance_decile'].unique())}",
    )
    top50 = pd.read_csv(SOURCE / "figure_5_target_subset_summary.tsv", sep="\t")
    fixed_top50 = top50.loc[
        top50["target_subset"].eq("training_defined_top50")
        & top50["contrast"].eq("pretrained_vs_random")
        & top50["endpoint"].eq("section_centered_within_gene_pcc")
    ]
    check("figure_4:fixed_top50_units", len(fixed_top50) == 8 and fixed_top50["n_units"].eq(50).all(), f"rows={len(fixed_top50)}")

    # Figure 5 and scGPT P0 audits.
    scgpt_absolute_path = STAGE / "scgpt_whole_human_decoder/summary/absolute_per_run.tsv"
    scgpt_absolute = pd.read_csv(scgpt_absolute_path, sep="\t")
    expected_conditions = {
        "pretrained_gene_tokens",
        "random_gene_vectors",
        "constant_gene_vector",
        "identity_shuffled_gene_tokens",
    }
    check(
        "figure_5:scgpt_decoder_complete",
        len(scgpt_absolute) == 48 and set(scgpt_absolute["condition"]) == expected_conditions,
        f"rows={len(scgpt_absolute)}, conditions={sorted(scgpt_absolute['condition'].unique())}",
    )
    max_scgpt_identity = float(
        np.abs(scgpt_absolute["overall_mse"] - scgpt_absolute["abundance_mse"] - scgpt_absolute["centered_mse"]).max()
    )
    check("figure_5:scgpt_exact_mse_identity", max_scgpt_identity < 1e-12, f"maximum absolute error={max_scgpt_identity:.3e}")
    constant = scgpt_absolute.loc[scgpt_absolute["condition"].eq("constant_gene_vector")]
    check("figure_5:constant_gene_mean_pcc_na", constant["abundance_pcc"].isna().all(), f"undefined rows={int(constant['abundance_pcc'].isna().sum())}/12")
    constant_audit = pd.read_csv(STAGE / "constant_gene_mean_pcc_audit/constant_condition_per_run.tsv", sep="\t")
    sd_column = [column for column in constant_audit.columns if "std" in column.lower() and "pred" in column.lower()]
    maximum_constant_sd = float(constant_audit[sd_column[0]].max()) if sd_column else np.nan
    check("figure_5:constant_predicted_gene_means_identical", bool(sd_column) and maximum_constant_sd <= 1e-6, f"maximum predicted-gene-mean SD={maximum_constant_sd:.3e}")
    number("Constant-vector predicted gene-mean standard deviation", "maximum across runs", maximum_constant_sd, "log-expression", STAGE / "constant_gene_mean_pcc_audit/constant_condition_per_run.tsv")

    indexed = scgpt_absolute.set_index(["cohort", "seed", "condition"])
    scgpt_rows: list[dict[str, object]] = []
    for cohort in COHORTS_INTERNAL:
        for seed in SEEDS:
            candidate = indexed.loc[(cohort, seed, "pretrained_gene_tokens")]
            for control, contrast in (
                ("random_gene_vectors", "pretrained_vs_random"),
                ("identity_shuffled_gene_tokens", "pretrained_vs_identity_shuffle"),
            ):
                reference = indexed.loc[(cohort, seed, control)]
                total = float(reference["overall_mse"] - candidate["overall_mse"])
                gene_mean = float(reference["abundance_mse"] - candidate["abundance_mse"])
                scgpt_rows.append(
                    {
                        "cohort": cohort,
                        "seed": seed,
                        "contrast": contrast,
                        "full_matrix_pcc": float(candidate["full_matrix_pcc"] - reference["full_matrix_pcc"]),
                        "gene_mean_pcc": float(candidate["abundance_pcc"] - reference["abundance_pcc"]),
                        "mean_within_gene_pcc": float(candidate["mean_gene_pcc"] - reference["mean_gene_pcc"]),
                        "gene_mean_share": gene_mean / total if total > 0 else np.nan,
                    }
                )
    scgpt_effects = pd.DataFrame(scgpt_rows)
    scgpt_random = scgpt_effects.loc[scgpt_effects["contrast"].eq("pretrained_vs_random")]
    scgpt_cohort = scgpt_random.groupby("cohort").mean(numeric_only=True)
    scgpt_identity = (
        scgpt_effects.loc[scgpt_effects["contrast"].eq("pretrained_vs_identity_shuffle")]
        .groupby("cohort")
        .mean(numeric_only=True)
    )
    check("figure_5:scgpt_gene_mean_dominance", bool((scgpt_cohort["gene_mean_share"] > 0.90).all()), f"shares={scgpt_cohort['gene_mean_share'].round(4).to_dict()}")
    check("figure_5:scgpt_matrix_gain", bool((scgpt_cohort["full_matrix_pcc"] > 0.25).all()), f"effects={scgpt_cohort['full_matrix_pcc'].round(4).to_dict()}")
    check("figure_5:scgpt_identity_control", bool((scgpt_effects.loc[scgpt_effects['contrast'].eq('pretrained_vs_identity_shuffle')].groupby('cohort')['gene_mean_pcc'].mean() > 0.60).all()), "correct token identity restores large gene-mean effects in all cohorts")
    for cohort, row in scgpt_cohort.iterrows():
        number("scGPT full-matrix PCC gain versus random", cohort, row["full_matrix_pcc"], "PCC", scgpt_absolute_path)
        number("scGPT gene-mean PCC gain versus random", cohort, row["gene_mean_pcc"], "PCC", scgpt_absolute_path)
        number("scGPT mean within-gene PCC gain versus random", cohort, row["mean_within_gene_pcc"], "PCC", scgpt_absolute_path)
        number("scGPT gene-mean share of MSE reduction", cohort, row["gene_mean_share"], "fraction", scgpt_absolute_path)
    for cohort, row in scgpt_identity.iterrows():
        number("scGPT full-matrix PCC gain versus identity shuffle", cohort, row["full_matrix_pcc"], "PCC", scgpt_absolute_path)
        number("scGPT gene-mean PCC gain versus identity shuffle", cohort, row["gene_mean_pcc"], "PCC", scgpt_absolute_path)
        number("scGPT mean within-gene PCC gain versus identity shuffle", cohort, row["mean_within_gene_pcc"], "PCC", scgpt_absolute_path)
        number("scGPT gene-mean share of MSE reduction versus identity shuffle", cohort, row["gene_mean_share"], "fraction", scgpt_absolute_path)

    provenance_manifest = json.loads((STAGE / "scgpt_whole_human_provenance/manifest.json").read_text())
    common_manifest = json.loads((STAGE / "common_target_representation_audit_whole_human/manifest.json").read_text())
    stratification_manifest = json.loads((STAGE / "representation_spatial_stratification_whole_human/manifest.json").read_text())
    ridge_manifest = json.loads((STAGE / "scgpt_whole_human_static_gene_vectors/manifest.json").read_text())
    check("representation:scgpt_provenance", provenance_manifest.get("status") == "PASS", f"status={provenance_manifest.get('status')}")
    check("representation:common_target_audit", common_manifest.get("status") == "PASS", f"status={common_manifest.get('status')}")
    check("representation:common_target_identity_control", bool(common_manifest.get("included_scgpt_identity_shuffle")), f"included={common_manifest.get('included_scgpt_identity_shuffle')}")
    check("representation:spatial_stratification", stratification_manifest.get("status") == "PASS", f"status={stratification_manifest.get('status')}")
    check("representation:no_image_ridge", ridge_manifest.get("status") == "PASS", f"status={ridge_manifest.get('status')}")
    provenance = json.loads((STAGE / "scgpt_whole_human_provenance/representation_provenance.json").read_text())
    check(
        "representation:scgpt_checkpoint_frozen",
        provenance.get("checkpoint_sha256") == "6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922"
        and int(provenance.get("embedding_dimension", 0)) == 512,
        f"dimension={provenance.get('embedding_dimension')}, checkpoint_sha256={provenance.get('checkpoint_sha256')}",
    )

    common_targets = pd.read_csv(STAGE / "common_target_representation_audit_whole_human/common_heldout_targets.tsv.gz", sep="\t")
    common_counts = common_targets.groupby("cohort").size().to_dict()
    expected_common = {"hippocampus_donor_disjoint": 3574, "dlpfc": 3574, "nac": 3574, "her2st": 3179}
    check("representation:exact_common_targets", common_counts == expected_common, f"counts={common_counts}")

    rep_deciles = pd.read_csv(SOURCE / "figure_5_variance_decile_summary.tsv", sep="\t")
    decile_contrast_counts = rep_deciles["contrast"].value_counts().to_dict()
    check(
        "figure_5:two_representation_deciles",
        len(rep_deciles) == 400
        and rep_deciles["representation"].nunique() == 2
        and decile_contrast_counts
        == {
            "pretrained_vs_constant": 160,
            "pretrained_vs_random": 160,
            "pretrained_vs_identity_shuffled": 80,
        },
        f"rows={len(rep_deciles)}, contrasts={decile_contrast_counts}",
    )
    scgpt_top50 = fixed_top50.loc[fixed_top50["representation"].eq("scGPT whole-human gene-token")].set_index("cohort")
    expected_direction = {
        "hippocampus_donor_disjoint": 1,
        "dlpfc": 1,
        "nac": 1,
        "her2st": 1,
    }
    direction_pass = all(np.sign(float(scgpt_top50.loc[cohort, "mean_effect"])) == direction for cohort, direction in expected_direction.items())
    check("figure_5:scgpt_top50_context_dependence", direction_pass, f"effects={scgpt_top50['mean_effect'].round(4).to_dict()}")
    for cohort, row in scgpt_top50.iterrows():
        number("scGPT top-50 section-centred within-gene PCC gain versus random", cohort, row["mean_effect"], "PCC", SOURCE / "figure_5_target_subset_summary.tsv")

    checkpoint_sensitivity_path = SOURCE / "supp_table_17_scgpt_checkpoint_sensitivity.tsv"
    checkpoint_sensitivity = pd.read_csv(checkpoint_sensitivity_path, sep="\t")
    check(
        "representation:scgpt_checkpoint_sensitivity_complete",
        len(checkpoint_sensitivity) == 24
        and set(checkpoint_sensitivity["checkpoint"]) == {"Whole-human", "Brain"},
        f"rows={len(checkpoint_sensitivity)}, checkpoints={sorted(checkpoint_sensitivity['checkpoint'].unique())}",
    )
    her2_checkpoint = (
        checkpoint_sensitivity.loc[checkpoint_sensitivity["cohort"].eq("her2st")]
        .groupby("checkpoint")["top50_section_centered_pcc_gain"]
        .mean()
    )
    check(
        "representation:her2st_checkpoint_spatial_sensitivity",
        float(her2_checkpoint["Whole-human"]) > 0 and float(her2_checkpoint["Brain"]) < 0,
        f"HER2ST top-50 effects={her2_checkpoint.round(4).to_dict()}",
    )
    for checkpoint, value in her2_checkpoint.items():
        number("scGPT HER2ST top-50 checkpoint-sensitivity effect", checkpoint, value, "PCC", checkpoint_sensitivity_path)

    # Text-level role and terminology checks.
    abstract = (SECTIONS / "abstract.tex").read_text()
    results = (SECTIONS / "results.tex").read_text()
    discussion = (SECTIONS / "discussion.tex").read_text()
    methods = (SECTIONS / "methods.tex").read_text()
    main_tex = (SUBMISSION / "main.tex").read_text()
    combined = "\n".join([abstract, results, discussion, methods])
    check(
        "text:conclusion_forward_title",
        "Pretrained gene representations transfer mean expression more broadly than spatial patterns" in main_tex,
        "current conclusion-forward title is present",
    )
    check("text:full_matrix_terminology", "Pooled PCC" not in combined and "pooled PCC" not in combined, "full-matrix terminology used in manuscript text")
    check("text:scgpt_role", "fixed token vectors" in methods and "scGPT whole-human checkpoint" in combined, "scGPT is identified as a frozen representation source")
    check(
        "text:cross_representation_error_range",
        (
            "91.5--95.8\\%" in combined
            or ("91.9--95.8\\%" in results and "91.5--95.3\\%" in results)
        ),
        "audited representation-specific gene-mean MSE-share ranges are present in the manuscript",
    )
    check(
        "text:context_dependent_spatial_transfer",
        "Spatial effects differed between representations and cohorts" in results
        and "tissue, assay platform and image modality were not independently varied" in discussion,
        "cohort-dependent spatial effect and restricted attribution are explicit",
    )
    check("text:no_representation_leaderboard", "Each representation was compared with its own dimension-matched control" in methods and "compare transfer profiles rather than rank representations" in methods, "within-representation comparison boundary explicit")
    check("text:constant_pcc_boundary", "undefined" in methods and "constant-vector" in methods, "constant-vector gene-mean PCC boundary documented")
    check("text:five_main_figures", all(f"figure_{index}.pdf" in results for index in range(1, 6)), "Results embeds Figures 1--5")

    # Figure 5b must use whole-human per-gene effects, not the older static-token
    # table whose filename predates the checkpoint change.
    gene_effects = pd.read_csv(SOURCE / "figure_5_whole_human_gene_effects_technical_mean.tsv.gz", sep="\t")
    selected = gene_effects.loc[gene_effects["training_defined_top50"]
        & gene_effects["contrast"].eq("pretrained_vs_random")
        & gene_effects["endpoint"].eq("section_centered_within_gene_pcc")]
    paired = selected.pivot(index=["cohort", "gene_id"], columns="representation", values="technical_mean_effect")
    concordance = pd.read_csv(SOURCE / "figure_5_gene_effect_concordance.tsv", sep="\t")
    for cohort in COHORTS_INTERNAL:
        data = paired.loc[cohort]
        r = data["Decima sequence-derived"].corr(data["scGPT whole-human gene-token"])
        expected = concordance.loc[concordance["cohort"].eq(cohort)
            & concordance["endpoint"].eq("section_centered_within_gene_pcc")
            & concordance["target_subset"].eq("training_defined_top50")].iloc[0]
        check(f"figure_5:whole_human_concordance:{cohort}",
              len(data) == 50 and not data.isna().any().any() and np.isclose(r, expected["pearson_r"], atol=1e-12),
              f"n={len(data)}; recomputed Pearson r={r:.12f}; archived r={expected['pearson_r']:.12f}")

    validation = pd.DataFrame(checks)
    numerical = pd.DataFrame(numbers)
    validation.to_csv(VALIDATION, sep="\t", index=False)
    numerical.to_csv(NUMERICAL, sep="\t", index=False)
    status = "PASS" if validation["status"].eq("PASS").all() else "FAIL"

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_commit = "unavailable"
    supplementary_outputs = [
        FIGURES / "extended_data_figure_9.pdf",
        FIGURES / "extended_data_figure_9.png",
        FIGURES / "extended_data_figure_9.svg",
        FIGURES / "extended_data_figure_10.pdf",
        FIGURES / "extended_data_figure_10.png",
        FIGURES / "extended_data_figure_10.svg",
        SUBMISSION / "tables/supp_table_15_representation_provenance.tex",
        SUBMISSION / "tables/supp_table_16_scgpt_replication.tex",
        SUBMISSION / "tables/supp_table_17_scgpt_checkpoint_sensitivity.tex",
        SOURCE / "supp_table_17_scgpt_checkpoint_sensitivity.tsv",
    ]
    manifest = {
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "manuscript_identity": "five-figure component-resolved virtual-spatial-transcriptomics submission",
        "central_claim": "Across two pretrained gene-representation families, held-out-gene matrix gains are consistently dominated by transfer of gene-mean log-expression; within-gene spatial transfer is smaller, signal dependent and context dependent.",
        "analysis_runs": SEEDS,
        "technical_run_boundary": "analysis seeds quantify optimization and control-vector realization variability; they are not biological replicates",
        "figures": figure_records,
        "extended_data_figures": extended_data_records,
        "supplementary_representation_audit": [record(path) for path in supplementary_outputs if path.exists()],
        "validation": {
            "n_checks": len(validation),
            "n_pass": int(validation["status"].eq("PASS").sum()),
            "n_fail": int(validation["status"].eq("FAIL").sum()),
            "table": str(VALIDATION.relative_to(ROOT)),
        },
        "numerical_audit": {
            "n_values": len(numerical),
            "table": str(NUMERICAL.relative_to(ROOT)),
        },
        "representation_boundaries": {
            "Decima": "sequence-derived frozen vector; held out only from downstream decoder fitting",
            "scGPT": "static token-table row from the whole-human checkpoint; no contextual or cell-specific output; exact upstream dataset overlap unresolved",
            "comparison": "parallel within-representation contrasts on exact common downstream targets; no cross-representation leaderboard",
            "constant_vector_gene_mean_pcc": "undefined (NA) because predicted gene means have zero across-gene variance",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    report_lines = [
        "# Five-figure results evidence audit",
        "",
        f"Status: **{status}**",
        "",
        f"Checks passed: {manifest['validation']['n_pass']}/{manifest['validation']['n_checks']}.",
        "",
        "## Central audited result",
        "",
        "Across Decima sequence-derived vectors and static scGPT gene-token vectors, held-out-gene full-matrix improvements were dominated by reduced gene-mean error. Spatial gains were smaller, increased with training-derived target variance and varied across cohorts and representations.",
        "",
        "## Critical semantic checks",
        "",
        "- Constant-vector gene-mean PCC is undefined and stored as NA, not zero.",
        "- scGPT is a frozen representation source, not the predictor introduced by this study.",
        "- Decima and scGPT are compared through within-representation effects on exact common downstream targets, not as a leaderboard.",
        "- Analysis seeds are computational runs; biological-individual effects are reported separately.",
        "",
        "## Validation table",
        "",
    ]
    for row in validation.itertuples(index=False):
        report_lines.append(f"- {row.status}: `{row.check}` — {row.detail}")
    REPORT.write_text("\n".join(report_lines) + "\n")

    # Refresh concise figure-specific manifests from the exact promoted files.
    for figure in range(1, 6):
        figure_checks = [row for row in checks if str(row["check"]).startswith(f"figure_{figure}:")]
        figure_manifest = {
            "status": "PASS" if figure_checks and all(row["status"] == "PASS" for row in figure_checks) else "FAIL",
            "overall_evidence_status": status,
            "figure": f"Figure {figure}",
            **figure_records[str(figure)],
            "analysis_runs": SEEDS if figure > 1 else None,
        }
        (SOURCE / f"figure_{figure}_manifest.json").write_text(json.dumps(figure_manifest, indent=2) + "\n")

    for figure in range(1, 11):
        manifest_path = SOURCE / f"extended_data_figure_{figure}_manifest.json"
        existing: dict[str, object] = {}
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
        prefix = f"extended_data_figure_{figure}:"
        figure_checks = [row for row in checks if str(row["check"]).startswith(prefix)]
        figure_status = "PASS" if figure_checks and all(row["status"] == "PASS" for row in figure_checks) else "FAIL"
        extended_data_manifest = {
            **existing,
            "status": figure_status,
            "figure": f"Extended Data Fig. {figure}",
            **extended_data_records[str(figure)],
            "analysis_runs": (
                {"GeneQuery": SEEDS, "DeepSpot-M": [42]}
                if figure == 10
                else SEEDS
            ),
        }
        manifest_path.write_text(json.dumps(extended_data_manifest, indent=2) + "\n")

    print(json.dumps({"status": status, "checks": manifest["validation"], "numerical_audit": manifest["numerical_audit"]}, indent=2))
    if status != "PASS":
        failed = validation.loc[validation["status"].eq("FAIL")]
        raise SystemExit(f"Evidence audit failed:\n{failed.to_string(index=False)}")


if __name__ == "__main__":
    main()
