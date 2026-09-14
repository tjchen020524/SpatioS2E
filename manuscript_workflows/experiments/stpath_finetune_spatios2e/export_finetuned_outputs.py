#!/usr/bin/env python3
"""Export finetuned STPath embeddings and baseline predictions for all splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.stpath_finetune_spatios2e.common import (
    STPathFinetuneDataset,
    STPathSubsetPredictor,
    build_gene2token_map,
    fit_or_load_adapter,
    load_aligned_sample,
    load_yaml_or_json,
    resolve_split,
)


def export_all(cfg_path: Path, ckpt_path: Path) -> None:
    cfg = load_yaml_or_json(cfg_path)
    split_map = resolve_split(cfg)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = cfg["paths"]
    hist_root = Path(paths["hist_root"])
    spatial_root = Path(paths["spatial_root"])
    expr_root = Path(paths["expression_root"])
    adapter_path = Path(paths["adapter_path"])
    gene_vocab = Path(paths["gene_vocab"])
    emb_out_root = Path(paths["embed_out_root"])
    pred_out_root = Path(paths["pred_out_root"])
    emb_out_root.mkdir(parents=True, exist_ok=True)
    pred_out_root.mkdir(parents=True, exist_ok=True)

    all_samples = split_map.get("train", []) + split_map.get("val", []) + split_map.get("test", [])
    fit_samples = split_map.get("train", [])
    if not fit_samples:
        raise RuntimeError("No train samples for adapter")
    first_x, _, _ = load_aligned_sample(hist_root, spatial_root, fit_samples[0])
    adapter = fit_or_load_adapter(
        adapter_path=adapter_path,
        fit_samples=fit_samples,
        hist_root=hist_root,
        spatial_root=spatial_root,
        in_dim=int(first_x.shape[1]),
        out_dim=1536,
    )
    gene2token = build_gene2token_map(gene_vocab)

    predictor = STPathSubsetPredictor(
        stpath_root=Path(paths["stpath_root"]),
        stpath_weight=Path(paths["stpath_weight"]),
        device=device,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    predictor.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    predictor.eval()

    tech_label = str(cfg["model"].get("tech_type", "Visium"))
    organ_label = str(cfg["model"].get("organ_type", "Brain"))
    tech_id = int(state.get("tech_id", predictor.tokenizer.tech_tokenizer.encode(tech_label, align_first=True)))
    organ_id = int(state.get("organ_id", predictor.tokenizer.organ_tokenizer.encode(organ_label, align_first=True)))

    spot_chunk = int(cfg.get("export", {}).get("spot_chunk_size", 2048))
    gene_chunk = int(cfg.get("export", {}).get("gene_chunk_size", 512))

    summary = {
        "config": str(cfg_path),
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "spot_chunk_size": spot_chunk,
        "gene_chunk_size": gene_chunk,
        "samples": {},
    }

    with torch.no_grad():
        for split_name, samples in split_map.items():
            for sample in samples:
                ds = STPathFinetuneDataset(
                    samples=[sample],
                    expression_root=expr_root,
                    hist_root=hist_root,
                    spatial_root=spatial_root,
                    adapter=adapter,
                    gene2token=gene2token,
                    split_name=split_name,
                )
                batch = ds[0]
                x = batch["x"].to(device)
                coords = batch["coords"].to(device)
                token_ids = batch["token_ids"].to(device)
                gene_ids = batch["gene_ids"]
                barcodes = batch["barcodes"]
                coords_np = batch["coords"].cpu().numpy().astype(np.float32)

                tech_tokens = torch.full((x.shape[0],), tech_id, dtype=torch.long, device=device)
                organ_tokens = torch.full((x.shape[0],), organ_id, dtype=torch.long, device=device)

                h_list = []
                for st in range(0, x.shape[0], spot_chunk):
                    ed = min(x.shape[0], st + spot_chunk)
                    h = predictor.encode_spots(x[st:ed], coords[st:ed], tech_tokens[st:ed], organ_tokens[st:ed])
                    h_list.append(h)
                h_spot = torch.cat(h_list, dim=0)
                h_np = h_spot.detach().cpu().numpy().astype(np.float32)

                pred = np.zeros((x.shape[0], len(gene_ids)), dtype=np.float32)
                valid_pos = torch.where(token_ids >= 0)[0]
                for st in range(0, valid_pos.numel(), gene_chunk):
                    ed = min(valid_pos.numel(), st + gene_chunk)
                    pos = valid_pos[st:ed]
                    tok = token_ids.index_select(0, pos)
                    pred_chunk = predictor.predict_from_hidden(h_spot, tok).detach().cpu().numpy().astype(np.float32)
                    pred[:, pos.cpu().numpy()] = pred_chunk

                emb_dir = emb_out_root / sample
                pred_dir = pred_out_root / sample
                emb_dir.mkdir(parents=True, exist_ok=True)
                pred_dir.mkdir(parents=True, exist_ok=True)

                np.savez_compressed(
                    emb_dir / "stpath_embeddings.npz",
                    embeddings=h_np,
                    barcodes=barcodes,
                    coords=coords_np,
                )
                np.savez_compressed(
                    pred_dir / "stpath_pred.npz",
                    pred=pred,
                    gene_ids=np.array(gene_ids, dtype="U"),
                    barcodes=barcodes,
                    token_ids=token_ids.detach().cpu().numpy(),
                )

                summary["samples"][sample] = {
                    "status": "ok",
                    "split": split_name,
                    "n_spots": int(x.shape[0]),
                    "n_genes": int(len(gene_ids)),
                    "mapped_genes": int(valid_pos.numel()),
                }
                print(
                    f"[export] {sample} split={split_name} spots={x.shape[0]} genes={len(gene_ids)} mapped={valid_pos.numel()}"
                )

    (emb_out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (pred_out_root / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args()
    export_all(args.config, args.ckpt)
