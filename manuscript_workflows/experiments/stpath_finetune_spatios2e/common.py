#!/usr/bin/env python3
"""Shared utilities for STPath finetuning on SpatioS2E."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_yaml_or_json(path: Path) -> Dict:
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def resolve_split(cfg: Dict) -> Dict[str, List[str]]:
    if "split" in cfg and cfg["split"]:
        return {k: [str(x) for x in v] for k, v in cfg["split"].items()}
    split_cfg_path = cfg.get("paths", {}).get("split_config")
    if not split_cfg_path:
        raise ValueError("No split found: set cfg['split'] or paths.split_config")
    split_cfg = load_yaml_or_json(Path(split_cfg_path))
    return {k: [str(x) for x in v] for k, v in split_cfg["split"].items()}


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    c = coords.astype(np.float32, copy=True)
    c[:, 0] = c[:, 0] - c[:, 0].min()
    c[:, 1] = c[:, 1] - c[:, 1].min()
    mn = c.min(axis=0)
    mx = c.max(axis=0)
    rg = np.maximum(mx - mn, 1.0e-6)
    return ((c - mn) / rg * 100.0).astype(np.float32)


def load_aligned_sample(
    hist_root: Path,
    spatial_root: Path,
    sample: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist_path = hist_root / sample / "embeddings.npz"
    spatial_path = spatial_root / sample / "spatial_features.csv.gz"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing histology embeddings: {hist_path}")
    if not spatial_path.exists():
        raise FileNotFoundError(f"Missing spatial features: {spatial_path}")

    h = np.load(hist_path, allow_pickle=True)
    hist_x = h["embeddings"].astype(np.float32)
    hist_bc = h["barcodes"].astype(str)

    s = pd.read_csv(spatial_path, usecols=["barcode", "pxl_col", "pxl_row"])
    s["barcode"] = s["barcode"].astype(str)
    s_map = {bc: i for i, bc in enumerate(s["barcode"].tolist())}

    keep_hist = []
    keep_spat = []
    for i, bc in enumerate(hist_bc):
        j = s_map.get(bc)
        if j is not None:
            keep_hist.append(i)
            keep_spat.append(j)

    if not keep_hist:
        raise ValueError(f"No aligned barcodes for sample {sample}")

    x = hist_x[keep_hist]
    coords = s.iloc[keep_spat][["pxl_col", "pxl_row"]].to_numpy(dtype=np.float32)
    bcs = hist_bc[keep_hist]
    return x, coords, bcs


def fit_or_load_adapter(
    adapter_path: Path,
    fit_samples: Sequence[str],
    hist_root: Path,
    spatial_root: Path,
    in_dim: int,
    out_dim: int = 1536,
) -> Dict[str, np.ndarray]:
    if in_dim == out_dim:
        return {
            "identity": np.array([1], dtype=np.int8),
            "in_dim": np.array([in_dim], dtype=np.int32),
            "out_dim": np.array([out_dim], dtype=np.int32),
        }

    if adapter_path.exists():
        arr = np.load(adapter_path, allow_pickle=True)
        return {k: arr[k] for k in arr.files}

    ipca = IncrementalPCA(n_components=out_dim, batch_size=4096)
    for s in fit_samples:
        x, _, _ = load_aligned_sample(hist_root, spatial_root, s)
        ipca.partial_fit(x)

    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        adapter_path,
        in_dim=np.array([in_dim], dtype=np.int32),
        out_dim=np.array([out_dim], dtype=np.int32),
        mean=ipca.mean_.astype(np.float32),
        components=ipca.components_.astype(np.float32),
        explained_ratio=np.array([float(ipca.explained_variance_ratio_.sum())], dtype=np.float32),
    )
    arr = np.load(adapter_path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def apply_adapter(x: np.ndarray, adapter: Dict[str, np.ndarray]) -> np.ndarray:
    if "identity" in adapter:
        return x.astype(np.float32, copy=False)
    mean = adapter["mean"]
    comp = adapter["components"]
    return ((x - mean) @ comp.T).astype(np.float32)


def load_expression(npz_path: Path) -> Dict[str, object]:
    arr = np.load(npz_path, allow_pickle=True)
    shape = tuple(arr["shape"].tolist())
    csr = (arr["data"], arr["indices"], arr["indptr"], shape)
    gene_ids = arr["gene_ids"].astype(str).tolist()
    barcodes = arr["barcodes"].astype(str).tolist()
    return {"csr": csr, "gene_ids": gene_ids, "barcodes": barcodes}


def csr_to_dense(csr_tuple) -> np.ndarray:
    data, indices, indptr, shape = csr_tuple
    from scipy import sparse

    csr = sparse.csr_matrix((data, indices, indptr), shape=shape)
    return csr.toarray().T.astype(np.float32)


def canon_gene_id(gid: str) -> str:
    gid = str(gid)
    if gid.startswith("ENSG") and "." in gid:
        return gid.split(".")[0]
    return gid


def build_gene2token_map(gene_vocab_path: Path) -> Dict[str, int]:
    symbol2gene = json.loads(gene_vocab_path.read_text())
    gene_ids = sorted(set(symbol2gene.values()))
    return {g: i + 2 for i, g in enumerate(gene_ids)}  # 0=<pad>, 1=<mask>


def map_expression_genes_to_tokens(expr_gene_ids: Sequence[str], gene2token: Dict[str, int]) -> np.ndarray:
    token_ids = np.full((len(expr_gene_ids),), -1, dtype=np.int64)
    for i, gid in enumerate(expr_gene_ids):
        tok = gene2token.get(gid)
        if tok is None:
            tok = gene2token.get(canon_gene_id(gid))
        if tok is not None:
            token_ids[i] = int(tok)
    return token_ids


class STPathFinetuneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[str],
        expression_root: Path,
        hist_root: Path,
        spatial_root: Path,
        adapter: Dict[str, np.ndarray],
        gene2token: Dict[str, int],
        split_name: str,
    ) -> None:
        self.samples = list(samples)
        self.expression_root = expression_root
        self.hist_root = hist_root
        self.spatial_root = spatial_root
        self.adapter = adapter
        self.gene2token = gene2token
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        x_raw, coords_raw, hist_barcodes = load_aligned_sample(self.hist_root, self.spatial_root, sample)
        x = apply_adapter(x_raw, self.adapter)
        coords = normalize_coords(coords_raw)

        expr = load_expression(self.expression_root / sample / f"{self.split_name}.npz")
        y_dense = csr_to_dense(expr["csr"])
        expr_gene_ids = expr["gene_ids"]
        expr_barcodes = expr["barcodes"]
        token_ids = map_expression_genes_to_tokens(expr_gene_ids, self.gene2token)

        expr_index = {bc: i for i, bc in enumerate(expr_barcodes)}
        keep_hist = []
        keep_expr = []
        for i, bc in enumerate(hist_barcodes.tolist()):
            j = expr_index.get(bc)
            if j is not None:
                keep_hist.append(i)
                keep_expr.append(j)
        if not keep_expr:
            raise ValueError(f"No overlapping barcodes for {sample}")

        x = x[keep_hist]
        coords = coords[keep_hist]
        y = y_dense[keep_expr]
        barcodes = hist_barcodes[keep_hist].astype(str)

        return {
            "sample": sample,
            "x": torch.tensor(x, dtype=torch.float32),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "gene_ids": expr_gene_ids,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "barcodes": barcodes,
        }


class STPathSubsetPredictor(torch.nn.Module):
    """STPath model wrapper with subset-gene prediction without full one-hot gene tokens."""

    def __init__(self, stpath_root: Path, stpath_weight: Path, device: torch.device):
        super().__init__()
        if str(stpath_root) not in sys.path:
            sys.path.append(str(stpath_root))

        from stpath.model.model import STFM  # type: ignore
        from stpath.model.nn_utils.config import ModelConfig  # type: ignore
        from stpath.tokenization import (  # type: ignore
            AnnotationTokenizer,
            GeneExpTokenizer,
            IDTokenizer,
            ImageTokenizer,
            TokenizerTools,
        )

        gene_vocab = stpath_root / "utils_data" / "symbol2ensembl.json"
        self.tokenizer = TokenizerTools(
            ge_tokenizer=GeneExpTokenizer(str(gene_vocab)),
            image_tokenizer=ImageTokenizer(feature_dim=1536),
            tech_tokenizer=IDTokenizer(id_type="tech"),
            specie_tokenizer=IDTokenizer(id_type="specie"),
            organ_tokenizer=IDTokenizer(id_type="organ"),
            cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
            domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
        )

        config = ModelConfig.get_default_config()
        config.feature_dim = 1536
        config.activation = "gelu"
        config.n_genes = self.tokenizer.ge_tokenizer.n_tokens
        config.n_tech = self.tokenizer.tech_tokenizer.n_tokens
        config.n_species = self.tokenizer.specie_tokenizer.n_tokens
        config.n_organs = self.tokenizer.organ_tokenizer.n_tokens
        config.backbone = "spatial_transformer"

        model = STFM(config).to(device)
        state = torch.load(stpath_weight, map_location=device)
        model.load_state_dict(state, strict=True)
        self.model = model
        self.mask_token_id = int(self.tokenizer.ge_tokenizer.mask_token_id)
        self.d_model = int(self.model.input_encoder.image_embed.out_features)

    def set_trainable(
        self,
        train_gene_head: bool,
        train_image_embed: bool,
        train_gene_embed: bool,
        unfreeze_last_n_blocks: int,
    ) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

        if train_gene_head:
            for p in self.model.gene_exp_head.parameters():
                p.requires_grad = True
        if train_image_embed:
            for p in self.model.input_encoder.image_embed.parameters():
                p.requires_grad = True
        if train_gene_embed:
            for p in self.model.input_encoder.gene_embed.parameters():
                p.requires_grad = True

        if unfreeze_last_n_blocks > 0 and hasattr(self.model.model, "blks"):
            blks = self.model.model.blks
            n = min(int(unfreeze_last_n_blocks), len(blks))
            for i in range(len(blks) - n, len(blks)):
                for p in blks[i].parameters():
                    p.requires_grad = True

    def encode_spots(
        self,
        img_tokens: torch.Tensor,
        coords: torch.Tensor,
        tech_tokens: torch.Tensor,
        organ_tokens: torch.Tensor,
    ) -> torch.Tensor:
        n = img_tokens.shape[0]
        img_embed = self.model.input_encoder.image_embed(img_tokens)
        ge_mask_vec = self.model.input_encoder.gene_embed.weight[:, self.mask_token_id]
        ge_embed = ge_mask_vec.unsqueeze(0).expand(n, -1)
        tech_embed = self.model.input_encoder.tech_embed(tech_tokens)
        organ_embed = self.model.input_encoder.organ_embed(organ_tokens)
        x = img_embed + ge_embed + tech_embed + organ_embed
        batch_idx = torch.zeros(n, dtype=torch.long, device=img_tokens.device)
        return self.model.model(x, coords, batch_idx)

    def predict_from_hidden(self, h_spot: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        h_norm = self.model.gene_exp_head[0](h_spot)
        w = self.model.gene_exp_head[1].weight.index_select(0, token_ids)
        b = self.model.gene_exp_head[1].bias.index_select(0, token_ids)
        return h_norm @ w.T + b
