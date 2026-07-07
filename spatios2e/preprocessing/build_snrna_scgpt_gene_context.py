#!/usr/bin/env python3
"""Build donor-matched scGPT gene-context embeddings from snRNA counts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from scipy import sparse
from scipy.io import mmread
from torch.utils.data import DataLoader, Dataset, SequentialSampler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda it, **kw: it  # type: ignore


class SimpleVocab:
    """Small vocab shim with the API needed by scGPT model/data collator."""

    def __init__(self, token2idx: Dict[str, int]):
        self.token2idx = dict(token2idx)

    def __len__(self) -> int:
        return len(self.token2idx)

    def __contains__(self, token: str) -> bool:
        return token in self.token2idx

    def __getitem__(self, token: str) -> int:
        return self.token2idx[token]

    def __call__(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2idx[token] for token in tokens]

    def append_token(self, token: str) -> int:
        if token not in self.token2idx:
            self.token2idx[token] = len(self.token2idx)
        return self.token2idx[token]


class SparseCellDataset(Dataset):
    def __init__(self, matrix_csc: sparse.csc_matrix, cls_id: int, pad_value: float):
        self.matrix_csc = matrix_csc
        self.cls_id = int(cls_id)
        self.pad_value = float(pad_value)

    def __len__(self) -> int:
        return self.matrix_csc.shape[1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        col = self.matrix_csc.getcol(idx)
        genes = np.concatenate(([self.cls_id], col.indices.astype(np.int64, copy=False)))
        values = np.concatenate(([self.pad_value], col.data.astype(np.float32, copy=False)))
        return {
            "id": torch.tensor(idx, dtype=torch.long),
            "genes": torch.tensor(genes, dtype=torch.long),
            "expressions": torch.tensor(values, dtype=torch.float32),
        }


def resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text())


def load_target_gene_ids(source_npz: Path) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    arr = np.load(source_npz, allow_pickle=False)
    return (
        arr["gene_ids"].astype(str).tolist(),
        arr["donor_ids"].astype(str).tolist(),
        arr["sample_ids"].astype(str),
        arr["sample_donor_ids"].astype(str),
    )


def load_sample_links(csv_path: Path, snrna_root: Path, root: Path) -> Dict[str, List[Path]]:
    donor_to_snrna: Dict[str, List[Path]] = defaultdict(list)
    with csv_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["modality"] != "snRNA":
                continue
            sample_dir = Path(row["path"]) if row.get("path") else snrna_root / row["gsm_id"]
            if not sample_dir.is_absolute():
                sample_dir = root / sample_dir
            donor_to_snrna[row["donor_id"]].append(sample_dir)
    return {donor: sorted(paths) for donor, paths in donor_to_snrna.items()}


def load_feature_rows(features_path: Path) -> Tuple[List[str], List[str], List[str]]:
    feature_ids: List[str] = []
    feature_names: List[str] = []
    feature_types: List[str] = []
    with gzip.open(features_path, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Expected at least 3 columns in {features_path}")
            feature_ids.append(fields[0])
            feature_names.append(fields[1])
            feature_types.append(fields[2])
    return feature_ids, feature_names, feature_types


def build_row_maps(
    feature_ids: Sequence[str],
    feature_names: Sequence[str],
    feature_types: Sequence[str],
    target_gene_ids: Sequence[str],
    vocab: SimpleVocab,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_vocab_ids = np.full(len(feature_ids), -1, dtype=np.int64)
    row_target_ids = np.full(len(feature_ids), -1, dtype=np.int64)
    row_is_gene_expression = np.zeros(len(feature_ids), dtype=bool)
    target_to_idx = {gene_id: idx for idx, gene_id in enumerate(target_gene_ids)}
    for row_idx, (gene_id, gene_name, feature_type) in enumerate(zip(feature_ids, feature_names, feature_types)):
        if feature_type != "Gene Expression":
            continue
        row_is_gene_expression[row_idx] = True
        if gene_name in vocab:
            row_vocab_ids[row_idx] = vocab[gene_name]
        target_idx = target_to_idx.get(gene_id)
        if target_idx is not None:
            row_target_ids[row_idx] = target_idx
    return row_vocab_ids, row_target_ids, row_is_gene_expression


def choose_cells(col_counts: np.ndarray, min_counts: int, max_cells: int) -> np.ndarray:
    keep = np.flatnonzero(col_counts >= min_counts)
    if max_cells > 0 and keep.size > max_cells:
        order = np.argsort(col_counts[keep])[::-1][:max_cells]
        keep = np.sort(keep[order])
    return keep.astype(np.int64, copy=False)


def make_scgpt_matrix(
    coo: sparse.coo_matrix,
    row_vocab_ids: np.ndarray,
    keep_cols: np.ndarray,
    vocab_size: int,
) -> sparse.csc_matrix:
    col_map = np.full(coo.shape[1], -1, dtype=np.int64)
    col_map[keep_cols] = np.arange(keep_cols.size, dtype=np.int64)
    row_ids = row_vocab_ids[coo.row]
    cols = col_map[coo.col]
    keep = (row_ids >= 0) & (cols >= 0)
    mat = sparse.csc_matrix(
        (coo.data[keep].astype(np.float32, copy=False), (row_ids[keep], cols[keep])),
        shape=(vocab_size, keep_cols.size),
        dtype=np.float32,
    )
    mat.sum_duplicates()
    return mat


def make_target_matrix(
    coo: sparse.coo_matrix,
    row_target_ids: np.ndarray,
    keep_cols: np.ndarray,
    n_target_genes: int,
) -> sparse.csr_matrix:
    col_map = np.full(coo.shape[1], -1, dtype=np.int64)
    col_map[keep_cols] = np.arange(keep_cols.size, dtype=np.int64)
    row_ids = row_target_ids[coo.row]
    cols = col_map[coo.col]
    keep = (row_ids >= 0) & (cols >= 0)
    mat = sparse.csr_matrix(
        (coo.data[keep].astype(np.float32, copy=False), (row_ids[keep], cols[keep])),
        shape=(n_target_genes, keep_cols.size),
        dtype=np.float32,
    )
    mat.sum_duplicates()
    return mat


def add_scgpt_to_path(model_dir: Path, scgpt_root: Path | None) -> None:
    candidates = []
    if scgpt_root is not None:
        candidates.append(scgpt_root)
    candidates.append(model_dir.parent)
    for candidate in candidates:
        if candidate and (candidate / "scgpt").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def load_model(
    model_dir: Path,
    device: torch.device,
    use_fast_transformer: bool,
    scgpt_root: Path | None,
):
    add_scgpt_to_path(model_dir, scgpt_root)
    try:
        from scgpt.model import TransformerModel
        from scgpt.utils import load_pretrained
    except ImportError as exc:
        raise ImportError(
            "scGPT is required to build scGPT gene-context profiles. "
            "Install scGPT or pass --scgpt-root pointing to a checkout that contains the scgpt package."
        ) from exc

    model_configs = json.loads((model_dir / "args.json").read_text())
    vocab = SimpleVocab(json.loads((model_dir / "vocab.json").read_text()))
    for token in ("<pad>", "<cls>", "<eoc>"):
        vocab.append_token(token)
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=int(model_configs["embsize"]),
        nhead=int(model_configs["nheads"]),
        d_hid=int(model_configs["d_hid"]),
        nlayers=int(model_configs["nlayers"]),
        nlayers_cls=int(model_configs["n_layers_cls"]),
        n_cls=1,
        vocab=vocab,
        dropout=float(model_configs["dropout"]),
        pad_token=str(model_configs["pad_token"]),
        pad_value=float(model_configs["pad_value"]),
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        pre_norm=False,
        input_emb_style=str(model_configs.get("input_emb_style", "continuous")),
        n_input_bins=int(model_configs.get("n_bins", 51)),
    )
    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    load_pretrained(model, state, verbose=False)
    model.to(device)
    model.eval()
    return model, vocab, model_configs


def embed_cells(
    matrix_csc: sparse.csc_matrix,
    model,
    vocab: SimpleVocab,
    model_configs: Dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    from scgpt.data_collator import DataCollator

    dataset = SparseCellDataset(
        matrix_csc=matrix_csc,
        cls_id=vocab["<cls>"],
        pad_value=float(model_configs["pad_value"]),
    )
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[str(model_configs["pad_token"])],
        pad_value=float(model_configs["pad_value"]),
        do_mlm=False,
        do_binning=True,
        max_length=int(model_configs.get("max_seq_len", 1200)),
        sampling=True,
        keep_first_n_tokens=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
    )
    embeddings = np.zeros((len(dataset), int(model_configs["embsize"])), dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding nuclei", leave=False):
            genes = batch["gene"].to(device, non_blocking=True)
            expr = batch["expr"].to(device, non_blocking=True)
            padding = genes.eq(vocab[str(model_configs["pad_token"])])
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                encoded = model._encode(genes, expr, src_key_padding_mask=padding)
                cell_emb = encoded[:, 0, :]
            cell_emb = torch.nn.functional.normalize(cell_emb.float(), p=2, dim=1)
            cell_emb = cell_emb.cpu().numpy().astype(np.float32, copy=False)
            embeddings[offset : offset + cell_emb.shape[0]] = cell_emb
            offset += cell_emb.shape[0]
    return embeddings


def build_artifact(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    cfg_path = resolve_path(args.config, root)
    cfg = load_yaml(cfg_path)
    paths = cfg["paths"]
    source_npz = resolve_path(paths["source_snrna_base_npz"], root)
    target_gene_ids, donor_ids, sample_ids, sample_donor_ids = load_target_gene_ids(source_npz)
    donor_to_idx = {donor: idx for idx, donor in enumerate(donor_ids)}
    sample_links_csv = resolve_path(paths.get("sample_links_csv", "metadata/sample_links.csv"), root)
    snrna_root = resolve_path(paths.get("snrna_root", "data/snRNA_raw"), root)
    donor_to_snrna = load_sample_links(sample_links_csv, snrna_root, root)

    missing = sorted(set(donor_ids) - set(donor_to_snrna))
    if missing:
        raise KeyError(f"Missing snRNA sample dirs for donors: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_dir = resolve_path(args.model_dir, root)
    scgpt_root = resolve_path(args.scgpt_root, root) if args.scgpt_root else None
    model, vocab, model_configs = load_model(model_dir, device, args.use_fast_transformer, scgpt_root)
    min_counts = int(args.min_barcode_counts)
    n_donor = len(donor_ids)
    n_gene = len(target_gene_ids)
    emb_dim = int(model_configs["embsize"])
    weighted_sum = np.zeros((n_donor, n_gene, emb_dim), dtype=np.float32)
    expr_sum = np.zeros((n_donor, n_gene), dtype=np.float64)
    expressing_cells = np.zeros((n_donor, n_gene), dtype=np.int32)
    donor_cell_sum = np.zeros((n_donor, emb_dim), dtype=np.float64)
    donor_cell_count = np.zeros(n_donor, dtype=np.int64)

    sample_summaries = []
    selected_samples = [(donor, sample_dir) for donor in donor_ids for sample_dir in donor_to_snrna[donor]]
    if args.sample_limit > 0:
        selected_samples = selected_samples[: args.sample_limit]

    for sample_idx, (donor, sample_dir) in enumerate(selected_samples, start=1):
        start = time.time()
        print(f"[{sample_idx}/{len(selected_samples)}] {donor} {sample_dir}", flush=True)
        features_path = sample_dir / "features.tsv.gz"
        matrix_path = sample_dir / "matrix.mtx.gz"
        if not features_path.exists() or not matrix_path.exists():
            raise FileNotFoundError(f"Missing 10x files under {sample_dir}")
        feature_ids, feature_names, feature_types = load_feature_rows(features_path)
        row_vocab_ids, row_target_ids, row_is_gene_expr = build_row_maps(
            feature_ids,
            feature_names,
            feature_types,
            target_gene_ids,
            vocab,
        )
        with gzip.open(matrix_path, "rb") as handle:
            coo = mmread(handle).tocoo()
        coo.data = coo.data.astype(np.float32, copy=False)
        gene_expr_nnz = row_is_gene_expr[coo.row]
        col_counts = np.bincount(
            coo.col[gene_expr_nnz],
            weights=coo.data[gene_expr_nnz],
            minlength=coo.shape[1],
        ).astype(np.float64)
        keep_cols = choose_cells(col_counts, min_counts=min_counts, max_cells=int(args.max_cells_per_sample))
        if keep_cols.size == 0:
            raise RuntimeError(f"No cells retained for {sample_dir} with min counts {min_counts}")
        scgpt_matrix = make_scgpt_matrix(coo, row_vocab_ids, keep_cols, len(vocab))
        target_matrix = make_target_matrix(coo, row_target_ids, keep_cols, n_gene)
        embeddings = embed_cells(
            scgpt_matrix,
            model=model,
            vocab=vocab,
            model_configs=model_configs,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            amp=bool(args.amp),
        )
        donor_idx = donor_to_idx[donor]
        donor_cell_sum[donor_idx] += embeddings.sum(axis=0, dtype=np.float64)
        donor_cell_count[donor_idx] += embeddings.shape[0]
        weighted = target_matrix @ embeddings
        weighted_sum[donor_idx] += np.asarray(weighted, dtype=np.float32)
        sample_expr_sum = np.asarray(target_matrix.sum(axis=1)).reshape(-1)
        expr_sum[donor_idx] += sample_expr_sum
        expressing_cells[donor_idx] += np.asarray((target_matrix > 0).sum(axis=1)).reshape(-1).astype(np.int32)
        sample_summaries.append(
            {
                "donor": donor,
                "sample_dir": str(sample_dir),
                "matrix_shape": [int(coo.shape[0]), int(coo.shape[1])],
                "n_cells_retained": int(keep_cols.size),
                "n_scgpt_input_nnz": int(scgpt_matrix.nnz),
                "n_target_nnz": int(target_matrix.nnz),
                "n_vocab_rows_matched": int(np.sum(row_vocab_ids >= 0)),
                "n_target_rows_matched": int(np.sum(row_target_ids >= 0)),
                "elapsed_sec": float(time.time() - start),
            }
        )
        del coo, scgpt_matrix, target_matrix, embeddings, weighted
        if device.type == "cuda":
            torch.cuda.empty_cache()

    donor_cell_mean = np.zeros_like(donor_cell_sum, dtype=np.float32)
    for donor_idx in range(n_donor):
        if donor_cell_count[donor_idx] > 0:
            donor_cell_mean[donor_idx] = (
                donor_cell_sum[donor_idx] / float(donor_cell_count[donor_idx])
            ).astype(np.float32)
    donor_gene_context = np.zeros_like(weighted_sum, dtype=np.float32)
    for donor_idx in range(n_donor):
        valid = expr_sum[donor_idx] > 0
        donor_gene_context[donor_idx, valid] = (
            weighted_sum[donor_idx, valid] / expr_sum[donor_idx, valid, np.newaxis].astype(np.float32)
        )
        donor_gene_context[donor_idx, ~valid] = donor_cell_mean[donor_idx]
    norm = np.linalg.norm(donor_gene_context, axis=2, keepdims=True)
    donor_gene_context = donor_gene_context / np.maximum(norm, 1.0e-8)

    out_path = resolve_path(args.output, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        gene_ids=np.asarray(target_gene_ids, dtype=str),
        donor_ids=np.asarray(donor_ids, dtype=str),
        sample_ids=sample_ids,
        sample_donor_ids=sample_donor_ids,
        donor_scgpt_gene_context=donor_gene_context.astype(np.float32),
        donor_scgpt_gene_expr_sum=expr_sum.astype(np.float32),
        donor_scgpt_gene_n_cells_expr=expressing_cells.astype(np.int32),
        donor_scgpt_cell_mean=donor_cell_mean.astype(np.float32),
        donor_scgpt_cell_count=donor_cell_count.astype(np.int64),
    )
    summary = {
        "output": str(out_path),
        "config": str(cfg_path),
        "model_dir": str(model_dir),
        "device": str(device),
        "n_genes": int(n_gene),
        "n_donors": int(n_donor),
        "embedding_dim": int(emb_dim),
        "min_barcode_counts": int(min_counts),
        "max_cells_per_sample": int(args.max_cells_per_sample),
        "sample_limit": int(args.sample_limit),
        "n_samples_processed": int(len(sample_summaries)),
        "donor_cell_count": {donor: int(donor_cell_count[donor_to_idx[donor]]) for donor in donor_ids},
        "n_zero_expr_donor_gene": int(np.sum(expr_sum <= 0)),
        "n_nonzero_expr_donor_gene": int(np.sum(expr_sum > 0)),
        "sample_summaries": sample_summaries,
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository/data root for relative config paths.")
    parser.add_argument("--model-dir", type=Path, default=Path("scGPT/brain_model"))
    parser.add_argument("--scgpt-root", type=Path, default=None, help="Optional checkout containing the scgpt package.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/snrna/snrna_scgpt_gene_context.npz"))
    parser.add_argument("--min-barcode-counts", type=int, default=500)
    parser.add_argument("--max-cells-per-sample", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-fast-transformer", action="store_true")
    build_artifact(parser.parse_args())


if __name__ == "__main__":
    main()
