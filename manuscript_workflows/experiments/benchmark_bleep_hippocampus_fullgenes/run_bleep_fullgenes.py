#!/usr/bin/env python3
from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
import os
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse


ROOT = DATA_ROOT
EXP_ROOT = ROOT / "experiments/benchmark_bleep_hippocampus_fullgenes"
DATA_ROOT = EXP_ROOT / "data"
EXPRESSION_ROOT = ROOT / "experiments/sample_split_fullgenes/expression_full"
BLEEP_ROOT = Path("/users/tchen2/SpatioS2E/third_party/BLEEP")


def patch_torch_pytree() -> None:
    import torch.utils._pytree as pytree

    if hasattr(pytree, "register_pytree_node") or not hasattr(pytree, "_register_pytree_node"):
        return

    def register_pytree_node(node_type, flatten_fn, unflatten_fn, *args, **kwargs):
        return pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

    pytree.register_pytree_node = register_pytree_node


patch_torch_pytree()
if str(BLEEP_ROOT) not in sys.path:
    sys.path.append(str(BLEEP_ROOT))

import config as BLEEP_CFG  # type: ignore  # noqa: E402
from models import CLIPModel  # type: ignore  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


@dataclass
class RunningStats:
    count: int = 0
    sum_pred: float = 0.0
    sum_true: float = 0.0
    sum_pred2: float = 0.0
    sum_true2: float = 0.0
    sum_pred_true: float = 0.0
    sum_sq_error: float = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        true = np.asarray(true, dtype=np.float32).reshape(-1)
        mask = np.isfinite(pred) & np.isfinite(true)
        if not np.any(mask):
            return
        p = pred[mask].astype(np.float64, copy=False)
        t = true[mask].astype(np.float64, copy=False)
        self.count += int(p.size)
        self.sum_pred += float(p.sum())
        self.sum_true += float(t.sum())
        self.sum_pred2 += float((p * p).sum())
        self.sum_true2 += float((t * t).sum())
        self.sum_pred_true += float((p * t).sum())
        self.sum_sq_error += float(((p - t) ** 2).sum())

    def mse(self) -> float:
        return self.sum_sq_error / self.count if self.count else float("nan")

    def corr(self) -> float:
        if self.count <= 1:
            return float("nan")
        mp = self.sum_pred / self.count
        mt = self.sum_true / self.count
        vp = self.sum_pred2 / self.count - mp * mp
        vt = self.sum_true2 / self.count - mt * mt
        if vp <= 0 or vt <= 0:
            return float("nan")
        cov = self.sum_pred_true / self.count - mp * mt
        return float(cov / np.sqrt(vp * vt))


class GeneVectorStats:
    def __init__(self, n_genes: int) -> None:
        self.count = np.zeros(n_genes, dtype=np.int64)
        self.sum_pred = np.zeros(n_genes, dtype=np.float64)
        self.sum_true = np.zeros(n_genes, dtype=np.float64)
        self.sum_pred2 = np.zeros(n_genes, dtype=np.float64)
        self.sum_true2 = np.zeros(n_genes, dtype=np.float64)
        self.sum_pred_true = np.zeros(n_genes, dtype=np.float64)
        self.sum_sq_error = np.zeros(n_genes, dtype=np.float64)

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        mask = np.isfinite(pred) & np.isfinite(true)
        self.count += mask.sum(axis=0).astype(np.int64)
        p = np.where(mask, pred, 0.0).astype(np.float64, copy=False)
        t = np.where(mask, true, 0.0).astype(np.float64, copy=False)
        self.sum_pred += p.sum(axis=0)
        self.sum_true += t.sum(axis=0)
        self.sum_pred2 += (p * p).sum(axis=0)
        self.sum_true2 += (t * t).sum(axis=0)
        self.sum_pred_true += (p * t).sum(axis=0)
        self.sum_sq_error += ((p - t) ** 2).sum(axis=0)

    def mse(self) -> np.ndarray:
        out = np.full(self.count.shape, np.nan, dtype=np.float64)
        mask = self.count > 0
        out[mask] = self.sum_sq_error[mask] / self.count[mask]
        return out

    def corr(self) -> np.ndarray:
        out = np.full(self.count.shape, np.nan, dtype=np.float64)
        mask = self.count > 1
        if not np.any(mask):
            return out
        mp = np.zeros_like(self.sum_pred)
        mt = np.zeros_like(self.sum_true)
        mp[mask] = self.sum_pred[mask] / self.count[mask]
        mt[mask] = self.sum_true[mask] / self.count[mask]
        vp = np.zeros_like(self.sum_pred)
        vt = np.zeros_like(self.sum_true)
        vp[mask] = self.sum_pred2[mask] / self.count[mask] - mp[mask] * mp[mask]
        vt[mask] = self.sum_true2[mask] / self.count[mask] - mt[mask] * mt[mask]
        good = mask & (vp > 0) & (vt > 0)
        cov = np.zeros_like(self.sum_pred)
        cov[good] = self.sum_pred_true[good] / self.count[good] - mp[good] * mt[good]
        out[good] = cov[good] / np.sqrt(vp[good] * vt[good])
        return out


class ExpressionStore:
    def __init__(self, gene_ids: list[str]) -> None:
        self.gene_ids = gene_ids
        self._cache: dict[str, tuple[sparse.csr_matrix, dict[str, int]]] = {}

    def _load(self, sample: str) -> tuple[sparse.csr_matrix, dict[str, int]]:
        cached = self._cache.get(sample)
        if cached is not None:
            return cached
        arr = np.load(EXPRESSION_ROOT / sample / "test.npz", allow_pickle=True)
        shape = tuple(arr["shape"].tolist())
        csr = sparse.csr_matrix((arr["data"], arr["indices"], arr["indptr"]), shape=shape)
        gene_ids = arr["gene_ids"].astype(str).tolist()
        if gene_ids != self.gene_ids:
            raise ValueError(f"Gene order mismatch for {sample}")
        barcodes = arr["barcodes"].astype(str).tolist()
        spot_csr = csr.T.tocsr()
        barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
        cached = (spot_csr, barcode_to_idx)
        self._cache[sample] = cached
        return cached

    def get_row(self, sample: str, barcode: str) -> np.ndarray:
        spot_csr, barcode_to_idx = self._load(sample)
        return spot_csr.getrow(barcode_to_idx[barcode]).toarray().ravel().astype(np.float32, copy=False)

    def get_weighted(self, refs: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.gene_ids), dtype=np.float32)
        for w, row in zip(weights.astype(np.float32, copy=False), refs.itertuples(index=False)):
            out += w * self.get_row(str(row.sample), str(row.barcode))
        return out


class BleepSpotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        expression_store: ExpressionStore,
        *,
        patch_size: int,
        augment: bool,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.expression_store = expression_store
        self.patch_size = patch_size
        self.augment = augment
        self._image_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_image(self, path: str) -> np.ndarray:
        cached = self._image_cache.get(path)
        if cached is not None:
            return cached
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._image_cache[path] = image
        return image

    def _crop(self, image: np.ndarray, center_row: int, center_col: int) -> np.ndarray:
        half = self.patch_size // 2
        padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode="constant", constant_values=255)
        row = center_row + half
        col = center_col + half
        crop = padded[row - half : row + half, col - half : col + half]
        if crop.shape[0] != self.patch_size or crop.shape[1] != self.patch_size:
            crop = cv2.resize(crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        return crop

    def _transform(self, crop: np.ndarray) -> torch.Tensor:
        if self.augment:
            if random.random() > 0.5:
                crop = np.ascontiguousarray(crop[:, ::-1])
            if random.random() > 0.5:
                crop = np.ascontiguousarray(crop[::-1])
            k = random.randint(0, 3)
            if k:
                crop = np.ascontiguousarray(np.rot90(crop, k))
        tensor = torch.tensor(crop.copy().tolist(), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.manifest.iloc[idx]
        image = self._load_image(str(row.image_path))
        crop = self._crop(image, int(row.pixel_row), int(row.pixel_col))
        expr = self.expression_store.get_row(str(row["sample"]), str(row.barcode))
        return {
            "image": self._transform(crop),
            "reduced_expression": torch.tensor(expr.tolist(), dtype=torch.float32),
            "row_index": idx,
            "sample": str(row["sample"]),
            "barcode": str(row.barcode),
        }


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([x["image"] for x in batch]),  # type: ignore[list-item]
        "reduced_expression": torch.stack([x["reduced_expression"] for x in batch]),  # type: ignore[list-item]
        "row_index": np.asarray([x["row_index"] for x in batch], dtype=np.int64),
        "sample": [x["sample"] for x in batch],
        "barcode": [x["barcode"] for x in batch],
    }


def build_model(n_genes: int, pretrained: bool, device: torch.device) -> torch.nn.Module:
    BLEEP_CFG.spot_embedding = n_genes
    BLEEP_CFG.pretrained = pretrained
    try:
        return CLIPModel(spot_embedding=n_genes).to(device)
    except Exception:
        if not pretrained:
            raise
        print("[warn] Could not initialize pretrained BLEEP image encoder; retrying with pretrained=False", flush=True)
        BLEEP_CFG.pretrained = False
        return CLIPModel(spot_embedding=n_genes).to(device)


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key.removeprefix("module."), value) for key, value in state_dict.items())


@torch.no_grad()
def validation_loss(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    loss_sum = 0.0
    for batch in val_loader:
        payload = {
            "image": batch["image"].to(device, non_blocking=True),
            "reduced_expression": batch["reduced_expression"].to(device, non_blocking=True),
        }
        loss = model(payload)
        n = int(payload["image"].shape[0])
        loss_sum += float(loss.detach().cpu()) * n
        total += n
    return loss_sum / max(total, 1)


def train(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Path:
    out_dir = EXP_ROOT / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        for batch in train_loader:
            payload = {
                "image": batch["image"].to(device, non_blocking=True),
                "reduced_expression": batch["reduced_expression"].to(device, non_blocking=True),
            }
            loss = model(payload)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = int(payload["image"].shape[0])
            loss_sum += float(loss.detach().cpu()) * n
            total += n
        avg = loss_sum / max(total, 1)
        val_avg = validation_loss(model, val_loader, device)
        print(f"epoch={epoch} train_loss={avg:.6f} val_loss={val_avg:.6f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch, "train_loss": avg, "val_loss": val_avg}, out_dir / "last.pt")
        if val_avg < best_loss:
            best_loss = val_avg
            torch.save({"model": model.state_dict(), "epoch": epoch, "train_loss": avg, "val_loss": val_avg}, best_path)
    return best_path


@torch.no_grad()
def embed(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, pd.DataFrame]:
    model.eval()
    image_embeddings = []
    spot_embeddings = []
    rows: list[dict[str, str]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        expr = batch["reduced_expression"].to(device, non_blocking=True)
        img = model.image_projection(model.image_encoder(images))
        spot = model.spot_projection(expr)
        image_embeddings.append(img.detach().cpu())
        spot_embeddings.append(spot.detach().cpu())
        rows.extend({"sample": s, "barcode": b} for s, b in zip(batch["sample"], batch["barcode"]))
    return (
        torch.cat(image_embeddings).numpy(),
        torch.cat(spot_embeddings).numpy(),
        pd.DataFrame(rows),
    )


def impute_and_score(
    query_image_embeddings: np.ndarray,
    key_spot_embeddings: np.ndarray,
    key_refs: pd.DataFrame,
    query_refs: pd.DataFrame,
    expression_store: ExpressionStore,
    gene_ids: list[str],
    *,
    top_k: int,
    chunk_size: int,
) -> None:
    save_dir = EXP_ROOT / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    key = F.normalize(torch.from_numpy(key_spot_embeddings).float(), dim=1)
    overall = RunningStats()
    sample_stats = {s: RunningStats() for s in sorted(query_refs["sample"].unique())}
    gene_stats = GeneVectorStats(len(gene_ids))

    pred_memmap = np.lib.format.open_memmap(
        save_dir / "predictions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(query_image_embeddings.shape[0], len(gene_ids)),
    )
    true_memmap = np.lib.format.open_memmap(
        save_dir / "targets.npy",
        mode="w+",
        dtype=np.float32,
        shape=(query_image_embeddings.shape[0], len(gene_ids)),
    )

    for start in range(0, query_image_embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, query_image_embeddings.shape[0])
        query = F.normalize(torch.from_numpy(query_image_embeddings[start:end]).float(), dim=1)
        sim = query @ key.T
        k = min(top_k, sim.shape[1])
        values, indices = torch.topk(sim, k=k, dim=1)
        weights = torch.softmax(values, dim=1).numpy()
        indices_np = indices.numpy()

        pred_block = np.zeros((end - start, len(gene_ids)), dtype=np.float32)
        true_block = np.zeros_like(pred_block)
        for i in range(end - start):
            pred_block[i] = expression_store.get_weighted(key_refs.iloc[indices_np[i]], weights[i])
            q = query_refs.iloc[start + i]
            true_block[i] = expression_store.get_row(str(q["sample"]), str(q.barcode))

        pred_memmap[start:end] = pred_block
        true_memmap[start:end] = true_block
        overall.update(pred_block, true_block)
        gene_stats.update(pred_block, true_block)
        for sample in sample_stats:
            mask = (query_refs.iloc[start:end]["sample"].to_numpy() == sample)
            if np.any(mask):
                sample_stats[sample].update(pred_block[mask], true_block[mask])
        print(f"scored {end}/{query_image_embeddings.shape[0]} test spots", flush=True)

    pred_memmap.flush()
    true_memmap.flush()
    gene_mse = gene_stats.mse()
    gene_corr = gene_stats.corr()
    finite_gene_corr = gene_corr[np.isfinite(gene_corr)]
    (save_dir / "test_overall.json").write_text(
        json.dumps(
            {
                "dataset": "hippocampus_fullgenes",
                "model": "BLEEP",
                "prediction": f"weighted top-{top_k} train-spot expression retrieval",
                "mse": overall.mse(),
                "corr": overall.corr(),
                "gene_corr_mean": float(np.mean(finite_gene_corr)) if finite_gene_corr.size else float("nan"),
                "gene_corr_median": float(np.median(finite_gene_corr)) if finite_gene_corr.size else float("nan"),
            },
            indent=2,
        )
    )
    pd.DataFrame(
        [{"sample": sample, "mse": stat.mse(), "corr": stat.corr()} for sample, stat in sample_stats.items()]
    ).to_csv(save_dir / "test_sample_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"gene_id": gid, "mse": float(m), "corr": float(c)}
            for gid, m, c in zip(gene_ids, gene_mse.tolist(), gene_corr.tolist())
        ]
    ).to_csv(save_dir / "test_gene_metrics.tsv", sep="\t", index=False)
    query_refs.to_csv(save_dir / "prediction_spots.tsv", sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patch-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--impute-chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    manifest = pd.read_csv(DATA_ROOT / "manifest.tsv", sep="\t")
    gene_ids = [line.strip() for line in (DATA_ROOT / "gene_ids.txt").read_text().splitlines() if line.strip()]
    train_manifest = manifest[manifest["split"] == "train"].copy()
    val_manifest = manifest[manifest["split"] == "val"].copy()
    test_manifest = manifest[manifest["split"] == "test"].copy()
    expression_store = ExpressionStore(gene_ids)

    train_ds = BleepSpotDataset(train_manifest, expression_store, patch_size=args.patch_size, augment=True)
    val_ds = BleepSpotDataset(val_manifest, expression_store, patch_size=args.patch_size, augment=False)
    train_eval_ds = BleepSpotDataset(train_manifest, expression_store, patch_size=args.patch_size, augment=False)
    test_ds = BleepSpotDataset(test_manifest, expression_store, patch_size=args.patch_size, augment=False)
    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate,
    }
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(gene_ids), args.pretrained, device)
    best_path = train(args, model, train_loader, val_loader, device)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(normalize_state_dict(ckpt["model"]))

    test_img, _test_spot, test_refs = embed(model, test_loader, device)
    _train_img, train_spot, train_refs = embed(model, train_eval_loader, device)
    impute_and_score(
        test_img,
        train_spot,
        train_refs,
        test_refs,
        expression_store,
        gene_ids,
        top_k=args.top_k,
        chunk_size=args.impute_chunk_size,
    )


if __name__ == "__main__":
    main()
