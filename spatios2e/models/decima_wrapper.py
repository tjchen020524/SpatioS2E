#!/usr/bin/env python3
"""
Decima sequence wrapper for SpatioS2E.

Loads DecimaModel, freezes embedding (Conv+Transformer), extracts gene embeddings
from 524,288bp sequences stored in an HDF5 file, and applies a lightweight
pseudobulk head.

Expected HDF5 layout (as used in decima.data.read_hdf5):
  - datasets: 'genes' (gene ids), 'sequences' (int indices), 'masks'
  - sequences are one-hot indices; masks binary
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Sequence, Tuple

import importlib
import sys
import types


def _patch_huggingface_hub_version_check() -> None:
    """Keep transformers importable when an over-new hub package is installed."""
    try:
        import importlib.metadata as metadata
    except Exception:
        return
    current = metadata.version
    if getattr(current, "_spatios2e_hf_hub_patch", False):
        return

    def _version(name: str) -> str:
        if name in {"huggingface-hub", "huggingface_hub"}:
            try:
                found = current(name)
            except Exception:
                raise
            major = found.split(".", 1)[0]
            if major.isdigit() and int(major) >= 1:
                return "0.34.0"
            return found
        return current(name)

    _version._spatios2e_hf_hub_patch = True  # type: ignore[attr-defined]
    metadata.version = _version  # type: ignore[assignment]


def _bootstrap_package(pkg_name: str, pkg_dir: Path) -> None:
    if pkg_name in sys.modules:
        return
    module = types.ModuleType(pkg_name)
    module.__path__ = [str(pkg_dir)]
    sys.modules[pkg_name] = module


def _find_package_dir(pkg_name: str) -> Path | None:
    parts = pkg_name.split("/")
    for base in sys.path:
        candidate = Path(base).joinpath(*parts)
        if candidate.exists():
            return candidate
    return None


def _patch_torch_pytree() -> None:
    try:
        import torch.utils._pytree as _pytree  # type: ignore
    except Exception:
        return
    if hasattr(_pytree, "_register_pytree_node"):
        def _safe_register_pytree_node(*args, **kwargs):  # type: ignore[override]
            # Drop newer kwargs unsupported by older torch
            kwargs.pop("serialized_type_name", None)
            kwargs.pop("namespace", None)
            kwargs.pop("flatten_with_keys_fn", None)
            try:
                return _pytree._register_pytree_node(*args, **kwargs)  # type: ignore[attr-defined]
            except TypeError:
                return _pytree._register_pytree_node(*args[:3])  # type: ignore[attr-defined]
        _pytree.register_pytree_node = _safe_register_pytree_node  # type: ignore[attr-defined]


def _load_decima_model() -> type:
    _patch_huggingface_hub_version_check()
    _patch_torch_pytree()
    try:
        from decima.model.decima_model import DecimaModel  # type: ignore
        return DecimaModel
    except Exception as exc:
        _patch_torch_pytree()
        pkg_dir = _find_package_dir("decima")
        if pkg_dir is None:
            raise ImportError(
                "Decima is required to instantiate DecimaSequenceWrapper. "
                "Install Decima or add its source directory to PYTHONPATH before training."
            ) from exc
        _bootstrap_package("decima", pkg_dir)
        _bootstrap_package("decima.model", pkg_dir / "model")
        module = importlib.import_module("decima.model.decima_model")
        return module.DecimaModel


def _indices_to_one_hot(indices: np.ndarray) -> np.ndarray:
    indices = indices.astype(np.int64, copy=False)
    length = indices.shape[0]
    out = np.zeros((4, length), dtype=np.float32)
    valid = (indices >= 0) & (indices < 4)
    out[indices[valid], np.nonzero(valid)[0]] = 1.0
    return out


def _extract_center(seq: np.ndarray, seq_len: int) -> np.ndarray:
    if seq.shape[1] <= seq_len:
        return seq
    start = (seq.shape[1] - seq_len) // 2
    return seq[:, start : start + seq_len]


EPS = 1e-8


class GeneSequenceLoader:
    """Load gene sequences either from H5 (genes/sequences/masks) or per-gene NPZ."""

    def __init__(self, h5_path: Path | None = None, npz_dir: Path | None = None, seq_len: int = 524288):
        self.h5_path = Path(h5_path) if h5_path else None
        self.npz_dir = Path(npz_dir) if npz_dir else None
        self.seq_len = seq_len
        self.gene_to_idx: Dict[str, int] = {}
        if self.h5_path and self.h5_path.exists():
            with h5py.File(self.h5_path, "r") as f:
                genes = np.array(f["genes"]).astype(str)
            self.gene_to_idx = {g[0] if isinstance(g, (np.ndarray, list)) else g: i for i, g in enumerate(genes)}

    def load(self, gene_id: str) -> torch.Tensor:
        if self.npz_dir:
            npz_path = self.npz_dir / f"{gene_id}.npz"
            if not npz_path.exists():
                raise KeyError(f"Gene {gene_id} npz not found at {npz_path}")
            data = np.load(npz_path)
            if not data.files:
                raise ValueError(f"{npz_path} is empty; expected array with shape (5, L)")
            # Prefer keys named like "input" or the first array with shape (5, L)
            candidate_keys = ["input"] + list(data.files)
            arr = None
            chosen = None
            for key in candidate_keys:
                if key not in data:
                    continue
                a = data[key]
                if a.ndim >= 2 and a.shape[0] == 5:
                    arr = a
                    chosen = key
                    break
            if arr is None:
                raise ValueError(
                    f"{npz_path} has keys {data.files} but none with shape (5, L); "
                    "ensure file stores a (5,524288) array (one-hot+mask)."
                )
            return torch.tensor(arr, dtype=torch.float32)
        if self.h5_path and self.gene_to_idx:
            if gene_id not in self.gene_to_idx:
                raise KeyError(f"Gene {gene_id} not found in {self.h5_path}")
            idx = self.gene_to_idx[gene_id]
            with h5py.File(self.h5_path, "r") as f:
                seq = np.array(f["sequences"][idx])
                mask = np.array(f["masks"][idx])
            seq = _indices_to_one_hot(seq)  # 4, L
            seq = np.concatenate([seq, mask[None, :]], axis=0)  # 5, L
            seq = _extract_center(seq, seq_len=self.seq_len)
            return torch.tensor(seq, dtype=torch.float32)
        raise ValueError("No valid sequence source provided (h5_path or npz_dir).")


class DecimaSequenceWrapper(nn.Module):
    """Encapsulates Decima embedding + trainable pseudobulk head."""

    def __init__(
        self,
        ckpt_path: Path,
        h5_path: Path | None = None,
        npz_dir: Path | None = None,
        freeze_backbone: bool = True,
        freeze_head: bool = False,
    ):
        super().__init__()
        self.seq_loader = GeneSequenceLoader(h5_path=h5_path, npz_dir=npz_dir)
        # Peek ckpt to infer n_tasks (head channels) so load_state_dict matches.
        ckpt_state = self._load_raw_state(ckpt_path)
        n_tasks = 1
        if "head.channel_transform.conv.layer.weight" in ckpt_state:
            n_tasks = ckpt_state["head.channel_transform.conv.layer.weight"].shape[0]
        # Instantiate DecimaModel lazily so importing spatios2e does not require
        # a local Decima checkout until a sequence-conditioned model is created.
        decima_model_cls = _load_decima_model()
        self.decima = decima_model_cls(n_tasks=n_tasks, init_borzoi=False)
        self._load_checkpoint(ckpt_state)
        if freeze_backbone:
            for p in self.decima.embedding.parameters():
                p.requires_grad = False
        # Infer embedding dim from ConvHead input (1920 in Decima)
        self.embed_dim = 1920
        self.pseudobulk_head = nn.Linear(self.embed_dim, 1)
        if freeze_head:
            for p in self.pseudobulk_head.parameters():
                p.requires_grad = False
        self.cache: Dict[str, torch.Tensor] = {}

    def _load_raw_state(self, ckpt_path: Path) -> Dict[str, torch.Tensor]:
        # ckpts saved with pickled objects; allow full load (trusted source).
        try:
            import numpy as _np  # local import to avoid unused warning

            torch.serialization.add_safe_globals([_np.core.multiarray._reconstruct])  # type: ignore[attr-defined]
        except Exception:
            pass
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
            # strip "model." prefix if present
            state = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in state.items()}
        return state

    def _load_checkpoint(self, state: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.decima.load_state_dict(state, strict=False)
        if missing:
            print(f"[DecimaWrapper] Missing keys: {missing}")
        if unexpected:
            print(f"[DecimaWrapper] Unexpected keys: {unexpected}")

    def encode_genes(self, gene_ids: Sequence[str], device: torch.device) -> torch.Tensor:
        embs = []
        for gid in gene_ids:
            if gid in self.cache:
                embs.append(self.cache[gid].to(device))
                continue
            seq = self.seq_loader.load(gid).unsqueeze(0).to(device)  # (1,5,L)
            with torch.set_grad_enabled(any(p.requires_grad for p in self.decima.embedding.parameters())):
                feat = self.decima.embedding(seq)  # expect (B,C,L)
                if feat.dim() == 3:
                    feat = feat.mean(dim=-1)
            feat = feat.squeeze(0)
            self.cache[gid] = feat.detach().cpu()
            embs.append(feat.to(device))
        return torch.stack(embs, dim=0)

    def forward_pseudobulk(self, e_gene: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.pseudobulk_head(e_gene).squeeze(-1)
        mu_pb = F.softplus(logits) + EPS
        return mu_pb, logits
