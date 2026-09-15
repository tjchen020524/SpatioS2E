#!/usr/bin/env python3
from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torchvision


PROJECT_ROOT = DATA_ROOT
STNET_ROOT = Path(os.environ.get("SPATIOS2E_STNET_ROOT", PROJECT_ROOT / "third_party/ST-Net")).resolve()
EXP_ROOT = PROJECT_ROOT / "experiments/benchmark_stnet_hippocampus_fullgenes"

if str(STNET_ROOT) not in sys.path:
    sys.path.append(str(STNET_ROOT))

import stnet  # type: ignore


class SafeToTensor:
    def __call__(self, pic):
        arr = np.asarray(pic, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        tensor = torch.tensor(arr.tolist(), dtype=torch.float32).permute(2, 0, 1)
        return tensor / 255.0


def configure_stnet_roots() -> tuple[Path, Path]:
    raw_root = EXP_ROOT / "raw"
    processed_root = EXP_ROOT / "processed"
    stnet.config.SPATIAL_RAW_ROOT = str(raw_root)
    stnet.config.SPATIAL_PROCESSED_ROOT = str(processed_root)
    return raw_root, processed_root


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor], *, want_module_prefix: bool) -> OrderedDict[str, torch.Tensor]:
    has_module_prefix = all(key.startswith("module.") for key in state_dict.keys())
    if want_module_prefix == has_module_prefix:
        return OrderedDict(state_dict)
    if want_module_prefix:
        return OrderedDict((f"module.{key}", value) for key, value in state_dict.items())
    return OrderedDict((key.removeprefix("module."), value) for key, value in state_dict.items())


def load_checkpoint_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    want_module_prefix = any(key.startswith("module.") for key in model.state_dict().keys())
    normalized_state = _normalize_state_dict_keys(state_dict, want_module_prefix=want_module_prefix)
    model.load_state_dict(normalized_state)


def load_meta() -> dict:
    return json.loads((EXP_ROOT / "meta/split.json").read_text())


def build_run_args(args: argparse.Namespace, train_patients: list[str], test_patients: list[str], checkpoint_root: Path, pred_root: Path, logfile: Path) -> list[str]:
    out = [
        "--seed",
        str(args.seed),
        "run_spatial",
        "--gene",
        "--logfile",
        str(logfile),
        "--epochs",
        str(args.epochs),
        "--checkpoint",
        str(checkpoint_root),
        "--checkpoint_every",
        "1",
        "--keep_checkpoints",
        *[str(epoch) for epoch in range(1, args.epochs + 1)],
        "--save_pred_every",
        "1",
        "--pred_root",
        str(pred_root),
        "--trainpatients",
        *train_patients,
        "--testpatients",
        *test_patients,
        "--window",
        str(args.window),
        "--model",
        args.model,
        "--pretrained",
        "--average",
        "--batch",
        str(args.batch),
        "--test-batch",
        str(args.test_batch),
        "--workers",
        str(args.workers),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--gene_filter",
        "none",
        "--gene_transform",
        "none",
    ]
    out.append("--cpu" if args.cpu else "--gpu")
    return out


def select_best_epoch(pred_root: Path, epochs: int) -> tuple[int, float]:
    best_epoch = 1
    best_loss = math.inf
    for epoch in range(1, epochs + 1):
        pred_path = Path(f"{pred_root}{epoch}.npz")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing validation prediction file: {pred_path}")
        data = np.load(pred_path, allow_pickle=True)
        pred = data["predictions"].astype(np.float64)
        true = data["counts"].astype(np.float64)
        loss = float(np.square(pred - true).sum())
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
    return best_epoch, best_loss


def estimate_image_stats(train_loader: torch.utils.data.DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    n_samples = 10
    mean = 0.0
    std = 0.0
    n = 0
    for i, (x, *_rest) in enumerate(train_loader):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        mean += torch.sum(x, dim=1)
        std += torch.sum(x ** 2, dim=1)
        if i > n_samples:
            break
    mean /= n
    std = torch.sqrt(std / n - mean ** 2)
    return mean, std


def export_test_predictions(args: argparse.Namespace, best_epoch: int) -> Path:
    raw_root, processed_root = configure_stnet_roots()
    meta = load_meta()
    train_patients = meta["train_patients"]
    test_patients = meta["test_patients"]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    safe_to_tensor = SafeToTensor()
    loader_kwargs = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 1

    train_dataset = stnet.datasets.Spatial(
        train_patients,
        window=args.window,
        root=str(processed_root),
        gene_filter="none",
        downsample=1,
        norm=None,
        gene_transform="none",
        transform=safe_to_tensor,
        feature=False,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, **loader_kwargs)

    mean, std = estimate_image_stats(train_loader)
    transform = torchvision.transforms.Compose(
        [
            stnet.transforms.EightSymmetry(),
            torchvision.transforms.Lambda(
                lambda symmetries: torch.stack(
                    [torchvision.transforms.Normalize(mean=mean, std=std)(safe_to_tensor(s)) for s in symmetries]
                )
            ),
        ]
    )

    test_dataset = stnet.datasets.Spatial(
        test_patients,
        transform,
        window=args.window,
        root=str(processed_root),
        gene_filter="none",
        downsample=1,
        norm=None,
        gene_transform="none",
        feature=False,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, **loader_kwargs)

    outputs = train_dataset[0][2].shape[0]
    model = torchvision.models.__dict__[args.model](pretrained=False)
    stnet.utils.nn.set_out_features(model, outputs)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    load_checkpoint_state(model, EXP_ROOT / f"output/val_checkpoints/epoch_{best_epoch}.pt", device)
    model.eval()

    predictions = []
    counts = []
    tumor = []
    coord = []
    patient = []
    section = []
    pixel = []

    with torch.no_grad():
        for x, y, gene, c, ind, pat, s, pix, f in test_loader:
            counts.append(gene.detach().cpu().numpy())
            tumor.append(y.detach().cpu().numpy())
            coord.append(c.detach().cpu().numpy())
            patient += list(pat)
            section += list(s)
            pixel.append(pix.detach().cpu().numpy())

            x = x.to(device)
            batch, n_sym, ch, h, w = x.shape
            x = x.view(-1, ch, h, w)
            pred = model(x)
            pred = pred.view(batch, n_sym, -1).mean(1)
            predictions.append(pred.detach().cpu().numpy())

    out_path = EXP_ROOT / "output/stnet_fullgenes_test.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": np.asarray("gene"),
        "tumor": np.asarray(np.concatenate(tumor), dtype=np.int64),
        "counts": np.asarray(np.concatenate(counts), dtype=np.float32),
        "predictions": np.asarray(np.concatenate(predictions), dtype=np.float32),
        "coord": np.asarray(np.concatenate(coord), dtype=np.int64),
        "patient": np.asarray(patient, dtype=str),
        "section": np.asarray(section, dtype=str),
        "pixel": np.asarray(np.concatenate(pixel), dtype=np.int64),
        "ensg_names": np.asarray(test_dataset.ensg_names, dtype=str),
        "gene_names": np.asarray(test_dataset.gene_names, dtype=str),
        "best_epoch": np.asarray([best_epoch], dtype=np.int32),
    }
    np.savez(str(out_path), **payload)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--select-epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--test-batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--window", type=int, default=224)
    ap.add_argument("--model", type=str, default="densenet121")
    ap.add_argument("--lr", type=float, default=1.0e-6)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    configure_stnet_roots()

    meta = load_meta()
    train_patients = meta["train_patients"]
    val_patients = meta["val_patients"]

    output_dir = EXP_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_root = output_dir / "val_epoch_"
    checkpoint_root = output_dir / "val_checkpoints/epoch_"
    logfile = output_dir / "stnet_train_val.log"

    if not args.skip_train:
        stnet.main(build_run_args(args, train_patients, val_patients, checkpoint_root, pred_root, logfile))

    select_epochs = args.select_epochs if args.select_epochs is not None else args.epochs
    best_epoch, best_loss = select_best_epoch(pred_root, select_epochs)

    selection = {"best_epoch": best_epoch, "best_val_sse": best_loss}
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    export_test_predictions(args, best_epoch)


if __name__ == "__main__":
    main()
