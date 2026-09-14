#!/usr/bin/env python3
from __future__ import annotations
from workflow_paths import DATA_ROOT

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = DATA_ROOT
EXP_ROOT = ROOT / "experiments/benchmark_stnet_hippocampus_fullgenes"
SAVE_DIR = EXP_ROOT / "results"
OUT_NPZ = EXP_ROOT / "output/stnet_fullgenes_test.npz"


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum_pred = 0.0
        self.sum_true = 0.0
        self.sum_pred2 = 0.0
        self.sum_true2 = 0.0
        self.sum_pred_true = 0.0
        self.sum_sq_error = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
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


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(EXP_ROOT / "meta/patient_mapping.tsv", sep="\t")
    patient_to_sample = dict(zip(mapping["patient"], mapping["sample"]))

    data = np.load(OUT_NPZ, allow_pickle=True)
    pred = data["predictions"].astype(np.float32)
    true = data["counts"].astype(np.float32)
    patients = data["patient"].astype(str)
    gene_ids = [str(x) for x in data["ensg_names"]]

    overall = RunningStats()
    overall.update(pred.ravel(), true.ravel())

    sample_rows = []
    for patient in sorted(set(patients)):
        sample = patient_to_sample[patient]
        mask = patients == patient
        rs = RunningStats()
        rs.update(pred[mask].ravel(), true[mask].ravel())
        sample_rows.append({"sample": sample, "mse": rs.mse(), "corr": rs.corr()})

    gene_stats = GeneVectorStats(len(gene_ids))
    gene_stats.update(pred, true)
    gene_mse = gene_stats.mse()
    gene_corr = gene_stats.corr()
    gene_rows = [
        {"gene_id": gid, "mse": float(m), "corr": float(c)}
        for gid, m, c in zip(gene_ids, gene_mse.tolist(), gene_corr.tolist())
    ]
    finite_gene_corr = gene_corr[np.isfinite(gene_corr)]

    (SAVE_DIR / "test_overall.json").write_text(
        json.dumps(
            {
                "dataset": "hippocampus_fullgenes",
                "mse": overall.mse(),
                "corr": overall.corr(),
                "gene_corr_mean": float(np.mean(finite_gene_corr)) if finite_gene_corr.size else float("nan"),
                "gene_corr_median": float(np.median(finite_gene_corr)) if finite_gene_corr.size else float("nan"),
            },
            indent=2,
        )
    )
    pd.DataFrame(sample_rows).to_csv(SAVE_DIR / "test_sample_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(gene_rows).to_csv(SAVE_DIR / "test_gene_metrics.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
