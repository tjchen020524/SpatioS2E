from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy import sparse

from spatios2e.utils.residual_scale import compute_training_tissue_mean


def _write_expression(path: Path, values: np.ndarray, gene_ids: list[str]) -> None:
    matrix = sparse.csr_matrix(values.T)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape),
        gene_ids=np.asarray(gene_ids),
        barcodes=np.asarray([f"spot_{index}" for index in range(values.shape[0])]),
    )


def test_training_tissue_mean_is_spot_weighted_and_training_only():
    genes = ["g1", "g2"]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_expression(root / "a/train.npz", np.asarray([[1.0, 2.0], [3.0, 4.0]]), genes)
        _write_expression(root / "b/train.npz", np.asarray([[8.0, 10.0]]), genes)
        config = {
            "paths": {"expression_root": str(root)},
            "split": {"train": ["a", "b"], "val": ["unused"], "test": ["unused"]},
        }

        mean, n_spots = compute_training_tissue_mean(config, genes)

    assert n_spots == 3
    assert np.allclose(mean, [4.0, 16.0 / 3.0])
