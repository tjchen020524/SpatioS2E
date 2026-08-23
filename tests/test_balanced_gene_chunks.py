import torch

from spatios2e.training.train import take_balanced_chunk


def test_balanced_chunks_cover_all_genes_before_reuse():
    order = torch.arange(10)
    cursor = 0
    chunks = []
    for _ in range(2):
        chunk, order, cursor = take_balanced_chunk(order, cursor, width=5, n_genes=10)
        chunks.append(chunk)

    assert torch.equal(torch.cat(chunks), torch.arange(10))
    assert cursor == 0


def test_balanced_chunk_wraps_to_requested_width():
    torch.manual_seed(3)
    order = torch.arange(5)
    chunk, next_order, cursor = take_balanced_chunk(order, cursor=3, width=4, n_genes=5)

    assert torch.equal(chunk[:2], torch.tensor([3, 4]))
    assert chunk.numel() == 4
    assert next_order.numel() == 5
    assert cursor == 2
