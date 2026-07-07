def test_public_imports():
    import spatios2e
    from spatios2e.data import SpatioS2EDataset, collate_graphs
    from spatios2e.models import GatedScGPTGeneContextPriorModel
    from spatios2e.models.factory import build_model

    assert spatios2e.__version__
    assert SpatioS2EDataset is not None
    assert collate_graphs is not None
    assert GatedScGPTGeneContextPriorModel is not None
    assert build_model is not None
