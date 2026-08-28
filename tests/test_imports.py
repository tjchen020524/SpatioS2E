def test_public_imports():
    import spatios2e
    from spatios2e.data import SpatioS2EDataset, collate_graphs
    from spatios2e.evaluation import component_metrics
    from spatios2e.models import FactorizedDotProductDecoder, GatedScGPTGeneContextPriorModel
    from spatios2e.models.factory import build_model
    from spatios2e.preprocessing.extract_scgpt_gene_tokens import layer_normalized_gene_tokens
    from spatios2e.training import fit_heldout_decoder

    assert spatios2e.__version__
    assert SpatioS2EDataset is not None
    assert collate_graphs is not None
    assert GatedScGPTGeneContextPriorModel is not None
    assert FactorizedDotProductDecoder is not None
    assert component_metrics is not None
    assert build_model is not None
    assert fit_heldout_decoder is not None
    assert layer_normalized_gene_tokens is not None
