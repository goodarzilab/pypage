"""Tests for SingleCellPAGE."""

import numpy as np
import pandas as pd
import pytest

from pypage import GeneSets
from pypage.sc import SingleCellPAGE


def _make_synthetic_data(n_cells=200, n_genes=500, n_planted=5, n_null=5,
                         pathway_size=30, seed=42):
    """Create synthetic single-cell data with planted and null pathways.

    Two clusters of cells. Planted pathways have member genes differentially
    expressed between clusters. Null pathways have random membership.
    """
    rng = np.random.RandomState(seed)
    gene_names = np.array([f"gene_{i}" for i in range(n_genes)])

    # Two clusters
    half = n_cells // 2
    expression = rng.randn(n_cells, n_genes) * 0.5

    # Planted pathways: genes are differentially expressed between clusters
    pathway_genes_list = []
    pathway_names_list = []
    for p in range(n_planted):
        start = p * pathway_size
        end = start + pathway_size
        # Upregulate in cluster 1, downregulate in cluster 2
        expression[:half, start:end] += 3.0
        expression[half:, start:end] -= 3.0
        for g_idx in range(start, end):
            pathway_genes_list.append(gene_names[g_idx])
            pathway_names_list.append(f"planted_{p}")

    # Null pathways: random genes, no differential expression
    used_genes = set(range(n_planted * pathway_size))
    for p in range(n_null):
        available = [i for i in range(n_genes) if i not in used_genes]
        chosen = rng.choice(available, size=pathway_size, replace=False)
        for g_idx in chosen:
            pathway_genes_list.append(gene_names[g_idx])
            pathway_names_list.append(f"null_{p}")
            used_genes.add(g_idx)

    gs = GeneSets(
        genes=np.array(pathway_genes_list),
        pathways=np.array(pathway_names_list),
    )

    labels = np.array([0] * half + [1] * half)

    return expression, gene_names, gs, labels


@pytest.fixture
def synthetic_data():
    """Fixture for synthetic single-cell data."""
    return _make_synthetic_data()


class TestSingleCellPAGEInit:
    """Test initialization and input validation."""

    def test_init_with_numpy(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs
        )
        assert sc.n_cells == 200
        assert sc.n_pathways == 10
        assert len(sc.shared_genes) > 0

    def test_init_with_anndata(self, synthetic_data):
        anndata = pytest.importorskip("anndata")
        expression, genes, gs, _ = synthetic_data
        adata = anndata.AnnData(
            X=expression,
            var=pd.DataFrame(index=genes),
        )
        sc = SingleCellPAGE(adata=adata, genesets=gs)
        assert sc.n_cells == 200

    def test_init_anndata_with_connectivity(self, synthetic_data):
        anndata = pytest.importorskip("anndata")
        from scipy.sparse import eye
        expression, genes, gs, _ = synthetic_data
        n = expression.shape[0]
        conn = eye(n, format='csr') * 0.5
        adata = anndata.AnnData(
            X=expression,
            var=pd.DataFrame(index=genes),
            obsp={'connectivities': conn},
        )
        sc = SingleCellPAGE(adata=adata, genesets=gs)
        assert sc.connectivity is not None

    def test_init_anndata_with_embeddings(self, synthetic_data):
        anndata = pytest.importorskip("anndata")
        expression, genes, gs, _ = synthetic_data
        embedding = np.random.randn(200, 2)
        adata = anndata.AnnData(
            X=expression,
            var=pd.DataFrame(index=genes),
            obsm={'X_umap': embedding},
        )
        sc = SingleCellPAGE(adata=adata, genesets=gs)
        assert 'X_umap' in sc.embeddings
        assert sc.embeddings['X_umap'].shape == (200, 2)

    def test_init_no_genesets_raises(self, synthetic_data):
        expression, genes, _, _ = synthetic_data
        with pytest.raises(ValueError, match="genesets is required"):
            SingleCellPAGE(expression=expression, genes=genes)

    def test_init_no_input_raises(self, synthetic_data):
        _, _, gs, _ = synthetic_data
        with pytest.raises(ValueError, match="Provide either adata"):
            SingleCellPAGE(genesets=gs)

    def test_init_bad_function(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        with pytest.raises(ValueError, match="function must be"):
            SingleCellPAGE(
                expression=expression, genes=genes, genesets=gs,
                function='bad'
            )

    def test_init_bad_bin_axis(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        with pytest.raises(ValueError, match="bin_axis must be"):
            SingleCellPAGE(
                expression=expression, genes=genes, genesets=gs,
                bin_axis='bad',
            )

    def test_init_bad_permutation_chunk_size(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        with pytest.raises(ValueError, match="permutation_chunk_size must be"):
            SingleCellPAGE(
                expression=expression, genes=genes, genesets=gs,
                permutation_chunk_size=0,
            )

    def test_init_bad_redundancy_ratio(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        with pytest.raises(ValueError, match="redundancy_ratio must be"):
            SingleCellPAGE(
                expression=expression, genes=genes, genesets=gs,
                redundancy_ratio=0,
            )

    def test_init_bad_redundancy_scope(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        with pytest.raises(ValueError, match="redundancy_scope must be"):
            SingleCellPAGE(
                expression=expression, genes=genes, genesets=gs,
                redundancy_scope="bad",
            )

    def test_init_bad_redundancy_fdr(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        with pytest.raises(ValueError, match="redundancy_fdr must be"):
            SingleCellPAGE(
                expression=expression, genes=genes, genesets=gs,
                redundancy_fdr=1.2,
            )

    def test_init_1d_expression_raises(self, synthetic_data):
        _, genes, gs, _ = synthetic_data
        with pytest.raises(ValueError, match="2D"):
            SingleCellPAGE(
                expression=np.ones(500), genes=genes, genesets=gs
            )

    def test_repr(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        sc = SingleCellPAGE(expression=expression, genes=genes, genesets=gs)
        r = repr(sc)
        assert "SingleCellPAGE" in r
        assert "n_cells=200" in r
        assert "bin_axis='cell'" in r


class TestSingleCellPAGERun:
    """Test the full run() pipeline."""

    @pytest.mark.slow
    def test_run_planted_vs_null(self, synthetic_data):
        """Planted pathways should have higher consistency than null ones."""
        expression, genes, gs, _ = synthetic_data
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=5, function='mi',
        )
        results = sc.run(n_permutations=200)

        assert isinstance(results, pd.DataFrame)
        assert 'pathway' in results.columns
        assert 'consistency' in results.columns
        assert 'p-value' in results.columns
        assert 'FDR' in results.columns

        # Planted pathways should have higher consistency
        planted = results[results['pathway'].str.startswith('planted')]
        null = results[results['pathway'].str.startswith('null')]
        assert planted['consistency'].mean() > null['consistency'].mean()

    @pytest.mark.slow
    def test_run_cmi(self, synthetic_data):
        """CMI mode should also identify planted pathways."""
        expression, genes, gs, _ = synthetic_data
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=5, function='cmi',
        )
        results = sc.run(n_permutations=100)

        planted = results[results['pathway'].str.startswith('planted')]
        null = results[results['pathway'].str.startswith('null')]
        assert planted['consistency'].mean() > null['consistency'].mean()

    def test_run_returns_correct_shape(self):
        """Results should have one row per pathway."""
        expression, genes, gs, _ = _make_synthetic_data(
            n_cells=50, n_genes=100, n_planted=2, n_null=2, pathway_size=10
        )
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi',
        )
        results = sc.run(n_permutations=10)
        assert len(results) == sc.n_pathways

    def test_scores_shape(self):
        """scores attribute should be (n_cells, n_pathways)."""
        expression, genes, gs, _ = _make_synthetic_data(
            n_cells=50, n_genes=100, n_planted=2, n_null=2, pathway_size=10
        )
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi',
        )
        sc.run(n_permutations=10)
        assert sc.scores.shape == (50, sc.n_pathways)

    def test_bin_axis_changes_discretization(self):
        """cell and gene binning should produce different discretizations."""
        expression, genes, gs, _ = _make_synthetic_data(
            n_cells=50, n_genes=100, n_planted=2, n_null=2, pathway_size=10
        )
        sc_cell = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi', bin_axis='cell',
        )
        sc_gene = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi', bin_axis='gene',
        )
        expr_subset = expression[:, sc_cell._expr_idxs]

        np.random.seed(7)
        bins_cell = sc_cell._discretize_expression(expr_subset)
        np.random.seed(7)
        bins_gene = sc_gene._discretize_expression(expr_subset)

        assert bins_cell.shape == bins_gene.shape
        assert not np.array_equal(bins_cell, bins_gene)

    def test_numpy_anndata_consistency(self):
        """AnnData and numpy inputs should produce identical results."""
        anndata = pytest.importorskip("anndata")
        expression, genes, gs, _ = _make_synthetic_data(
            n_cells=40, n_genes=80, n_planted=2, n_null=1, pathway_size=8,
            seed=123,
        )

        # Use fixed seed for reproducibility
        np.random.seed(99)
        sc_np = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi',
        )
        res_np = sc_np.run(n_permutations=10)

        adata = anndata.AnnData(
            X=expression, var=pd.DataFrame(index=genes)
        )
        np.random.seed(99)
        sc_ad = SingleCellPAGE(adata=adata, genesets=gs, n_bins=3, function='mi')
        res_ad = sc_ad.run(n_permutations=10)

        # Consistency values should match
        np.testing.assert_allclose(
            sc_np.consistency, sc_ad.consistency, rtol=1e-10
        )

    def test_permutation_chunking_matches_unchunked(self):
        """Chunked permutation testing should match unchunked with fixed seed."""
        expression, genes, gs, _ = _make_synthetic_data(
            n_cells=50, n_genes=100, n_planted=2, n_null=2, pathway_size=10,
            seed=321,
        )
        np.random.seed(13)
        sc_full = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi', permutation_chunk_size=100,
        )
        res_full = sc_full.run(n_permutations=20)

        np.random.seed(13)
        sc_chunked = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi', permutation_chunk_size=5,
        )
        res_chunked = sc_chunked.run(n_permutations=20)

        pd.testing.assert_frame_equal(
            res_full.sort_values('pathway').reset_index(drop=True),
            res_chunked.sort_values('pathway').reset_index(drop=True),
        )

    def test_redundancy_log_schema_without_run(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        sc = SingleCellPAGE(expression=expression, genes=genes, genesets=gs)
        log = sc.get_redundancy_log()
        assert list(log.columns) == ['rejected_pathway', 'killed_by', 'min_ratio']
        assert len(log) == 0

    def test_redundancy_filtering_kills_duplicate_pathway(self):
        rng = np.random.RandomState(0)
        n_cells = 60
        n_genes = 40
        genes = np.array([f"g{i}" for i in range(n_genes)])
        expression = rng.normal(size=(n_cells, n_genes))
        expression[: n_cells // 2, :10] += 2.0
        expression[n_cells // 2:, :10] -= 2.0

        gs_genes = []
        gs_pathways = []
        for g in genes[:10]:
            gs_genes.extend([g, g])
            gs_pathways.extend(["dup_a", "dup_b"])
        for g in genes[10:20]:
            gs_genes.append(g)
            gs_pathways.append("other")

        gs = GeneSets(np.array(gs_genes), np.array(gs_pathways))
        sc = SingleCellPAGE(
            expression=expression,
            genes=genes,
            genesets=gs,
            function='mi',
            n_bins=4,
            filter_redundant=True,
            redundancy_scope='all',
            redundancy_ratio=5.0,
        )
        np.random.seed(7)
        results = sc.run(n_permutations=20)
        killed = sc.get_redundancy_log()

        assert len(killed) >= 1
        killed_set = set(killed['rejected_pathway'].tolist())
        assert ('dup_a' in killed_set) or ('dup_b' in killed_set)
        assert len(results) < sc.n_pathways
        assert hasattr(sc, 'full_results')
        assert 'redundant' in sc.full_results.columns


class TestRunNeighborhoods:
    """Test the run_neighborhoods() alternative mode."""

    def test_run_neighborhoods_with_labels(self, synthetic_data):
        expression, genes, gs, labels = synthetic_data
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=5, function='mi',
        )
        summary, group_results = sc.run_neighborhoods(labels=labels)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2  # two clusters
        assert 'group' in summary.columns
        assert 'n_cells' in summary.columns
        assert isinstance(group_results, dict)
        assert len(group_results) == 2

    def test_run_neighborhoods_with_npools(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=5, function='mi',
        )
        summary, group_results = sc.run_neighborhoods(n_pools=4)
        assert len(summary) == 4

    def test_run_neighborhoods_with_invalid_npools_raises(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=5, function='mi',
        )
        with pytest.raises(ValueError, match="n_pools must be > 0"):
            sc.run_neighborhoods(n_pools=0)
        with pytest.raises(ValueError, match="n_pools must be > 0"):
            sc.run_neighborhoods(n_pools=-3)

    def test_run_neighborhoods_no_args_raises(self, synthetic_data):
        expression, genes, gs, _ = synthetic_data
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs
        )
        with pytest.raises(ValueError, match="Provide either labels or n_pools"):
            sc.run_neighborhoods()


class TestPlottingWrappers:
    """Test plotting wrapper methods."""

    def test_plot_consistency_ranking(self):
        """Should create a figure without error."""
        import matplotlib
        matplotlib.use('Agg')

        expression, genes, gs, _ = _make_synthetic_data(
            n_cells=30, n_genes=60, n_planted=2, n_null=1, pathway_size=6
        )
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi',
        )
        sc.run(n_permutations=5)
        ax = sc.plot_consistency_ranking(top_n=3)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_pathway_on_embedding(self):
        """Should create a scatter plot when embeddings available."""
        import matplotlib
        matplotlib.use('Agg')

        expression, genes, gs, _ = _make_synthetic_data(
            n_cells=30, n_genes=60, n_planted=2, n_null=1, pathway_size=6
        )
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi',
        )
        sc.run(n_permutations=5)
        # Add a fake embedding
        sc.embeddings['X_umap'] = np.random.randn(30, 2)

        pw_name = sc.pathway_names[0]
        ax = sc.plot_pathway_on_embedding(pw_name, embedding_key='X_umap')
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_pathway_on_embedding_no_embedding_raises(self):
        expression, genes, gs, _ = _make_synthetic_data(
            n_cells=30, n_genes=60, n_planted=2, n_null=1, pathway_size=6
        )
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi',
        )
        sc.run(n_permutations=5)
        with pytest.raises(ValueError, match="not available"):
            sc.plot_pathway_on_embedding("planted_0", embedding_key='X_umap')

    def test_plot_pathway_heatmap(self):
        """Should create a heatmap without error."""
        import matplotlib
        matplotlib.use('Agg')

        expression, genes, gs, labels = _make_synthetic_data(
            n_cells=30, n_genes=60, n_planted=2, n_null=1, pathway_size=6
        )
        sc = SingleCellPAGE(
            expression=expression, genes=genes, genesets=gs,
            n_bins=3, function='mi',
        )
        sc.run(n_permutations=5)
        ax = sc.plot_pathway_heatmap(labels)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')
