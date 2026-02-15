"""Single-cell PAGE analysis.

Provides SingleCellPAGE for computing pathway activity scores per cell
using MI/CMI, testing spatial coherence via Geary's C on a cell-cell
KNN graph, and optionally running standard PAGE on neighborhood aggregates.
"""

import math
import numpy as np
import numba as nb
import pandas as pd
from tqdm import tqdm
from scipy.sparse import issparse, csr_matrix
from typing import Optional, Tuple

from .io import GeneSets, ExpressionProfile
from .page import PAGE
from .information import (
    mutual_information, conditional_mutual_information,
    batch_mutual_information_2d, batch_conditional_mutual_information_2d,
)
from .spatial import geary_c, geary_c_batch, build_knn_graph
from .utils import benjamini_hochberg


class SingleCellPAGE:
    """Single-cell pathway analysis using information-theoretic scoring
    and spatial autocorrelation on a cell-cell graph.

    Computes MI or CMI per cell for each pathway, then tests whether
    pathway scores are spatially coherent across the cell manifold
    using Geary's C statistic.

    Parameters
    ----------
    adata : anndata.AnnData, optional
        AnnData object. If provided, expression matrix, gene names,
        precomputed connectivity, and embeddings are extracted.
    expression : np.ndarray, optional
        (n_cells, n_genes) expression matrix. Alternative to adata.
    genes : np.ndarray, optional
        Gene names array of length n_genes. Required if expression is given.
    genesets : GeneSets
        Gene sets / pathway annotations.
    n_neighbors : int, optional
        Number of neighbors for KNN graph. Default: ceil(sqrt(n_cells)),
        capped at 100.
    n_bins : int, optional
        Number of bins for expression discretization. Default: 10.
    bin_axis : str, optional
        Discretization axis: 'cell' (per-cell across genes, VISION-like)
        or 'gene' (per-gene across cells). Default: 'cell'.
    permutation_chunk_size : int, optional
        Number of random gene sets to process per permutation chunk.
        Lower values reduce peak memory; default is an automatic
        memory-aware chunk size.
    connectivity : scipy.sparse matrix, optional
        Precomputed cell-cell connectivity matrix.
    function : str, optional
        'mi' or 'cmi'. Default: 'cmi'.
    n_jobs : int, optional
        Number of threads for Numba parallel execution. Default: 1.
        Set to 0 or None to use all available threads.
    """

    def __init__(
        self,
        adata=None,
        expression=None,
        genes=None,
        genesets=None,
        n_neighbors=None,
        n_bins=10,
        connectivity=None,
        function='cmi',
        bin_axis='cell',
        permutation_chunk_size=None,
        n_jobs=1,
    ):
        if genesets is None:
            raise ValueError("genesets is required")
        if function not in ('mi', 'cmi'):
            raise ValueError(f"function must be 'mi' or 'cmi', got {function!r}")
        if bin_axis not in ('cell', 'gene'):
            raise ValueError(f"bin_axis must be 'cell' or 'gene', got {bin_axis!r}")
        if permutation_chunk_size is not None and int(permutation_chunk_size) <= 0:
            raise ValueError(
                f"permutation_chunk_size must be > 0 when provided, got {permutation_chunk_size!r}"
            )

        self.genesets = genesets
        self.n_bins = n_bins
        self.function = function
        self.bin_axis = bin_axis
        self.permutation_chunk_size = (
            None if permutation_chunk_size is None else int(permutation_chunk_size)
        )
        self.n_jobs = n_jobs
        self.embeddings = {}

        if adata is not None:
            self._extract_from_adata(adata, connectivity)
        elif expression is not None and genes is not None:
            self.expression_matrix = np.asarray(expression, dtype=np.float64)
            self.genes = np.asarray(genes)
            self.connectivity = connectivity
        else:
            raise ValueError(
                "Provide either adata or both expression and genes"
            )

        if self.expression_matrix.ndim != 2:
            raise ValueError(
                f"Expression must be 2D (n_cells, n_genes), "
                f"got shape {self.expression_matrix.shape}"
            )
        if len(self.genes) != self.expression_matrix.shape[1]:
            raise ValueError(
                f"Number of genes ({len(self.genes)}) does not match "
                f"expression columns ({self.expression_matrix.shape[1]})"
            )

        self.n_cells = self.expression_matrix.shape[0]
        self.n_genes_total = self.expression_matrix.shape[1]

        # Default n_neighbors
        if n_neighbors is not None:
            self.n_neighbors = int(n_neighbors)
        else:
            self.n_neighbors = min(int(math.ceil(math.sqrt(self.n_cells))), 100)

        # Intersect genes with gene sets
        self._intersect_genes()

    def _extract_from_adata(self, adata, connectivity=None):
        """Extract data from an AnnData object."""
        import anndata

        if not isinstance(adata, anndata.AnnData):
            raise TypeError(f"Expected AnnData, got {type(adata)}")

        # Expression matrix
        X = adata.X
        if issparse(X):
            X = X.toarray()
        self.expression_matrix = np.asarray(X, dtype=np.float64)

        # Gene names
        self.genes = np.asarray(adata.var_names)

        # Precomputed connectivity
        if connectivity is not None:
            self.connectivity = connectivity
        elif 'connectivities' in adata.obsp:
            self.connectivity = adata.obsp['connectivities']
        else:
            self.connectivity = None

        # Embeddings
        for key in ('X_umap', 'X_tsne', 'X_pca'):
            if key in adata.obsm:
                self.embeddings[key] = np.asarray(adata.obsm[key])

    def _intersect_genes(self):
        """Find shared genes between expression data and gene sets."""
        self.shared_genes = np.sort(
            np.intersect1d(self.genes, self.genesets.genes)
        )
        if len(self.shared_genes) == 0:
            raise ValueError(
                "No shared genes between expression data and gene sets"
            )

        # Build index maps for subsetting
        gene_to_expr_idx = {g: i for i, g in enumerate(self.genes)}
        self._expr_idxs = np.array(
            [gene_to_expr_idx[g] for g in self.shared_genes]
        )

        # Subset gene sets
        self.ont_bool = self.genesets.get_gene_subset(self.shared_genes)
        self.membership_bins = self.genesets.get_membership_subset(
            self.shared_genes
        )
        self.pathway_names = self.genesets.pathways.copy()
        self.n_pathways = self.ont_bool.shape[0]

    def _set_jobs(self):
        """Set the number of threads for Numba parallel execution."""
        if self.n_jobs:
            self.n_jobs = int(self.n_jobs)
            nb.set_num_threads(self.n_jobs)
        else:
            nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)

    def _build_knn_graph(self):
        """Build or use precomputed KNN graph."""
        if self.connectivity is not None:
            W = self.connectivity
            if not isinstance(W, csr_matrix):
                W = csr_matrix(W)
            self.W = W
        else:
            expr_subset = self.expression_matrix[:, self._expr_idxs]
            self.W = build_knn_graph(expr_subset, self.n_neighbors)

        # Ensure float64 for Numba
        self.W = self.W.astype(np.float64)

    def _score_pathways(self):
        """Compute MI or CMI for each cell and pathway.

        Returns
        -------
        np.ndarray
            Shape (n_cells, n_pathways) with information scores.
        """
        expr_subset = self.expression_matrix[:, self._expr_idxs]
        exp_bins = self._discretize_expression(expr_subset)
        # exp_bins shape: (n_cells, n_shared_genes)

        x_bins = int(exp_bins.max()) + 1
        y_bins = int(self.ont_bool.max()) + 1

        exp_bins_i32 = exp_bins.astype(np.int32)
        ont_bool_i32 = self.ont_bool.astype(np.int32)

        if self.function == 'mi':
            return batch_mutual_information_2d(
                exp_bins_i32, ont_bool_i32, x_bins, y_bins)
        else:
            z_bins = int(self.membership_bins.max()) + 1
            return batch_conditional_mutual_information_2d(
                exp_bins_i32, ont_bool_i32,
                self.membership_bins.astype(np.int32),
                x_bins, y_bins, z_bins)

    def _compute_geary_c(self, scores):
        """Compute Geary's C' for each pathway.

        Parameters
        ----------
        scores : np.ndarray
            Shape (n_cells, n_pathways).

        Returns
        -------
        np.ndarray
            Shape (n_pathways,) with C' = 1 - C values.
        """
        W = self.W
        return geary_c_batch(
            scores,
            W.indices.astype(np.int64),
            W.indptr.astype(np.int64),
            W.data.astype(np.float64),
            self.n_cells,
        )

    def _permutation_test(self, n_permutations=1000):
        """Permutation test for Geary's C using size-matched random gene sets.

        Generates random gene sets of matching sizes, computes their per-cell
        scores and Geary's C, then computes empirical p-values.

        Parameters
        ----------
        n_permutations : int
            Number of random gene sets per size group.

        Returns
        -------
        np.ndarray
            Shape (n_pathways,) p-values.
        """
        pathway_sizes = self.ont_bool.sum(axis=1)
        unique_sizes = np.unique(pathway_sizes)

        # Group pathways by size for efficiency
        size_to_pathways = {}
        for size in unique_sizes:
            size_to_pathways[int(size)] = np.where(pathway_sizes == size)[0]

        pvalues = np.ones(self.n_pathways)
        n_shared = len(self.shared_genes)

        # Pre-compute discretized expression (same as _score_pathways)
        expr_subset = self.expression_matrix[:, self._expr_idxs]
        exp_bins = self._discretize_expression(expr_subset)
        x_bins = int(exp_bins.max()) + 1
        y_bins = 2  # binary membership
        z_bins = int(self.membership_bins.max()) + 1
        membership = self.membership_bins.astype(np.int32)

        exp_bins_i32 = exp_bins.astype(np.int32)

        W_indices = self.W.indices.astype(np.int64)
        W_indptr = self.W.indptr.astype(np.int64)
        W_data = self.W.data.astype(np.float64)

        if n_permutations <= 0:
            raise ValueError(f"n_permutations must be > 0, got {n_permutations}")

        # Keep random membership matrix chunks near ~256 MB by default.
        bytes_per_perm = max(n_shared * 4, 1)  # int32 matrix
        auto_chunk = max(1, int((256 * 1024 * 1024) // bytes_per_perm))
        chunk_size = self.permutation_chunk_size or min(n_permutations, auto_chunk)
        chunk_size = max(1, min(int(chunk_size), n_permutations))

        for size in tqdm(unique_sizes, desc="permutation testing (pathway sizes)"):
            size = int(size)
            pw_idxs = size_to_pathways[size]
            exceed_counts = np.zeros(len(pw_idxs), dtype=np.int64)
            chunk_starts = range(0, n_permutations, chunk_size)
            chunk_bar = tqdm(
                chunk_starts,
                total=(n_permutations + chunk_size - 1) // chunk_size,
                desc=f"  size={size} ({len(pw_idxs)} pathways)",
                leave=False,
            )
            for start in chunk_bar:
                end = min(start + chunk_size, n_permutations)
                n_chunk = end - start

                # Generate random gene sets for this chunk: (n_chunk, n_shared)
                rand_bools = np.zeros((n_chunk, n_shared), dtype=np.int32)
                for perm in range(n_chunk):
                    genes = np.random.choice(n_shared, size=size, replace=False)
                    rand_bools[perm, genes] = 1

                # One batch JIT call: (n_cells, n_chunk) score matrix
                if self.function == 'mi':
                    rand_scores = batch_mutual_information_2d(
                        exp_bins_i32, rand_bools, x_bins, y_bins)
                else:
                    rand_scores = batch_conditional_mutual_information_2d(
                        exp_bins_i32, rand_bools, membership,
                        x_bins, y_bins, z_bins)

                # Batch Geary's C for all permutations in this chunk
                null_c_primes = geary_c_batch(
                    rand_scores, W_indices, W_indptr, W_data, self.n_cells)

                for i, p_idx in enumerate(pw_idxs):
                    observed = self.consistency[p_idx]
                    exceed_counts[i] += np.sum(null_c_primes >= observed)
                chunk_bar.set_postfix_str(f"perm {end}/{n_permutations}")

            # Final empirical p-values for pathways of this size
            for i, p_idx in enumerate(pw_idxs):
                pvalues[p_idx] = (exceed_counts[i] + 1) / (n_permutations + 1)

        return pvalues

    def run_manual(self, pathway_names):
        """Score specific pathways without permutation testing.

        Parameters
        ----------
        pathway_names : list of str
            Pathway names to analyze. Must exist in gene set annotations.

        Returns
        -------
        pd.DataFrame
            Results with columns: pathway, consistency, p-value (NaN), FDR (NaN).
        """
        unknown = [n for n in pathway_names if n not in self.pathway_names]
        if unknown:
            raise ValueError(f"Unknown pathway(s): {unknown}")

        pathway_indices = np.array([
            int(np.flatnonzero(self.pathway_names == name)[0])
            for name in pathway_names
        ])

        self._set_jobs()
        self._build_knn_graph()

        all_scores = self._score_pathways()
        self.scores = all_scores[:, pathway_indices]
        self.consistency = self._compute_geary_c(self.scores)
        self.pvalues = np.full(len(pathway_indices), np.nan)
        self.fdr = np.full(len(pathway_indices), np.nan)

        self.results = pd.DataFrame({
            'pathway': [self.pathway_names[i] for i in pathway_indices],
            'consistency': self.consistency,
            'p-value': self.pvalues,
            'FDR': self.fdr,
        }).sort_values('consistency', ascending=False).reset_index(drop=True)

        return self.results

    def run(self, n_permutations=1000):
        """Run the single-cell PAGE analysis.

        Computes MI/CMI per cell, tests spatial coherence via Geary's C,
        and returns a results DataFrame.

        Parameters
        ----------
        n_permutations : int
            Number of permutations for significance testing. Default: 1000.

        Returns
        -------
        pd.DataFrame
            Results with columns: pathway, consistency, p-value, FDR.
        """
        pipeline_bar = tqdm(total=4, desc="single-cell pipeline", leave=True)
        self._set_jobs()
        self._build_knn_graph()
        pipeline_bar.update(1)
        pipeline_bar.set_postfix_str("knn graph ready")

        self.scores = self._score_pathways()
        pipeline_bar.update(1)
        pipeline_bar.set_postfix_str("pathway scores ready")

        self.consistency = self._compute_geary_c(self.scores)
        pipeline_bar.update(1)
        pipeline_bar.set_postfix_str("geary c ready")

        self.pvalues = self._permutation_test(n_permutations=n_permutations)
        self.fdr = benjamini_hochberg(self.pvalues)
        pipeline_bar.update(1)
        pipeline_bar.set_postfix_str("permutation test ready")
        pipeline_bar.close()

        self.results = pd.DataFrame({
            'pathway': self.pathway_names,
            'consistency': self.consistency,
            'p-value': self.pvalues,
            'FDR': self.fdr,
        }).sort_values('consistency', ascending=False).reset_index(drop=True)

        return self.results

    def run_neighborhoods(self, labels=None, n_pools=None):
        """Run standard PAGE on aggregated neighborhoods.

        Groups cells by label or micro-pooling, computes mean expression
        per group, then runs standard bulk PAGE on the resulting pseudo-bulk
        profiles.

        Parameters
        ----------
        labels : array-like, optional
            Cell group assignments (e.g., cluster labels from adata.obs['leiden']).
            Length must match n_cells.
        n_pools : int, optional
            Number of random micro-pools to create (simple random partitioning).
            Used only if labels is not provided.

        Returns
        -------
        summary : pd.DataFrame
            Summary with mean information per group.
        group_results : dict
            Dict mapping group name to (results_df, heatmap) from standard PAGE.
        """
        if labels is None and n_pools is None:
            raise ValueError("Provide either labels or n_pools")

        if labels is not None:
            labels = np.asarray(labels)
            if len(labels) != self.n_cells:
                raise ValueError(
                    f"Labels length ({len(labels)}) != n_cells ({self.n_cells})"
                )
            unique_labels = np.unique(labels)
        else:
            if n_pools is None or int(n_pools) <= 0:
                raise ValueError(f"n_pools must be > 0, got {n_pools!r}")
            n_pools = int(n_pools)
            # Simple random partitioning
            indices = np.random.permutation(self.n_cells)
            pool_size = max(1, self.n_cells // n_pools)
            labels = np.zeros(self.n_cells, dtype=int)
            for i in range(n_pools):
                start = i * pool_size
                end = min((i + 1) * pool_size, self.n_cells)
                labels[indices[start:end]] = i
            unique_labels = np.arange(n_pools)

        expr_subset = self.expression_matrix[:, self._expr_idxs]
        group_results = {}
        summary_rows = []

        for label in unique_labels:
            mask = labels == label
            if mask.sum() == 0:
                continue

            # Mean expression for this group
            mean_expr = expr_subset[mask].mean(axis=0)

            ep = ExpressionProfile(
                self.shared_genes, mean_expr, n_bins=self.n_bins
            )
            page = PAGE(
                ep, self.genesets,
                function=self.function,
                n_shuffle=1000,
            )
            results_df, hm = page.run()
            group_results[str(label)] = (results_df, hm)

            if len(results_df) > 0:
                summary_rows.append({
                    'group': str(label),
                    'n_cells': int(mask.sum()),
                    'n_significant': len(results_df),
                    'top_pathway': results_df.iloc[0]['pathway']
                    if len(results_df) > 0 else None,
                })
            else:
                summary_rows.append({
                    'group': str(label),
                    'n_cells': int(mask.sum()),
                    'n_significant': 0,
                    'top_pathway': None,
                })

        summary = pd.DataFrame(summary_rows)
        return summary, group_results

    # --- Visualization wrappers ---

    def plot_pathway_on_embedding(
        self, pathway_name, embedding_key='X_umap', ax=None, **kwargs
    ):
        """Plot pathway scores on a 2D embedding.

        Parameters
        ----------
        pathway_name : str
            Name of the pathway to plot.
        embedding_key : str
            Key in self.embeddings dict. Default: 'X_umap'.
        ax : matplotlib.axes.Axes, optional
        **kwargs
            Passed to plot_pathway_embedding.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .plotting import plot_pathway_embedding

        p_idx = np.where(self.pathway_names == pathway_name)[0]
        if len(p_idx) == 0:
            raise ValueError(f"Pathway not found: {pathway_name!r}")
        p_idx = p_idx[0]

        if embedding_key not in self.embeddings:
            raise ValueError(
                f"Embedding {embedding_key!r} not available. "
                f"Available: {list(self.embeddings.keys())}"
            )

        return plot_pathway_embedding(
            scores=self.scores[:, p_idx],
            embedding=self.embeddings[embedding_key],
            pathway_name=pathway_name,
            ax=ax,
            **kwargs,
        )

    def plot_pathway_heatmap(self, labels, **kwargs):
        """Plot heatmap of pathway scores across cell groups.

        Parameters
        ----------
        labels : array-like
            Cell group assignments.
        **kwargs
            Passed to plot_pathway_heatmap.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .plotting import plot_pathway_heatmap

        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        group_means = np.zeros((len(unique_labels), self.n_pathways))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            group_means[i] = self.scores[mask].mean(axis=0)

        return plot_pathway_heatmap(
            results_df=self.results,
            scores_matrix=group_means,
            group_names=unique_labels.astype(str),
            pathway_names=self.pathway_names,
            **kwargs,
        )

    def plot_consistency_ranking(self, **kwargs):
        """Plot top pathways ranked by consistency score.

        Parameters
        ----------
        **kwargs
            Passed to plot_consistency_ranking.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .plotting import plot_consistency_ranking

        return plot_consistency_ranking(results=self.results, **kwargs)

    def __repr__(self):
        return (
            f"SingleCellPAGE(\n"
            f"  n_cells={self.n_cells},\n"
            f"  n_genes={self.n_genes_total},\n"
            f"  n_shared_genes={len(self.shared_genes)},\n"
            f"  n_pathways={self.n_pathways},\n"
            f"  n_neighbors={self.n_neighbors},\n"
            f"  bin_axis='{self.bin_axis}',\n"
            f"  function='{self.function}'\n"
            f")"
        )

    @staticmethod
    def _discretize_1d(inp_array, bins, noise_std=1e-9):
        """Discretize one vector into equal-frequency bins."""
        length = len(inp_array)
        to_discr = inp_array + np.random.normal(0, noise_std, length)
        bins_for_discr = np.interp(
            np.linspace(0, length, bins + 1),
            np.arange(length),
            np.sort(to_discr),
        )
        bins_for_discr[-1] += 1
        return np.digitize(to_discr, bins_for_discr) - 1

    def _discretize_expression(self, expr_subset):
        """Discretize expression by cell or by gene.

        Returns
        -------
        np.ndarray
            Integer array of shape (n_cells, n_genes).
        """
        axis = 1 if self.bin_axis == 'cell' else 0
        return np.apply_along_axis(
            lambda x: self._discretize_1d(x, self.n_bins),
            axis,
            expr_subset,
        )
