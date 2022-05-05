"""Data input and output handling for ontologies
"""


import sys
import numpy as np
from typing import Optional


class GeneOntology:
    """Container for a Set of Pathways

    Attributes
    ==========
    genes: np.ndarray
        the sorted list of genes found in the pathways
    pathways: np.ndarray
        the sorted list of pathways found in the index
    n_genes: int
        the number of genes found
    n_pathways: int
        the number of pathways found
    bool_array: np.ndarray
        a (n_pathways, n_genes) bit array representing the bin-position of each gene
    pathway_sizes: np.ndarray
        a (n_pathways, ) array where each value represents the number of genes in that pathway index
    avg_p_size: float
        the mean number of genes across pathways

    Methods
    -------
    get_gene_subset:
        returns a subset of the `bool_array` for the provided gene list
    filter_pathways:
        filters pathways based on membership
    """
    def __init__(
            self,
            genes: np.ndarray,
            pathways: np.ndarray):
        """
        Parameters
        ==========
        genes: np.ndarray
            an array of gene names
        pathways: np.ndarray
            an array associated pathways
        """

        self._validate_inputs(genes, pathways)
        self._load_genes(genes)
        self._load_pathways(pathways)
        self._build_bool_array(genes, pathways)
    
    def _validate_inputs(
            self,
            x: np.ndarray,
            y: np.ndarray):
        """validates inputs are as expected
        """
        assert x.size > 0,\
            "provided array must not be empty"
        assert x.size == y.size,\
            "genes and pathway arrays must be equal sized"
        assert x.shape == y.shape,\
            "genes and pathway arrays must be equally shaped"

    def _load_genes(
            self, 
            genes: np.ndarray):
        """load genes and associated indices
        """
        self._gene_indices = {n: idx for idx, n in enumerate(np.sort(np.unique(genes)))}
        self.genes = np.array(list(self._gene_indices.keys()))
        self.n_genes = self.genes.size

    def _load_pathways(
            self, 
            pathways: np.ndarray):
        """load pathways and associated indices
        """
        self._pathway_indices = {n: idx for idx, n in enumerate(np.sort(np.unique(pathways)))}
        self.pathways = np.array(list(self._pathway_indices.keys()))
        self.n_pathways = self.pathways.size

    def _calculate_pathway_sizes(self):
        """calculates and sets pathway sizes
        """
        self.pathway_sizes = self.bool_array.sum(axis=1)
        self.avg_p_size = self.pathway_sizes.mean()

    def _build_bool_array(
            self,
            genes: np.ndarray,
            pathways: np.ndarray):
        """create the bool array of genes/pathway interactions
        """
        self.bool_array = np.zeros((self.n_pathways, self.n_genes), dtype=int)
        for g, p in zip(genes, pathways):
            self.bool_array[self._pathway_indices[p]][self._gene_indices[g]] += 1

        self._calculate_pathway_sizes()

        # these may change with filtering so better to remove them
        del self._pathway_indices
        del self._gene_indices

    def get_gene_subset(
            self,
            gene_subset: np.ndarray) -> np.ndarray:
        """
        Index the bool-array for the required gene subset. 
        Expects the subset to be sorted
        
        Parameters
        ==========
        gene_subset: np.ndarray
            a list of genes to subset the bin_array to

        Returns
        =======
        np.ndarray
            the bool_array subsetted to the indices of the `gene_subset`
        """
        mask = np.isin(self.genes, gene_subset)
        return self.bool_array[:, mask]

    def filter_pathways(
            self,
            min_size: Optional[int] = None,
            max_size: Optional[int] = None):
        """
        Filters pathways to those within a specified range

        Parameters
        ----------
        min_size: Optional[int]
            the minimum size of the pathways, defaults to 0
        max_size: Optional[int]
            the maximum size of the pathways, default to current maximum
        """
        if not min_size and not max_size:
            print("No minimum or maximum size provided. Doing nothing.", file=sys.stderr)
            return 
        if not min_size:
            min_size = 0
        if not max_size:
            max_size = np.max(self.pathway_sizes)
        assert min_size >= 0, "Provided minimum must be >= 0"
        assert max_size > min_size, f"Provided maximum must be > min_size: {min_size}"
        
        # determine pathway-level mask
        p_mask = (self.pathway_sizes >= min_size) & (self.pathway_sizes <= max_size)
        
        # filter pathways
        self.bool_array = self.bool_array[p_mask]
        self.pathway_sizes = self.pathway_sizes[p_mask]
        self.pathways = self.pathways[p_mask]

        # determine gene level mask
        g_mask = self.bool_array.sum(axis=0) != 0
        
        # filter genes with no pathways
        self.bool_array = self.bool_array[:, g_mask]
        self.genes = self.genes[g_mask]
    
    def __repr__(self) -> str:
        """
        """
        s = ""
        s += "Gene Ontology\n"
        s += f">> num_genes: {self.n_genes}\n"
        s += f">> num_pathways: {self.n_pathways}\n"
        s += ">> avg_pathway_size: {:.2f}\n".format(self.avg_p_size)
        return s


