"""Data input and output handling for ontologies
"""


import numpy as np
import pandas as pd
from typing import Optional


class GeneOntology:
    """Container for a Set of Pathways

    Attributes
    ==========
    genes: np.ndarray
        the sorted list of genes found in the pathways
    gene_indices: np.ndarray
        a dictionary where keys are gene names and values are associated indices
    pathways: np.ndarray
        the sorted list of pathways found in the index
    pathway_indices: np.ndarray
        a dictionary where keys are pathway names and values are associated indices
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

        self._load_genes(genes)
        self._load_pathways(pathways)
        self._build_bool_array(genes, pathways)

    def _load_genes(
            self, 
            genes: np.ndarray):
        """load genes and associated indices
        """
        self.gene_indices = {n: idx for idx, n in enumerate(np.sort(np.unique(genes)))}
        self.genes = np.array(list(self.gene_indices.keys()))
        self.n_genes = self.genes.size

    def _load_pathways(
            self, 
            pathways: np.ndarray):
        """load pathways and associated indices
        """
        self.pathway_indices = {n: idx for idx, n in enumerate(np.sort(np.unique(pathways)))}
        self.pathways = np.array(list(self.pathway_indices.keys()))
        self.n_pathways = self.pathways.size

    def _build_bool_array(
            self,
            genes: np.ndarray,
            pathways: np.ndarray):
        """create the bool array of genes/pathway interactions
        """
        self.bool_array = np.zeros((self.n_pathways, self.n_genes), dtype=int)
        for g, p in zip(genes, pathways):
            self.bool_array[self.pathway_indices[p]][self.gene_indices[g]] += 1

        self.pathway_sizes = self.bool_array.sum(axis=1)
        self.avg_p_size = self.pathway_sizes.mean()

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
    
    def __repr__(self) -> str:
        """
        """
        s = ""
        s += "Gene Ontology\n"
        s += f">> num_genes: {self.n_genes}\n"
        s += f">> num_pathways: {self.n_pathways}\n"
        s += ">> avg_pathway_size: {:.2f}\n".format(self.avg_p_size)
        return s


