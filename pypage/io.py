"""Data input and output handling
"""

import numpy as np
import pandas as pd


class ExpressionProfile:
    """Container for a Sample Expression Profile
    """
    def __init__(
            self,
            expression_profile: pd.DataFrame):
        """
        """

        genes = {n: idx for idx, n in enumerate(np.sort(expression_profile.gene.unique()))}
        bins = {n: idx for idx, n in enumerate(np.sort(expression_profile.bin.unique()))}

        self.genes = np.array(list(genes.keys()))
        self.bins= np.array(list(bins.keys()))

        self.n_genes = self.genes.size
        self.n_bins = self.bins.size

        self.bool_array = np.zeros((self.n_bins, self.n_genes), dtype=int)
        for g, b in expression_profile.values:
            self.bool_array[bins[b]][genes[g]] += 1
        
        self.bin_sizes = self.bool_array.sum(axis=1)

    def __repr__(
            self) -> str:
        s = ""
        s += "Expression Profile\n"
        s += f">> num_genes: {self.n_genes}\n"
        s += f">> num_bins: {self.n_bins}\n"
        s += f">> bin_sizes: {' '.join(self.bin_sizes.astype(str))}\n"
        return s

    def get_gene_subset(
            self,
            gene_subset: np.ndarray) -> np.ndarray:
        """Indexed the bool array for the required gene subset. Expects the subset to be sorted
        """
        mask = np.isin(self.genes, gene_subset)
        return self.bool_array[:, mask]


class GeneOntology:
    """Container for a Set of Pathways
    """
    def __init__(
            self,
            index_filename: str,
            names_filename: str):

        self.index_filename = index_filename

        # load files
        self._load_index()

    def _load_index(self):
        """
        loads the index file representing the gene to pathway mapping
        """
        index = pd.read_csv(
                self.index_filename, 
                sep="\t",
                header=None,
                names=["gene", "pathway"])
        
        pathways = {n: idx for idx, n in enumerate(np.sort(index.pathway.unique()))}
        genes = {n: idx for idx, n in enumerate(np.sort(index.gene.unique()))}

        self.genes = np.array(list(genes.keys()))
        self.pathways = np.array(list(pathways.keys()))

        self.n_genes = self.genes.size
        self.n_pathways = self.pathways.size

        # builds bool array
        self.bool_array = np.zeros((self.n_pathways, self.n_genes), dtype=int)
        for g, p in index.values:
            self.bool_array[pathways[p]][genes[g]] += 1

        self.pathway_sizes = self.bool_array.sum(axis=1)
        self.avg_p_size = self.pathway_sizes.mean()

    def get_gene_subset(
            self,
            gene_subset: np.ndarray) -> np.ndarray:
        """Indexed the bool array for the required gene subset. Expects the subset to be sorted
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


