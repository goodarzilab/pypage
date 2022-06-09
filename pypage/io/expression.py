"""Data input and output handling for expression
"""


import numpy as np
import pandas as pd
from typing import Optional


class ExpressionProfile:
    """Container for a Sample Expression Profile

    Attributes
    ==========
    genes: np.ndarray
        the sorted list of genes provided
    gene_indices: dict
        a dictionary where keys are gene names and values are indices
    bins: np.ndarray
        the sorted list of bins provided / calculated
    bin_indices: dict
        a dictionary where keys are bin names and values are indices
    n_genes: int
        the number of genes found
    n_bins: int
        the number of bins found / calculated
    bool_array: np.ndarray
        a (n_bins, n_genes) bit array representing the bin-position of each gene
    bin_array: np.ndarray
        a (n_genes, ) array where each value represents the bin index of that gene
    bin_sizes: np.ndarray
        a (n_bins, ) array where each value represents the number of genes in that bin index
    bin_ranges: Optional[np.ndarray]
        a (n_bins, ) array describing the ranges of the scores for each bin. only present if `is_bin` is True 

    Methods
    -------
    get_gene_subset:
        returns a subset of the `bin_array` for the provided gene list
    """
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            is_bin: bool = False,
            bin_strategy: Optional[str] = None,
            n_bins: Optional[int] = None):
        """
        Parameters
        ==========
        x: np.ndarray
            The array representing the gene names
        y: np.ndarray
            The array representing either the continuous expression value
            of a specific gene, or the bin/cluster that gene belongs to.
        is_bin: bool
            whether the provided dataframe is prebinned. 
        bin_strategy: str
            the method to bin expression, choices are ['hist', 'split'].
            'hist' will create a histogram with `n_bins` which will group
            genes of similar counts more closely together but with unequal bin sizes. 
            'split' will create bins of equivalent sizes. 
            default = 'hist'
        n_bins: int
        """
        self._is_bin = is_bin
        self._validate_inputs(x, y)
        self._set_bin_strategy(bin_strategy)

        self._load_genes(x)
        bins = self._load_expression(y, n_bins)
        self._build_bool_array(x, bins)
        self._build_bin_array()

    def _validate_inputs(
            self,
            x: np.ndarray,
            y: np.ndarray):
        """validates inputs are as expected
        """
        assert x.size > 0,\
            "provided array must not be empty"
        assert x.size == y.size,\
            "genes and expression/bin arrays must be equal sized"
        assert x.shape == y.shape,\
            "genes and expression/bin arrays must be equally shaped"

    def _set_bin_strategy(
            self,
            bin_strategy: str):
        """validates and sets bin strategy
        """
        known_strategy = ["hist", "split"]
        if not bin_strategy:
            self._bin_strategy = "hist"
        else:
            assert bin_strategy in known_strategy,\
                f"unknown bin strategy: `{bin_strategy}`. Known strategies : {', '.join(known_strategy)}"
            self._bin_strategy = bin_strategy

    def _load_genes(
            self,
            genes: np.ndarray) -> np.ndarray:
        """loads the genes from the dataframe
        """
        self.gene_indices = {n: idx for idx, n in enumerate(np.sort(np.unique(genes)))}
        self.genes = np.array(list(self.gene_indices.keys()))
        self.n_genes = self.genes.size

    def _load_expression(
            self, 
            expression: np.ndarray,
            n_bins: Optional[int]) -> np.ndarray:
        """loads the expression/bin data from the dataframe
        """
        if self._is_bin:
            self.n_bins = np.unique(expression).size
            return self._load_bins(expression)
        else:
            if not n_bins:
                self.n_bins = 10
            else:
                assert n_bins > 1,\
                    "number of bins must be greater than 1"
                self.n_bins = n_bins
            bins = self._build_bin_indices(expression)
            return self._load_bins(bins)

    def _build_bin_indices(
            self,
            expression: np.ndarray,
            epsilon: float = 1e-6) -> np.ndarray:
        """converts expression data to binned data
        """
        if self._bin_strategy == "hist":
            return self._build_bin_hist(expression)
        elif self._bin_strategy == "split":
            return self._build_bin_split(expression)

    def _build_bin_hist(
            self,
            expression: np.ndarray,
            epsilon: float = 1e-6) -> np.ndarray:
        """converts expression data to binned data using histogram method
        """
        self.bin_sizes, self.bin_ranges = np.histogram(expression, bins=self.n_bins)
        self.bin_ranges[-1] += epsilon # added because digitize is not inclusive at maximum 
        return np.digitize(expression, self.bin_ranges)

    def _build_bin_split(
            self,
            expression: np.ndarray) -> np.ndarray:
        """converts expression data to binned data using equivlanet split method
        """
        argidx = np.argsort(expression)
        max_size = expression.size
        bin_size = int(max_size / self.n_bins)

        bin_identities = np.zeros(max_size, dtype=int)
        self.bin_sizes = np.zeros(self.n_bins, dtype=int)
        self.bin_ranges = np.zeros(self.n_bins)
        self.bin_ranges[-1] = expression.max()
        
        for i in np.arange(0, self.n_bins):
            lower_bound = expression[argidx[bin_size * i]]

            if i < self.n_bins - 1:
                upper_bound = expression[argidx[bin_size * (i + 1)]]
                mask = (expression >= lower_bound) & (expression < upper_bound)

            # put remaining into last bin
            else:
                mask = (expression >= lower_bound)

            self.bin_sizes[i] = mask.sum()
            self.bin_ranges[i] = lower_bound
            bin_identities[mask] = i

        return bin_identities
        
    def _load_bins(
            self, 
            bins: np.ndarray) -> np.ndarray:
        """loads the bin data from an array
        """
        self.bin_indices = {n: idx for idx, n in enumerate(np.sort(np.unique(bins)))}
        self.bins = np.array(list(self.bin_indices.keys()))
        return bins

    def _build_bool_array(
            self,
            genes: np.ndarray,
            bins: np.ndarray):
        """creates the internal bool array
        """
        self.bool_array = np.zeros((self.n_bins, self.n_genes), dtype=int)
        for g, b in zip(genes, bins):
            self.bool_array[self.bin_indices[b]][self.gene_indices[g]] += 1

    def _build_bin_array(self):
        """creates the internal bin array
        """
        self.bin_array = np.argmax(self.bool_array, axis=0)

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
        """
        Index the bin-array for the required gene subset. 
        Expects the subset to be sorted
        
        Parameters
        ==========
        gene_subset: np.ndarray
            a list of genes to subset the bin_array to

        Returns
        =======
        np.ndarray
            the bin_array subsetted to the indices of the `gene_subset`
        """

        idxs = [np.where(self.genes == gene)[0][0] for gene in gene_subset]
        return self.bin_array[idxs]

    def convert_from_to(self,
                        input_format: str,
                        output_format: str,
                        species: Optional[str] = 'human'):
        """
        A function which changes accessions
        Parameters
        ----------
        input_format
            input accession type, takes 'enst', 'ensg', 'refseq', 'entrez', 'gs', 'ext'
        output_format
            output accession type, takes 'enst', 'ensg', 'refseq', 'entrez', 'gs', 'ext'
        species
            analyzed species, takes either 'human' or 'mouse'
        """
        if input_format in ['ensg', 'enst', 'refseq']:  # for ex., remove '.1' from 'ENSG00000128016.1'
            self.genes = np.array([gene.split('.')[0] for gene in self.genes])
        self.genes = change_accessions(self.genes,
                                       input_format,
                                       output_format,
                                       species)
