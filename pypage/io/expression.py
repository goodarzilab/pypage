"""Data input and output handling for expression
"""


import numpy as np
import pandas as pd
from typing import Optional
from .accession_types import change_accessions
import numba


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
            n_bins: Optional[int] = 10):
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
        self._process_input(x, y)
        self.n_bins = n_bins

    def _process_input(self, x, y):
        self.genes = np.array(x)
        self.n_genes = self.genes.size
        self.raw_expression = np.array(y)
        self.genes = self.genes[~np.isnan(self.raw_expression)]
        self.raw_expression = self.raw_expression[~np.isnan(self.raw_expression)]

    def _validate_inputs(
            self,
            x: np.ndarray,
            y: np.ndarray):
        """validates inputs are as expected
        """
        assert x.size > 0, \
            "provided array must not be empty"
        assert x.size == y.size, \
            "genes and expression/bin arrays must be equal sized"
        assert x.shape == y.shape, \
            "genes and expression/bin arrays must be equally shaped"


    def discretize(self, inp_array, bins, noise_std=0.000000001, new_seed=False):
        if not new_seed:
            np.random.seed(57)
        length = len(inp_array)
        to_discr = inp_array + np.random.normal(0, noise_std, length)

        bins_for_discr = np.interp(np.linspace(0, length, bins + 1),
                                   np.arange(length),
                                   np.sort(to_discr))
        bins_for_discr[-1] += 1  # otherwise numpy creates one extra bin with only 1 point
        digitized = np.digitize(to_discr, bins_for_discr)
        digitized = digitized - 1
        return digitized

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

        sub_expression = self.raw_expression[idxs]
        bin_array = self.discretize(sub_expression, self.n_bins)

        return bin_array

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

    def __repr__(
            self) -> str:
        s = ""
        s += "Expression Profile\n"
        s += f">> num_genes: {self.n_genes}\n"
        s += f">> num_bins: {self.n_bins}\n"
        s += f">> bin_sizes: {' '.join(self.bin_sizes.astype(str))}\n"
        return s