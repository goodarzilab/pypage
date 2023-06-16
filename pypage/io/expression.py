"""Data input and output handling for expression
"""


import numpy as np
from typing import Optional
from .accession_types import change_accessions


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
            genes: np.ndarray,
            expression: np.ndarray,
            is_bin: bool = False,
            n_bins: Optional[int] = 10):
        """
        Parameters
        ==========
        genes: np.ndarray
            The array with the gene names
        expression: np.ndarray
            The array representing either the continuous expression value
            of a specific gene, or the bin/cluster that gene belongs to.
        is_bin: bool
            whether the provided dataframe is prebinned.
        n_bins: int
        """
        self.modified = False
        self._is_bin = is_bin
        self._validate_inputs(genes, expression)
        self._process_input(genes, expression)
        self.n_bins = n_bins

    def _process_input(self,
                       x: np.ndarray,
                       y: np.ndarray):
        """
        sets the attributes accordingly to the inputs
        Parameters
        ----------
        x: np.ndarray
            input genes
        y: np.ndarray
            input expression
        """
        self.genes = np.array(x)
        self.n_genes = self.genes.size
        self.raw_expression = np.array(y)
        notnan_mask = ~ np.isnan(self.raw_expression)
        if len(self.raw_expression.shape) == 2:
            notnan_mask = notnan_mask.any(0)
            self.genes = self.genes[notnan_mask]
            self.raw_expression = self.raw_expression[:, notnan_mask]
        else:
            self.genes = self.genes[notnan_mask]
            self.raw_expression = self.raw_expression[notnan_mask]

    def _validate_inputs(
            self,
            x: np.ndarray,
            y: np.ndarray):
        """validates inputs are as expected
        """
        assert x.size > 0, \
            "provided array must not be empty"
        assert x.shape[0] == y.shape[-1], \
            "genes and expression/bin arrays must be equally shaped"

    def _discretize(self,
                    inp_array: np.ndarray,
                    bins: int,
                    noise_std: Optional[float] = 0.000000001):
        """
        discretizes the expression profile
        Parameters
        ----------
        inp_array: np.ndarray
        bins: int
        noise_std: float
        Returns
        -------
        np.ndarray
            discretized expression profile
        """

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

        if len(self.raw_expression.shape) == 1:
            sub_expression = self.raw_expression[idxs]
            bin_array = self._discretize(sub_expression, self.n_bins)
        else:
            sub_expression = self.raw_expression[:, idxs]
            bin_array = np.apply_along_axis(lambda x: self._discretize(x, self.n_bins), 0, sub_expression)

        self.bin_array = bin_array
        self.modified = True
        return self.bin_array

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

    def reset(self):
        self.modified = False

    def __repr__(
            self) -> str:
        s = ""
        s += "Expression Profile\n"
        s += f">> num_genes: {self.n_genes}\n"
        s += f">> num_bins: {self.n_bins}\n"
        return s