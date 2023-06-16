"""Data input and output handling for ontologies
"""


import sys
import numpy as np
from typing import Optional
from .accession_types import change_accessions
import gzip


class GeneSets:
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
    membership: np.ndarray
        array representing the membership of each gene in an annotation
    n_bins: int
        number of bins to use when binning membership array
    bin_array: np.ndarray
        binned membership array

    Methods
    -------
    get_gene_subset:
        returns a subset of the `bool_array` for the provided gene list
    filter_pathways:
        filters pathways based on membership
    """
    def __init__(
            self,
            genes: Optional[np.ndarray] = None,
            pathways: Optional[np.ndarray] = None,
            ann_file: Optional[str] = None,
            n_bins: Optional[int] = 3,
            first_col_is_genes: Optional[bool] = False):
        """
        Parameters
        ==========
        genes: np.ndarray
            an array of gene names
        pathways: np.ndarray
            an array associated pathways
        """
        self.modified = False
        if ann_file:
            self._read_annotation_file(ann_file, first_col_is_genes)
        else:
            self._validate_inputs(genes, pathways)
            self._load_genes(genes)
            self._load_pathways(pathways)
            self._build_bool_array(genes, pathways)
        self._make_membership_profile()
        self.n_bins = n_bins

    def _read_annotation_file(self,
                              ann_file: str,
                              first_col_is_genes: Optional[bool] = False):
        """
         Reads annotation files which are in a descriptive format,
         i.e, "Gene1 Pathway1 Pathway2..." or "Pathway1 Gene1 Gene2..."

         Parameters
         ----------
         ann_file
             a file name of the annotation
         first_is_gene
             a parameter which specifies whether a gene is
         Returns
         -------
         pd.DataFrame
             a dataframe in a long format
         """
        if ann_file[-2:] == 'gz':
            f = gzip.open(ann_file, 'r')
        else:
            f = open(ann_file)

        row_names = []
        column_names = set()
        for line in f:
            if ann_file[-2:] == 'gz':
                line = line.decode('ASCII')
            els = line.rstrip().split('\t')
            row_names.append(els[0])
            els.pop(0)
            if 'http://' in els[0]:
                els.pop(0)
            column_names |= set(els)
        column_names = list(column_names)

        positions = dict(zip(column_names, np.arange(len(column_names))))
        db_profiles = np.zeros((len(row_names), len(column_names)), dtype=int)

        if ann_file[-2:] == 'gz':
            f = gzip.open(ann_file, 'r')
        else:
            f = open(ann_file)

        i = 0
        for line in f:
            if ann_file[-2:] == 'gz':
                line = line.decode('ASCII')
            els = line.rstrip().split('\t')[1:]
            if 'http://' in els[0]:
                els.pop(0)
            indices = [positions[el] for el in els]
            db_profiles[i, indices] = 1
            i += 1

        if first_col_is_genes:
            db_profiles = db_profiles.T
            db_genes = row_names
            db_names = column_names
        else:
            db_genes = column_names
            db_names = row_names
        self.pathways = np.array(db_names)
        self.genes = np.array(db_genes)
        self.bool_array = db_profiles
        self.n_genes = len(self.genes)
        self.n_pathways = len(self.pathways)

    def _validate_inputs(
            self,
            x: np.ndarray,
            y: np.ndarray):
        """validates inputs are as expected
        """
        assert x.size > 0, \
            "provided array must not be empty"
        assert x.size == y.size, \
            "genes and pathway arrays must be equal sized"
        assert x.shape == y.shape, \
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

    def _build_bin_split(
            self,
            membership: np.ndarray,
            n_bins: int) -> np.ndarray:
        """converts membership array to binned data using equivlanet split method
        """
        argidx = np.argsort(membership)
        max_size = membership.size
        bin_size = int(max_size / n_bins)

        bin_identities = np.zeros(max_size, dtype=int)
        self.bin_sizes = np.zeros(n_bins, dtype=int)
        self.bin_ranges = np.zeros(n_bins)
        self.bin_ranges[-1] = membership.max()

        for i in np.arange(0, n_bins):
            lower_bound = membership[argidx[bin_size * i]]

            if i < n_bins - 1:
                upper_bound = membership[argidx[bin_size * (i + 1)]]
                mask = (membership >= lower_bound) & (membership < upper_bound)

            # put remaining into last bin
            else:
                mask = (membership >= lower_bound)

            self.bin_sizes[i] = mask.sum()
            self.bin_ranges[i] = lower_bound
            bin_identities[mask] = i

        return bin_identities

    def _make_membership_profile(self) -> np.ndarray:
        """create a gene membership array"""
        self.membership = self.bool_array.sum(0)

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
        idxs = [np.where(self.genes == gene)[0][0] for gene in gene_subset]
        self.sub_bool_array = self.bool_array[:, idxs]
        self.modified = True
        return self.sub_bool_array

    def get_membership_subset(
            self,
            gene_subset: np.ndarray) -> np.ndarray:
        """
        Index the bool-array for the required gene membership subset and bin it.
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
        idxs = [np.where(self.genes == gene)[0][0] for gene in gene_subset]
        sub_membership = self.membership[idxs]
        sub_membership_binned = self._build_bin_split(sub_membership, self.n_bins)
        return sub_membership_binned

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

    def __repr__(self) -> str:
        """
        """
        s = ""
        s += "Gene Ontology\n"
        s += f">> num_genes: {self.n_genes}\n"
        s += f">> num_pathways: {self.n_pathways}\n"
        return s


