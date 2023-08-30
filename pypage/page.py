"""Implementation of the PAGE Algorithm
"""

from .io import (
    GeneSets,
    ExpressionProfile)
from .utils import (
    empirical_pvalue,
    hypergeometric_test,
    benjamini_hochberg)
from .information import (
    mutual_information,
    calculate_mi_permutations,
    calculate_cmi_permutations,
    measure_redundancy,
    conditional_mutual_information)
from .heatmap import Heatmap

import numpy as np
import numba as nb
import pandas as pd
from tqdm import tqdm
from typing import Optional


class PAGE:
    """
    Pathway Analysis of Gene Expression [1]_


    Methods
    -------
    run:
        Perform the PAGE algorithm on the provided data

    Attributes
    ----------
    expression: ExpressionProfile
        the provided expression profile
    ontology: GeneSets
        the provided ontology
    shared_genes: np.ndarray
        the shared genes between the `ExpressionProfile` and `GeneOntology`
    exp_bins: np.ndarray
        the expression bins for the intersection of `shared_genes`.
        of shape (`shared_genes`.size, )
    ont_bool: np.ndarray
        the ontology bins for the intersection of `shared_genes`.
        of shape (`num_pathways`, `shared_genes`.size)
    x_bins: int
        the number of expression bins in `exp_bins`
    y_bins: int
        the number of expression bins in `ont_bool`
    num_pathways: int
        the number of pathways in `ontology`
    information: np.ndarray
        the calculated mutual information for each pathway.
        of shape (`num_pathways`, )
    informative: np.ndarray
        a `bool` array representing whether that pathway was considered informative.
        of shape (`num_pathways`, )
    n_shuffle: int
        the number of performed permutation tests
        (`default = 1e4`)
    alpha: float
        the maximum p-value threshold to consider a pathway informative
        with respect to the permuted mutual information distribution
        (`default = 5e-3`)
    k: int
        the number of contiguous uninformative pathways to consider before
        stopping the informative pathway search
        (`default = 20`)
    r: float
        the ratio of the conditional mutual information of a new accepted
        pathway against the mutual information of that pathway against
        all other accepted pathways. Only active when filter_redundant == True.
        (`default = 5.0`)
    base: int
        the base of the logarithm used when calculating entropy
        (`default = 2`)
    filter_redundant: bool
        whether to perform the pathway redundancy search
        (`default = True`)
    n_jobs: int
        The number of parallel jobs to use in the analysis
        (`default = all available cores`)


    Notes
    =====
    .. [1] H. Goodarzi, O. Elemento, S. Tavazoie, "Revealing Global Regulatory Perturbations across Human Cancers." https://doi.org/10.1016/j.molcel.2009.11.016
    """

    def __init__(
            self,
            expression: ExpressionProfile,
            genesets: GeneSets,
            n_shuffle: int = 1e3,
            alpha: float = 1e-2,
            k: int = 10,
            filter_redundant: bool = False,
            n_jobs: Optional[int] = 1,
            function: Optional[str] = 'cmi',
            redundancy_ratio: Optional[float] = .1):
        """
        Initialize object

        Parameters
        ----------
        expression: ExpressionProfile
            The provided `ExpressionProfile`

        genesets: GeneSets
            the provided `GeneOntology`

        n_shuffle: int
            the number of performed permutation tests
            (`default = 1e4`)

        alpha: float
            the maximum p-value threshold to consider a pathway informative
            with respect to the permuted mutual information distribution
            (`default = 5e-2`)

        k: int
            the number of contiguous uninformative pathways to consider before
            stopping the informative pathway search
            (`default = 20`)

        base: int
            the base of the logarithm used when calculating entropy
            (`default = 2`)

        filter_redundant: bool
            whether to perform the pathway redundancy search
            (`default = True`)

        n_jobs: int
            The number of parallel jobs to use in the analysis
            (`default = all available cores`)
        """

        self.expression = expression
        self.ontology = genesets

        self.n_shuffle = int(n_shuffle)
        self.alpha = float(alpha)
        self.k = int(k)
        self.base = 2
        self.filter_redundant = filter_redundant
        self.n_jobs = n_jobs
        self._set_jobs()
        self.function = function
        self.redundancy_ratio = redundancy_ratio

        if not self.expression.modified and not self.ontology.modified:
            self._intersect_genes()
            self._subset_matrices()
            self._set_sizes()

    def _set_jobs(self):
        """Sets the number of available jobs for numba parallel
        """
        if self.n_jobs:
            self.n_jobs = int(self.n_jobs)
            nb.set_num_threads(self.n_jobs)
        else:
            # default to using all available threads
            nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)

    def _intersect_genes(self):
        """Intersects to genes found in both sets
        """
        self.shared_genes = np.sort(np.intersect1d(
                                    self.expression.genes,
                                    self.ontology.genes))

    def _subset_matrices(self):
        """Subsets the bool arrays to the gene intersection
        """
        self.exp_bins = self.expression.get_gene_subset(self.shared_genes)
        self.ont_bool = self.ontology.get_gene_subset(self.shared_genes)
        self.membership_bins = self.ontology.get_membership_subset(self.shared_genes)

    def _set_sizes(self):
        """Sets the number of bins for the expression and ontology
        as well as the number of pathways found
        """
        self.x_bins = self.exp_bins.max() + 1
        self.y_bins = self.ont_bool.max() + 1
        self.z_bins = self.membership_bins.max() + 1
        self.num_pathways = self.ont_bool.shape[0]

    def _calculate_information(self) -> np.ndarray:
        """Calculates mutual or conditional mutual information for each pathway
        """
        information = np.zeros(self.num_pathways)
        if self.function == 'mi':
            desc = "calculating mutual information"
        else:
            desc = "calculating conditional mutual information"
        pbar = tqdm(range(self.num_pathways), desc=desc)
        for idx in pbar:
            if self.function == 'mi':
                information[idx] = mutual_information(
                    self.exp_bins,
                    self.ont_bool[idx],
                    self.x_bins,
                    self.y_bins,
                    base=self.base)
            else:
                information[idx] = conditional_mutual_information(
                    self.exp_bins,
                    self.ont_bool[idx],
                    self.membership_bins,
                    self.x_bins,
                    self.y_bins,
                    self.z_bins,
                    base=self.base)
        return information

    def _calculate_information_2D(self) -> np.ndarray:
        """Calculates mutual or conditional mutual information for each pathway
        """
        information = np.zeros((self.exp_bins.shape[0], self.num_pathways))
        if self.function == 'mi':
            desc = "calculating mutual information"
        else:
            desc = "calculating conditional mutual information"
        pbar = tqdm(range(self.exp_bins.shape[0]), desc=desc)
        for exp_idx in pbar:
            for idx in range(self.num_pathways):
                if self.function == 'mi':
                    information[exp_idx, idx] = mutual_information(
                        self.exp_bins[exp_idx],
                        self.ont_bool[idx],
                        self.x_bins,
                        self.y_bins,
                        base=self.base)
                else:
                    information[exp_idx, idx] = conditional_mutual_information(
                        self.exp_bins[exp_idx],
                        self.ont_bool[idx],
                        self.membership_bins,
                        self.x_bins,
                        self.y_bins,
                        self.z_bins,
                        base=self.base)
        information = pd.DataFrame(information, columns=self.ontology.pathways)

        return information

    def _calculate_enrichment(self) -> (np.ndarray, np.ndarray):
        """
        Iterates through informative pathways to calculate hypergeometric pvalues
        """
        overrep_pvals = np.zeros((self.pathway_indices.size, self.x_bins))
        underrep_pvals = np.zeros_like(overrep_pvals)

        pbar = tqdm(enumerate(self.pathway_indices), desc="hypergeometric tests")
        for idx, info_idx in pbar:
            pvals = hypergeometric_test(
                self.exp_bins,
                self.ont_bool[info_idx])
            overrep_pvals[idx, :] = pvals[0]
            underrep_pvals[idx, :] = pvals[1]

        return overrep_pvals, underrep_pvals

    def _calculate_informative(self) -> (np.ndarray, np.ndarray):
        """Calculates the informative categories
        """
        n_break = 0
        informative = np.zeros_like(self.information)
        pvalues = np.ones_like(self.information)

        # iterate through most informative pathways
        pbar = tqdm(np.argsort(self.information)[::-1], desc="permutation testing")
        for idx in pbar:

            # calculate mutual information of random permutations
            if self.function == 'mi':
                permutations = calculate_mi_permutations(
                    self.exp_bins,
                    self.ont_bool[idx],
                    self.x_bins,
                    self.y_bins,
                    n=self.n_shuffle)
            else:
                permutations = calculate_cmi_permutations(
                    self.exp_bins,
                    self.ont_bool[idx],
                    self.membership_bins,
                    self.x_bins,
                    self.y_bins,
                    self.z_bins,
                    n=self.n_shuffle)

            # calculate empirical pvalue against randomization

            pvalues[idx] = empirical_pvalue(
                permutations,
                self.information[idx])

            if pvalues[idx] > self.alpha:
                n_break += 1
                if n_break == self.k:
                    break
            else:
                informative[idx] = 1
                n_break = 0

        return (informative, pvalues)

    def _consolidate_pathways(self) -> np.ndarray:
        """Consolidate redundant pathways
        """
        existing = []
        inf_idx = np.flatnonzero(self.informative)

        # iterate through cmi in descending order
        pbar = tqdm(np.argsort(self.information)[::-1], desc="consolidating redundant pathways")
        for idx in pbar:

            # skip indices that are not informative
            if idx not in inf_idx:
                continue

            # if there are no existing pathways yet start the chain
            if len(existing) == 0:
                existing.append(idx)
                continue

            # initialize reduncancy information array
            all_ri = np.zeros(len(existing))

            # calculate redundancy
            for i, e in enumerate(existing):
                all_ri[i] = measure_redundancy(
                    self.exp_bins,
                    self.ont_bool[idx],
                    self.ont_bool[e],
                    self.x_bins,
                    self.y_bins,
                    self.y_bins)

            if all(all_ri > self.redundancy_ratio):
                existing.append(idx)
            else:
                pass

        return np.array(existing)

    def _gather_results(self) -> pd.DataFrame:
        """Gathers the results from the experiment into a single dataframe
        """
        # estimate sign
        self.log_overrep_pvals = np.log10(self.overrep_pvals)
        self.log_underrep_pvals = np.log10(self.underrep_pvals)
        self.graphical_ar = np.minimum(self.log_overrep_pvals, self.log_underrep_pvals)
        self.graphical_ar[self.log_overrep_pvals < self.log_underrep_pvals] *= -1  # make overrepresented positive
        n_bins = self.graphical_ar.shape[1]
        s1 = self.graphical_ar[:, :n_bins // 3].copy()
        s2 = self.graphical_ar[:, -n_bins // 3:].copy()
        s1[s1 < 0] = 0
        s2[s2 < 0] = 0
        sign = s1.sum(1) <= s2.sum(1)
        sign = sign.astype(int)
        sign[sign == 0] = -1
        results = pd.DataFrame({"pathway": self.ontology.pathways[self.pathway_indices],
                                "CMI": self.information[self.pathway_indices],
                                "p-value": self.pvalues[self.pathway_indices],
                                "Regulation pattern": sign}
                               )
        return results

    def _make_heatmap(self):
        """

        Returns
        -------

        """

        hm = Heatmap(np.array(self.results['pathway']),
                     self.graphical_ar)

        hm.add_gene_expression(self.expression.genes, self.expression.raw_expression)
        return hm

    def run(self) -> (pd.DataFrame, Heatmap):
        """
        Perform the PAGE algorithm

        Returns
        =======
        pd.DataFrame
            A dataframe representing the results of the analysis

        Examples
        ========
        >>> p = PAGE(exp, ont)
        >>> p.run()
        """

        # calculate mutual information
        if len(self.exp_bins.shape) == 2:
            self.information = self._calculate_information_2D()
            return self.information

        self.information = self._calculate_information()

        # select informative pathways
        self.informative, self.pvalues = self._calculate_informative()

        # filter redundant pathways
        if self.filter_redundant:
            self.pathway_indices = self._consolidate_pathways()
        else:
            self.pathway_indices = np.flatnonzero(self.informative)
        if len(self.pathway_indices) != 0:
            # hypergeometric testing over selected pathways
            self.overrep_pvals, self.underrep_pvals = self._calculate_enrichment()

            self.results = self._gather_results()
            self.hm = self._make_heatmap()

            return self.results, self.hm
        else:
            return pd.DataFrame(columns=["pathway", "CMI", "p-value", "Regulation pattern"]), None

    def get_enriched_genes(self, name) -> list:
        assert name in self.ontology.pathways, "pathway not present"
        pathway_idx = np.where(self.ontology.pathways == name)[0][0]
        pathway_binary = self.ont_bool[pathway_idx]
        res = []
        for bin in set(self.exp_bins):
            genes_in_bin = self.shared_genes[np.where(pathway_binary & (self.exp_bins == bin))[0]]
            res.append(genes_in_bin)
        return res

    def get_es_matrix(self):
        """
        Returns the enrichment score matrix for each pathway and bin

        enrichment score is defined as the log10 of the hypergeometric p-value
        (overrepresented scores per bin are positive and underrepresented scores are negative)

        Returns
        =======
        pd.DataFrame
            A dataframe representing the results of the analysis as a matrix (rows are pathways and columns are bins)
        """
        hm = self._make_heatmap()
        return pd.DataFrame(
            hm._subset_and_sort_pathways()[1],
            index=hm._subset_and_sort_pathways()[0]
        )