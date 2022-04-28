"""Implementation of the PAGE Algorithm
"""

from .io import (
        GeneOntology, 
        ExpressionProfile)
from .utils import (
        empirical_pvalue,
        hypergeometric_test,
        benjamini_hochberg)
from .information import (
        mutual_information,
        calculate_mi_permutations,
        measure_redundancy)

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

    Notes
    =====
    .. [1] H. Goodarzi, O. Elemento, S. Tavazoie, "Revealing Global Regulatory Perturbations across Human Cancers." https://doi.org/10.1016/j.molcel.2009.11.016
    """
    def __init__(
            self,
            n_shuffle: int = 1e4,
            alpha: float = 5e-3,
            k: int = 20,
            r: float = 5.,
            base: int = 2,
            filter_redundant: bool = True,
            n_jobs: Optional[int] = None):
        """
        Initialize object
        
        Parameters
        ----------
        
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
        """

        self.n_shuffle = int(n_shuffle)
        self.alpha = float(alpha)
        self.k = int(k)
        self.r = float(r)
        self.base = int(base)
        self.filter_redundant = filter_redundant
        self.n_jobs = n_jobs
        self._set_jobs()

    def _set_jobs(self):
        """Sets the number of available jobs for numba parallel
        """
        if self.n_jobs:
            self.n_jobs = int(self.n_jobs)
            nb.set_num_threads(self.n_jobs)
        else:
            # default to using all available threads
            nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)

    def _intersect_genes(
            self, 
            exp: ExpressionProfile,
            ont: GeneOntology) -> np.ndarray:
        """Intersects to genes found in both sets
        """
        shared_genes = np.sort(np.intersect1d(exp.genes, ont.genes))
        return shared_genes

    def _subset_matrices(
            self,
            exp: ExpressionProfile,
            ont: GeneOntology,
            ix: np.ndarray) -> (np.ndarray, np.ndarray):
        """Subsets the bool arrays to the gene intersection
        """
        exp_bins = exp.get_gene_subset(ix)
        ont_bool = ont.get_gene_subset(ix)
        return exp_bins, ont_bool

    def _calculate_mutual_information(
            self,
            exp_bins: np.ndarray,
            ont_bool: np.ndarray,
            x_bins: int,
            y_bins: int) -> np.ndarray:
        """Calculates the mutual information for each pathway
        """
        num_pathways = ont_bool.shape[0]
        information = np.zeros(num_pathways)
        for idx in tqdm(range(num_pathways), desc="calculating mutual information"):
            information[idx] = mutual_information(
                    exp_bins, 
                    ont_bool[idx], 
                    x_bins, 
                    y_bins, 
                    base=self.base)
        return information

    def _significance_testing(
            self,
            indices: np.ndarray,
            exp_bins: np.ndarray,
            ont_bool: np.ndarray,
            x_bins: int,
            y_bins: int):
        """
        Iterates through informative pathways to calculate hypergeometric pvalues
        """
        overrep_pvals = np.zeros((x_bins, indices.size))
        underrep_pvals = np.zeros_like(overrep_pvals)

        pbar = tqdm(enumerate(indices), desc="hypergeometric tests")
        for idx, info_idx in pbar:
            pvals = hypergeometric_test(
                    exp_bins, 
                    ont_bool[info_idx])
            overrep_pvals[:, idx] = pvals[0]
            underrep_pvals[:, idx] = pvals[1]

        return overrep_pvals, underrep_pvals

    def _gather_results(
            self,
            exp: ExpressionProfile,
            ont: GeneOntology,
            info: np.ndarray,
            over_pvals: np.ndarray,
            under_pvals: np.ndarray) -> pd.DataFrame:
        """Gathers the results from the experiment into a single dataframe
        """
        results = []
        for info_idx, path_idx in enumerate(info):
            for bin_idx in range(over_pvals.shape[0]):
                results.append({
                    "bin": exp.bins[bin_idx],
                    "pathway": ont.pathways[path_idx],
                    "over_pval": over_pvals[bin_idx, info_idx],
                    "under_pval": under_pvals[bin_idx, info_idx]})

        results = pd.DataFrame(results)
        results["sign"] = results.apply(lambda x: 1 if x.over_pval < x.under_pval else -1, axis=1)
        results["pvalue"] = results.apply(lambda x: np.min([x.over_pval, x.under_pval]), axis=1)
        results["adj_pval"] = benjamini_hochberg(results.pvalue)
        results["nlp"] = -np.log10(results.adj_pval + np.min(results.adj_pval[results.adj_pval > 0]))
        results["snlp"] = results.sign * results.nlp
        return results

    def _calculate_informative(
            self,
            information: np.ndarray,
            exp_bins: np.ndarray,
            ont_bool: np.ndarray,
            x_bins: np.ndarray,
            y_bins: np.ndarray) -> (np.ndarray, np.ndarray):
        """Calculates the informative categories
        """
        n_break = 0
        informative = np.zeros_like(information)
        pvalues = np.zeros_like(information)

        # iterate through most informative pathways
        pbar = tqdm(np.argsort(information)[::-1], desc="permutation testing")
        for idx in pbar:
            
            # calculate mutual information of random permutations
            permutations = calculate_mi_permutations(
                    exp_bins,
                    ont_bool[idx],
                    x_bins,
                    y_bins,
                    n=self.n_shuffle)
            
            # calculate empirical pvalue against randomization
            pval = empirical_pvalue(
                    permutations, 
                    information[idx])

            if pval > self.alpha:
                n_break += 1
                if n_break == self.k:
                    break
            else:
                informative[idx] = 1
                n_break = 0

        return (informative, pvalues)

    def _consolidate_pathways(
            self,
            informative: np.ndarray,
            pvalues: np.ndarray,
            exp_bins: np.ndarray,
            ont_bool: np.ndarray,
            x_bins: int,
            y_bins: int) -> np.ndarray:
        """Consolidate redundant pathways
        """
        existing = []
        inf_idx = np.flatnonzero(informative)

        # iterate through pvalues in ascending order
        pbar = tqdm(np.argsort(pvalues), desc="consolidating redundant pathways")
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
                        exp_bins, 
                        ont_bool[idx], 
                        ont_bool[e], 
                        x_bins, 
                        y_bins, 
                        y_bins)


            # accept if informative above all existing accepted pathways
            if np.all(all_ri >= self.r):
                existing.append(idx)
            else:
                pass
        
        return np.array(existing)

    def run(
            self,
            exp: ExpressionProfile,
            ont: GeneOntology) -> pd.DataFrame:
        """
        Perform the PAGE algorithm

        Parameters
        ==========
        exp: ExpressionProfile
            The ExpressionProfile to consider in the analysis
        ont: GeneOntology
            the GeneOntology to consider in the analysis

        Returns
        =======
        pd.DataFrame
            A dataframe representing the results of the analysis

        Examples
        ========
        >>> p = PAGE()
        >>> p.run(exp, ont)
        """
        shared_genes = self._intersect_genes(exp, ont)
        exp_bins, ont_bool = self._subset_matrices(exp, ont, shared_genes)

        x_bins = exp_bins.max() + 1
        y_bins = 2

        # calculate mutual information
        information = self._calculate_mutual_information(
                exp_bins,
                ont_bool,
                x_bins,
                y_bins)
        
        # select informative pathways
        informative, pvalues = self._calculate_informative(
                information, 
                exp_bins, 
                ont_bool, 
                x_bins,
                y_bins)

        # filter redundant pathways
        if self.filter_redundant:
            pathway_indices = self._consolidate_pathways(
                    informative,
                    pvalues,
                    exp_bins,
                    ont_bool,
                    x_bins,
                    y_bins)
        else:
            pathway_indices = np.flatnonzero(informative)

        # hypergeometric testing over selected pathways
        overrep_pvals, underrep_pvals = self._significance_testing(
                pathway_indices,
                exp_bins,
                ont_bool,
                x_bins,
                y_bins)

        results = self._gather_results(
                exp,
                ont,
                pathway_indices,
                overrep_pvals,
                underrep_pvals)

        return results
