"""Implementation of the PAGE Algorithm
"""

from .io import (
        GeneOntology, 
        ExpressionProfile)
from .utils import (
        contingency_table, 
        shuffle_bool_array,
        empirical_pvalue,
        hypergeometric_test)
from .information import mutual_information

import numpy as np
from tqdm import tqdm


class PAGE:
    def __init__(
            self,
            n_shuffle: int = 1e4,
            alpha: float = 5e-3,
            k: int = 20):
        """
        """
        self.n_shuffle = int(n_shuffle)
        self.alpha = float(alpha)
        self.k = int(k)

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
        exp_bool = exp.get_gene_subset(ix)
        ont_bool = ont.get_gene_subset(ix)
        return exp_bool, ont_bool

    def _build_contingency(
            self,
            exp_bool: np.ndarray,
            ont_bool: np.ndarray) -> np.ndarray:
        """creates a contingency table tensor for each pathway
        """
        num_pathways = ont_bool.shape[0]
        num_bins = exp_bool.shape[0]
        cont_tensor = np.zeros((num_pathways, 2, num_bins))
        for idx in tqdm(range(num_pathways), desc="building contingency tables"):
            cont_tensor[idx] = contingency_table(exp_bool, ont_bool[idx])
        return cont_tensor

    def _calculate_mutual_information(
            self,
            cont_tensor: np.ndarray) -> np.ndarray:
        """Calculates the mutual information for each pathway
        """
        num_pathways = cont_tensor.shape[0]
        information = np.zeros(num_pathways)
        for idx in tqdm(range(num_pathways), desc="calculating mutual information"):
            information[idx] = mutual_information(cont_tensor[idx])
        return information

    def _permutation_test_mi(
            self,
            exp_bool: np.ndarray,
            ont_bool: np.ndarray,
            information: float) -> float:
        """performs a permutation test and calculates an empirical p-value
        """
        mi_shuf = np.zeros(self.n_shuffle)
        for idx in np.arange(self.n_shuffle):
            b_shuf = shuffle_bool_array(exp_bool)
            c_shuf = contingency_table(b_shuf, ont_bool)
            mi_shuf[idx] = mutual_information(c_shuf)
        emp_pval = empirical_pvalue(mi_shuf, information)
        return emp_pval

    def _filter_informative(
            self,
            exp_bool: np.ndarray,
            ont_bool: np.ndarray,
            information: np.ndarray):
        """iterates through most informative pathways to perform permutation testing
        """
        # sort I by most informative
        qvals = np.argsort(information)[::-1]
        
        informative = []
        uninformative = 0
        pbar = tqdm(qvals, desc="Permutation Tests")
        for q_idx in pbar:

            pval = self._permutation_test_mi(
                exp_bool,
                ont_bool[q_idx],
                information[q_idx])

            # gene set is sufficiently informative
            if pval <= self.alpha:
                informative.append(q_idx)
                q_idx += 1
                uninformative = 0

            # gene set in uninformative
            else:
                uninformative += 1
                if uninformative == self.k:
                    pbar.set_description(
                        desc=f"Permutation Tests: {self.k} uninformative pathways reached")
                    break

        return np.array(informative)

    def _identify_informative(
            self,
            exp: ExpressionProfile,
            ont: GeneOntology):
        """Identifies the informative pathways
        """
        shared_genes = self._intersect_genes(exp, ont)
        exp_bool, ont_bool = self._subset_matrices(exp, ont, shared_genes)
        cont_tensor = self._build_contingency(exp_bool, ont_bool)
        information = self._calculate_mutual_information(cont_tensor)
        informative = self._filter_informative(exp_bool, ont_bool, information)
        return informative, exp_bool, ont_bool

    def _significance_testing(
            self,
            informative: np.ndarray,
            exp_bool: np.ndarray,
            ont_bool: np.ndarray):
        """
        Iterates through informative pathways to calculate hypergeometric pvalues
        """
        overrep_pvals = np.zeros((exp_bool.shape[0], informative.size))
        underrep_pvals = np.zeros_like(overrep_pvals)

        pbar = tqdm(enumerate(informative), desc="hypergeometric tests")
        for idx, info_idx in pbar:
            pvals = hypergeometric_test(
                    exp_bool, 
                    ont_bool[info_idx])
            overrep_pvals[:, idx] = pvals[0]
            underrep_pvals[:, idx] = pvals[1]

        return overrep_pvals, underrep_pvals


    def run(
            self,
            exp: ExpressionProfile,
            ont: GeneOntology):
        """
        """
        informative, e_bool, o_bool = self._identify_informative(exp, ont)
        overrep_pvals, underrep_pvals = self._significance_testing(informative, e_bool, o_bool)
        return overrep_pvals, underrep_pvals



