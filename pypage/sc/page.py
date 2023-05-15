import pandas as pd
import numpy as np
from ..page import PAGE
from ..utils import empirical_pvalue, hypergeometric_test
from ..information import (
    mutual_information,
    calculate_mi_permutations,
    calculate_cmi_permutations,
    conditional_mutual_information)
from tqdm import tqdm


class scPAGE(PAGE):
    def _calculate_information(self, pathway_idx, cell_idx) -> np.ndarray:
        """Calculates mutual or conditional mutual information for each pathway
        """
        if self.function == 'mi':
            desc = "calculating mutual information"
        else:
            desc = "calculating conditional mutual information"

        if self.function == 'mi':
            information = mutual_information(
                self.exp_bins[cell_idx],
                self.ont_bool[pathway_idx],
                self.x_bins,
                self.y_bins,
                base=self.base)
        else:
            information = conditional_mutual_information(
                self.exp_bins[cell_idx],
                self.ont_bool[pathway_idx],
                self.membership_bins,
                self.x_bins,
                self.y_bins,
                self.z_bins,
                base=self.base)
        return information

    '''def _significance_testing(self, pathway_idx, cell_idx) -> (np.ndarray, np.ndarray):
        """
        Iterates through informative pathways to calculate hypergeometric pvalues
        """

        #overrep_pvals, underrep_pvals = hypergeometric_test(
        #    self.exp_bins[cell_idx],
        #    self.ont_bool[pathway_idx])
        pathway = self.ont_bool[pathway_idx]
        profile = self.exp_bins[cell_idx]
        pathway = pathway[np.argsort(profile)]

        s1 = sum(pathway[:len(pathway) // 4])
        s2 = sum(pathway[-len(pathway) // 4:])
        if 0.49 < s1 / (s1+s2) < 0.51:
            sign = 0
        elif s1 < s2:
            sign = 1
        else:
            sign = -1

        #log_overrep_pvals = np.log10(overrep_pvals)
        #log_underrep_pvals = np.log10(underrep_pvals)
        #graphical_ar = np.minimum(log_overrep_pvals, log_underrep_pvals)
        #graphical_ar[log_overrep_pvals < log_underrep_pvals] *= -1  # make overrepresented positive
        #n_bins = graphical_ar.shape[0]
        #sign = graphical_ar[:n_bins // 2].sum() <= graphical_ar[n_bins // 2:].sum()
        #sign = sign.astype(int)
        #if sign == 0:
        #    sign = -1
        return sign'''

    def _estimate_uncertainty(self, information, pathway_idx, cell_idx) -> (np.ndarray, np.ndarray):

        # calculate mutual information of random permutations
        if self.function == 'mi':
            permutations = calculate_mi_permutations(
                self.exp_bins[cell_idx],
                self.ont_bool[pathway_idx],
                self.x_bins,
                self.y_bins,
                n=self.n_shuffle)
        else:
            permutations = calculate_cmi_permutations(
                self.exp_bins[cell_idx],
                self.ont_bool[pathway_idx],
                self.membership_bins,
                self.x_bins,
                self.y_bins,
                self.z_bins,
                n=self.n_shuffle)

        # calculate empirical pvalue against randomization

        pvalue = empirical_pvalue(
            permutations,
            information)

        return pvalue

    def run(self, pathway, estimate_uncertainty=False) -> (pd.DataFrame):
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
        assert pathway in self.ontology.pathways, 'Pathway not found'
        pathway_idx = np.where(self.ontology.pathways == pathway)[0][0]

        info_vals = np.zeros(len(self.exp_bins))
        # sign = np.zeros(len(self.exp_bins))
        uncertainty_vals = np.zeros(len(self.exp_bins))
        for cell_idx in np.arange(len(self.exp_bins)):
            info_vals[cell_idx] = self._calculate_information(pathway_idx, cell_idx)
            # sign[cell_idx] = self._significance_testing(pathway_idx, cell_idx)
            if estimate_uncertainty:
                uncertainty_vals[cell_idx] = self._estimate_uncertainty(info_vals[cell_idx], pathway_idx, cell_idx)

        info_scores = (info_vals - np.mean(info_vals)) / np.std(info_vals)  # * sign

        if estimate_uncertainty:
            return info_scores, info_vals, uncertainty_vals
        else:
            return info_scores, info_vals
