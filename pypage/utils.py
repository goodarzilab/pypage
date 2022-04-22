"""Utility functions
"""

import numpy as np
import numba as nb
from scipy.stats import hypergeom


@nb.jit(nopython=True)
def contingency_table(
        bool_arr: np.ndarray,
        bool_set: np.ndarray) -> np.ndarray:
    """calculates the contingency table for a given gene set
    """
    C = np.zeros((2, bool_arr.shape[0]))
    for idx in range(bool_arr.shape[0]):
        ix = np.sum((bool_arr[idx] + bool_set) == 2)
        C[0][idx] = ix
        C[1][idx] = np.sum(bool_arr[idx]) - ix
    return C


@nb.jit(nopython=True)
def shuffle_bool_array(
        bool_arr: np.ndarray) -> np.ndarray:
    """Shuffles the genes for each observation independently
    """
    shuf = bool_arr.copy()
    for idx in range(bool_arr.shape[0]):
        np.random.shuffle(shuf[idx])
    return shuf


@nb.jit(nopython=True)
def empirical_pvalue(
        array: np.ndarray,
        value: float) -> float:
    """Calculates the empirical pvalue of a value given a distribution
    """
    m = array.max()
    if value > m:
        return 0.
    else:
        return np.mean(array > value)


def hypergeometric_test(
        e_bool: np.ndarray, 
        o_bool: np.ndarray) -> np.ndarray:
    """
    translating from: https://ars.els-cdn.com/content/image/1-s2.0-S1097276509008570-mmc1.pdf
    
    m = n: number of genes in the pathway
    x = k: number of genes in the bin which are also in the pathway
    n = N: number of genes in the bin
    N = M: total number of genes
    """
    n = o_bool.sum()
    k = (e_bool & o_bool).sum(axis=1)
    N = e_bool.sum(axis=1)
    M = o_bool.size

    # parameterize hypergeometric distribution
    hg = hypergeom(M, n, N)
    
    # measure overrepresentation
    sf = hg.sf(k)
    
    # measure underrepresentation
    cdf = hg.cdf(k)
    
    return np.stack([sf, cdf])
