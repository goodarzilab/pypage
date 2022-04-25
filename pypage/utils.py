"""Utility functions
"""

import numpy as np
import numba as nb
from scipy.stats import hypergeom

@nb.jit(
    nopython=True, 
    nogil=True, 
    cache=True, 
    fastmath=True)
def hist1D(
        arr_a: np.ndarray, 
        bin_a: int) -> np.ndarray:
    """calculates the number of events in all bins 

    inputs:
        arr_a: np.ndarray
            a 1D array where each value represents a the cluster
            identity of a specific gene
        bin_a: np.ndarray
            the number of clusters in array 1. Should be equivalent
            to the maximum value in `arr_a` + 1

    outputs:
        ct: np.ndarray
            a 1 dimensional array where each index represents the
            number of events in each bin
    """
    ct = np.zeros((bin_a, 1), dtype=np.int32)
    for x in np.arange(arr_a.size):
        ct[arr_a[x]] += 1
    return ct


@nb.jit(
    nopython=True, 
    nogil=True, 
    cache=True, 
    fastmath=True)
def hist2D(
        arr_a: np.ndarray, 
        arr_b: np.ndarray, 
        bin_a: int, 
        bin_b: int) -> np.ndarray:
    """calculates the cluster intersections between two arrays

    e.g. ct[0][2]: 
    the intersection of `arr_a` bin 0 & `arr_b` bin 2

    inputs:
        arr_a: np.ndarray
            a 1D array where each value represents a the cluster
            identity of a specific gene
        arr_b: np.ndarray
            a 1D array where each value represents a the cluster
            identity of a specific gene
        bin_a: np.ndarray
            the number of clusters in array 1. Should be equivalent
            to the maximum value in `arr_a` + 1
        bin_b: np.ndarray
            the number of clusters in array 1. Should be equivalent
            to the maximum value in `arr_b` + 1

    outputs:
        ct: np.ndarray
            a 2 dimensional array where each index represents
            the intersection of the cluster identities in
            `arr_a` and `arr_b`
    """
    assert arr_a.size == arr_b.size
    ct = np.zeros((bin_a, bin_b), dtype=np.int32)
    for x in np.arange(arr_a.size):
        ct[arr_a[x]][arr_b[x]] += 1
    return ct

@nb.jit(
    nopython=True, 
    nogil=True, 
    cache=True, 
    fastmath=True)
def hist3D(
        arr_a: np.ndarray, 
        arr_b: np.ndarray, 
        arr_c: np.ndarray, 
        bin_a: int, 
        bin_b: int, 
        bin_c: int) -> np.ndarray:
    """calculates the bin intersections between three arrays

    e.g. ct[0][2][1]: 
    the intersection of `arr_a` bin 0 & `arr_b` bin 2 & `arr_c` bin 1

    inputs:
        arr_a: np.ndarray
            a 1D array where each value represents a the cluster
            identity of a specific gene
        arr_b: np.ndarray
            a 1d array where each value represents a the cluster
            identity of a specific gene
        arr_c: np.ndarray
            a 1d array where each value represents a the cluster
            identity of a specific gene
        bin_a: np.ndarray
            the number of clusters in array 1. Should be equivalent
            to the maximum value in `arr_a` + 1
        bin_b: np.ndarray
            the number of clusters in array 1. Should be equivalent
            to the maximum value in `arr_b` + 1
        bin_c: np.ndarray
            the number of clusters in array 1. Should be equivalent
            to the maximum value in `arr_c` + 1

    outputs:
        ct: np.ndarray
            a 3 dimensional array where each index represents
            the intersection of the cluster identities in
            `arr_a` and `arr_b` and `arr_c`
    """
    assert arr_a.size == arr_b.size
    assert arr_b.size == arr_c.size
    ct = np.zeros((bin_a, bin_b, bin_c), dtype=np.int32)
    for x in np.arange(arr_a.size):
        ct[arr_a[x]][arr_b[x]][arr_c[x]] += 1
    return ct

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
def shuffle_bin_array(
        bin_array: np.ndarray) -> np.ndarray:
    """Shuffles the bin identities for a provided bin array
    """
    shuf = bin_array.copy()
    np.random.shuffle(shuf)
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
        return np.mean(array >= value)


def hypergeometric_test(
        e_bins: np.ndarray, 
        o_bool: np.ndarray) -> np.ndarray:
    """
    translating from: https://ars.els-cdn.com/content/image/1-s2.0-S1097276509008570-mmc1.pdf
    
    m = n: number of genes in the pathway
    x = k: number of genes in the bin which are also in the pathway
    n = N: number of genes in the bin
    N = M: total number of genes
    """
    n_bins = e_bins.max() + 1
    ct = hist2D(e_bins, o_bool, n_bins, 2)
    n = o_bool.sum()
    k = ct[:, 1]
    N = ct.sum(axis=1)
    M = o_bool.size

    # parameterize hypergeometric distribution
    hg = hypergeom(M, n, N)
    
    # measure overrepresentation
    sf = hg.sf(k)
    
    # measure underrepresentation
    cdf = hg.cdf(k)
    
    return np.stack([sf, cdf])


def benjamini_hochberg(
        p: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg p-value correction for multiple hypothesis testing.
    https://stackoverflow.com/a/33532498
    """
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]
