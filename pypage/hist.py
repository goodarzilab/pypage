"""Histogram functions
"""

import numpy as np
import numba as nb


@nb.jit(
    nopython=True, 
    nogil=True, 
    cache=True, 
    fastmath=True)
def hist1D(
        arr_a: np.ndarray, 
        bin_a: int) -> np.ndarray:
    """calculates the number of events in all bins 

    Parameters
	----------
    arr_a: np.ndarray
        a 1D array where each value represents a the cluster
        identity of a specific gene
    bin_a: np.ndarray
        the number of clusters in array 1. Should be equivalent
        to the maximum value in `arr_a` + 1

    Returns
	-------
    np.ndarray
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

    Parameters
	----------
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

    Returns
	-------
    np.ndarray
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

    Parameters
	----------
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

    Returns
	-------
    np.ndarray
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
