"""Implementation of Information Utils
"""

import math
import numpy as np
import numba as nb
from typing import Optional

from .hist import (
        hist1D,
        hist2D,
        hist3D)
from .utils import (
        shuffle_bin_array)


@nb.jit(
    cache=True,
    nogil=True,
    nopython=True)
def entropy(
        X: np.ndarray,
        x_bins: int,
        base: int = 2) -> float:
    """Calculates the empirical entropy of an array.
    
    Calculated using the form:

    .. math::
        H(X) = -\sum_{i=1}^{n}{ P( X_{i} ) \log{P(X_{i})} }
        
    
    Parameters
    ----------
    X: np.ndarray
        The unquantized array to calculate entropy over
    x_bins: int
        The number of bins to split the array into
    base: int
        The base of the logarithm

    Returns
    -------
    float
        The calculated entropy: H(X)
    """
    c_x = hist1D(X, x_bins).ravel()
    p_x = c_x / c_x.sum()
    log_base = np.log(base)
    info = 0.
    for i in range(x_bins):
        if p_x[i] == 0:
            continue
        info -= p_x[i] * np.log(p_x[i]) / log_base
    return info


@nb.jit(
    cache=True,
    nogil=True,
    nopython=True)
def joint_entropy(
        X: np.ndarray,
        Y: np.ndarray,
        x_bins: int,
        y_bins: int,
        base: int = 2) -> float:
    """Calculates the joint entropy of two random variables
    
    Calculated using the form:

    .. math::
        H(X,Y) = - \sum_{x∈X}\sum_{y∈Y} P(x,y) \log{P(x,y)}

    Parameters
    ----------
    X: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Y: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    x_bins: int 
        the number of bins in `X`. equivalent to `max(X) + 1`
    y_bins: int,
        the number of bins in `Y`. equivalent to `max(Y) + 1`
    base: int
        the base of the logarithm

    Returns
    -------
    float
        The calculated joint entropy H(X,Y)
    """
    c_xy = hist2D(X, Y, x_bins, y_bins)
    p_xy = c_xy / c_xy.sum()
    log_base = np.log(base)
    info = 0.
    for x in range(x_bins):
        for y in range(y_bins):
            if p_xy[x][y] == 0:
                continue
            info -= p_xy[x][y] * np.log(p_xy[x][y]) / log_base
    return info


@nb.jit(
    cache=True,
    nogil=True,
    nopython=True)
def joint_entropy_3d(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        x_bins: int,
        y_bins: int,
        z_bins: int,
        base: int = 2) -> float:
    """Calculates the joint entropy of two random variables

    Calculated using the form:

    .. math::
        H(X,Y,Z) = - \sum_{x∈X}\sum_{y∈Y}\sum_{z∈Z} P(x,y,z) \log{P(x,y,z)}

    Parameters
    ----------
    X: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Y: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Z: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    x_bins: int 
        the number of bins in `X`. equivalent to `max(X) + 1`
    y_bins: int,
        the number of bins in `Y`. equivalent to `max(Y) + 1`
    z_bins: int,
        the number of bins in `Z`. equivalent to `max(Z) + 1`
    base: int
        the base of the logarithm

    Returns
    -------
    float
        The calculated joint entropy
    """
    c_xyz = hist3D(X, Y, Z, x_bins, y_bins, z_bins)
    p_xyz = c_xyz / c_xyz.sum()
    log_base = np.log(base)
    info = 0.
    for x in range(x_bins):
        for y in range(y_bins):
            for z in range(z_bins):
                if p_xyz[x][y][z] == 0:
                    continue
                info -= p_xyz[x][y][z] \
                        * np.log( p_xyz[x][y][z] ) \
                        / log_base

    return info


@nb.jit(
    cache=True,
    nogil=True,
    nopython=True)
def conditional_entropy(
        X: np.ndarray,
        Y: np.ndarray,
        x_bins: int,
        y_bins: int,
        base: int = 2) -> float:
    """Calculates the conditional entropy of two random variables

    Calculated using the form:

    .. math::
        H(X \mid Y) = - \sum_{x∈X,y∈Y} P(x,y) log{ \\frac {P(x,y)} {P(y)} }

    Parameters
    ----------
    X: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Y: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    x_bins: int 
        the number of bins in `X`. equivalent to `max(X) + 1`
    y_bins: int,
        the number of bins in `Y`. equivalent to `max(Y) + 1`
    base: int
        the base of the logarithm

    Returns
    -------
    float
        The calculated conditional entropy
    """
    c_xy = hist2D(X, Y, x_bins, y_bins)
    c_y = c_xy.sum(axis=0)

    p_xy = c_xy / c_xy.sum()
    p_y = c_y / c_y.sum()

    log_base = np.log(base)
    info = 0.
    for x in range(x_bins):
        for y in range(y_bins):
            if p_xy[x][y] == 0:
                continue
            info -= p_xy[x][y] *\
                    np.log( (p_xy[x][y]) / p_y[y] ) \
                    / log_base
    return info


@nb.jit(
    cache=True,
    nogil=True,
    nopython=True)
def mutual_information(
        X: np.ndarray,
        Y: np.ndarray,
        x_bins: int,
        y_bins: int,
        base: int = 2) -> float:
    """Calculates mutual information for two arrays. 
    
    Calculated using the form:

    .. math::
        I(X;Y) = \sum_{x∈X,y∈Y} P_{X,Y}(x,y) \log{ \\frac {P_{X,Y}(x,y)} {P_X(x)P_Y(y)} }

    Parameters
    ----------
    X: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Y: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    x_bins: int 
        the number of bins in `X`. equivalent to `max(X) + 1`
    y_bins: int,
        the number of bins in `Y`. equivalent to `max(Y) + 1`
    base: int
        the base of the logarithm

    Returns
    -------
    float
        The calculated mutual information
    """
    c_xy = hist2D(X, Y, x_bins, y_bins)
    c_x = c_xy.sum(axis=1)
    c_y = c_xy.sum(axis=0)

    p_xy = c_xy / c_xy.sum()
    p_x = c_x / c_x.sum()
    p_y = c_y / c_y.sum()

    log_base = np.log(base)
    info = 0.
    for x in range(x_bins):
        for y in range(y_bins):
            if p_xy[x][y] == 0:
                continue
            info += p_xy[x][y] * np.log(p_xy[x][y] / (p_x[x] * p_y[y])) / log_base

    return info


@nb.jit(
    cache=True,
    nogil=True,
    nopython=True)
def conditional_mutual_information(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        x_bins: int,
        y_bins: int,
        z_bins: int,
        base: int = 2) -> float:
    """Calculates conditional mutual information for three arrays.
    
    Calculated using the form:

    .. math::
        I(X;Y \mid Z) = \sum_{x∈X,y∈Y,z∈Z} P_{X,Y,Z}(x,y,z) log { \\frac {P_Z(z)P_{X,Y,Z}(x,y,z)} {P_{X,Z}(x,z)P_{Y,Z}(y,z)}}

    Parameters
    ----------
    X: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Y: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Z: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    x_bins: int 
        the number of bins in `X`. equivalent to `max(X) + 1`
    y_bins: int,
        the number of bins in `Y`. equivalent to `max(Y) + 1`
    z_bins: int,
        the number of bins in `Z`. equivalent to `max(Z) + 1`
    base: int
        the base of the logarithm

    Returns
    -------
    float
        The calculated conditional mutual information
    """

    c_xyz = hist3D(X, Y, Z, x_bins, y_bins, z_bins)

    # Derive marginals from c_xyz via axis summation (O(bins) instead of O(n))
    c_xz = np.zeros((x_bins, z_bins), dtype=np.int32)
    for x in range(x_bins):
        for z in range(z_bins):
            for y in range(y_bins):
                c_xz[x, z] += c_xyz[x, y, z]
    c_yz = np.zeros((y_bins, z_bins), dtype=np.int32)
    for y in range(y_bins):
        for z in range(z_bins):
            for x in range(x_bins):
                c_yz[y, z] += c_xyz[x, y, z]
    c_z = np.zeros(z_bins, dtype=np.int32)
    for z in range(z_bins):
        for x in range(x_bins):
            for y in range(y_bins):
                c_z[z] += c_xyz[x, y, z]

    n = c_xyz.sum()
    p_xyz = c_xyz / n
    p_xz = c_xz / n
    p_yz = c_yz / n
    p_z = c_z / n


    log_base = np.log(base)
    info = 0.
    for x in range(x_bins):
        for y in range(y_bins):
            for z in range(z_bins):
                if p_xyz[x][y][z] == 0:
                    continue
                numer = p_z[z] * p_xyz[x][y][z]
                denom = p_xz[x][z] * p_yz[y][z]
                info += p_xyz[x][y][z] * np.log(numer / denom) / log_base
    return info


@nb.jit(
    cache=True,
    nogil=True,
    nopython=True,
    fastmath=True,
    parallel=True)
def calculate_mi_permutations(
        X: np.ndarray, 
        Y: np.ndarray,
        x_bins: int,
        y_bins: int,
        base: int = 2,
        n: int = 10000) -> np.ndarray:
    """calculates the MI for `n` permutations of X

    Parameters
    ----------
    X: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Y: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    x_bins: int 
        the number of bins in `X`. equivalent to `max(X) + 1`
    y_bins: int,
        the number of bins in `Y`. equivalent to `max(Y) + 1`
    base: int
        the base of the logarithm
    n: int
        the number of permutations to perform (default = 10000)

    Returns
    -------
    np.ndarray 
        The calculated joint entropy for each of the permutations
    """
    permutations = np.zeros(n)
    for idx in nb.prange(n):
        tmp_X = shuffle_bin_array(X)
        permutations[idx] = mutual_information(
                tmp_X, Y, x_bins, y_bins, base=base)
    return permutations

@nb.jit(
    cache=True,
    nogil=True,
    nopython=True,
    fastmath=True,
    parallel=True)
def calculate_cmi_permutations(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        x_bins: int,
        y_bins: int,
        z_bins: int,
        base: int = 2,
        n: int = 10000) -> np.ndarray:
    """calculates the MI for `n` permutations of X

    Parameters
    ----------
    X: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Y: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Z: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    x_bins: int
        the number of bins in `X`. equivalent to `max(X) + 1`
    y_bins: int,
        the number of bins in `Y`. equivalent to `max(Y) + 1`
    z_bins: int,
        the number of bins in `Z`. equivalent to `max(Z) + 1`
    base: int
        the base of the logarithm
    n: int
        the number of permutations to perform (default = 10000)

    Returns
    -------
    np.ndarray
        The calculated CMI for each of the permutations
    """
    permutations = np.zeros(n)
    for idx in nb.prange(n):
        tmp_X = shuffle_bin_array(X)
        permutations[idx] = conditional_mutual_information(
            tmp_X, Y, Z, x_bins, y_bins, z_bins, base=base)
    return permutations

@nb.jit(
    cache=True,
    nogil=True,
    nopython=True,
    fastmath=True)
def measure_redundancy(
        X: np.ndarray, 
        Y: np.ndarray,
        Z: np.ndarray,
        x_bins: int,
        y_bins: int,
        z_bins: int,
        base: int = 2) -> float:
    """Measures the redundancy of a pathway via a ratio of
    conditional mutual information and mutual information

    calculated using the form:

    .. math::
        R_i = \\frac {I(Y; X \mid Z)} {I(Y;Z)}

    where:
        X: Gene Expression

        Y: Candidate Pathway
        
        Z: Accepted Pathway

    Parameters
    ----------
    X: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Y: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    Z: np.ndarray
        a 1D array where each value represents the bin index
        for a gene
    x_bins: int 
        the number of bins in `X`. equivalent to `max(X) + 1`
    y_bins: int,
        the number of bins in `Y`. equivalent to `max(Y) + 1`
    z_bins: int,
        the number of bins in `Z`. equivalent to `max(Z) + 1`
    base: int
        the base of the logarithm

    Returns
    -------
    float
        The calculated R-value
    """
    cmi = conditional_mutual_information(
            Y, 
            X, 
            Z, 
            y_bins, 
            x_bins, 
            z_bins,
            base=base)

    mi = mutual_information(
            Y,
            Z,
            y_bins,
            z_bins,
            base=base)

    if mi <= 0:
        return 1e12  # independent pathways have no redundancy → always accept
    return cmi / mi


@nb.jit(
    cache=True,
    nogil=True,
    nopython=True,
    parallel=True)
def batch_mutual_information(
        exp_bins: np.ndarray,
        ont_bool: np.ndarray,
        x_bins: int,
        y_bins: int,
        base: int = 2) -> np.ndarray:
    """Calculates mutual information for all pathways in parallel.

    Parameters
    ----------
    exp_bins: np.ndarray
        a 1D array of expression bin indices (n_genes,)
    ont_bool: np.ndarray
        a 2D boolean/int array of pathway memberships (n_pathways, n_genes)
    x_bins: int
        the number of expression bins
    y_bins: int
        the number of ontology bins (typically 2)
    base: int
        the base of the logarithm

    Returns
    -------
    np.ndarray
        MI values for each pathway (n_pathways,)
    """
    n = ont_bool.shape[0]
    result = np.zeros(n)
    for i in nb.prange(n):
        result[i] = mutual_information(exp_bins, ont_bool[i], x_bins, y_bins, base)
    return result


@nb.jit(
    cache=True,
    nogil=True,
    nopython=True,
    parallel=True)
def batch_conditional_mutual_information(
        exp_bins: np.ndarray,
        ont_bool: np.ndarray,
        membership_bins: np.ndarray,
        x_bins: int,
        y_bins: int,
        z_bins: int,
        base: int = 2) -> np.ndarray:
    """Calculates conditional mutual information for all pathways in parallel.

    Parameters
    ----------
    exp_bins: np.ndarray
        a 1D array of expression bin indices (n_genes,)
    ont_bool: np.ndarray
        a 2D boolean/int array of pathway memberships (n_pathways, n_genes)
    membership_bins: np.ndarray
        a 1D array of membership bin indices (n_genes,)
    x_bins: int
        the number of expression bins
    y_bins: int
        the number of ontology bins (typically 2)
    z_bins: int
        the number of membership bins
    base: int
        the base of the logarithm

    Returns
    -------
    np.ndarray
        CMI values for each pathway (n_pathways,)
    """
    n = ont_bool.shape[0]
    result = np.zeros(n)
    for i in nb.prange(n):
        result[i] = conditional_mutual_information(
            exp_bins, ont_bool[i], membership_bins,
            x_bins, y_bins, z_bins, base)
    return result

