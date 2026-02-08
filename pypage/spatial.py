"""Spatial statistics for single-cell analysis.

Geary's C autocorrelation on KNN graphs and KNN graph construction.
"""

import numpy as np
import numba as nb
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.neighbors import NearestNeighbors


@nb.jit(nopython=True, cache=True, nogil=True)
def geary_c(scores, W_indices, W_indptr, W_data, n):
    """Compute Geary's C for a single score vector on a sparse KNN graph.

    C = (n-1) * sum_ij w_ij (x_i - x_j)^2 / (2 * W * sum_i (x_i - x_bar)^2)

    Returns C' = 1 - C (consistency score: higher = more autocorrelated).

    Parameters
    ----------
    scores : np.ndarray
        1D array of shape (n,) with pathway scores per cell.
    W_indices : np.ndarray
        CSR indices array of the weight matrix.
    W_indptr : np.ndarray
        CSR indptr array of the weight matrix.
    W_data : np.ndarray
        CSR data array of the weight matrix.
    n : int
        Number of cells.

    Returns
    -------
    float
        C' = 1 - C (consistency score).
    """
    x_bar = 0.0
    for i in range(n):
        x_bar += scores[i]
    x_bar /= n

    # Denominator: variance term
    var_sum = 0.0
    for i in range(n):
        diff = scores[i] - x_bar
        var_sum += diff * diff

    if var_sum == 0.0:
        return 0.0

    # Numerator: weighted squared differences
    W_total = 0.0
    weighted_sq_diff = 0.0
    for i in range(n):
        start = W_indptr[i]
        end = W_indptr[i + 1]
        for idx in range(start, end):
            j = W_indices[idx]
            w = W_data[idx]
            W_total += w
            diff = scores[i] - scores[j]
            weighted_sq_diff += w * diff * diff

    if W_total == 0.0:
        return 0.0

    C = (n - 1) * weighted_sq_diff / (2.0 * W_total * var_sum)
    return 1.0 - C


@nb.jit(nopython=True, cache=True, nogil=True, parallel=True)
def geary_c_batch(score_matrix, W_indices, W_indptr, W_data, n):
    """Compute Geary's C for multiple score columns in parallel.

    Parameters
    ----------
    score_matrix : np.ndarray
        2D array of shape (n_cells, n_pathways).
    W_indices : np.ndarray
        CSR indices array of the weight matrix.
    W_indptr : np.ndarray
        CSR indptr array of the weight matrix.
    W_data : np.ndarray
        CSR data array of the weight matrix.
    n : int
        Number of cells.

    Returns
    -------
    np.ndarray
        1D array of shape (n_pathways,) with C' values.
    """
    n_pathways = score_matrix.shape[1]
    results = np.empty(n_pathways)
    for p in nb.prange(n_pathways):
        results[p] = geary_c(score_matrix[:, p], W_indices, W_indptr, W_data, n)
    return results


def build_knn_graph(X, k):
    """Build a KNN graph with Gaussian kernel weights.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_cells, n_features).
    k : int
        Number of neighbors.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse (n_cells, n_cells) symmetric weight matrix.
    """
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Remove self-connections (first column is the point itself)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Gaussian kernel: w_ij = exp(-d_ij^2 / sigma_i^2)
    # sigma_i = distance to k-th neighbor (last column)
    sigma = distances[:, -1].copy()
    sigma[sigma == 0] = 1e-10  # avoid division by zero

    W = lil_matrix((n, n))
    for i in range(n):
        for j_idx in range(k):
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            w = np.exp(-(d ** 2) / (sigma[i] ** 2))
            W[i, j] = w

    # Symmetrize: W = (W + W^T) / 2
    W = W.tocsr()
    W = (W + W.T) / 2.0
    return W
