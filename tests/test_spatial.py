"""Tests for spatial statistics module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pypage.spatial import geary_c, geary_c_batch, build_knn_graph


@pytest.fixture
def simple_graph():
    """A small 4-node graph with known weights."""
    # 4 nodes in a line: 0-1-2-3
    # Weights: w(0,1)=1, w(1,2)=1, w(2,3)=1
    row = [0, 1, 1, 2, 2, 3]
    col = [1, 0, 2, 1, 3, 2]
    data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    W = csr_matrix((data, (row, col)), shape=(4, 4))
    return W


def test_geary_c_perfect_autocorrelation(simple_graph):
    """When neighbors have identical values, C should be 0 (C' = 1)."""
    W = simple_graph
    # All same value => no variance => returns 0
    scores = np.array([5.0, 5.0, 5.0, 5.0])
    c_prime = geary_c(scores, W.indices, W.indptr, W.data.astype(np.float64), 4)
    assert c_prime == 0.0  # zero variance case


def test_geary_c_smooth_gradient(simple_graph):
    """Smoothly varying scores on a line graph should give positive C'."""
    W = simple_graph
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    c_prime = geary_c(scores, W.indices, W.indptr, W.data.astype(np.float64), 4)
    # For a smooth gradient, C < 1, so C' > 0
    assert c_prime > 0.0


def test_geary_c_alternating_pattern(simple_graph):
    """Alternating high/low values should give negative C' (C > 1)."""
    W = simple_graph
    scores = np.array([10.0, 0.0, 10.0, 0.0])
    c_prime = geary_c(scores, W.indices, W.indptr, W.data.astype(np.float64), 4)
    # Alternating pattern: neighbors are maximally different, C > 1, C' < 0
    assert c_prime < 0.0


def test_geary_c_known_value():
    """Test Geary's C against a hand-calculated value."""
    # 3 nodes in a triangle, all weights = 1
    row = [0, 0, 1, 1, 2, 2]
    col = [1, 2, 0, 2, 0, 1]
    data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    W = csr_matrix((data, (row, col)), shape=(3, 3))

    scores = np.array([1.0, 2.0, 3.0])
    # x_bar = 2.0
    # var_sum = (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2
    # W_total = 6.0
    # weighted_sq_diff = w01*(1-2)^2 + w02*(1-3)^2 + w10*(2-1)^2 + w12*(2-3)^2
    #                  + w20*(3-1)^2 + w21*(3-2)^2
    #                  = 1 + 4 + 1 + 1 + 4 + 1 = 12
    # C = (3-1) * 12 / (2 * 6 * 2) = 24 / 24 = 1.0
    # C' = 1 - 1 = 0.0
    c_prime = geary_c(scores, W.indices, W.indptr, W.data.astype(np.float64), 3)
    assert abs(c_prime - 0.0) < 1e-10


def test_geary_c_batch(simple_graph):
    """Batch computation should match individual calls."""
    W = simple_graph
    n = 4
    scores1 = np.array([1.0, 2.0, 3.0, 4.0])
    scores2 = np.array([10.0, 0.0, 10.0, 0.0])
    score_matrix = np.column_stack([scores1, scores2])

    batch_results = geary_c_batch(
        score_matrix, W.indices, W.indptr, W.data.astype(np.float64), n
    )

    c1 = geary_c(scores1, W.indices, W.indptr, W.data.astype(np.float64), n)
    c2 = geary_c(scores2, W.indices, W.indptr, W.data.astype(np.float64), n)

    assert abs(batch_results[0] - c1) < 1e-10
    assert abs(batch_results[1] - c2) < 1e-10


def test_build_knn_graph_shape():
    """KNN graph should have correct shape and be symmetric."""
    np.random.seed(42)
    X = np.random.randn(50, 10)
    k = 5
    W = build_knn_graph(X, k)

    assert W.shape == (50, 50)
    # Check symmetry
    diff = abs(W - W.T)
    assert diff.max() < 1e-10
    # Check non-negative weights
    assert W.data.min() >= 0


def test_build_knn_graph_connectivity():
    """Each node should have at least k neighbors."""
    np.random.seed(42)
    X = np.random.randn(30, 5)
    k = 3
    W = build_knn_graph(X, k)

    # Each row should have at least k nonzero entries
    # (could have more due to symmetrization)
    for i in range(30):
        nnz = W[i].nnz
        assert nnz >= k


def test_build_knn_graph_weights_positive():
    """All weights should be positive (Gaussian kernel)."""
    np.random.seed(42)
    X = np.random.randn(20, 3)
    k = 4
    W = build_knn_graph(X, k)
    assert np.all(W.data > 0)


def test_geary_c_with_knn_graph():
    """Integration test: build graph and compute Geary's C."""
    np.random.seed(42)
    n = 100
    # Two clusters with different scores
    X = np.vstack([np.random.randn(50, 5) - 2, np.random.randn(50, 5) + 2])
    scores = np.concatenate([np.ones(50) * 0.0, np.ones(50) * 1.0])

    W = build_knn_graph(X, k=10)
    c_prime = geary_c(
        scores, W.indices, W.indptr, W.data.astype(np.float64), n
    )

    # Two well-separated clusters with consistent scores should have high C'
    assert c_prime > 0.5


def test_geary_c_random_scores():
    """Random scores on a structured graph should have C' near 0."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 5)
    scores = np.random.randn(n)

    W = build_knn_graph(X, k=10)
    c_prime = geary_c(
        scores, W.indices, W.indptr, W.data.astype(np.float64), n
    )

    # Random scores should give C' near 0 (not strongly positive or negative)
    assert abs(c_prime) < 0.3
