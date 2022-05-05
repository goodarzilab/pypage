"""Testing Expression IO 
"""


import numpy as np
from pypage import ExpressionProfile


N_GENES=1000
N_BINS=5
T = 100


def get_expression() -> (np.ndarray, np.ndarray):
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    scores = np.random.normal(size=N_GENES)
    return genes, scores


def get_bins() -> (np.ndarray, np.ndarray):
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    bins = np.random.choice(N_BINS, size=N_GENES)
    return genes, bins


def test_load_expression_nobins():
    for _ in np.arange(T):
        genes, scores = get_expression()
        exp = ExpressionProfile(genes, scores)
        assert exp.n_genes == N_GENES
        assert exp.n_bins == 10


def test_load_expression():
    for _ in np.arange(T):
        genes, expression = get_expression()
        exp = ExpressionProfile(genes, expression, n_bins=N_BINS)
        assert exp.n_genes == N_GENES
        assert exp.n_bins == N_BINS


def test_load_expression_strategy_hist():
    for _ in np.arange(T):
        genes, expression = get_expression()

        # test hist binning
        hist_exp = ExpressionProfile(
                genes, 
                expression, 
                n_bins=N_BINS, 
                bin_strategy='hist')
        
        assert hist_exp.n_genes == N_GENES
        assert hist_exp.n_bins == N_BINS

        # test split binning
        split_exp = ExpressionProfile(
                genes, 
                expression, 
                n_bins=N_BINS, 
                bin_strategy='split')
        
        assert split_exp.n_genes == N_GENES
        assert split_exp.n_bins == N_BINS

        # assert bins are different
        assert np.any(hist_exp.bin_sizes != split_exp.bin_sizes)


def test_load_bins():
    genes, bins = get_bins()
    exp = ExpressionProfile(genes, bins, n_bins=N_BINS)
    assert exp.n_genes == N_GENES
    assert exp.n_bins == N_BINS
