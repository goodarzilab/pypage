"""Testing Expression IO 
"""


import numpy as np
import pandas as pd
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


def test_load_expression_assertion():
    for i in np.arange(T):
        genes, expression = get_expression()

        if i % 2 == 0:
            genes = genes[np.random.random(N_GENES) < 0.3]
        else:
            expression = expression[np.random.random(N_GENES) < 0.3]

        try:
            ont = ExpressionProfile(genes, expression)
        except AssertionError:
            continue

        assert False


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


def test_subsetting():
    for _ in np.arange(T):
        genes, expression = get_expression()

        exp = ExpressionProfile(genes, expression)
        subset = genes[np.random.random(genes.size) < 0.5]

        bin_sub = exp.get_gene_subset(subset)
        assert bin_sub.size == subset.size


def test_expression_conversion():
    df = pd.read_csv('example_data/bladder_refseq.tsv.gz',
                     sep="\t",
                     header=0,
                     names=["gene", "exp"])
    exp = ExpressionProfile(df.iloc[:, 0],
                            df.iloc[:, 1],
                            bin_strategy='split',
                            n_bins=20)
    exp.convert_from_to('refseq', 'ensg', 'human')
