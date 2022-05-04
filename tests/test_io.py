"""Testing IO 
"""

import pytest
import numpy as np
import pandas as pd
from pypage import (
        ExpressionProfile,
        GeneOntology)

N_GENES=1000
N_BINS=5

def get_expression() -> (np.ndarray, np.ndarray):
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    scores = np.random.normal(size=N_GENES)
    return genes, scores

def get_bins() -> (np.ndarray, np.ndarray):
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    bins = np.random.choice(N_BINS, size=N_GENES)
    return genes, bins

def test_load_expression_nobins():
    genes, scores = get_expression()
    exp = ExpressionProfile(genes, scores)
    assert exp.n_genes == N_GENES
    assert exp.n_bins == 10

def test_load_expression():
    genes, bins = get_bins()
    exp = ExpressionProfile(genes, bins, n_bins=N_BINS)
    assert exp.n_genes == N_GENES
    assert exp.n_bins == N_BINS

def test_load_expression_strategy():
    genes, bins = get_bins()

    hist_exp = ExpressionProfile(genes, bins, n_bins=N_BINS, bin_strategy='hist')
    assert hist_exp.n_genes == N_GENES
    assert hist_exp.n_bins == N_BINS

    split_exp = ExpressionProfile(genes, bins, n_bins=N_BINS, bin_strategy='split')
    assert split_exp.n_genes == N_GENES
    assert split_exp.n_bins == N_BINS

    assert np.any(hist_exp.bin_sizes != split_exp.bin_sizes)

def test_load_bins():
    genes, bins = get_bins()
    exp = ExpressionProfile(genes, bins, n_bins=N_BINS)
    assert exp.n_genes == N_GENES
    assert exp.n_bins == N_BINS


def test_load_ontology():
    ont = GeneOntology("example_data/GO_BP_2021_index.txt.gz")
