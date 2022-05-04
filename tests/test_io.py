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

@pytest.fixture()
def expression_dataframe():
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    scores = np.random.normal(size=N_GENES)
    return pd.DataFrame({
        "gene": genes,
        "score": scores})

@pytest.fixture()
def bins_dataframe():
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    scores = np.random.choice(N_BINS, size=N_GENES)
    return pd.DataFrame({
        "gene": genes,
        "score": scores})

def test_load_expression_nobins(expression_dataframe):
    exp = ExpressionProfile(expression_dataframe)
    assert exp.n_genes == N_GENES
    assert exp.n_bins == 10

def test_load_expression(expression_dataframe):
    exp = ExpressionProfile(expression_dataframe, n_bins=N_BINS)
    assert exp.n_genes == N_GENES
    assert exp.n_bins == N_BINS

def test_load_bins(bins_dataframe):
    exp = ExpressionProfile(bins_dataframe, n_bins=N_BINS)
    assert exp.n_genes == N_GENES
    assert exp.n_bins == N_BINS


def test_load_ontology():
    ont = GeneOntology("example_data/GO_BP_2021_index.txt.gz")
