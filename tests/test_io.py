"""Testing IO 
"""

import pytest
import pandas as pd
from pypage.io import (
        ExpressionProfile,
        GeneOntology)

@pytest.fixture()
def load_expression_dataframe():
    return pd.read_csv(
        "example_data/AP2S1.tab.gz", 
        sep="\t",
        header=None,
        names=["gene", "bin"])


def test_load_expression(load_expression_dataframe):
    frame = load_expression_dataframe
    exp = ExpressionProfile(frame)
    assert exp.n_genes == frame.gene.unique().size
    assert exp.n_bins == frame.bin.unique().size


def test_load_ontology():
    ont = GeneOntology("example_data/GO_BP_2021_index.txt.gz")
