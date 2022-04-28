"""Testing an example usage of the PAGE algorithm
"""

import pytest 
import pandas as pd
from pypage import (
        PAGE,
        ExpressionProfile, 
        GeneOntology)

@pytest.fixture()
def load_expression():
    frame = pd.read_csv(
        "example_data/AP2S1.tab.gz", 
        sep="\t",
        header=None,
        names=["gene", "bin"])
    return ExpressionProfile(frame)

@pytest.fixture()
def load_ontology():
    return GeneOntology("example_data/GO_BP_2021_index.txt.gz")


def test_run(load_expression, load_ontology):
    p = PAGE(n_shuffle=5, k=2)
    results = p.run(
            load_expression,
            load_ontology)

