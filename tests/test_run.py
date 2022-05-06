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

    return ExpressionProfile(
            frame.iloc[:, 0], 
            frame.iloc[:, 1], 
            is_bin=True)

@pytest.fixture()
def load_ontology():
    frame = pd.read_csv(
            "example_data/GO_BP_2021_index.txt.gz",
            sep="\t",
            header=None,
            names=["gene", "pathway"])

    return GeneOntology(
            frame.iloc[:, 0],
            frame.iloc[:, 1])


def test_run(load_expression, load_ontology):
    p = PAGE(
        load_expression, 
        load_ontology, 
        n_shuffle=5, 
        k=2)
    results = p.run()

