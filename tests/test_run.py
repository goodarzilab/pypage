"""Testing an example usage of the PAGE algorithm
"""

import pytest 
import pandas as pd
from pypage import (
    PAGE,
    ExpressionProfile,
    GeneSets)

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

    return GeneSets(
            frame.iloc[:, 0],
            frame.iloc[:, 1])


def test_run(load_expression, load_ontology):
    p = PAGE(
        load_expression, 
        load_ontology, 
        n_shuffle=5, 
        k=2)
    results = p.run()


def test_run_reuse_inputs(load_expression, load_ontology):
    p1 = PAGE(
        load_expression,
        load_ontology,
        n_shuffle=2,
        k=1)
    p1.run()

    p2 = PAGE(
        load_expression,
        load_ontology,
        n_shuffle=2,
        k=1)
    results = p2.run()
    assert isinstance(results, tuple)
