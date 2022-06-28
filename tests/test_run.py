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


def test_init(load_expression, load_ontology):
    p = PAGE(
        load_expression,
        load_ontology,
        n_shuffle=5,
        k=2)


def test_run(load_expression, load_ontology):
    p = PAGE(
        load_expression, 
        load_ontology, 
        n_shuffle=5, 
        k=2)
    results = p.run()


def test_norun_heatmap(load_expression, load_ontology):
    p = PAGE(
        load_expression, 
        load_ontology, 
        n_shuffle=5, 
        k=2)

    try:
        p.heatmap()
        assert False
    except AttributeError:
        assert True


def test_empty_heatmap(load_expression, load_ontology):
    p = PAGE(
        load_expression, 
        load_ontology, 
        n_shuffle=5, 
        k=2)
    p.run()

    # set empty dataframe
    p.results = pd.DataFrame([])

    try:
        p.heatmap()
        assert False
    except ValueError:
        assert True


