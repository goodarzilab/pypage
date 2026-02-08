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
        k=1,
        filter_redundant=False)
    p1.run()

    p2 = PAGE(
        load_expression,
        load_ontology,
        n_shuffle=2,
        k=1,
        filter_redundant=False)
    results = p2.run()
    assert isinstance(results, tuple)


def test_run_manual(load_expression, load_ontology):
    p = PAGE(
        load_expression,
        load_ontology,
        n_shuffle=5,
        k=2)
    # Pick first 3 pathways
    pathways = list(p.ontology.pathways[:3])
    results, hm = p.run_manual(pathways)
    assert len(results) == 3
    assert set(results['pathway']) == set(pathways)
    assert hm is not None


def test_run_manual_unknown_pathway(load_expression, load_ontology):
    p = PAGE(
        load_expression,
        load_ontology,
        n_shuffle=5,
        k=2)
    import pytest
    with pytest.raises(ValueError, match="Unknown pathway"):
        p.run_manual(['nonexistent_pathway_xyz'])
