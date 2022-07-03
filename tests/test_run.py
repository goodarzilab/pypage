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


@pytest.fixture()
def load_expression_bladder():
    df = pd.read_csv('example_data/bladder_refseq.tsv.gz',
                     sep="\t",
                     header=0,
                     names=["gene", "exp"])
    exp = ExpressionProfile(df.iloc[:, 0],
                             df.iloc[:, 1],
                            n_bins=20)
    exp.convert_from_to('refseq', 'ensg', 'human')
    return exp


@pytest.fixture()
def load_ontology_cistrome():
    ont = GeneOntology(ann_file='example_data/hg38_cistrome_index_truncated.txt.gz')
    return ont


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
    summary = p.summary()


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


def test_norun_summary(load_expression, load_ontology):
    p = PAGE(
        load_expression, 
        load_ontology, 
        n_shuffle=5, 
        k=2)

    try:
        p.summary()
        assert False
    except AttributeError:
        assert True


def test_empty_summary(load_expression, load_ontology):
    p = PAGE(
        load_expression, 
        load_ontology, 
        n_shuffle=5, 
        k=2)
    p.run()

    # set empty dataframe
    p.results = pd.DataFrame([])

    try:
        p.summary()
        assert False
    except ValueError:
        assert True


def test_heatmap_conversion(load_expression_bladder, load_ontology_cistrome):
    p = PAGE(
        load_expression_bladder,
        load_ontology_cistrome,
        n_shuffle=10,
        k=5)
    results = p.run()
    hm = p.heatmap()
    hm.convert_from_to('gs', 'ensg', 'human')


def test_heatmap_save(load_expression_bladder, load_ontology_cistrome):
    p = PAGE(
        load_expression_bladder,
        load_ontology_cistrome,
        n_shuffle=10,
        k=5)
    results = p.run()
    hm = p.heatmap()
    hm.save('test_heatmap', show_reg=True)

