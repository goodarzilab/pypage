"""Testing an example usage of the PAGE algorithm after its upgrade
"""

import pytest
import pandas as pd
import numpy as np
from pypage import (
    PAGE,
    ExpressionProfile,
    GeneOntology)


@pytest.fixture()
def load_expression():
    df = pd.read_csv('example_data/bladder_refseq.tsv.gz',
                     sep="\t",
                     header=0,
                     names=["gene", "exp"])
    exp_array = np.array([df.iloc[:, 1]] * 5)
    exp = ExpressionProfile(df.iloc[:, 0],
                            exp_array,
                            bin_strategy='split',
                            n_bins=10)
    exp.convert_from_to('refseq', 'ensg', 'human')
    return exp


@pytest.fixture()
def load_ontology():

    ont = GeneOntology(ann_file='example_data/hg38_cistrome_index.txt.gz', n_bins=6)
    return ont


def test_run(load_expression, load_ontology):
    p = PAGE(
        load_expression,
        load_ontology,
        n_shuffle=100,
        k=7,
        filter_redundant=True
    )
    results = p.run()
