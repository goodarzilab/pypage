"""Testing an example usage of the PAGE algorithm after its upgrade
"""

import pytest
import os
import pandas as pd
from pypage import (
    PAGE,
    ExpressionProfile,
    GeneSets)

RUN_ONLINE_TESTS = os.getenv("PYPAGE_RUN_ONLINE_TESTS") == "1"
pytestmark = [
    pytest.mark.online,
    pytest.mark.skipif(
        not RUN_ONLINE_TESTS,
        reason="Requires network access to Ensembl. Set PYPAGE_RUN_ONLINE_TESTS=1 to enable.",
    ),
]


@pytest.fixture()
def load_expression():
    df = pd.read_csv('example_data/bladder_refseq.tsv.gz',
                     sep="\t",
                     header=0,
                     names=["gene", "exp"])
    exp = ExpressionProfile(df.iloc[:, 0],
                             df.iloc[:, 1],
                            n_bins=10)
    exp.convert_from_to('refseq', 'ensg', 'human')
    return exp


@pytest.fixture()
def load_ontology():

    ont = GeneSets(ann_file='example_data/hg38_cistrome_index.txt.gz', n_bins=6)
    return ont


def test_enrichment(load_expression, load_ontology):
    p = PAGE(
        load_expression,
        load_ontology,
        n_shuffle=100,
        k=7,
        filter_redundant=True
        )
    results = p.get_enriched_genes('NFKB2')
