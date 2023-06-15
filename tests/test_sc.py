import pytest
import pandas as pd
from pypage import (
    scPAGE,
    scExpressionProfile,
    GeneSets)
import numpy as np
import pandas as pd
from scipy import sparse


@pytest.fixture()
def load_expression():
    df = pd.read_csv('example_data/bladder_refseq.tsv.gz',
                     sep="\t",
                     header=0,
                     names=["gene", "exp"]).iloc[:1000]

    simulated_sc = np.stack([df.iloc[:, 1]] * 3)
    simulated_sc = sparse.csr_matrix(simulated_sc)

    exp = scExpressionProfile(df.iloc[:, 0],
                              simulated_sc,
                              bin_strategy='split',
                              n_bins=10)
    exp.convert_from_to('refseq', 'ensg', 'human')
    return exp


@pytest.fixture()
def load_ontology():
    ont = GeneSets(ann_file='example_data/hg38_cistrome_index.txt.gz', n_bins=6)
    return ont

def test_run(load_expression, load_ontology):
    p = scPAGE(
        load_expression,
        load_ontology,
        n_shuffle=100,
        k=7,
        filter_redundant=True
    )
    info_scores, info_vals, uncertainty_vals = p.run('NFKB2', estimate_uncertainty=True)

