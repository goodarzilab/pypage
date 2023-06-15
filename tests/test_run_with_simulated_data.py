"""Testing an example usage of the PAGE algorithm with the simulated data
"""

import pytest
import pandas as pd
from pypage import (
    PAGE,
    ExpressionProfile,
    GeneSets)

@pytest.fixture()
def load_expression():
    df = pd.read_csv('example_data/simulated_expr.csv.gz', header=0, index_col=0, compression='gzip')
    exp = ExpressionProfile(df.iloc[:, 0],
                            df.iloc[:, 1],
                            n_bins=10)
    return exp

@pytest.fixture()
def load_ontology():
    frame = pd.read_csv('example_data/simulated_df.csv.gz', header=0, index_col=0, compression='gzip')
    ont = GeneSets(
        frame.iloc[:, 0],
        frame.iloc[:, 1],
        n_bins=6)
    return ont


def test_run(load_expression, load_ontology):
    p = PAGE(
        load_expression,
        load_ontology,
        n_shuffle=100,
        k=10,
        alpha=0.01)
    results, hm = p.run()

    positive = set(['upregulated_' + str(i) for i in range(50)] + ['downregulated_' + str(i) for i in range(50)])
    negative = set(['random_' + str(i) for i in range(1000)])
    count_true_positive = lambda series: len(set(series) & positive)
    count_true_negative = lambda series: len(negative - set(series))
    tpr = count_true_positive(results['pathway']) / 100
    tnr = count_true_negative(results['pathway']) / 1000
    print('tpr: %.2f\ntnr: %.2f' % (tpr, tnr))
    assert tpr > .9
    assert tnr > .9
