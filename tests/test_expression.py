"""Testing Expression IO 
"""


import os
import pytest
import pandas as pd
import numpy as np
from pypage import ExpressionProfile


N_GENES=1000
N_BINS=5
T = 100
RUN_ONLINE_TESTS = os.getenv("PYPAGE_RUN_ONLINE_TESTS") == "1"


def get_expression() -> (np.ndarray, np.ndarray):
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    scores = np.random.normal(size=N_GENES)
    return genes, scores


def get_bins() -> (np.ndarray, np.ndarray):
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    bins = np.random.choice(N_BINS, size=N_GENES)
    return genes, bins


def test_load_expression_nobins():
    for _ in np.arange(T):
        genes, scores = get_expression()
        exp = ExpressionProfile(genes, scores)
        assert exp.n_genes == N_GENES
        assert exp.n_bins == 10


def test_load_expression():
    for _ in np.arange(T):
        genes, expression = get_expression()
        exp = ExpressionProfile(genes, expression, n_bins=N_BINS)
        assert exp.n_genes == N_GENES
        assert exp.n_bins == N_BINS


def test_load_expression_assertion():
    for i in np.arange(T):
        genes, expression = get_expression()

        if i % 2 == 0:
            genes = genes[np.random.random(N_GENES) < 0.3]
        else:
            expression = expression[np.random.random(N_GENES) < 0.3]

        try:
            ont = ExpressionProfile(genes, expression)
        except AssertionError:
            continue

        assert False


"""def test_load_expression_strategy_hist():
    for _ in np.arange(T):
        genes, expression = get_expression()

        # test hist binning
        hist_exp = ExpressionProfile(
                genes, 
                expression, 
                n_bins=N_BINS, 
                bin_strategy='hist')
        
        assert hist_exp.n_genes == N_GENES
        assert hist_exp.n_bins == N_BINS

        # test split binning
        split_exp = ExpressionProfile(
                genes, 
                expression, 
                n_bins=N_BINS, 
                bin_strategy='split')
        
        assert split_exp.n_genes == N_GENES
        assert split_exp.n_bins == N_BINS

        # assert bins are different
        assert np.any(hist_exp.bin_sizes != split_exp.bin_sizes)"""


def test_load_bins():
    genes, bins = get_bins()
    exp = ExpressionProfile(genes, bins, n_bins=N_BINS)
    assert exp.n_genes == N_GENES
    assert exp.n_bins == N_BINS


def test_load_bins_is_bin_keeps_values():
    genes = np.array([f"g.{g}" for g in np.arange(30)])
    bins = np.random.choice(np.arange(N_BINS), size=30, p=np.array([0.6, 0.1, 0.1, 0.1, 0.1]))
    exp = ExpressionProfile(genes, bins, is_bin=True, n_bins=N_BINS)
    subset = exp.get_gene_subset(genes)
    assert np.array_equal(subset, bins)


def test_expression_conversion_offline(monkeypatch):
    def fake_change_accessions(ids, input_format, output_format, species):
        assert input_format == "refseq"
        assert output_format == "ensg"
        assert species == "human"
        return np.array([f"ENSG_{x}" for x in ids])

    monkeypatch.setattr("pypage.io.expression.change_accessions", fake_change_accessions)
    genes = np.array(["NM_0001.1", "NM_0002.2", "NM_0003.3"])
    exp = ExpressionProfile(genes, np.array([1.0, 2.0, 3.0]), n_bins=20)
    exp.convert_from_to("refseq", "ensg", "human")
    assert np.array_equal(exp.genes, np.array(["ENSG_NM_0001", "ENSG_NM_0002", "ENSG_NM_0003"]))


@pytest.mark.skipif(
    not RUN_ONLINE_TESTS,
    reason="Requires network access to Ensembl. Set PYPAGE_RUN_ONLINE_TESTS=1 to enable.",
)
@pytest.mark.online
def test_expression_conversion_online():
    df = pd.read_csv('example_data/bladder_refseq.tsv.gz',
                     sep="\t",
                     header=0,
                     names=["gene", "exp"],
                     compression='gzip')
    exp = ExpressionProfile(df.iloc[:, 0],
                            df.iloc[:, 1],
                            n_bins=20)
    exp.convert_from_to('refseq', 'ensg', 'human')
