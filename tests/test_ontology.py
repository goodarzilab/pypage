"""Testing Ontology IO
"""


import os
import pytest
import numpy as np
from pypage import GeneSets


N_GENES=1000
N_PATHWAYS=50
T=100
RUN_ONLINE_TESTS = os.getenv("PYPAGE_RUN_ONLINE_TESTS") == "1"


def get_ontology() -> (np.ndarray, np.ndarray):
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    pathways = np.random.choice(N_PATHWAYS, size=N_GENES)
    return genes, pathways


def filter_assertions(
        ont: GeneSets,
        mask: np.ndarray, 
        unique_pathways: np.ndarray, 
        counts: np.ndarray):

    filtered_pathways = unique_pathways[mask]
    filtered_counts = counts[mask]

    assert ont.pathways.size == filtered_pathways.size
    assert np.all(ont.pathways == filtered_pathways)
    assert np.all(ont.pathway_sizes == filtered_counts)


def test_load_ontology():
    for _ in np.arange(T):
        genes, pathways = get_ontology()
        ont = GeneSets(genes, pathways)


def test_load_ontology_assertion():
    for i in np.arange(T):
        genes, pathways = get_ontology()

        if i % 2 == 0:
            genes = genes[np.random.random(N_GENES) < 0.3]
        else:
            pathways = pathways[np.random.random(N_GENES) < 0.3]

        try:
            ont = GeneSets(genes, pathways)
        except AssertionError:
            continue

        assert False


def test_ontology_min_filtering():
    for _ in np.arange(T):
        min_size = np.random.choice(np.arange(0, 3))
        genes, pathways = get_ontology()
        ont = GeneSets(genes, pathways)

        unique_pathways, counts = np.unique(pathways, return_counts=True)
        mask = counts >= min_size
        ont.filter_pathways(min_size=min_size)

        filter_assertions(ont, mask, unique_pathways, counts)


def test_ontology_max_filtering():
    for _ in np.arange(T):
        max_size = np.random.choice(np.arange(8, 10))
        genes, pathways = get_ontology()
        ont = GeneSets(genes, pathways)

        unique_pathways, counts = np.unique(pathways, return_counts=True)
        mask = counts <= max_size
        ont.filter_pathways(max_size=max_size)
        filter_assertions(ont, mask, unique_pathways, counts)


def test_ontology_bandpass_filtering():
    for _ in np.arange(T):
        min_size = np.random.choice(np.arange(0, 3))
        max_size = np.random.choice(np.arange(8, 10))
        genes, pathways = get_ontology()
        ont = GeneSets(genes, pathways)

        unique_pathways, counts = np.unique(pathways, return_counts=True)
        mask = (counts >= min_size) & (counts <= max_size)
        ont.filter_pathways(min_size=min_size, max_size=max_size)
        filter_assertions(ont, mask, unique_pathways, counts)


def test_ontology_null_filtering():
    for _ in np.arange(T):
        genes, pathways = get_ontology()
        ont = GeneSets(genes, pathways)
        
        unique_pathways, counts = np.unique(pathways, return_counts=True)
        mask = np.ones(unique_pathways.size, dtype=bool)
        ont.filter_pathways()
        filter_assertions(ont, mask, unique_pathways, counts)


def test_ontology_min_assertion():
    for _ in np.arange(T):
        genes, pathways = get_ontology()
        ont = GeneSets(genes, pathways)

        try:
            ont.filter_pathways(min_size = -1)
        except AssertionError:
            continue

        assert False


def test_ontology_max_assertion():
    for _ in np.arange(T):
        genes, pathways = get_ontology()
        ont = GeneSets(genes, pathways)

        try:
            ont.filter_pathways(min_size = 10, max_size=5)
        except AssertionError:
            continue

        assert False


def test_read_annotation_file():
    ont = GeneSets(ann_file='example_data/hg38_cistrome_index.txt.gz')
    assert ont.n_pathways > 0
    assert ont.n_genes > 0


def test_convert_from_to_offline(monkeypatch):
    def fake_change_accessions(ids, input_format, output_format, species):
        assert input_format == "ensg"
        assert output_format == "gs"
        assert species == "human"
        return np.array([f"GS_{x}" for x in ids])

    monkeypatch.setattr("pypage.io.ontology.change_accessions", fake_change_accessions)
    ont = GeneSets(genes=np.array(["ENSG1.1", "ENSG2.2"]), pathways=np.array(["P1", "P2"]))
    ont.convert_from_to("ensg", "gs", "human")
    assert np.array_equal(ont.genes, np.array(["GS_ENSG1", "GS_ENSG2"]))


@pytest.mark.skipif(
    not RUN_ONLINE_TESTS,
    reason="Requires network access to Ensembl. Set PYPAGE_RUN_ONLINE_TESTS=1 to enable.",
)
@pytest.mark.online
def test_convert_from_to_online():
    ont = GeneSets(ann_file='example_data/hg38_cistrome_index.txt.gz')
    ont.convert_from_to('ensg', 'gs', 'human')
