"""Testing Ontology IO
"""


import numpy as np
from pypage import GeneOntology


N_GENES=1000
N_PATHWAYS=50
T=100


def get_ontology() -> (np.ndarray, np.ndarray):
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    pathways = np.random.choice(N_PATHWAYS, size=N_GENES)
    return genes, pathways


def filter_assertions(
        ont: GeneOntology, 
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
        ont = GeneOntology(genes, pathways)


def test_load_ontology_assertion():
    for i in np.arange(T):
        genes, pathways = get_ontology()

        if i % 2 == 0:
            genes = genes[np.random.random(N_GENES) < 0.3]
        else:
            pathways = pathways[np.random.random(N_GENES) < 0.3]

        try:
            ont = GeneOntology(genes, pathways)
        except AssertionError:
            continue

        assert False


def test_ontology_min_filtering():
    for _ in np.arange(T):
        min_size = np.random.choice(np.arange(0, 3))
        genes, pathways = get_ontology()
        ont = GeneOntology(genes, pathways)

        unique_pathways, counts = np.unique(pathways, return_counts=True)
        mask = counts >= min_size
        ont.filter_pathways(min_size=min_size)

        filter_assertions(ont, mask, unique_pathways, counts)


def test_ontology_max_filtering():
    for _ in np.arange(T):
        max_size = np.random.choice(np.arange(8, 10))
        genes, pathways = get_ontology()
        ont = GeneOntology(genes, pathways)

        unique_pathways, counts = np.unique(pathways, return_counts=True)
        mask = counts <= max_size
        ont.filter_pathways(max_size=max_size)
        filter_assertions(ont, mask, unique_pathways, counts)


def test_ontology_bandpass_filtering():
    for _ in np.arange(T):
        min_size = np.random.choice(np.arange(0, 3))
        max_size = np.random.choice(np.arange(8, 10))
        genes, pathways = get_ontology()
        ont = GeneOntology(genes, pathways)

        unique_pathways, counts = np.unique(pathways, return_counts=True)
        mask = (counts >= min_size) & (counts <= max_size)
        ont.filter_pathways(min_size=min_size, max_size=max_size)
        filter_assertions(ont, mask, unique_pathways, counts)


def test_ontology_null_filtering():
    for _ in np.arange(T):
        genes, pathways = get_ontology()
        ont = GeneOntology(genes, pathways)
        
        unique_pathways, counts = np.unique(pathways, return_counts=True)
        mask = np.ones(unique_pathways.size, dtype=bool)
        ont.filter_pathways()
        filter_assertions(ont, mask, unique_pathways, counts)


def test_ontology_min_assertion():
    for _ in np.arange(T):
        genes, pathways = get_ontology()
        ont = GeneOntology(genes, pathways)

        try:
            ont.filter_pathways(min_size = -1)
        except AssertionError:
            continue

        assert False


def test_ontology_max_assertion():
    for _ in np.arange(T):
        genes, pathways = get_ontology()
        ont = GeneOntology(genes, pathways)

        try:
            ont.filter_pathways(min_size = 10, max_size=5)
        except AssertionError:
            continue

        assert False


def test_read_annotation_file():
    ont = GeneOntology(ann_file='example_data/hg38_cistrome_index_truncated.txt.gz')
    return ont


def test_convert_from_to():
    ont = GeneOntology(ann_file='example_data/hg38_cistrome_index_truncated.txt.gz')
    ont.convert_from_to('ensg', 'gs', 'human')
    return ont
