"""Testing Ontology IO
"""


import numpy as np
from pypage import GeneOntology


N_GENES=1000
N_PATHWAYS=100
T=100


def get_ontology() -> (np.ndarray, np.ndarray):
    genes = np.array([f"g.{g}" for g in np.arange(N_GENES)])
    pathways = np.random.choice(N_GENES, size=N_GENES)
    return genes, pathways


def test_load_ontology():
    for _ in np.arange(T):
        genes, pathways = get_ontology()
        ont = GeneOntology(genes, pathways)
