"""Tests for GeneMapper (offline gene ID mapping)."""

import os
import tempfile
import numpy as np
import pytest
from pypage.io.gene_map import GeneMapper


# --- Fixtures: create a small mock mapping table ---

MOCK_TSV = """\
ensg\tsymbol\tentrez
ENSG00000141510\tTP53\t7157
ENSG00000012048\tBRCA1\t672
ENSG00000171862\tPTEN\t5728
ENSG00000157764\tBRAF\t673
ENSG00000146648\tEGFR\t1956
"""


@pytest.fixture
def mock_cache(tmp_path):
    """Create a temporary cache directory with a mock mapping table."""
    cache_file = tmp_path / 'gene_map_human.tsv'
    cache_file.write_text(MOCK_TSV)
    return str(tmp_path)


@pytest.fixture
def mapper(mock_cache):
    """Create a GeneMapper with mock data (no network)."""
    # Since the cache already exists, __init__ won't call build()
    return GeneMapper(species='human', cache_dir=mock_cache)


# --- Tests ---

def test_mapper_loads(mapper):
    assert mapper.species == 'human'


def test_convert_ensg_to_symbol(mapper):
    ids = ['ENSG00000141510', 'ENSG00000012048']
    result, unmapped = mapper.convert(ids, from_type='ensg', to_type='symbol')
    assert result.tolist() == ['TP53', 'BRCA1']
    assert len(unmapped) == 0


def test_convert_symbol_to_ensg(mapper):
    ids = ['TP53', 'PTEN']
    result, unmapped = mapper.convert(ids, from_type='symbol', to_type='ensg')
    assert result.tolist() == ['ENSG00000141510', 'ENSG00000171862']
    assert len(unmapped) == 0


def test_convert_ensg_to_entrez(mapper):
    ids = ['ENSG00000141510']
    result, unmapped = mapper.convert(ids, from_type='ensg', to_type='entrez')
    assert result.tolist() == ['7157']
    assert len(unmapped) == 0


def test_convert_entrez_to_symbol(mapper):
    ids = ['672', '1956']
    result, unmapped = mapper.convert(ids, from_type='entrez', to_type='symbol')
    assert result.tolist() == ['BRCA1', 'EGFR']
    assert len(unmapped) == 0


def test_convert_strips_version(mapper):
    """ENSG IDs with version suffixes should still resolve."""
    ids = ['ENSG00000141510.18', 'ENSG00000012048.23']
    result, unmapped = mapper.convert(ids, from_type='ensg', to_type='symbol')
    assert result.tolist() == ['TP53', 'BRCA1']
    assert len(unmapped) == 0


def test_convert_unmapped_ids(mapper):
    ids = ['ENSG00000141510', 'ENSG_FAKE_123', 'ENSG00000012048']
    result, unmapped = mapper.convert(ids, from_type='ensg', to_type='symbol')
    assert result[0] == 'TP53'
    assert result[1] is None
    assert result[2] == 'BRCA1'
    assert 'ENSG_FAKE_123' in unmapped


def test_convert_all_unmapped(mapper):
    ids = ['FAKE1', 'FAKE2']
    result, unmapped = mapper.convert(ids, from_type='ensg', to_type='symbol')
    assert all(r is None for r in result)
    assert len(unmapped) == 2


def test_convert_empty_input(mapper):
    result, unmapped = mapper.convert([], from_type='ensg', to_type='symbol')
    assert len(result) == 0
    assert len(unmapped) == 0


def test_convert_invalid_from_type(mapper):
    with pytest.raises(ValueError, match="from_type"):
        mapper.convert(['X'], from_type='invalid', to_type='symbol')


def test_convert_invalid_to_type(mapper):
    with pytest.raises(ValueError, match="to_type"):
        mapper.convert(['X'], from_type='ensg', to_type='invalid')


def test_invalid_species():
    with pytest.raises(ValueError, match="Unsupported species"):
        GeneMapper(species='zebrafish')


def test_cache_path(mapper, mock_cache):
    assert mapper.cache_path == os.path.join(mock_cache, 'gene_map_human.tsv')


def test_map_genes_integration(mock_cache):
    """Test GeneSets.map_genes() with mock mapper."""
    from pypage import GeneSets

    mapper = GeneMapper(species='human', cache_dir=mock_cache)

    # Create a GeneSets with ENSG IDs
    genes = np.array([
        'ENSG00000141510', 'ENSG00000012048', 'ENSG00000171862',
        'ENSG00000141510', 'ENSG00000012048',
    ])
    pathways = np.array(['PathA', 'PathA', 'PathA', 'PathB', 'PathB'])
    gs = GeneSets(genes=genes, pathways=pathways)

    gs.map_genes(mapper, from_type='ensg', to_type='symbol')

    assert 'TP53' in gs.genes
    assert 'BRCA1' in gs.genes
    assert 'PTEN' in gs.genes
    assert gs.n_pathways == 2


def test_map_genes_drops_unmapped(mock_cache):
    """Unmapped genes should be dropped with a warning."""
    from pypage import GeneSets

    mapper = GeneMapper(species='human', cache_dir=mock_cache)

    genes = np.array(['ENSG00000141510', 'FAKE_GENE'])
    pathways = np.array(['PathA', 'PathA'])
    gs = GeneSets(genes=genes, pathways=pathways)

    with pytest.warns(UserWarning, match="Dropped 1 unmapped"):
        gs.map_genes(mapper, from_type='ensg', to_type='symbol')

    assert gs.n_genes == 1
    assert gs.genes[0] == 'TP53'


def test_map_genes_all_unmapped(mock_cache):
    """If all genes are unmapped, raise ValueError."""
    from pypage import GeneSets

    mapper = GeneMapper(species='human', cache_dir=mock_cache)

    genes = np.array(['FAKE1', 'FAKE2'])
    pathways = np.array(['PathA', 'PathA'])
    gs = GeneSets(genes=genes, pathways=pathways)

    with pytest.raises(ValueError, match="No genes could be mapped"):
        gs.map_genes(mapper, from_type='ensg', to_type='symbol')
