"""Tests for GMT file reader and writer."""

import os
import gzip
import tempfile
import numpy as np
import pytest
from pypage import GeneSets


GMT_FILE = 'example_data/example.gmt'


def test_from_gmt_loads():
    gs = GeneSets.from_gmt(GMT_FILE)
    assert gs.n_pathways == 5
    assert gs.n_genes > 0
    assert gs.bool_array.shape == (gs.n_pathways, gs.n_genes)


def test_from_gmt_pathway_names():
    gs = GeneSets.from_gmt(GMT_FILE)
    expected = sorted([
        'HALLMARK_APOPTOSIS', 'HALLMARK_P53_PATHWAY',
        'HALLMARK_TNFA_SIGNALING', 'HALLMARK_GLYCOLYSIS',
        'HALLMARK_HYPOXIA',
    ])
    assert sorted(gs.pathways.tolist()) == expected


def test_from_gmt_descriptions():
    gs = GeneSets.from_gmt(GMT_FILE)
    assert hasattr(gs, 'descriptions')
    assert 'HALLMARK_APOPTOSIS' in gs.descriptions
    assert 'gsea-msigdb.org' in gs.descriptions['HALLMARK_APOPTOSIS']


def test_from_gmt_gene_contents():
    gs = GeneSets.from_gmt(GMT_FILE)
    # CASP3 should be in HALLMARK_APOPTOSIS
    gene_list = gs.genes.tolist()
    assert 'CASP3' in gene_list
    assert 'TP53' in gene_list

    # Check that BAX is in both APOPTOSIS and P53 (shared gene)
    apop_idx = np.where(gs.pathways == 'HALLMARK_APOPTOSIS')[0][0]
    p53_idx = np.where(gs.pathways == 'HALLMARK_P53_PATHWAY')[0][0]
    bax_idx = np.where(gs.genes == 'BAX')[0][0]
    assert gs.bool_array[apop_idx, bax_idx] == 1
    assert gs.bool_array[p53_idx, bax_idx] == 1


def test_from_gmt_pathway_sizes():
    gs = GeneSets.from_gmt(GMT_FILE)
    # Each pathway in the example file has 20 genes
    for i in range(gs.n_pathways):
        assert gs.bool_array[i].sum() == 20


def test_from_gmt_min_max_filter():
    # max_size=19 should leave none (all pathways have 20 genes)
    gs = GeneSets.from_gmt(GMT_FILE, max_size=19)
    assert gs.pathways.size == 0

    # min_size=10, max_size=25 should keep all (all have 20)
    gs = GeneSets.from_gmt(GMT_FILE, min_size=10, max_size=25)
    assert gs.pathways.size == 5

    # min_size=21, max_size=30 should leave none
    gs = GeneSets.from_gmt(GMT_FILE, min_size=21, max_size=30)
    assert gs.pathways.size == 0


def test_from_gmt_gzip():
    """Test loading from a gzipped GMT file."""
    with tempfile.NamedTemporaryFile(suffix='.gmt.gz', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create gzipped version
        with open(GMT_FILE) as f:
            content = f.read()
        with gzip.open(tmp_path, 'wt') as f:
            f.write(content)

        gs = GeneSets.from_gmt(tmp_path)
        gs_orig = GeneSets.from_gmt(GMT_FILE)
        assert gs.n_pathways == gs_orig.n_pathways
        assert gs.n_genes == gs_orig.n_genes
    finally:
        os.unlink(tmp_path)


def test_to_gmt_roundtrip():
    """Test that to_gmt -> from_gmt preserves gene sets."""
    gs1 = GeneSets.from_gmt(GMT_FILE)

    with tempfile.NamedTemporaryFile(suffix='.gmt', delete=False, mode='w') as tmp:
        tmp_path = tmp.name

    try:
        gs1.to_gmt(tmp_path)
        gs2 = GeneSets.from_gmt(tmp_path)

        assert gs1.n_pathways == gs2.n_pathways
        assert gs1.n_genes == gs2.n_genes
        assert sorted(gs1.pathways.tolist()) == sorted(gs2.pathways.tolist())
        assert sorted(gs1.genes.tolist()) == sorted(gs2.genes.tolist())

        # Check bool_array contents match (after aligning indices)
        for pathway in gs1.pathways:
            idx1 = np.where(gs1.pathways == pathway)[0][0]
            idx2 = np.where(gs2.pathways == pathway)[0][0]
            genes1 = set(gs1.genes[gs1.bool_array[idx1] > 0])
            genes2 = set(gs2.genes[gs2.bool_array[idx2] > 0])
            assert genes1 == genes2, f"Gene mismatch for {pathway}"
    finally:
        os.unlink(tmp_path)


def test_to_gmt_gzip_roundtrip():
    """Test round-trip through gzipped GMT."""
    gs1 = GeneSets.from_gmt(GMT_FILE)

    with tempfile.NamedTemporaryFile(suffix='.gmt.gz', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        gs1.to_gmt(tmp_path)
        gs2 = GeneSets.from_gmt(tmp_path)
        assert gs1.n_pathways == gs2.n_pathways
        assert gs1.n_genes == gs2.n_genes
    finally:
        os.unlink(tmp_path)


def test_to_gmt_preserves_descriptions():
    """Test that descriptions survive round-trip."""
    gs1 = GeneSets.from_gmt(GMT_FILE)

    with tempfile.NamedTemporaryFile(suffix='.gmt', delete=False, mode='w') as tmp:
        tmp_path = tmp.name

    try:
        gs1.to_gmt(tmp_path)
        gs2 = GeneSets.from_gmt(tmp_path)
        for pathway in gs1.pathways:
            assert gs2.descriptions[pathway] == gs1.descriptions[pathway]
    finally:
        os.unlink(tmp_path)


def test_to_gmt_custom_descriptions():
    """Test writing with custom descriptions."""
    gs = GeneSets.from_gmt(GMT_FILE)

    custom_desc = {p: f"Custom desc for {p}" for p in gs.pathways}

    with tempfile.NamedTemporaryFile(suffix='.gmt', delete=False, mode='w') as tmp:
        tmp_path = tmp.name

    try:
        gs.to_gmt(tmp_path, descriptions=custom_desc)
        gs2 = GeneSets.from_gmt(tmp_path)
        for pathway in gs.pathways:
            assert gs2.descriptions[pathway] == f"Custom desc for {pathway}"
    finally:
        os.unlink(tmp_path)


def test_from_gmt_with_constructor_genesets():
    """Test that from_gmt GeneSets works with PAGE-compatible operations."""
    gs = GeneSets.from_gmt(GMT_FILE)

    # Should have membership computed
    assert hasattr(gs, 'membership')
    assert gs.membership.shape == (gs.n_genes,)

    # Should be able to get gene subset
    subset = gs.genes[:10]
    result = gs.get_gene_subset(subset)
    assert result.shape == (gs.n_pathways, 10)
