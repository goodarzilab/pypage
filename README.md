# pyPAGE

`pyPAGE` is a Python implementation of the conditional-information PAGE framework for gene-set enrichment analysis.

It is designed to infer differential activity of pathways and regulons while accounting for annotation and membership biases using information-theoretic methods.

## Installation

Install from PyPI:

```bash
pip install bio-pypage
```

Or install from source:

```bash
git clone https://github.com/goodarzilab/pypage
cd pypage
pip install -e .
```

## Quick Start

```python
import pandas as pd
from pypage import PAGE, ExpressionProfile, GeneSets

# 1) Load expression profile (gene, score)
expr = pd.read_csv(
    "example_data/AP2S1.tab.gz",
    sep="\t",
    header=None,
    names=["gene", "score"],
)
exp = ExpressionProfile(expr["gene"], expr["score"], is_bin=True)

# 2) Load annotation (gene, pathway)
ann = pd.read_csv(
    "example_data/GO_BP_2021_index.txt.gz",
    sep="\t",
    header=None,
    names=["gene", "pathway"],
)
gs = GeneSets(ann["gene"], ann["pathway"])

# 3) Run pyPAGE
p = PAGE(exp, gs, n_shuffle=100, k=7, filter_redundant=True)
results, heatmap = p.run()

print(results.head())
heatmap.show()
```

`results` contains:
- `pathway`
- `CMI` — conditional mutual information score
- `p-value` — empirical p-value from permutation test
- `Regulation pattern` (`1` for up, `-1` for down)

## Loading Gene Sets

Gene sets can be loaded from multiple sources:

**Paired arrays** — gene and pathway name arrays of equal length:
```python
gs = GeneSets(genes=gene_array, pathways=pathway_array)
```

**Annotation index files** — tab-delimited files where each line starts with a pathway name followed by its member genes (supports gzip):
```python
gs = GeneSets(ann_file="GO_BP_2021_index.txt.gz")
# If the first column is genes (not pathways):
gs = GeneSets(ann_file="annotations.txt", first_col_is_genes=True)
```

**GMT files** — MSigDB `.gmt` format (plain or gzipped), with optional size filtering:
```python
gs = GeneSets.from_gmt("h.all.v2024.1.Hs.symbols.gmt")
gs = GeneSets.from_gmt("c2.cp.kegg.gmt", min_size=15, max_size=500)

# Export back to GMT
gs.to_gmt("filtered_pathways.gmt")
```

## Loading Expression Data

Expression input can be:
- **Continuous differential scores** (`is_bin=False`, default) — auto-discretized into `n_bins` equal-frequency bins
- **Pre-binned integer labels** (`is_bin=True`) — used as-is

```python
# Continuous scores
exp = ExpressionProfile(genes, scores, n_bins=10)

# Pre-binned labels
exp = ExpressionProfile(genes, bin_labels, is_bin=True)
```

## Gene ID Conversion

### GeneMapper (Recommended)

`GeneMapper` downloads a gene ID mapping table from Ensembl once and caches it locally (~5 MB at `~/.pypage/`) for fast offline lookups. Supported ID types: `'ensg'`, `'symbol'`, `'entrez'`.

```python
from pypage import GeneMapper, GeneSets

# First call downloads from Ensembl; subsequent calls use cache
mapper = GeneMapper(species='human')

# Convert gene IDs
symbols, unmapped = mapper.convert(
    ['ENSG00000141510', 'ENSG00000012048'],
    from_type='ensg', to_type='symbol',
)
# symbols -> ['TP53', 'BRCA1']

# Convert genes in-place on a GeneSets object
gs = GeneSets.from_gmt("kegg_entrez.gmt")
gs.map_genes(mapper, from_type='entrez', to_type='symbol')
```

### Legacy: convert_from_to() (Requires Network)

`ExpressionProfile.convert_from_to()`, `GeneSets.convert_from_to()`, and `Heatmap.convert_from_to()` use Ensembl BioMart (`pybiomart`) and require an active internet connection:

```python
exp.convert_from_to("refseq", "ensg", "human")
```

## Bulk PAGE Analysis

The `PAGE` class performs pathway enrichment analysis with permutation testing and optional redundancy filtering:

```python
p = PAGE(exp, gs,
    function='cmi',         # 'cmi' (default, corrects annotation bias) or 'mi'
    n_shuffle=1000,         # permutation count
    alpha=0.01,             # p-value threshold
    k=10,                   # early-stopping parameter
    filter_redundant=True,  # remove redundant pathways
    n_jobs=1,               # parallel threads
)
results, heatmap = p.run()

# Enriched genes per pathway
enriched = p.get_enriched_genes("pathway_name")

# Enrichment score matrix (log10 hypergeometric p-values)
es_matrix = p.get_es_matrix()
```

## Single-Cell Analysis

`SingleCellPAGE` brings per-cell pathway scoring and spatial coherence testing to pyPAGE, inspired by [VISION](https://github.com/YosefLab/VISION). It accepts AnnData objects or raw numpy arrays.

```python
import anndata
from pypage import GeneSets, SingleCellPAGE

adata = anndata.read_h5ad("my_data.h5ad")
gs = GeneSets(ann_file="annotations.txt.gz")

sc = SingleCellPAGE(adata=adata, genesets=gs, function='cmi')
results = sc.run(n_permutations=1000)

print(results.head())
```

`results` contains:
- `pathway`
- `consistency` — spatial autocorrelation score (C' = 1 - Geary's C; higher = more coherent)
- `p-value` — empirical p-value from size-matched random gene sets
- `FDR` — Benjamini-Hochberg corrected p-value

### Visualization

```python
sc.plot_pathway_on_embedding("MyPathway", embedding_key='X_umap')
sc.plot_consistency_ranking(top_n=20)
sc.plot_pathway_heatmap(adata.obs['leiden'])
```

### Neighborhood Mode

Aggregate cells by cluster labels and run standard bulk PAGE per group:

```python
summary, group_results = sc.run_neighborhoods(labels=adata.obs['leiden'])
```

### Input Options

| Input | How |
|-------|-----|
| AnnData | `SingleCellPAGE(adata=adata, genesets=gs)` |
| Numpy arrays | `SingleCellPAGE(expression=X, genes=gene_names, genesets=gs)` |
| Precomputed KNN | `SingleCellPAGE(adata=adata, genesets=gs, connectivity=W)` |

## Parameter Reference

### PAGE

| Parameter | Default | Description |
|-----------|---------|-------------|
| `function` | `'cmi'` | `'cmi'` (conditional MI, corrects annotation bias) or `'mi'` |
| `n_shuffle` | `1000` | Number of permutations for significance testing |
| `alpha` | `0.01` | P-value threshold for informative pathways |
| `k` | `10` | Early-stopping: stop after k consecutive non-significant pathways |
| `filter_redundant` | `False` | Remove redundant pathways via CMI |
| `redundancy_ratio` | `0.1` | Redundancy threshold (higher = stricter filtering) |
| `n_jobs` | `1` | Number of parallel threads |

### SingleCellPAGE

| Parameter | Default | Description |
|-----------|---------|-------------|
| `function` | `'cmi'` | `'cmi'` or `'mi'` |
| `n_bins` | `10` | Number of bins for expression discretization |
| `n_neighbors` | `ceil(sqrt(n_cells))` | KNN neighbors (capped at 100) |
| `connectivity` | `None` | Precomputed cell-cell connectivity matrix |

### GeneMapper

| Parameter | Default | Description |
|-----------|---------|-------------|
| `species` | `'human'` | `'human'` or `'mouse'` |
| `cache_dir` | `'~/.pypage/'` | Directory for cached mapping file |

## Reproducibility Tips

For deterministic benchmark-style runs:

```python
import numpy as np
np.random.seed(0)
p = PAGE(exp, gs, n_shuffle=100, n_jobs=1)
```

## Tutorials

- **[Comprehensive Tutorial](notebooks/pypage_tutorial.ipynb)** — End-to-end walkthrough covering all features (GMT, GeneMapper, bulk PAGE, single-cell PAGE)
- **[Bulk & SC Example](notebooks/pypage_bulk_and_sc_example.ipynb)** — Bulk and single-cell analysis demo
- **[Single-Cell Tutorial](notebooks/single_cell_page_tutorial.ipynb)** — Detailed single-cell PAGE walkthrough

## Testing

Fast local test profile (default CI profile):

```bash
pytest -q -m "not slow and not online"
```

Full test profile (includes long and network-dependent tests):

```bash
PYPAGE_RUN_ONLINE_TESTS=1 pytest -q
```

## Documentation

For full API details, see `MANUAL.md`.

## Citation

Bakulin A, Teyssier NB, Kampmann M, Khoroshkin M, Goodarzi H (2024)
*pyPAGE: A framework for Addressing biases in gene-set enrichment analysis—A case study on Alzheimer's disease.*
PLoS Computational Biology 20(9): e1012346.
https://doi.org/10.1371/journal.pcbi.1012346

## License

MIT

## About

pyPAGE was developed in the Goodarzi Lab at UCSF by Artemy Bakulin, Noam B. Teyssier, and Hani Goodarzi.
