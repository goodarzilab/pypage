# pyPAGE

`pyPAGE` is a Python implementation of the conditional-information PAGE framework for gene-set enrichment analysis.

It is designed to infer differential activity of pathways and regulons while accounting for annotation and membership biases.

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
- `CMI`
- `p-value`
- `Regulation pattern` (`1` for up, `-1` for down)

## Single-Cell Analysis

`SingleCellPAGE` brings per-cell pathway scoring and spatial coherence testing to pyPAGE, inspired by [VISION](https://github.com/YosefLab/VISION). It accepts AnnData objects or raw numpy arrays.

```python
import anndata
from pypage import GeneSets, SingleCellPAGE

# Load your single-cell data
adata = anndata.read_h5ad("my_data.h5ad")
gs = GeneSets(ann_file="annotations.txt.gz")

# Run per-cell MI/CMI scoring + Geary's C spatial coherence test
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
# Pathway scores on UMAP embedding
sc.plot_pathway_on_embedding("MyPathway", embedding_key='X_umap')

# Top pathways ranked by spatial consistency
sc.plot_consistency_ranking(top_n=20)

# Heatmap of pathway scores across cell clusters
sc.plot_pathway_heatmap(adata.obs['leiden'])
```

### Neighborhood mode

Aggregate cells by cluster labels and run standard bulk PAGE per group:

```python
summary, group_results = sc.run_neighborhoods(labels=adata.obs['leiden'])
```

### Input options

| Input | How |
|-------|-----|
| AnnData | `SingleCellPAGE(adata=adata, genesets=gs)` |
| Numpy arrays | `SingleCellPAGE(expression=X, genes=gene_names, genesets=gs)` |
| Precomputed KNN | `SingleCellPAGE(adata=adata, genesets=gs, connectivity=W)` |

See `notebooks/single_cell_page_tutorial.ipynb` for a full walkthrough.

## Bulk Analysis — Input Notes

- Expression input can be:
  - continuous differential scores (set `is_bin=False`, default), or
  - pre-binned labels (set `is_bin=True`)
- Annotation can be loaded from:
  - paired `(genes, pathways)` arrays, or
  - index files via `GeneSets(ann_file=..., first_col_is_genes=...)`

## Gene ID Conversion

`ExpressionProfile.convert_from_to(...)`, `GeneSets.convert_from_to(...)`, and `Heatmap.convert_from_to(...)` use Ensembl BioMart (`pybiomart`) and require network access.

Example:

```python
exp.convert_from_to("refseq", "ensg", "human")
```

## Reproducibility Tips

For deterministic benchmark-style runs:

```python
import numpy as np
np.random.seed(0)
p = PAGE(exp, gs, n_shuffle=100, n_jobs=1)
```

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
*pyPAGE: A framework for Addressing biases in gene-set enrichment analysis—A case study on Alzheimer’s disease.*  
PLoS Computational Biology 20(9): e1012346.  
https://doi.org/10.1371/journal.pcbi.1012346

## License

MIT

## About

pyPAGE was developed in the Goodarzi Lab at UCSF by Artemy Bakulin, Noam B. Teyssier, and Hani Goodarzi.
