# pyPAGE

pyPAGE is a novel research tool for the analysis of differential gene expression.

Its aim is to provide unbiased estimates of changes in activity of gene regulatory programs.
pyPAGE can be used to detect differentially active biological pathways, targets of transcription factors
and post-transcriptional regulons controlled either by RNA binding proteins or miRNAs.

Moreover, pyPAGE is applicable both to the analysis of bulk and single-cell RNA-seq data. 




## Installation
Downloading and installing pyPAGE is as easy as typing the following command in your terminal:

```bash
pip install bio-pypage
```

Or alternatively you can install it from the source:

```bash
git clone https://github.com/goodarzilab/pypage
cd pypage
pip install -e .
```

## Preliminaries

pyPAGE is implemented as a python package.

To run it, you need to, first, download the relevant gene-set annotations and precompute differential gene expression.

You can download preprocessed annotations of TF and RBP regulons from [here](https://drive.google.com/drive/folders/1CTRQvTzhFANi45PqHfPNxEiN5NiKro11?usp=sharing ).<br/>
Moreover, [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H) is a great resource to search for an annotation of interest.
For example, you might want to use "GO: Gene Ontology gene sets" and "MIR: microRNA targets".
Also there is a [useful resource](https://bmcresnotes.biomedcentral.com/articles/10.1186/s13104-018-3856-x/tables/1) of TF targets annotations.

Then you need to compute differential gene expression. 
For the analysis of bulk data, you would typically use DESeq2, limma or EdgeR. These packages provide user with the log fold changes and their significance.
Then differential gene expression can be calculated as 'sign(log_fold_change) * (1-p-value)'.

To compute differential expression in single cell data, a good strategy is to use Mann-Whitney U test:

```python3
import numpy as np
from scipy.stats import mannwhitneyu


def count_difexp_with_u_test(cells_A, cells_B):
    """
    Computes differential gene expression between two groups of cells
    Parameters
    ----------
    cells_A: np.ndarray
        expression of genes in the first group of cells (n*m matrix where n is the number of cells and m the number of genes)
    cells_B: np.ndarray
        expression of genes in the second group of cells
    """
    
    n_genes = cells_A.shape[1]
    sign = np.sign(cells_A.mean(axis=0) - cells_B.mean(axis=0))
    p_values = np.empty(n_genes)
    p_values[:] = np.nan
    for i in range(n_genes):
        if not np.array_equal(np.unique(cells_A[:, i]), np.unique(cells_B[:, i])):
            p_values[i] = mannwhitneyu(cells_A[:, i], cells_B[:, i])[1]
    difexp = sign * (1 - p_values)
    return difexp
```

Note, that to perform the analysis post-transcriptional regulation, you need to estimate differential trancript stability.
To do this, you need to first compute sample-wise stability estimates of transcripts using [REMBRANDTS](https://github.com/csglab/REMBRANDTS) and then compare them between two conditions with a t-test.

## Usage

Now that we have completed the preparation, we can proceed directly to using pyPAGE.

First, load the relevant gene-set annotation:

```python3
from pypage import GeneSets

def load_annotation(ann_file):
    gs_ann = GeneSets(ann_file=ann_file, n_bins=3)
    return gs_ann

```

Next, load the expression data and convert gene names to the same format as the one used in the annotation:

```python3
import pandas as pd
from pypage import ExpressionProfile

def load_expression(expression_file):
    df = pd.read_csv(expression_file,
                     sep="\t",
                     header=0,
                     names=["gene", "exp"])
    exp = ExpressionProfile(df.iloc[:, 0],
                            df.iloc[:, 1],
                            n_bins=10)
    exp.convert_from_to('refseq', 'ensg', 'human')
    return exp
```


Finally, run pyPAGE algorithm and visualize the results:

```python3
from pypage import PAGE

def run_pyPAGE(expression, annotation):
    
    p = PAGE(
        expression,
        annotation,
        n_shuffle=1000,
        k=7,
        filter_redundant=True
    )
    results, hm = p.run()
    hm.convert_from_to('gs', 'ensg', 'human')
    hm.show(show_reg=True)
    
    return results
```

## Manual

For an exhaustive description of pyPAGE functions refer to the [manual](https://github.com/goodarzilab/pyPAGE/MANUAL.md).

## License
MIT license

## Citing
See the paper

## About pyPAGE
pyPAGE was developed in Goodarzi lab at UCSF by Artemy Bakulin, Noam B Teyssier and Hani Goodarzi.
