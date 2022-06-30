# pypage
python implementation of the PAGE algorithm

# Installation
```bash
git clone https://github.com/noamteyssier/pypage
cd pypage
pip install -e .
```

# Building Documentation
```bash
cd docs/
make html
firefox _build/html/index.html
```

# Usage
This implementation is meant to be used as a python module. The process of using the module is broken up into 3 major steps.

1. Load the Expression Data
2. Load the Ontology Data
3. Perform the Statistical Test

```python3
from pypage.io import (ExpressionProfile, GeneOntology)
from pypage.page import PAGE

# 1. load expression data
exp_frame = pd.read_csv(
  "example_data/AP2S1.tab.gz", 
  sep="\t", 
  header=None, 
  names=["gene", "bin"])
exp = ExpressionProfile(exp_frame.gene, exp_frame.bin)

# 2. load ontology data
ont_frame = pd.read_csv(
  "example_data/GO_BP_2021_index.txt.gz", 
  sep="\t", 
  header=None, 
  names=["gene", "pathway"])
ont = GeneOntology(ont_frame.gene, ont_frame.pathway)

# 3. Perform the statistical test
p = PAGE(exp, ont, n_shuffle=500)
results = p.run()
summary = p.summary()

# bin/pathway-level pvalues
print(results)

# pathway-level pvalues
print(summary)
```
