#pyPAGE


##pypage.GeneSets

```python3
class pypage.GeneSets(genes: Optional[np.ndarray] = None,
               pathways: Optional[np.ndarray] = None,
               ann_file: Optional[str] = None,
               n_bins: Optional[int] = 3,
               first_col_is_genes: Optional[bool] = False)
```

Objects of this class store information about gene-sets and should be passed to PAGE object as an input.
To initialize GeneSets user should provide either tab delimited annotation file in index format 
(each line starts with a gene-set name, followed by genes) 
or a binary matrix encoding gene membership.

```
    ann_file: str
        tab delimited annotation file in index format
    first_col_is_genes:
        specifies whether first element in each line of the annotation file is a gene
    genes: np.ndarray
        gene names in pathways matrix, alternative to ann_file
    pathways: np.ndarray
        binary matrix encoding gene-set membership, alternative to ann_file
    n_bins: int
        number of bins to use when binning membership array
```

####GeneSets.convert_from_to

```python3
GeneSets.convert_from_to(input_format: str,
                         output_format: str,
                         species: Optional[str] = 'human')
```

This function is used to convert gene names in the annotation to another format.

Available formats: ensg (ensemble gene ids), enst (ensemble transcript ids), refseq, entrez (gene ids), gs (gene symbol).

```
    input_format: str
        input format of the annotation
    output_format: str
        output format of the annotation
    species: str
        species, available: human, mouse
```

##pypage.ExpressionProfile

```python3
class pypage.ExpressionProfile(genes: np.ndarray,
                        expression: np.ndarray,
                        is_bin: bool = False,
                        n_bins: Optional[int] = 10)
```

Objects of this class store information about gene expression and should be passed to PAGE object as an input.

```
    genes: np.ndarray
        The array with gene names.
    expression: np.ndarray
        The array representing either the continuous expression value
        of genes, or the bin/cluster that gene belongs to.
    is_bin: bool
        Specifies that the provided array is already prebinned.
    n_bins: int
        number of bins to bin the expression array into.
```

####ExpressionProfile.convert_from_to

```python3
ExpressionProfile.convert_from_to(input_format: str,
                         output_format: str,
                         species: Optional[str] = 'human')
```

This function is used to convert gene names in the expression profile to another format.

Available formats: ensg (ensemble gene ids), enst (ensemble transcript ids), refseq, entrez (gene ids), gs (gene symbol).

```
    input_format: str
        input format of the annotation
    output_format: str
        output format of the annotation
    species: str
        species, available: human, mouse
```

##pypage.PAGE

```python3
class pypage.PAGE(
            expression: ExpressionProfile,
            genesets: GeneSets,
            n_shuffle: int = 1e3,
            alpha: float = 1e-2,
            k: int = 10,
            filter_redundant: bool = False,
            n_jobs: Optional[int] = 1,
            function: Optional[str] = 'cmi',
            redundancy_ratio: Optional[float] = .1)
```
The main object of the package that performs the computation of differentially active genes and stores the results.

```
        expression: ExpressionProfile
            ExpressionProfile object containing differential gene expression.

        genesets: GeneSets
            GeneSets object containing gene annotations.

        n_shuffle: int
            The number of permutations in the statistical test.

        alpha: float
            The maximum p-value threshold to consider a pathway informative
            with respect to the permuted mutual information distribution

        k: int
            The number of contiguous uninformative pathways to consider before
            stopping the informative pathway search

        filter_redundant: bool
            Specify whether to perform the pathway redundancy search

        n_jobs: int
            The number of parallel jobs to use in the analysis
            (`default = all available cores`)
            
        function: str
            Specify whether conditional mutual information ('cmi') or mutual information ('mi') should be calculated.
        
        redundancy_ratio: float
            The redundacy ratio to use (the bigger the threshold the lesser number of gene-sets will be in the output). To understand it refer to the paper.
```

####PAGE.run
```python3
PAGE.run()
```
The function to run computation of differentially active genes.
As a result it produces a pandas dataframe and a Heatmap object which can also be accessed as PAGE.results and PAGE.hm attributes.

####PAGE.get_enriched_genes
```python3
PAGE.get_enriched_genes(pathway: str)
```

The function that returns the information about which gene-set genes are present in which expression bin. 

```
    name: str
        The name of the gene-set.
```

##pypage.Heatmap

Heatmap objects are used to produce graphical representations of pyPAGE results.

Objects of this class are automatically generated by PAGE, so we will not concentrate on its input parameters here.

####Heatmap.show

```python3
Heatmap.show(max_rows: Optional[int] = 50,
             show_reg: Optional[bool] = False,
             max_val: Optional[int] = 5,
             title: str = '')
```

Show the heatmap representation pyPAGE results.

``` 
    max_rows: int
        Maximal number of rows in the ouput
    show_reg: bool
        Specifies whether expression of a regulator should be used (works only if gene-sets are named by their regulators).
    max_val: int
        Max value for a colorbar.
    title: str
        Title of the heatmap
```

####Heatmap.save

```python3
Heatmap.save(output_name:str,
             max_rows: Optional[int] = 50,
             show_reg: Optional[bool] = False,
             max_val: Optional[int] = 5,
             title: str = '')
```

Save the heatmap representation pyPAGE results.

``` 
    output_name: str
        The name of the output file.
    max_rows: int
        Maximal number of rows in the ouput
    show_reg: bool
        Specifies whether expression of a regulator should be used (works only if gene-sets are named by their regulators).
    max_val: int
        Max value for a colorbar.
    title: str
        Title of the heatmap
```

####Heatmap.convert_from_to

```python3
Heatmap.convert_from_to(input_format: str,
                        output_format: str,
                        species: Optional[str] = 'human')
```

This function is used to convert regulator gene names to another format (the one that is used in the differential expression profile).

Available formats: ensg (ensemble gene ids), enst (ensemble transcript ids), refseq, entrez (gene ids), gs (gene symbol).

```
    input_format: str
        input format of the annotation
    output_format: str
        output format of the annotation
    species: str
        species, available: human, mouse
```

####Heatmap.add_gene_expression
```python3
Heatmap.add_gene_expression(genes: np.ndarray,
                            expression: np.ndarray)
```
This function should be used if you want to visualize the expression of the regulators in cases when it is not contained in PAGE object. 
For example, when PAGE is run to identify RBP regulons using differential stability and you want to add RBP expression to the heatmap output.

``` 
genes: np.ndarray
    Array of gene names
expression: np.ndarray
    Array of differential expression values.
```