# pyPAGE


## pypage.GeneSets

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

#### GeneSets.from_gmt

```python3
@classmethod
GeneSets.from_gmt(gmt_file: str,
                  n_bins: int = 3,
                  min_size: Optional[int] = None,
                  max_size: Optional[int] = None) -> GeneSets
```

Load gene sets from a GMT file (e.g., MSigDB). Supports plain `.gmt` and gzipped `.gmt.gz` files.

```
    gmt_file: str
        Path to .gmt or .gmt.gz file.
    n_bins: int
        Number of bins for membership binning (default: 3).
    min_size: int, optional
        Minimum pathway size. Pathways with fewer genes are removed after loading.
    max_size: int, optional
        Maximum pathway size. Pathways with more genes are removed after loading.
```

After loading, pathway descriptions from the GMT file are available via the `descriptions` attribute (dict mapping pathway name to description string).

#### GeneSets.to_gmt

```python3
GeneSets.to_gmt(output_file: str,
                descriptions: Optional[dict] = None)
```

Export gene sets to GMT format.

```
    output_file: str
        Path to output .gmt or .gmt.gz file.
    descriptions: dict, optional
        Mapping of pathway name to description string.
        Falls back to self.descriptions if available, otherwise 'na'.
```

#### GeneSets.map_genes

```python3
GeneSets.map_genes(mapper: GeneMapper,
                   from_type: str = 'ensg',
                   to_type: str = 'symbol')
```

Convert gene IDs in-place using a GeneMapper instance. Genes that cannot be mapped are dropped. Updates `self.genes`, `self.bool_array`, and `self.membership` in place. Pathways that become empty after dropping genes are also removed.

```
    mapper: GeneMapper
        A GeneMapper instance with a cached mapping table.
    from_type: str
        Source ID type: 'ensg', 'symbol', or 'entrez'.
    to_type: str
        Target ID type: 'ensg', 'symbol', or 'entrez'.
```

#### GeneSets.convert_from_to

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

## pypage.ExpressionProfile

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

#### ExpressionProfile.convert_from_to

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

## pypage.PAGE

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

#### PAGE.run
```python3
PAGE.run()
```
The function to run computation of differentially active genes.
As a result it produces a pandas dataframe and a Heatmap object which can also be accessed as PAGE.results and PAGE.hm attributes.

#### PAGE.get_enriched_genes
```python3
PAGE.get_enriched_genes(pathway: str)
```

The function that returns the information about which gene-set genes are present in which expression bin. 

```
    name: str
        The name of the gene-set.
```

## pypage.Heatmap

Heatmap objects are used to produce graphical representations of pyPAGE results.

Objects of this class are automatically generated by PAGE, so we will not concentrate on its input parameters here.

#### Heatmap.show

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

#### Heatmap.save

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

#### Heatmap.convert_from_to

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

#### Heatmap.add_gene_expression
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

## pypage.GeneMapper

```python3
class pypage.GeneMapper(species: str = 'human',
                         cache_dir: Optional[str] = None)
```

Offline gene ID conversion using a locally cached mapping table. On first instantiation, downloads a mapping table from Ensembl BioMart (~5 MB) and caches it locally. Subsequent instantiations load from cache with no network required.

Supported ID types: `'ensg'` (Ensembl gene IDs), `'symbol'` (gene symbols), `'entrez'` (Entrez gene IDs).

```
    species: str
        Species to use: 'human' or 'mouse'.
    cache_dir: str, optional
        Directory for the cache file. Defaults to ~/.pypage/.
```

#### GeneMapper.convert

```python3
GeneMapper.convert(ids: array-like,
                   from_type: str = 'ensg',
                   to_type: str = 'symbol') -> (np.ndarray, dict)
```

Convert gene IDs using the cached mapping table.

```
    ids: array-like
        Gene IDs to convert.
    from_type: str
        Source ID type: 'ensg', 'symbol', or 'entrez'.
    to_type: str
        Target ID type: 'ensg', 'symbol', or 'entrez'.

    Returns:
        np.ndarray: Converted IDs. Unmapped entries are set to None.
        dict: Mapping of input IDs that failed to convert (input_id -> None).
```

Versioned Ensembl IDs (e.g., `ENSG00000141510.15`) are automatically stripped of the version suffix when `from_type='ensg'`.

#### GeneMapper.build

```python3
@staticmethod
GeneMapper.build(species: str = 'human',
                 cache_dir: Optional[str] = None)
```

Force (re)download the mapping table from Ensembl BioMart. Requires `pybiomart` and network access.

```
    species: str
        Species to use: 'human' or 'mouse'.
    cache_dir: str, optional
        Directory for the cache file. Defaults to ~/.pypage/.
```

#### GeneMapper.cache_path

```python3
@property
GeneMapper.cache_path -> str
```

Path to the cached mapping file (e.g., `~/.pypage/gene_map_human.tsv`).