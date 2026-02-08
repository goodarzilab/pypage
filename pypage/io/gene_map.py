"""Offline gene ID mapping using a locally cached mapping table."""

import os
import warnings
import numpy as np


# Column names in the cached TSV
_COLUMNS = ['ensg', 'symbol', 'entrez']

# Ensembl dataset names per species
_DATASETS = {
    'human': 'hsapiens_gene_ensembl',
    'mouse': 'mmusculus_gene_ensembl',
}

# BioMart attribute names
_ATTRIBUTES = ['ensembl_gene_id', 'external_gene_name', 'entrezgene_id']


def _default_cache_dir():
    return os.path.join(os.path.expanduser('~'), '.pypage')


class GeneMapper:
    """Offline gene ID conversion using a locally cached mapping table.

    Usage
    -----
    # First time (downloads ~5 MB from Ensembl, caches to ~/.pypage/):
    mapper = GeneMapper(species='human')

    # Convert IDs:
    symbols, unmapped = mapper.convert(
        ['ENSG00000141510', 'ENSG00000012048'],
        from_type='ensg', to_type='symbol',
    )
    # symbols -> ['TP53', 'BRCA1']

    # Use with GeneSets:
    gs = GeneSets.from_gmt("hallmark.gmt")
    gs.map_genes(mapper, from_type='ensg', to_type='symbol')
    """

    def __init__(self, species='human', cache_dir=None):
        """Load or build the mapping table.

        On first call: downloads from Ensembl BioMart and caches locally.
        On subsequent calls: loads from cache (no network needed).

        Parameters
        ----------
        species : str
            'human' or 'mouse'.
        cache_dir : str, optional
            Directory for the cache file. Defaults to ~/.pypage/.
        """
        if species not in _DATASETS:
            raise ValueError(f"Unsupported species '{species}'. Use 'human' or 'mouse'.")

        self.species = species
        self._cache_dir = cache_dir or _default_cache_dir()

        if not os.path.exists(self.cache_path):
            self.build(species=species, cache_dir=self._cache_dir)

        self._load()

    def _load(self):
        """Load the cached TSV into lookup dicts."""
        self._maps = {}
        for col in _COLUMNS:
            self._maps[col] = {}

        with open(self.cache_path) as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) != 3:
                    continue
                ensg, symbol, entrez = parts

                # Store forward and reverse mappings
                row = {'ensg': ensg, 'symbol': symbol, 'entrez': entrez}
                for from_col in _COLUMNS:
                    key = row[from_col]
                    if key:
                        # Only keep first mapping per key (dedup)
                        if key not in self._maps[from_col]:
                            self._maps[from_col][key] = row

    def convert(self, ids, from_type='ensg', to_type='symbol'):
        """Convert gene IDs using the cached mapping.

        Parameters
        ----------
        ids : array-like
            Gene IDs to convert.
        from_type : str
            'ensg', 'symbol', or 'entrez'.
        to_type : str
            'ensg', 'symbol', or 'entrez'.

        Returns
        -------
        np.ndarray
            Converted IDs. Unmapped IDs are set to None.
        dict
            Mapping of input IDs that failed to convert (input -> None).
        """
        if from_type not in _COLUMNS:
            raise ValueError(f"from_type must be one of {_COLUMNS}, got '{from_type}'")
        if to_type not in _COLUMNS:
            raise ValueError(f"to_type must be one of {_COLUMNS}, got '{to_type}'")

        ids = np.asarray(ids)
        lookup = self._maps[from_type]
        result = []
        unmapped = {}

        for id_ in ids:
            id_str = str(id_)
            # Strip version suffix for ENSG IDs
            if from_type == 'ensg' and '.' in id_str:
                id_str = id_str.split('.')[0]

            row = lookup.get(id_str)
            if row is not None and row[to_type]:
                result.append(row[to_type])
            else:
                result.append(None)
                unmapped[id_] = None

        return np.array(result, dtype=object), unmapped

    @staticmethod
    def build(species='human', cache_dir=None):
        """Force (re)download the mapping table from Ensembl.

        Parameters
        ----------
        species : str
            'human' or 'mouse'.
        cache_dir : str, optional
            Directory for the cache file. Defaults to ~/.pypage/.
        """
        import pybiomart

        if species not in _DATASETS:
            raise ValueError(f"Unsupported species '{species}'. Use 'human' or 'mouse'.")

        cache_dir = cache_dir or _default_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)

        dataset = pybiomart.Dataset(
            name=_DATASETS[species],
            host='http://www.ensembl.org',
        )
        df = dataset.query(attributes=_ATTRIBUTES)

        # Rename columns to our standard names
        df.columns = _COLUMNS

        # Strip version suffixes from ENSG IDs
        df['ensg'] = df['ensg'].astype(str).str.split('.').str[0]

        # Clean up entrez: convert float to int string, drop NaN
        df['entrez'] = df['entrez'].apply(
            lambda x: str(int(x)) if not (x != x) else ''  # x != x is NaN check
        )

        # Drop rows with no ensg
        df = df[df['ensg'].str.startswith('ENS')]

        # Drop rows where symbol is empty/NaN
        df['symbol'] = df['symbol'].fillna('')
        df = df[df['symbol'] != '']

        # Deduplicate: keep first mapping per ENSG
        df = df.drop_duplicates(subset='ensg', keep='first')

        cache_path = os.path.join(cache_dir, f'gene_map_{species}.tsv')
        df.to_csv(cache_path, sep='\t', index=False)

    @property
    def cache_path(self):
        """Path to the cached mapping file."""
        return os.path.join(self._cache_dir, f'gene_map_{self.species}.tsv')
