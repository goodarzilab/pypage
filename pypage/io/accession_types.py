import pybiomart
import numpy as np


def change_accessions(ids: np.ndarray,
                      input_format: str,
                      output_format: str,
                      species: str) -> np.ndarray:
    """
    A function which changes accessions
    Parameters
    ----------
    ids
    input_format
        input accession type, takes 'enst', 'ensg', 'refseq', 'entrez', 'gs', 'ext'
    output_format
        output accession type, takes 'enst', 'ensg', 'refseq', 'entrez', 'gs', 'ext'
    species
        analyzed species, takes either 'human' or 'mouse'
    Returns
    -------

    """
    if input_format == output_format:
        return ids
    else:
        if species == 'mouse':
            dataset = pybiomart.Dataset(name='mmusculus_gene_ensembl', host='http://www.ensembl.org')
        elif species == 'human':
            dataset = pybiomart.Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
        # print(*dataset.attributes.keys(), sep='\n')
        mart_attributes = {'enst': ['ensembl_transcript_id'], 'ensg': ['ensembl_gene_id'],
                           'refseq': ['refseq_mrna', 'refseq_mrna_predicted', 'refseq_ncrna',
                                      'refseq_ncrna_predicted'], 'entrez': ['entrezgene_id'],
                           'gs': ['entrezgene_accession'], 'ext': ['external_gene_name']}
        input_to_output = {}
        output_attributes = mart_attributes[output_format]
        if output_format == 'refseq':
            output_attributes = [output_attributes[0]]
        for mart in mart_attributes[input_format]:
            df1 = dataset.query(attributes=[mart] + output_attributes)
            df1 = df1[df1.iloc[:, 0].notna()]
            df1 = df1[df1.iloc[:, 1].notna()]
            if input_format == 'entrez' or output_format == 'entrez':
                df1['NCBI gene ID'] = df1['NCBI gene ID'].apply(lambda x: '%.f' % x)
            if input_format == 'gene_symbol' or output_format == 'gene_symbol':
                upper = lambda x: x.upper() if type(x) == str else x
                df1['NCBI gene accession'] = df1['NCBI gene accession'].apply(upper)
            input_to_output = {**input_to_output, **dict(zip(df1.iloc[:, 0], df1.iloc[:, 1]))}

        new_ids = []
        for id_ in ids:
            if id_ in input_to_output.keys():
                new_ids.append(input_to_output[id_])
            else:
                new_ids.append('-')
        return np.array(new_ids)
