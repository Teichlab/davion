import pandas as pd
import numpy as np
from pathlib import Path

from davion.utils import BioMartQuery

tmp_dir = Path('.tmp')


def get_effective_length(species='human', index='ensembl_gene_id'):

    species_mapper = {'human': 'hsapiens_gene_ensembl',
                      'mouse': 'mmusculus_gene_ensembl'}
    ensembl_species = species_mapper[species]
    q = BioMartQuery(ensembl_species)
    q.add_attributes(index, "start_position", "end_position")
    length_df = q.stream()
    length_df.set_index(index, inplace=True)
    gene_length = (length_df['end_position'] - length_df['start_position']) / 1000

    # Deal with repeats
    print(gene_length.head())
    gene_length = gene_length.groupby(level=0).agg(np.mean)
    print(gene_length)
    return gene_length


def count_to_tpm(counts, effective_lengths=None, **kwargs):
    if effective_lengths is None:
        effective_lengths = get_effective_length(**kwargs)

    common_genes = list(set(counts.index)
                        .intersection(set(effective_lengths.index)))
    count_vals = counts.loc[common_genes, :].values
    efflen_vals = effective_lengths.loc[common_genes].values

    rate = np.log(count_vals + 1) - np.log(efflen_vals)
    denom = np.log(np.exp(rate).sum(axis=1))
    out = np.exp((rate.T - denom).T + np.log(1e6))

    return pd.DataFrame(out, index=counts.index, columns=counts.columns)
