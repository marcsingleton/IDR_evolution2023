"""Make meta alignments of ordered and disordered regions."""

import os
import re

import numpy as np
import skbio
from src.utils import read_fasta


def is_redundant(column, cutoff):
    count = 0
    for sym in column:
        if sym in ['-', '.', 'X']:
            count += 1
    return count <= (1 - cutoff) * len(column)


spid_regex = r'spid=([a-z]+)'

tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)

rng = np.random.default_rng(seed=930715)
num_columns = int(1E5)  # Scientific notation defaults to float
num_samples = 25

# Load spids
spids = sorted([tip.name for tip in tree_template.tips() if tip.name != 'sleb'])
spid2idx = {spid: i for i, spid in enumerate(spids)}

# Load regions
OGid2regions = {}
with open('../../IDRpred/region_filter/out/regions_30.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        try:
            OGid2regions[OGid].append((start, stop, disorder))
        except KeyError:
            OGid2regions[OGid] = [(start, stop, disorder)]

# Extract column pools
column_pools = [('100R_disorder', True, lambda column: is_redundant(column, 1), []),
                ('100R_order', False, lambda column: is_redundant(column, 1), []),
                ('50R_disorder', True, lambda column: is_redundant(column, 0.5), []),
                ('50R_order', False, lambda column: is_redundant(column, 0.5), []),
                ('0R_disorder', True, lambda column: is_redundant(column, 0), []),
                ('0R_order', False, lambda column: is_redundant(column, 0), [])]
for OGid, regions in sorted(OGid2regions.items()):
    msa = read_fasta(f'../../../data/alignments/fastas/{OGid}.afa')
    msa = sorted([(re.search(spid_regex, header).group(1), seq) for header, seq in msa], key=lambda x: spid2idx[x[0]])
    if len(msa) < len(spids):  # Only use alignments with all species
        continue

    for start, stop, region_disorder in regions:
        for i in range(start, stop):
            column = [seq[i] for _, seq in msa]
            for _, pool_disorder, condition, column_pool in column_pools:
                if (region_disorder is pool_disorder) and condition(column):
                    column_pool.append(column)

if not os.path.exists('out/'):
    os.mkdir('out/')

# Write column pool sizes
with open('out/pool_sizes.tsv', 'w') as file:
    file.write('pool_name\tpool_size\n')
    for label, _, _, column_pool in column_pools:
        file.write(f'{label}\t{len(column_pool)}\n')

# Make meta alignments
for label, _, _, column_pool in column_pools:
    for sample_id in range(num_samples):
        sample = rng.choice(column_pool, size=num_columns)
        seqs = {spid: [] for spid in spids}
        for column in sample:
            for i, sym in enumerate(column):
                seqs[spids[i]].append(sym)

        with open(f'out/{label}_{sample_id:02}.afa', 'w') as file:
            for spid, seq in sorted(seqs.items()):
                seqstring = '\n'.join([''.join(seq[i:i+80]) for i in range(0, len(seq), 80)])
                file.write(f'>{spid} {label}\n{seqstring}\n')
