"""Extract basic statistics from AUCpreD scores."""

import os
import re

import numpy as np
import scipy.ndimage as ndimage
from src.utils import read_fasta


def load_scores(path):
    with open(path) as file:
        scores = []
        for line in file:
            if not line.startswith('#'):
                score = line.split()[3]
                scores.append(float(score))
    return np.array(scores)


ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
gnid_regex = r'gnid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'
cutoff = 0.5

# Load error flags
OGid2flags = {}
with open('../aucpred_scores/out/errors.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, ppid, error_flag = fields['OGid'], fields['ppid'], fields['error_flag'] == 'True'
        try:
            OGid2flags[OGid].append(error_flag)
        except KeyError:
            OGid2flags[OGid] = [error_flag]

# Convert error flags to successful OGids
OGids = []
for OGid, error_flags in sorted(OGid2flags.items()):
    if not any(error_flags):
        OGids.append(OGid)

records = []
for OGid in OGids:
    for header, _ in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        gnid = re.search(gnid_regex, header).group(1)
        spid = re.search(spid_regex, header).group(1)

        scores = load_scores(f'../../IDRpred/aucpred_scores/out/{OGid}/{ppid}.diso_noprof')
        binary = (scores >= cutoff)
        scores_sum = scores.sum()
        binary_sum = binary.sum()
        binary_regions = ndimage.label(binary)[1]  # Second element is number of objects
        length = len(scores)
        records.append({'OGid': OGid, 'ppid': ppid, 'gnid': gnid, 'spid': spid,
                        'scores_sum': scores_sum, 'binary_sum': binary_sum,
                        'binary_regions': binary_regions,
                        'length': length})

# Write segments to file
if not os.path.exists('out/'):
    os.mkdir('out/')

columns = ['OGid', 'ppid', 'gnid', 'spid', 'scores_sum', 'binary_sum', 'binary_regions', 'length']
with open('out/stats.tsv', 'w') as file:
    file.write('\t'.join(columns) + '\n')
    for record in records:
        file.write('\t'.join([str(record[column]) for column in columns]) + '\n')
