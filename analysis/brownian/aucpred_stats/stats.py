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

records = []
for OGid in [path.removesuffix('.afa') for path in os.listdir('../../../data/alignments/fastas/') if path.endswith('.afa')]:
    for header, _ in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        gnid = re.search(gnid_regex, header).group(1)
        spid = re.search(spid_regex, header).group(1)

        try:
            scores = load_scores(f'../aucpred_scores/out/{OGid}/{ppid.split(".")[0]}.diso_noprof')  # Remove anything after trailing .
            binary = (scores >= cutoff)
            scores_sum = scores.sum()
            binary_sum = binary.sum()
            binary_regions = ndimage.label(binary)[1]  # Second element is number of objects
            length = len(scores)
            records.append({'OGid': OGid, 'ppid': ppid, 'gnid': gnid, 'spid': spid,
                            'scores_sum': scores_sum, 'binary_sum': binary_sum,
                            'binary_regions': binary_regions,
                            'length': length})
        except FileNotFoundError:
            records.append({'OGid': OGid, 'ppid': ppid, 'gnid': gnid, 'spid': spid})

# Write segments to file
if not os.path.exists('out/'):
    os.mkdir('out/')

columns = ['OGid', 'ppid', 'gnid', 'spid', 'scores_sum', 'binary_sum', 'binary_regions', 'length']
with open('out/stats.tsv', 'w') as file:
    file.write('\t'.join(columns) + '\n')
    for record in sorted(records, key=lambda x: (x['OGid'], x['spid'])):
        file.write('\t'.join([str(record.get(column, 'NaN')) for column in columns]) + '\n')
