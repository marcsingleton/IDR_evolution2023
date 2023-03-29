"""Extract Brownian motion parameters from AUCpreD scores at the level of proteins."""

import os
import re

import numpy as np
import pandas as pd
import skbio
from src.phylo import get_contrasts
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
spid_regex = r'spid=([a-z]+)'
cutoff = 0.5

tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_template.tips())}

# Load error flags
OGid2flags = {}
with open('../get_scores/out/errors.tsv') as file:
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

roots_records = []
contrasts_records = []
for OGid in OGids:
    spid2value = {}
    for header, _ in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        spid = re.search(spid_regex, header).group(1)

        scores = load_scores(f'../get_scores/out/{OGid}/{ppid}.diso_noprof')
        scores_fraction = scores.mean()
        binary_fraction = (scores >= cutoff).mean()
        spid2value[spid] = pd.Series({'scores_fraction': scores_fraction, 'binary_fraction': binary_fraction})

    # Map features to tips
    tree = tree_template.shear(spid2value)
    for tip in tree.tips():
        tip.value = spid2value[tip.name]

    # Get contrasts
    roots, contrasts = get_contrasts(tree)

    # Convert to dataframes
    root_ids = pd.Series({'OGid': OGid})
    roots = pd.concat([root_ids, roots])

    contrast_ids = []
    for contrast_id in range(len(contrasts)):
        contrast_ids.append({'OGid': OGid, 'contrast_id': contrast_id})
    contrasts = pd.concat([pd.DataFrame(contrast_ids), pd.DataFrame(contrasts)], axis=1)

    roots_records.append(roots)
    contrasts_records.append(contrasts)

# Write to file
if not os.path.exists('out/'):
    os.mkdir('out/')

pd.DataFrame(roots_records).to_csv(f'out/roots.tsv', sep='\t', index=False)
pd.concat(contrasts_records).to_csv(f'out/contrasts.tsv', sep='\t', index=False)
