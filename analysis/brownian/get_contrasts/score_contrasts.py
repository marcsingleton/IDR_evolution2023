"""Calculate score contrasts using segments which pass quality filters."""

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
min_lengths = [30, 60, 90]
cutoff = 0.5

tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)

for min_length in min_lengths:
    # Load regions
    OGid2regions = {}
    with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
            ppids = set(fields['ppids'].split(','))
            try:
                OGid2regions[OGid].append((start, stop, disorder, ppids))
            except KeyError:
                OGid2regions[OGid] = [(start, stop, disorder, ppids)]

    roots_records = []
    contrasts_records = []
    for OGid, regions in OGid2regions.items():
        ppid2spid = {}
        spid2scores = {}
        for header, seq in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
            ppid = re.search(ppid_regex, header).group(1)
            spid = re.search(spid_regex, header).group(1)

            ppid2spid[ppid] = spid

            aligned_scores = np.full(len(seq), np.nan)
            scores = load_scores(f'../../IDRpred/get_scores/out/{OGid}/{ppid}.diso_noprof')
            idx = 0
            for j, sym in enumerate(seq):
                if sym not in ['-', '.']:
                    aligned_scores[j] = scores[idx]
                    idx += 1
            spid2scores[spid] = aligned_scores

        for start, stop, disorder, ppids in regions:
            # Map features to tips
            spids = [ppid2spid[ppid] for ppid in ppids]
            tree = tree_template.shear(spids)
            for tip in tree.tips():
                scores = spid2scores[tip.name][start:stop]
                scores = scores[~np.isnan(scores)]
                score_fraction = scores.mean()
                binary_fraction = (scores >= cutoff).mean()
                tip.value = pd.Series({'score_fraction': score_fraction, 'binary_fraction': binary_fraction})

            # Get contrasts
            roots, contrasts = get_contrasts(tree)

            # Convert to dataframes
            root_ids = pd.Series({'OGid': OGid, 'start': start, 'stop': stop})
            roots = pd.concat([root_ids, roots])

            contrast_ids = []
            for contrast_id in range(len(contrasts)):
                contrast_ids.append({'OGid': OGid, 'start': start, 'stop': stop, 'contrast_id': contrast_id})
            contrasts = pd.concat([pd.DataFrame(contrast_ids), pd.DataFrame(contrasts)], axis=1)

            roots_records.append(roots)
            contrasts_records.append(contrasts)

    # Write to file
    if not os.path.exists('out/scores/'):
        os.mkdir('out/scores/')

    pd.DataFrame(roots_records).to_csv(f'out/scores/roots_{min_length}.tsv', sep='\t', index=False)
    pd.concat(contrasts_records).to_csv(f'out/scores/contrasts_{min_length}.tsv', sep='\t', index=False)
