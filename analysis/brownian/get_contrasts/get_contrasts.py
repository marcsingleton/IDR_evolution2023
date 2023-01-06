"""Calculate contrasts using segments which pass quality filters."""

import multiprocessing as mp
import os
import re

import pandas as pd
import skbio
from src.utils import get_contrasts, read_fasta


def get_args(grouped, tree, feature_labels):
    for name, group in grouped:
        yield name, group, tree.copy(), feature_labels


def apply_contrasts(args):
    name, group, tree, feature_labels = args

    # Map features to tips
    tree = tree.shear(group['spid'])
    spid2idx = {spid: idx for idx, spid in zip(group.index, group['spid'])}
    for tip in tree.tips():
        idx = spid2idx[tip.name]
        tip.value = group.loc[idx, feature_labels]

    # Get contrasts
    roots, contrasts = get_contrasts(tree)

    return name, roots, contrasts


num_processes = int(os.environ.get('SLURM_CPUS_ON_NODE', 1))
tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)

ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'
length_regex = r'regions_([0-9]+).tsv'

if __name__ == '__main__':
    # Load sequence data
    ppid2spid = {}
    OGids = sorted([path.removesuffix('.afa') for path in os.listdir('../../../data/alignments/fastas/') if path.endswith('.afa')])
    for OGid in OGids:
        for header, _ in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
            ppid = re.search(ppid_regex, header).group(1)
            spid = re.search(spid_regex, header).group(1)
            ppid2spid[ppid] = spid

    # Get minimum lengths
    min_lengths = []
    for path in os.listdir('../regions_filter/out/'):
        match = re.search(length_regex, path)
        if match:
            min_lengths.append(int(match.group(1)))
    min_lengths = sorted(min_lengths)

    # Load features
    features = pd.read_table('../get_features/out/features.tsv')
    features.loc[features['kappa'] == -1, 'kappa'] = 1
    features.loc[features['omega'] == -1, 'omega'] = 1
    features['radius_gyration'] = features['length'] ** 0.6

    feature_labels = list(features.columns.drop(['OGid', 'start', 'stop', 'ppid', 'length']))

    # Load regions
    rows = []
    for min_length in min_lengths:
        with open(f'../regions_filter/out/regions_{min_length}.tsv') as file:
            field_names = file.readline().rstrip('\n').split('\t')
            for line in file:
                fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
                OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
                for ppid in fields['ppids'].split(','):
                    rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder,
                                 'ppid': ppid, 'spid': ppid2spid[ppid], 'min_length': min_length})
    df = pd.DataFrame(rows)

    if not os.path.exists('out/'):
        os.mkdir('out/')

    for min_length in min_lengths:
        segments = df[df['min_length'] == min_length].merge(features, how='left', on=['OGid', 'start', 'stop', 'ppid']).drop('min_length', axis=1)
        regions = segments.groupby(['OGid', 'start', 'stop', 'disorder'])

        # Apply contrasts
        args = get_args(regions, tree_template, feature_labels)
        with mp.Pool(processes=num_processes) as pool:
            # Using imap distributes the construction of the args tuples
            # However to force computation before pool is closed we must call list on it
            records = list(pool.imap(apply_contrasts, args, chunksize=50))

        # Write contrasts and means to file
        with open(f'out/contrasts_{min_length}.tsv', 'w') as file1, open(f'out/roots_{min_length}.tsv', 'w') as file2:
            file1.write('\t'.join(['OGid', 'start', 'stop', 'contrast_id'] + feature_labels) + '\n')
            file2.write('\t'.join(['OGid', 'start', 'stop'] + feature_labels) + '\n')
            for name, roots, contrasts in records:
                # Write contrasts
                for idx, contrast in enumerate(contrasts):
                    fields1 = [name[0], str(name[1]), str(name[2]), str(idx)]
                    for feature_label in feature_labels:
                        fields1.append(str(contrast[feature_label]))
                    file1.write('\t'.join(fields1) + '\n')

                # Write means
                fields2 = [name[0], str(name[1]), str(name[2])]
                for feature_label in feature_labels:
                    fields2.append(str(roots[feature_label]))
                file2.write('\t'.join(fields2) + '\n')
