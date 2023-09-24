"""Calculate feature contrasts using segments which pass quality filters."""

import multiprocessing as mp
import os
import re

import pandas as pd
import skbio
from src.phylo import get_contrasts
from src.utils import read_fasta


def get_args(grouped, tree, feature_labels, group_labels):
    for name, group in grouped:
        yield name, group, tree, feature_labels, group_labels


def apply_contrasts(args):
    name, group, tree, feature_labels, group_labels = args
    OGid, start, stop, disorder = name

    # Map features to tips
    tree = tree.shear(group['spid'])
    spid2idx = {spid: idx for idx, spid in zip(group.index, group['spid'])}
    for tip in tree.tips():
        idx = spid2idx[tip.name]
        tip.value = group.loc[idx, feature_labels]

    # Get contrasts
    roots, contrasts = get_contrasts(tree)

    # Convert to dataframes
    root_ids = pd.Series({'OGid': OGid, 'start': start, 'stop': stop})
    roots = pd.concat([root_ids, roots])
    roots.index = pd.MultiIndex.from_arrays([roots.index, 3*['ids_group'] + group_labels])

    contrast_ids = []
    for contrast_id in range(len(contrasts)):
        contrast_ids.append({'OGid': OGid, 'start': start, 'stop': stop, 'contrast_id': contrast_id})
    contrasts = pd.concat([pd.DataFrame(contrast_ids), pd.DataFrame(contrasts)], axis=1)
    contrasts.columns = pd.MultiIndex.from_arrays([contrasts.columns, 4*['ids_group'] + group_labels])

    return roots, contrasts


num_processes = int(os.environ.get('SLURM_CPUS_ON_NODE', 10))

ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'
min_lengths = [30, 60, 90]

tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)

if __name__ == '__main__':
    # Load sequence data
    ppid2spid = {}
    OGids = sorted([path.removesuffix('.afa') for path in os.listdir('../../../data/alignments/fastas/') if path.endswith('.afa')])
    for OGid in OGids:
        for header, _ in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
            ppid = re.search(ppid_regex, header).group(1)
            spid = re.search(spid_regex, header).group(1)
            ppid2spid[ppid] = spid

    # Load features
    all_features = pd.read_table('../get_features/out/features.tsv', header=[0, 1])
    all_features.loc[all_features[('kappa', 'charge_group')] == -1, 'kappa'] = 1  # Need to specify full column index to get slicing to work
    all_features.loc[all_features[('omega', 'charge_group')] == -1, 'omega'] = 1
    all_features['length'] = all_features['length'] ** 0.6
    all_features.rename(columns={'length': 'radius_gyration'}, inplace=True)

    feature_labels = [feature_label for feature_label, group_label in all_features.columns if group_label != 'ids_group']
    group_labels = [group_label for _, group_label in all_features.columns if group_label != 'ids_group']
    all_features = all_features.droplevel(1, axis=1)

    # Load regions as segments
    rows = []
    for min_length in min_lengths:
        with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
            field_names = file.readline().rstrip('\n').split('\t')
            for line in file:
                fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
                OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
                for ppid in fields['ppids'].split(','):
                    rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder,
                                 'ppid': ppid, 'spid': ppid2spid[ppid], 'min_length': min_length})
    all_segments = pd.DataFrame(rows)

    if not os.path.exists('out/features/'):
        os.makedirs('out/features/')

    for min_length in min_lengths:
        segment_keys = all_segments[all_segments['min_length'] == min_length].drop('min_length', axis=1)
        features = segment_keys.merge(all_features, how='left', on=['OGid', 'start', 'stop', 'ppid'])
        regions = features.groupby(['OGid', 'start', 'stop', 'disorder'])

        # Apply contrasts
        args = get_args(regions, tree_template, feature_labels, group_labels)
        with mp.Pool(processes=num_processes) as pool:
            records = pool.map(apply_contrasts, args, chunksize=50)

        roots, contrasts = zip(*records)
        pd.DataFrame(roots).to_csv(f'out/features/roots_{min_length}.tsv', sep='\t', index=False)
        pd.concat(contrasts).to_csv(f'out/features/contrasts_{min_length}.tsv', sep='\t', index=False)
