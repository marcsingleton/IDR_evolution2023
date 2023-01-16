"""Fit models of evolution to data."""

import os
import re

import numpy as np
import pandas as pd
import skbio
import utils
from src.utils import read_fasta

num_processes = int(os.environ.get('SLURM_CPUS_ON_NODE', 1))
tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)

ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'
length_regex = r'regions_([0-9]+).tsv'

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
all_features = pd.read_table('../get_features/out/features.tsv')
all_features.loc[all_features['kappa'] == -1, 'kappa'] = 1
all_features.loc[all_features['omega'] == -1, 'omega'] = 1
all_features['length'] = all_features['length'] ** 0.6
all_features.rename(columns={'length': 'radius_gyration'}, inplace=True)

feature_labels = list(all_features.columns.drop(['OGid', 'ppid', 'start', 'stop']))

# Load regions as segments
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
all_segments = pd.DataFrame(rows)

if not os.path.exists('out/'):
    os.mkdir('out/')

for min_length in min_lengths:
    segment_keys = all_segments[all_segments['min_length'] == min_length].drop('min_length', axis=1)
    features = segment_keys.merge(all_features, how='left', on=['OGid', 'start', 'stop', 'ppid'])
    regions = features.groupby(['OGid', 'start', 'stop', 'disorder'])

    records = []
    for name, group in regions:
        OGid, start, stop, disorder = name
        spid2idx = {spid: idx for idx, spid in zip(group.index, group['spid'])}
        tree = tree_template.shear(group['spid'])
        tips, cov = utils.get_brownian_covariance(tree)
        inv = np.linalg.inv(cov)

        record = {'OGid': OGid, 'start': start, 'stop': stop}
        for feature_label in feature_labels:
            # Assign values to tips and extract vector
            # This is done in two ways because the MLE functions have different call signatures
            # as a result of some technical details relating to how they are implemented
            for tip in tips:
                idx = spid2idx[tip.name]
                tip.value = group.loc[idx, feature_label]
            values = np.array([tip.value for tip in tips])

            if np.all(values[0] == values):
                # If values are constant the model behaviors are technically undefined.
                # The Brownian case has a reasonable limit of a constant random variable with mean of the observed
                # value. The log-likelihood can be taken as 0 since the observed value occurs with certainty.
                #
                # The OU case is more subtle because the model is technically unidentifiable because sigma2 = 0 or
                # alpha = inf could cause zero observed variance. The convention taken here is to assume sigma2 = 0, and
                # consider alpha as unidentifiable from the data, i.e. set its value to nan. Under this interpretation,
                # the log-likelihood is 0 because it also reduces to a constant random variable.
                mu, sigma2 = values[0], 0
                loglikelihood = 0

                mu_OU, sigma2_OU, alpha_OU = values[0], 0, np.nan
                loglikelihood_OU = 0
            else:
                mu, sigma2 = utils.get_brownian_mles(cov=cov, inv=inv, values=values)
                loglikelihood = utils.get_brownian_loglikelihood(mu, sigma2, cov=cov, inv=inv, values=values)

                mu_OU, sigma2_OU, alpha_OU = utils.get_OU_mles(tips=tips, ts=cov)
                loglikelihood_OU = utils.get_OU_loglikelihood(mu_OU, sigma2_OU, alpha_OU, tips=tips, ts=cov)

            record.update({f'{feature_label}_mu': mu, f'{feature_label}_sigma2': sigma2,
                           f'{feature_label}_loglikelihood': loglikelihood,
                           f'{feature_label}_mu_OU': mu_OU, f'{feature_label}_sigma2_OU': sigma2_OU,
                           f'{feature_label}_alpha_OU': alpha_OU,
                           f'{feature_label}_loglikelihood_OU': loglikelihood_OU})
        records.append(record)

    with open(f'out/models_{min_length}.tsv', 'w') as file:
        if records:
            header = records[0]
            file.write('\t'.join(header) + '\n')
        for record in records:
            file.write('\t'.join(str(record.get(field, 'nan')) for field in header) + '\n')
