"""Fit models of evolution to features at tips."""

import multiprocessing as mp
import os
import re

import numpy as np
import pandas as pd
import skbio
import src.phylo as phylo
import src.utils as utils


def get_args(grouped, tree, feature_labels, group_labels):
    for name, group in grouped:
        yield name, group, tree, feature_labels, group_labels


def get_models(args):
    # Unpack variables
    name, group, tree, feature_labels, group_labels = args
    OGid, start, stop, disorder = name

    # Calculate some common quantities for all features
    spid2idx = {spid: idx for idx, spid in zip(group.index, group['spid'])}
    tree = tree_template.shear(group['spid'])
    tips, cov = phylo.get_brownian_covariance(tree)
    inv = np.linalg.inv(cov)

    record = {('OGid', 'ids_group'): OGid,
              ('start', 'ids_group'): start,
              ('stop', 'ids_group'): stop}
    for feature_label, group_label in zip(feature_labels, group_labels):
        # Assign values to tips and extract vector
        # This is done in two ways because the MLE functions have different call signatures
        # as a result of some technical details relating to how they are implemented
        for tip in tips:
            idx = spid2idx[tip.name]
            tip.value = group.loc[idx, feature_label]
        values = np.array([tip.value for tip in tips])

        if np.allclose(values, values.mean(), rtol=0, atol=1E-10):  # Use only absolute tolerance
            # If values are constant the model behaviors are technically undefined.
            # The Brownian case has a reasonable limit of a constant random variable with mean of the observed
            # value. The log-likelihood can be taken as 0 since the observed value occurs with certainty.
            #
            # The OU case is more subtle because the model is technically unidentifiable because sigma2 = 0 or
            # alpha = inf could cause zero observed variance. The convention taken here is to assume sigma2 = 0, and
            # consider alpha as unidentifiable from the data, i.e. set its value to nan. Under this interpretation,
            # the log-likelihood is 0 because it also reduces to a constant random variable.
            mu_BM, sigma2_BM = values[0], 0
            loglikelihood_BM = 0

            mu_OU, sigma2_OU, alpha_OU = values[0], 0, np.nan
            loglikelihood_OU = 0
        else:
            mu_BM, sigma2_BM = phylo.get_brownian_mles(cov=cov, inv=inv, values=values)
            loglikelihood_BM = phylo.get_brownian_loglikelihood(mu_BM, sigma2_BM, cov=cov, inv=inv, values=values)

            mu_OU, sigma2_OU, alpha_OU = phylo.get_OU_mles(tips=tips, ts=cov)
            loglikelihood_OU = phylo.get_OU_loglikelihood(mu_OU, sigma2_OU, alpha_OU, tips=tips, ts=cov)

        record.update({(f'{feature_label}_mu_BM', group_label): mu_BM,
                       (f'{feature_label}_sigma2_BM', group_label): sigma2_BM,
                       (f'{feature_label}_loglikelihood_BM', group_label): loglikelihood_BM,
                       (f'{feature_label}_mu_OU', group_label): mu_OU,
                       (f'{feature_label}_sigma2_OU', group_label): sigma2_OU,
                       (f'{feature_label}_alpha_OU', group_label): alpha_OU,
                       (f'{feature_label}_loglikelihood_OU', group_label): loglikelihood_OU})

    return record


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
        for header, _ in utils.read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
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
        with open(f'../../IDRpred/regions_filter/out/regions_{min_length}.tsv') as file:
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

        args = get_args(regions, tree_template, feature_labels, group_labels)
        with mp.Pool(processes=num_processes) as pool:
            records = pool.map(get_models, args, chunksize=10)

        with open(f'out/models_{min_length}.tsv', 'w') as file:
            if records:
                field_names = list(records[0])
                file.write('\t'.join([feature_label for feature_label, _ in field_names]) + '\n')
                file.write('\t'.join([group_label for _, group_label in field_names]) + '\n')
            for record in records:
                file.write('\t'.join(str(record.get(field_name, 'nan')) for field_name in field_names) + '\n')

"""
NOTES
get_models defines a "magic number" atol in its function body as part of checking if all tip values are effectively the
same. This was not refactored into a global variable to keep the function body self-contained.

This approximate equality checking was necessary because some regions have tip values of kappa and omega that are very
nearly one to the 15th or 16th decimal place. It's unclear why this happens, and for time the root causes were not
investigated at the level of individual sequences. However, it's likely these sequences are near the boundaries where
the definitions of kappa and omega break down, i.e. there are few residues belonging to either of the classes whose
separation is measured by these features.
"""