"""Plot statistics related to TFs and CFs."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import fisher_exact

pdidx = pd.IndexSlice
min_lengths = [30, 60, 90]

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 1
min_indel_rate = 0.1

with open('../../brownian/simulate_stats/out/BM/critvals.json') as file:
    critvals = json.load(file)

for min_length in min_lengths:
    prefix = f'out/regions_{min_length}/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Load regions
    rows = []
    with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
            rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder})
    all_regions = pd.DataFrame(rows)

    # Load TFs and CFs
    TFs = pd.read_table('../OG_intersect/out/TFs.tsv')
    CFs = pd.read_table('../OG_intersect/out/CFs.tsv')

    # Filter by rates
    asr_rates = pd.read_table(f'../../evofit/asr_stats/out/regions_{min_length}/rates.tsv')
    asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
    row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
    asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

    row_idx = (asr_rates['aa_rate_mean'] > min_aa_rate) | (asr_rates['indel_rate_mean'] > min_indel_rate)
    column_idx = ['OGid', 'start', 'stop', 'disorder']
    region_keys = asr_rates.loc[row_idx, column_idx]

    # Load models
    models = pd.read_table(f'../../brownian/model_compute/out/models_{min_length}.tsv', header=[0, 1])
    models = region_keys.merge(models.droplevel(1, axis=1), how='left', on=['OGid', 'start', 'stop'])
    models = models.set_index(['OGid', 'start', 'stop', 'disorder'])

    # Extract labels
    feature_groups = {}
    feature_labels = []
    nonmotif_labels = []
    with open(f'../../brownian/model_compute/out/models_{min_length}.tsv') as file:
        column_labels = file.readline().rstrip('\n').split('\t')
        group_labels = file.readline().rstrip('\n').split('\t')
    for column_label, group_label in zip(column_labels, group_labels):
        if not column_label.endswith('_loglikelihood_BM') or group_label == 'ids_group':
            continue
        feature_label = column_label.removesuffix('_loglikelihood_BM')
        try:
            feature_groups[group_label].append(feature_label)
        except KeyError:
            feature_groups[group_label] = [feature_label]
        feature_labels.append(feature_label)
        if group_label != 'motifs_group':
            nonmotif_labels.append(feature_label)

    # Calculate delta loglikelihood
    columns = {}
    for feature_label in feature_labels:
        columns[f'{feature_label}_delta_loglikelihood'] = models[f'{feature_label}_loglikelihood_OU'] - models[f'{feature_label}_loglikelihood_BM']
    models = pd.concat([models, pd.DataFrame(columns)], axis=1)

    # Get models subsets
    models_NF = models[~models.index.get_level_values('OGid').isin(set().union(TFs['OGid'], CFs['OGid']))]
    models_TF = models.reset_index().merge(TFs['OGid'], on='OGid').set_index(['OGid', 'start', 'stop', 'disorder'])
    models_CF = models.reset_index().merge(CFs['OGid'], on='OGid').set_index(['OGid', 'start', 'stop', 'disorder'])
    models_stack = [models_NF, models_TF, models_CF]
    models_labels = ['non-TF/CF', 'TF', 'CF']
    models_colors = ['C0', 'C1', 'C2']

    # Bar graph of delta fraction of regions with a significant feature
    column_labels = [f'{feature_label}_delta_loglikelihood' for feature_label in feature_labels]
    ys_NF = (models_NF.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).mean()
    ys_TF = (models_TF.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).mean()
    ys_CF = (models_CF.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).mean()
    xs = list(range(len(column_labels)))
    xs_labels = [label.removesuffix('_delta_loglikelihood') for label in column_labels]

    plots = [(ys_TF - ys_NF, 'TF - NF'),
             (ys_CF - ys_NF, 'CF - NF')]
    fig, axs = plt.subplots(len(plots), 1, figsize=(7.5, 4),
                            gridspec_kw={'left': 0.1, 'right': 0.995, 'bottom': 0.25, 'top': 0.925, 'hspace': 0.5})
    for ax, (data, data_label) in zip(axs, plots):
        ax.bar(xs, data)
        ax.set_xmargin(0.005)
        ax.set_xticks(xs, ['' for _ in xs])
        ax.set_ylabel('Difference in fraction')
        ax.set_title(data_label)
    ax.set_xticks(xs, xs_labels, fontsize=5.5,
                  rotation=60, rotation_mode='anchor', ha='right', va='center')
    fig.savefig(f'{prefix}/bar_regionfracdiff-feature.png')
    plt.close()

    # Bar graph of delta probability that feature is conserved given a feature is conserved
    column_labels = [f'{feature_label}_delta_loglikelihood' for feature_label in feature_labels]
    n_NF = (models_NF.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).sum()
    n_TF = (models_TF.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).sum()
    n_CF = (models_CF.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).sum()
    xs = list(range(len(column_labels)))
    xs_labels = [label.removesuffix('_delta_loglikelihood') for label in column_labels]

    plots = [(n_TF, n_NF, 'TF - NF'),
             (n_CF, n_NF, 'CF - NF')]
    fig, axs = plt.subplots(len(plots), 1, figsize=(7.5, 4),
                            gridspec_kw={'left': 0.1, 'right': 0.995, 'bottom': 0.25, 'top': 0.925, 'hspace': 0.5})
    for ax, (n1, n2, data_label) in zip(axs, plots):
        N1, N2 = n1.sum(), n2.sum()
        ys = n1/N1 - n2/N2

        ax.bar(xs, ys)
        ax.set_xticks(xs, ['' for _ in xs])
        ax.set_xmargin(0.005)
        ax.set_ymargin(0.1)
        ax.set_ylabel('Difference in fraction')
        ax.set_title(data_label)

        # Compute and plot significance tests for each feature
        ps = []
        for column_label in column_labels:
            table = [[n1[column_label], N1 - n1[column_label]],
                     [n2[column_label], N2 - n2[column_label]]]
            result = fisher_exact(table, alternative='two-sided')
            ps.append(result.pvalue)

        alpha = 0.01
        offset = (ys.max() - ys.min()) / 100
        for x, y, p in zip(xs, ys, ps):
            if p <= alpha:
                sign = 1 if y >= 0 else -1
                rotation = 0 if y >= 0 else -180
                ax.text(x, y + sign * offset, '*', fontsize=6, va='center', ha='center', rotation=rotation)
    ax.set_xticks(xs, xs_labels, fontsize=5.5,
                  rotation=60, rotation_mode='anchor', ha='right', va='center')
    fig.savefig(f'{prefix}/bar_featurefracdiff-feature.png')
    plt.close()

    # Distribution of number of significant features in regions
    column_labels = [f'{feature_label}_delta_loglikelihood' for feature_label in feature_labels]
    fig, axs = plt.subplots(3, 1, sharex=True, layout='constrained')
    for ax, data, data_label, color in zip(axs, models_stack, models_labels, models_colors):
        counts = (data.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).sum(axis=1).value_counts()
        ax.bar(counts.index, counts.values, label=data_label, color=color)
        ax.set_xmargin(0.01)
        ax.set_ylabel('Number of regions')
        ax.legend(fontsize=8)
    ax.set_xlabel('Number of significant features')
    fig.savefig(f'{prefix}/bar_regionnum-numfeatures.png')
    plt.close()
