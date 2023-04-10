"""Plot statistics associated with score contrasts."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pdidx = pd.IndexSlice
min_lengths = [30, 60, 90]

for min_length in min_lengths:
    # Load regions as segments
    rows = []
    with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
            for ppid in fields['ppids'].split(','):
                rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder, 'ppid': ppid})
    all_segments = pd.DataFrame(rows)
    all_regions = all_segments.drop('ppid', axis=1).drop_duplicates()

    feature_roots = pd.read_table(f'../get_contrasts/out/features/roots_{min_length}.tsv', header=[0, 1])
    feature_labels = [feature_label for feature_label, group_label in feature_roots.columns
                      if group_label != 'ids_group']
    nonmotif_labels = [feature_label for feature_label, group_label in feature_roots.columns
                       if group_label not in ['ids_group', 'motifs_group']]

    feature_roots = feature_roots.droplevel(1, axis=1)
    feature_roots = all_regions.merge(feature_roots, how='left', on=['OGid', 'start', 'stop'])
    feature_roots = feature_roots.set_index(['OGid', 'start', 'stop', 'disorder'])

    feature_contrasts = pd.read_table(f'../get_contrasts/out/features/contrasts_{min_length}.tsv', skiprows=[1])  # Skip group row
    feature_contrasts = all_regions.merge(feature_contrasts, how='left', on=['OGid', 'start', 'stop'])
    feature_contrasts = feature_contrasts.set_index(['OGid', 'start', 'stop', 'disorder', 'contrast_id'])

    score_roots = pd.read_table(f'../get_contrasts/out/scores/roots_{min_length}.tsv')
    score_roots = all_regions.merge(score_roots, how='left', on=['OGid', 'start', 'stop'])
    score_roots = score_roots.set_index(['OGid', 'start', 'stop', 'disorder'])

    score_contrasts = pd.read_table(f'../get_contrasts/out/scores/contrasts_{min_length}.tsv')
    score_contrasts = all_regions.merge(score_contrasts, how='left', on=['OGid', 'start', 'stop'])
    score_contrasts = score_contrasts.set_index(['OGid', 'start', 'stop', 'disorder', 'contrast_id'])

    score_rates = (score_contrasts ** 2).groupby(['OGid', 'start', 'stop', 'disorder']).mean()
    feature_rates = (feature_contrasts ** 2).groupby(['OGid', 'start', 'stop', 'disorder']).mean()

    prefix = f'out/regions_{min_length}/scores/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Hexbin of scores with features contrasts
    for feature_label in feature_labels:
        fig, ax = plt.subplots()
        hb = ax.hexbin(score_contrasts['score_fraction'], feature_contrasts[feature_label],
                       gridsize=75, mincnt=1, linewidth=0, bins='log')
        ax.set_xlabel('Average disorder score contrasts')
        ax.set_ylabel(f'{feature_label} contrasts')
        fig.colorbar(hb)
        fig.savefig(f'{prefix}/hexbin_{feature_label}-score_fraction.png')
        plt.close()

    # Correlation of score contrasts with feature contrasts
    corr_stack = []
    rng = np.random.default_rng(1)
    for _ in range(10000):
        x = rng.permutation(score_contrasts['score_fraction'].to_numpy())
        y = feature_contrasts.to_numpy()
        corr = np.corrcoef(x, y, rowvar=False)
        corr_stack.append(corr[1:, 0])  # Remove score_fraction self correlation
    corr_stack = np.stack(corr_stack)

    ys = np.arange(len(feature_labels))
    xs = np.corrcoef(score_contrasts['score_fraction'], feature_contrasts, rowvar=False)[1:, 0]  # Remove score_fraction self correlation

    # Calculate two-sided permutation p-values using conventions from SciPy permutation_test
    right = xs <= corr_stack
    left = xs >= corr_stack
    ps_right = (right.sum(axis=0) + 1) / (len(right) + 1)
    ps_left = (left.sum(axis=0) + 1) / (len(left) + 1)
    ps = 2 * np.minimum(ps_right, ps_left)
    pvalues = pd.DataFrame({'feature_label': feature_labels, 'pvalue': ps})
    pvalues.to_csv(f'{prefix}/pvalues.tsv', sep='\t', index=False)

    fig, ax = plt.subplots(figsize=(4.8, 8), layout='constrained')
    ax.invert_yaxis()
    ax.set_ymargin(0.01)
    ax.barh(ys, xs)
    ax.set_yticks(ys, feature_labels, fontsize=6)
    ax.set_xlabel('Correlation between feature and score contrasts')
    ax.set_ylabel('Feature')
    ax.set_title('All regions')
    alpha = 0.001
    offset = (xs.max() - xs.min()) / 200
    for y, x, p in zip(ys, xs, ps):
        if p <= alpha:
            sign = 1 if x >= 0 else -1
            rotation = -90 if x >= 0 else 90
            ax.text(x + sign * offset, y, '*', fontsize=6, va='center', ha='center', rotation=rotation)
    fig.savefig(f'{prefix}/bar_feature-score_contrast_corr.png')
    plt.close()

    # Correlation of scores rate with features
    ys = np.arange(len(feature_labels))
    xs = np.corrcoef(score_rates['score_fraction'], feature_roots, rowvar=False)[1:, 0]  # Remove score_fraction self correlation

    fig, ax = plt.subplots(figsize=(4.8, 8), layout='constrained')
    ax.invert_yaxis()
    ax.set_ymargin(0.01)
    ax.barh(ys, xs)
    ax.set_yticks(ys, feature_labels, fontsize=6)
    ax.set_xlabel('Correlation between feature roots and score rates')
    ax.set_ylabel('Feature')
    ax.set_title('All regions')
    fig.savefig(f'{prefix}/bar_feature_root-score_rate_corr.png')
    plt.close()

    disorder = score_rates.loc[pdidx[:, :, :, True], :]
    order = score_rates.loc[pdidx[:, :, :, False], :]
    fig, axs = plt.subplots(2, 1, sharex=True)
    xmin, xmax = score_rates['score_fraction'].min(), score_rates['score_fraction'].max()
    axs[0].hist(disorder['score_fraction'], bins=np.linspace(xmin, xmax, 150), color='C0', label='disorder')
    axs[1].hist(order['score_fraction'], bins=np.linspace(xmin, xmax, 150), color='C1', label='order')
    axs[1].set_xlabel('Score rate')
    axs[0].set_title(f'minimum length ≥ {min_length}')
    for ax in axs:
        ax.set_ylabel('Number of regions')
        ax.legend()
    fig.savefig(f'{prefix}/hist_numregions-score_rate.png')
    for ax in axs:
        ax.set_yscale('log')
    fig.savefig(f'{prefix}/hist_numregions-score_rate_log.png')
    plt.close()
