"""Plot statistics associated with score contrasts."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

min_lengths = [30, 60, 90]

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 1
min_indel_rate = 1

for min_length in min_lengths:
    # Load regions as segments
    rows = []
    with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields[
                'disorder'] == 'True'
            for ppid in fields['ppids'].split(','):
                rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder, 'ppid': ppid})
    all_segments = pd.DataFrame(rows)
    all_regions = all_segments.drop('ppid', axis=1).drop_duplicates()

    # Load and format data
    asr_rates = pd.read_table(f'../../evofit/asr_stats/out/regions_{min_length}/rates.tsv')
    asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
    row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
    asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

    row_idx = (asr_rates['aa_rate_mean'] > min_aa_rate) | (asr_rates['indel_rate_mean'] > min_indel_rate)
    column_idx = ['OGid', 'start', 'stop', 'disorder']
    region_keys = asr_rates.loc[row_idx, column_idx]
    segment_keys = all_segments.merge(region_keys, how='right', on=['OGid', 'start', 'stop', 'disorder'])

    feature_roots = pd.read_table(f'../get_contrasts/out/features/roots_{min_length}.tsv', header=[0, 1])
    feature_labels = [feature_label for feature_label, group_label in feature_roots.columns
                      if group_label != 'ids_group']
    nonmotif_labels = [feature_label for feature_label, group_label in feature_roots.columns
                       if group_label not in ['ids_group', 'motifs_group']]

    feature_roots = feature_roots.droplevel(1, axis=1)
    feature_roots = region_keys.merge(feature_roots, how='left', on=['OGid', 'start', 'stop'])
    feature_roots = feature_roots.set_index(['OGid', 'start', 'stop', 'disorder'])

    feature_contrasts = pd.read_table(f'../get_contrasts/out/features/contrasts_{min_length}.tsv', skiprows=[1])  # Skip group row
    feature_contrasts = region_keys.merge(feature_contrasts, how='left', on=['OGid', 'start', 'stop'])
    feature_contrasts = feature_contrasts.set_index(['OGid', 'start', 'stop', 'disorder', 'contrast_id'])

    score_roots = pd.read_table(f'../get_contrasts/out/scores/roots_{min_length}.tsv', skiprows=[1])  # Skip group row
    score_roots = region_keys.merge(score_roots, how='left', on=['OGid', 'start', 'stop'])
    score_roots = score_roots.set_index(['OGid', 'start', 'stop', 'disorder'])

    score_contrasts = pd.read_table(f'../get_contrasts/out/scores/contrasts_{min_length}.tsv', skiprows=[1])  # Skip group row
    score_contrasts = region_keys.merge(score_contrasts, how='left', on=['OGid', 'start', 'stop'])
    score_contrasts = score_contrasts.set_index(['OGid', 'start', 'stop', 'disorder', 'contrast_id'])

    prefix = f'out/regions_{min_length}/scores/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Hexbin of scores with features contrasts
    for feature_label in feature_labels:
        fig, ax = plt.subplots()
        hb = ax.hexbin(score_contrasts['scores_fraction'], feature_contrasts[feature_label],
                       gridsize=75, mincnt=1, linewidth=0, bins='log')
        ax.set_xlabel('Average disorder score contrasts')
        ax.set_ylabel(f'{feature_label} contrasts')
        fig.colorbar(hb)
        fig.savefig(f'{prefix}/hexbin_{feature_label}-scores_fraction.png')
        plt.close()

    # Correlation of score contrasts with feature contrasts
    corr_stack = []
    rng = np.random.default_rng(1)
    for _ in range(1000):
        x = rng.permutation(score_contrasts['scores_fraction'].to_numpy())
        y = feature_contrasts.to_numpy()
        corr = np.corrcoef(x, y, rowvar=False)
        corr_stack.append(corr[1:, 0])  # Remove scores_fraction self correlation
    corr_stack = np.stack(corr_stack)

    ys = np.arange(len(feature_labels))
    ws = np.corrcoef(score_contrasts['scores_fraction'], feature_contrasts, rowvar=False)[1:, 0]  # Remove scores_fraction self correlation

    # Calculate two-sided permutation p-values using conventions from SciPy permutation_test
    right = ws <= corr_stack
    left = ws >= corr_stack
    pvalues_right = (right.sum(axis=0) + 1) / (len(right) + 1)
    pvalues_left = (left.sum(axis=0) + 1) / (len(left) + 1)
    pvalues = 2 * np.minimum(pvalues_right, pvalues_left)

    fig, ax = plt.subplots(figsize=(4.8, 8), layout='constrained')
    ax.invert_yaxis()
    ax.set_ymargin(0.01)
    ax.barh(ys, ws)
    ax.set_yticks(ys, feature_labels, fontsize=6)
    ax.set_xlabel('Correlation between feature and score contrasts')
    ax.set_ylabel('Feature')
    ax.set_title('All regions')
    alpha = 0.01
    offset = (ws.max() - ws.min()) / 200
    for y, w, pvalue in zip(ys, ws, pvalues):
        if pvalue <= alpha:
            sign = 1 if w >= 0 else -1
            rotation = -90 if w >= 0 else 90
            ax.text(w + sign * offset, y, '*', fontsize=6, va='center', ha='center', rotation=rotation)
    fig.savefig(f'{prefix}/bar_feature-score_contrast_corr.png')
    plt.close()

    # Correlation of scores rate with features
    scores_rates = (score_contrasts ** 2).groupby(['OGid', 'start', 'stop']).mean()
    feature_rates = (feature_contrasts ** 2).groupby(['OGid', 'start', 'stop']).mean()

    ys = np.arange(len(feature_labels))
    ws = np.corrcoef(scores_rates['scores_fraction'], feature_roots, rowvar=False)[1:, 0]  # Remove scores_fraction self correlation

    fig, ax = plt.subplots(figsize=(4.8, 8), layout='constrained')
    ax.invert_yaxis()
    ax.set_ymargin(0.01)
    ax.barh(ys, ws)
    ax.set_yticks(ys, feature_labels, fontsize=6)
    ax.set_xlabel('Correlation between feature roots and score rates')
    ax.set_ylabel('Feature')
    ax.set_title('All regions')
    fig.savefig(f'{prefix}/bar_feature_root-score_rate_corr.png')
    plt.close()
