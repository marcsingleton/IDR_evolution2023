"""Plot statistics associated with feature contrasts."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import linregress
from sklearn.decomposition import PCA
from src.brownian.pca import plot_pca, plot_pca_arrows, plot_pca2, plot_pca2_arrows


def zscore(df):
    return (df - df.mean()) / df.std()


pdidx = pd.IndexSlice
min_lengths = [30, 60, 90]

min_indel_columns = 5  # Indel rates below this value are set to 0

pca_components = 10
cmap1, cmap2, cmap3 = plt.colormaps['Blues'], plt.colormaps['Oranges'], plt.colormaps['Purples']
color1, color2, color3 = '#4e79a7', '#f28e2b', '#b07aa1'
hexbin_kwargs = {'gridsize': 75, 'mincnt': 1, 'linewidth': 0}
hexbin_kwargs_log = {'gridsize': 75, 'mincnt': 1, 'linewidth': 0, 'bins': 'log'}
handle_markerfacecolor = 0.6
legend_kwargs = {'fontsize': 8, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
arrow_colors = ['#e15759', '#499894', '#59a14f', '#f1ce63', '#b07aa1', '#d37295', '#9d7660', '#bab0ac',
                '#ff9d9a', '#86bcb6', '#8cd17d', '#b6992d', '#d4a6c8', '#fabfd2', '#d7b5a6', '#79706e']

# Load features
all_features = pd.read_table('../feature_compute/out/features.tsv', header=[0, 1])
all_features.loc[all_features[('kappa', 'charge_group')] == -1, 'kappa'] = 1  # Need to specify full column index to get slicing to work
all_features.loc[all_features[('omega', 'charge_group')] == -1, 'omega'] = 1
all_features['length'] = all_features['length'] ** 0.6
all_features.rename(columns={'length': 'radius_gyration'}, inplace=True)

feature_labels = [feature_label for feature_label, group_label in all_features.columns
                  if group_label != 'ids_group']
nonmotif_labels = [feature_label for feature_label, group_label in all_features.columns
                   if group_label not in ['ids_group', 'motifs_group']]
all_features = all_features.droplevel(1, axis=1)

for min_length in min_lengths:
    if not os.path.exists(f'out/regions_{min_length}/'):
        os.makedirs(f'out/regions_{min_length}/')

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

    # Load and format data
    asr_rates = pd.read_table(f'../../evofit/asr_stats/out/regions_{min_length}/rates.tsv')
    asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
    row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
    asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

    features = all_segments.merge(all_features, how='left', on=['OGid', 'start', 'stop', 'ppid'])
    features = features.groupby(['OGid', 'start', 'stop', 'disorder']).mean()
    features = all_regions.merge(features, how='left', on=['OGid', 'start', 'stop', 'disorder'])
    features = features.set_index(['OGid', 'start', 'stop', 'disorder'])

    roots = pd.read_table(f'../contrast_compute/out/features/roots_{min_length}.tsv', skiprows=[1])  # Skip group row
    roots = all_regions.merge(roots, how='left', on=['OGid', 'start', 'stop'])
    roots = roots.set_index(['OGid', 'start', 'stop', 'disorder'])

    contrasts = pd.read_table(f'../contrast_compute/out/features/contrasts_{min_length}.tsv', skiprows=[1])  # Skip group row
    contrasts = all_regions.merge(contrasts, how='left', on=['OGid', 'start', 'stop'])
    contrasts = contrasts.set_index(['OGid', 'start', 'stop', 'disorder', 'contrast_id'])

    rates = (contrasts ** 2).groupby(['OGid', 'start', 'stop', 'disorder']).mean()

    # 1 CONTRASTS
    prefix = f'out/regions_{min_length}/contrasts/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    # 1.1 Plot contrast distributions
    disorder = contrasts.loc[pdidx[:, :, :, True, :], :]
    order = contrasts.loc[pdidx[:, :, :, False, :], :]
    for feature_label in feature_labels:
        fig, axs = plt.subplots(2, 1, sharex=True)
        xmin, xmax = contrasts[feature_label].min(), contrasts[feature_label].max()
        axs[0].hist(disorder[feature_label], bins=np.linspace(xmin, xmax, 150), color=color1, label='disorder')
        axs[1].hist(order[feature_label], bins=np.linspace(xmin, xmax, 150), color=color2, label='order')
        axs[1].set_xlabel(f'Contrast value ({feature_label})')
        axs[0].set_title(f'minimum length ≥ {min_length}')
        for ax in axs:
            ax.set_ylabel('Number of contrasts')
            ax.legend()
        fig.savefig(f'{prefix}/hist_numcontrasts-{feature_label}.png')
        for ax in axs:
            ax.set_yscale('log')
        fig.savefig(f'{prefix}/hist_numcontrasts-{feature_label}_log.png')
        plt.close()

    # 1.2 Plot correlation heatmaps
    plots = [(contrasts[nonmotif_labels], 'all', 'All regions'),
             (disorder[nonmotif_labels], 'disorder', 'Disorder regions'),
             (order[nonmotif_labels], 'order', 'Order regions')]
    for data, data_label, title_label in plots:
        corr = np.corrcoef(data, rowvar=False)

        fig, ax = plt.subplots(figsize=(7.5, 6), gridspec_kw={'left': 0.075, 'right': 0.99, 'top': 0.95, 'bottom': 0.125})
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu')
        ax.set_xticks(range(len(corr)), nonmotif_labels, fontsize=6,
                      rotation=60, rotation_mode='anchor', ha='right', va='center')
        ax.set_yticks(range(len(corr)), nonmotif_labels, fontsize=6)
        ax.set_title(title_label)
        fig.colorbar(im)
        fig.savefig(f'{prefix}/heatmap_corr_{data_label}.png')
        plt.close()

    corr1 = np.corrcoef(disorder[nonmotif_labels], rowvar=False)
    corr2 = np.corrcoef(order[nonmotif_labels], rowvar=False)
    corr = corr1 - corr2
    vext = np.abs(corr).max()

    fig, ax = plt.subplots(figsize=(7.5, 6), gridspec_kw={'left': 0.075, 'right': 0.99, 'top': 0.95, 'bottom': 0.125})
    im = ax.imshow(corr, vmin=-vext, vmax=vext, cmap='RdBu')
    ax.set_xticks(range(len(corr)), nonmotif_labels, fontsize=6,
                  rotation=60, rotation_mode='anchor', ha='right', va='center')
    ax.set_yticks(range(len(corr)), nonmotif_labels, fontsize=6)
    ax.set_title('Difference between disorder and order regions')
    fig.colorbar(im)
    fig.savefig(f'{prefix}/heatmap_corr_delta.png')
    plt.close()

    # 2 RATES
    prefix = f'out/regions_{min_length}/rates/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    # 2.1 Plot rate distributions
    rates_nonmotif = rates[nonmotif_labels]
    disorder = rates.loc[pdidx[:, :, :, True], :]
    order = rates.loc[pdidx[:, :, :, False], :]
    disorder_nonmotif = disorder[nonmotif_labels]
    order_nonmotif = order[nonmotif_labels]
    for feature_label in feature_labels:
        fig, axs = plt.subplots(2, 1, sharex=True)
        xmin, xmax = rates[feature_label].min(), rates[feature_label].max()
        axs[0].hist(disorder[feature_label], bins=np.linspace(xmin, xmax, 150), color=color1, label='disorder')
        axs[1].hist(order[feature_label], bins=np.linspace(xmin, xmax, 150), color=color2, label='order')
        axs[1].set_xlabel(f'Rate ({feature_label})')
        axs[0].set_title(f'minimum length ≥ {min_length}')
        for ax in axs:
            ax.set_ylabel('Number of regions')
            ax.legend()
        fig.savefig(f'{prefix}/hist_numregions-{feature_label}.png')
        for ax in axs:
            ax.set_yscale('log')
        fig.savefig(f'{prefix}/hist_numregions-{feature_label}_log.png')
        plt.close()

    # 2.2 Plot correlation heatmaps
    plots = [(rates[nonmotif_labels], 'all', 'All regions'),
             (disorder[nonmotif_labels], 'disorder', 'Disorder regions'),
             (order[nonmotif_labels], 'order', 'Order regions')]
    for data, data_label, title_label in plots:
        corr = np.corrcoef(data, rowvar=False)

        fig, ax = plt.subplots(figsize=(7.5, 6), gridspec_kw={'left': 0.075, 'right': 0.99, 'top': 0.95, 'bottom': 0.125})
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu')
        ax.set_xticks(range(len(corr)), nonmotif_labels, fontsize=6,
                      rotation=60, rotation_mode='anchor', ha='right', va='center')
        ax.set_yticks(range(len(corr)), nonmotif_labels, fontsize=6)
        ax.set_title(title_label)
        fig.colorbar(im)
        fig.savefig(f'{prefix}/heatmap_corr_{data_label}.png')
        plt.close()

    corr1 = np.corrcoef(disorder[nonmotif_labels], rowvar=False)
    corr2 = np.corrcoef(order[nonmotif_labels], rowvar=False)
    corr = corr1 - corr2
    vext = np.abs(corr).max()

    fig, ax = plt.subplots(figsize=(7.5, 6), gridspec_kw={'left': 0.075, 'right': 0.99, 'top': 0.95, 'bottom': 0.125})
    im = ax.imshow(corr, vmin=-vext, vmax=vext, cmap='RdBu')
    ax.set_xticks(range(len(corr)), nonmotif_labels, fontsize=6,
                  rotation=60, rotation_mode='anchor', ha='right', va='center')
    ax.set_yticks(range(len(corr)), nonmotif_labels, fontsize=6)
    ax.set_title('Difference between disorder and order regions')
    fig.colorbar(im)
    fig.savefig(f'{prefix}/heatmap_corr_delta.png')
    plt.close()

    # 2.3.1 Plot rate PCAs (combined)
    plots = [(rates, 'all', f'minimum length ≥ {min_length}, no norm, all features', 'nonorm_all'),
             (zscore(rates), 'all', f'minimum length ≥ {min_length}, z-score, all features', 'zscore_all'),
             (rates_nonmotif, 'all', f'minimum length ≥ {min_length}, no norm, non-motif features', 'nonorm_nonmotif'),
             (zscore(rates_nonmotif), 'all', f'minimum length ≥ {min_length}, z-score, non-motif features', 'zscore_nonmotif')]
    for data, data_label, title_label, file_label in plots:
        pca = PCA(n_components=pca_components)
        transform = pca.fit_transform(data.to_numpy())
        idx = data.index.get_level_values('disorder').array.astype(bool)
        color = color3

        # Feature variance bar chart
        var = data.var().sort_values(ascending=False)
        var = var / var.sum()
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        fig, ax = plt.subplots(gridspec_kw={'bottom': 0.3})
        ax.bar(range(len(truncate.index)), truncate.values)
        ax.set_xticks(range(len(truncate.index)), truncate.index,
                      rotation=60, rotation_mode='anchor', ha='right', va='center')
        ax.set_ylabel('Explained variance ratio')
        fig.savefig(f'{prefix}/bar_variance_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        fig, ax = plt.subplots()
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
               label=data_label, color=color)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(title_label)
        ax.legend()
        fig.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
        plt.close()

        # PCA scatters
        fig = plot_pca2(transform, 0, 1, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
                        hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca2_arrows(pca, transform, data.columns, 0, 1, idx, ~idx, cmap1, cmap2, title_label,
                               hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png')
        plt.close()

        fig = plot_pca2(transform, 1, 2, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
                        hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor)
        fig.savefig(f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca2_arrows(pca, transform, data.columns, 1, 2, idx, ~idx, cmap1, cmap2, title_label,
                               hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)
        fig.savefig(f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png')
        plt.close()

        # PC1 against ASR rates
        pca = PCA(n_components=pca_components)
        df = data.merge(asr_rates, left_index=True, right_on=['OGid', 'start', 'stop', 'disorder'])
        transform = pca.fit_transform(df[data.columns].to_numpy())

        xs = transform[:, 0]
        ys = df['aa_rate_mean'] + df['indel_rate_mean']
        result = linregress(xs, ys)
        m, b = result.slope, result.intercept
        r2 = result.rvalue ** 2
        xmin, xmax = xs.min(), xs.max()
        xpos = xmax if m >= 0 else xmin
        ha = 'right' if m >= 0 else 'left'

        fig, ax = plt.subplots()
        hb = ax.hexbin(xs, ys, gridsize=75, mincnt=1, linewidth=0)
        ax.plot([xmin, xmax], [m * xmin + b, m * xmax + b], color='black', linewidth=1)
        ax.annotate(r'$\mathregular{R^2}$' + f' = {r2:.2f}', (xpos, m * xpos + b), ha=ha, va='bottom')
        ax.set_xlabel('PC1')
        ax.set_ylabel('Sum of average amino acid and indel rates in region')
        ax.set_title(title_label)
        fig.colorbar(hb)

        fig.savefig(f'{prefix}/hexbin_pc1-rate_{data_label}_{file_label}.png')
        plt.close()

    # 2.3.2 Plot rate PCAs (individual)
    plots = [(disorder, 'disorder', f'minimum length ≥ {min_length}, no norm, all features', 'nonorm_all'),
             (order, 'order', f'minimum length ≥ {min_length}, no norm, all features', 'nonorm_all'),
             (zscore(disorder), 'disorder', f'minimum length ≥ {min_length}, z-score, all features', 'zscore_all'),
             (zscore(order), 'order', f'minimum length ≥ {min_length}, z-score, all features', 'zscore_all'),
             (disorder_nonmotif, 'disorder', f'minimum length ≥ {min_length}, no norm, non-motif features', 'nonorm_nonmotif'),
             (order_nonmotif, 'order', f'minimum length ≥ {min_length}, no norm, non-motif features', 'nonorm_nonmotif'),
             (zscore(disorder_nonmotif), 'disorder', f'minimum length ≥ {min_length}, z-score, non-motif features', 'zscore_nonmotif'),
             (zscore(order_nonmotif), 'order', f'minimum length ≥ {min_length}, z-score, non-motif features', 'zscore_nonmotif')]
    for data, data_label, title_label, file_label in plots:
        pca = PCA(n_components=pca_components)
        transform = pca.fit_transform(data.to_numpy())
        cmap = cmap1 if data_label == 'disorder' else cmap2
        color = color1 if data_label == 'disorder' else color2

        # Feature variance bar chart
        var = data.var().sort_values(ascending=False)
        var = var / var.sum()
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        fig, ax = plt.subplots(gridspec_kw={'bottom': 0.3})
        ax.bar(range(len(truncate.index)), truncate.values)
        ax.set_xticks(range(len(truncate.index)), truncate.index,
                      rotation=60, rotation_mode='anchor', ha='right', va='center')
        ax.set_ylabel('Explained variance ratio')
        fig.savefig(f'{prefix}/bar_variance_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        fig, ax = plt.subplots()
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
               label=data_label, color=color)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(title_label)
        ax.legend()
        fig.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
        plt.close()

        # PCA scatters
        fig = plot_pca(transform, 0, 1, cmap, data_label, title_label,
                       hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca_arrows(pca, transform, data.columns, 0, 1, cmap, title_label,
                              hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png')
        plt.close()

        fig = plot_pca(transform, 1, 2, cmap, data_label, title_label,
                       hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor)
        fig.savefig(f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca_arrows(pca, transform, data.columns, 1, 2, cmap, title_label,
                              hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)
        fig.savefig(f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png')
        plt.close()

        # PC1 against ASR rates
        pca = PCA(n_components=pca_components)
        df = data.merge(asr_rates, left_index=True, right_on=['OGid', 'start', 'stop', 'disorder'])
        transform = pca.fit_transform(df[data.columns].to_numpy())

        xs = transform[:, 0]
        ys = df['aa_rate_mean'] + df['indel_rate_mean']
        result = linregress(xs, ys)
        m, b = result.slope, result.intercept
        r2 = result.rvalue ** 2
        xmin, xmax = xs.min(), xs.max()
        xpos = xmax if m >= 0 else xmin
        ha = 'right' if m >= 0 else 'left'

        fig, ax = plt.subplots()
        hb = ax.hexbin(xs, ys, gridsize=75, mincnt=1, linewidth=0)
        ax.plot([xmin, xmax], [m * xmin + b, m * xmax + b], color='black', linewidth=1)
        ax.annotate(r'$\mathregular{R^2}$' + f' = {r2:.2f}', (xpos, m * xpos + b), ha=ha, va='bottom')
        ax.set_xlabel('PC1')
        ax.set_ylabel('Sum of average amino acid and indel rates in region')
        ax.set_title(title_label)
        fig.colorbar(hb)

        fig.savefig(f'{prefix}/hexbin_pc1-rate_{data_label}_{file_label}.png')
        plt.close()

    # 3 ROOTS
    prefix = f'out/regions_{min_length}/roots/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    # 3.1 Plot root distributions
    roots_nonmotif = roots[nonmotif_labels]
    disorder = roots.loc[pdidx[:, :, :, True], :]
    order = roots.loc[pdidx[:, :, :, False], :]
    disorder_nonmotif = disorder[nonmotif_labels]
    order_nonmotif = order[nonmotif_labels]
    for feature_label in feature_labels:
        fig, axs = plt.subplots(2, 1, sharex=True)
        xmin, xmax = roots[feature_label].min(), roots[feature_label].max()
        axs[0].hist(disorder[feature_label], bins=np.linspace(xmin, xmax, 75), color=color1, label='disorder')
        axs[1].hist(order[feature_label], bins=np.linspace(xmin, xmax, 75), color=color2, label='order')
        axs[1].set_xlabel(f'Root value ({feature_label})')
        axs[0].set_title(f'minimum length ≥ {min_length}')
        for ax in axs:
            ax.set_ylabel('Number of regions')
            ax.legend()
        fig.savefig(f'{prefix}/hist_numregions-{feature_label}.png')
        plt.close()

    # 3.2 Plot correlation heatmaps
    plots = [(roots[nonmotif_labels], 'all', 'All regions'),
             (disorder[nonmotif_labels], 'disorder', 'Disorder regions'),
             (order[nonmotif_labels], 'order', 'Order regions')]
    for data, data_label, title_label in plots:
        corr = np.corrcoef(data, rowvar=False)

        fig, ax = plt.subplots(figsize=(7.5, 6), gridspec_kw={'left': 0.075, 'right': 0.99, 'top': 0.95, 'bottom': 0.125})
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu')
        ax.set_xticks(range(len(corr)), nonmotif_labels, fontsize=6,
                      rotation=60, rotation_mode='anchor', ha='right', va='center')
        ax.set_yticks(range(len(corr)), nonmotif_labels, fontsize=6)
        ax.set_title(title_label)
        fig.colorbar(im)
        fig.savefig(f'{prefix}/heatmap_corr_{data_label}.png')
        plt.close()

    corr1 = np.corrcoef(disorder[nonmotif_labels], rowvar=False)
    corr2 = np.corrcoef(order[nonmotif_labels], rowvar=False)
    corr = corr1 - corr2
    vext = np.abs(corr).max()

    fig, ax = plt.subplots(figsize=(7.5, 6), gridspec_kw={'left': 0.075, 'right': 0.99, 'top': 0.95, 'bottom': 0.125})
    im = ax.imshow(corr, vmin=-vext, vmax=vext, cmap='RdBu')
    ax.set_xticks(range(len(corr)), nonmotif_labels, fontsize=6,
                  rotation=60, rotation_mode='anchor', ha='right', va='center')
    ax.set_yticks(range(len(corr)), nonmotif_labels, fontsize=6)
    ax.set_title('Difference between disorder and order regions')
    fig.colorbar(im)
    fig.savefig(f'{prefix}/heatmap_corr_delta.png')
    plt.close()

    # 3.3.1 Plot root PCAs (combined)
    plots = [(roots, 'all', f'minimum length ≥ {min_length}, no norm, all features', 'nonorm_all'),
             (zscore(roots), 'all', f'minimum length ≥ {min_length}, z-score, all features', 'zscore_all'),
             (roots_nonmotif, 'all', f'minimum length ≥ {min_length}, no norm, non-motif features', 'nonorm_nonmotif'),
             (zscore(roots_nonmotif), 'all', f'minimum length ≥ {min_length}, z-score, non-motif features', 'zscore_nonmotif')]
    for data, data_label, title_label, file_label in plots:
        pca = PCA(n_components=pca_components)
        transform = pca.fit_transform(data.to_numpy())
        idx = data.index.get_level_values('disorder').array.astype(bool)
        cmap, color = cmap3, color3

        # Feature variance bar chart
        var = data.var().sort_values(ascending=False)
        var = var / var.sum()
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        fig, ax = plt.subplots(gridspec_kw={'bottom': 0.3})
        ax.bar(range(len(truncate.index)), truncate.values)
        ax.set_xticks(range(len(truncate.index)), truncate.index,
                      rotation=60, rotation_mode='anchor', ha='right', va='center')
        ax.set_ylabel('Explained variance ratio')
        fig.savefig(f'{prefix}/bar_variance_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        fig, ax = plt.subplots()
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
               label=data_label, color=color)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(title_label)
        ax.legend()
        fig.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
        plt.close()

        # PCA scatters
        fig = plot_pca2(transform, 0, 1, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
                        hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca2_arrows(pca, transform, data.columns, 0, 1, idx, ~idx, cmap1, cmap2, title_label,
                               hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png')
        plt.close()

    # 3.3.2 Plot root PCAs (individual)
    plots = [(disorder, 'disorder', f'minimum length ≥ {min_length}, no norm, all features', 'nonorm_all'),
             (order, 'order', f'minimum length ≥ {min_length}, no norm, all features', 'nonorm_all'),
             (zscore(disorder), 'disorder', f'minimum length ≥ {min_length}, z-score, all features', 'zscore_all'),
             (zscore(order), 'order', f'minimum length ≥ {min_length}, z-score, all features', 'zscore_all'),
             (disorder_nonmotif, 'disorder', f'minimum length ≥ {min_length}, no norm, non-motif features', 'nonorm_nonmotif'),
             (order_nonmotif, 'order', f'minimum length ≥ {min_length}, no norm, non-motif features', 'nonorm_nonmotif'),
             (zscore(disorder_nonmotif), 'disorder', f'minimum length ≥ {min_length}, z-score, non-motif features', 'zscore_nonmotif'),
             (zscore(order_nonmotif), 'order', f'minimum length ≥ {min_length}, z-score, non-motif features', 'zscore_nonmotif')]
    for data, data_label, title_label, file_label in plots:
        pca = PCA(n_components=pca_components)
        transform = pca.fit_transform(data.to_numpy())
        cmap = cmap1 if data_label == 'disorder' else cmap2
        color = color1 if data_label == 'disorder' else color2

        # Feature variance bar chart
        var = data.var().sort_values(ascending=False)
        var = var / var.sum()
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        fig, ax = plt.subplots(gridspec_kw={'bottom': 0.3})
        ax.bar(range(len(truncate.index)), truncate.values)
        ax.set_xticks(range(len(truncate.index)), truncate.index,
                      rotation=60, rotation_mode='anchor', ha='right', va='center')
        ax.set_ylabel('Explained variance ratio')
        fig.savefig(f'{prefix}/bar_variance_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        fig, ax = plt.subplots()
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
               label=data_label, color=color)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(title_label)
        ax.legend()
        fig.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
        plt.close()

        # PCA scatters
        fig = plot_pca(transform, 0, 1, cmap, data_label, title_label,
                       hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca_arrows(pca, transform, data.columns, 0, 1, cmap, title_label,
                              hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png')
        plt.close()

    # 4 MERGE
    prefix = f'out/regions_{min_length}/merge/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    # 4.1 Plot correlations of roots and feature means
    merge = features.merge(roots, how='inner', on=['OGid', 'start', 'stop', 'disorder'], suffixes=('_mean', '_root'))
    for feature_label in feature_labels:
        xs, ys = merge[f'{feature_label}_root'], merge[f'{feature_label}_mean']
        result = linregress(xs, ys)
        m, b = result.slope, result.intercept
        r2 = result.rvalue ** 2
        xmin, xmax = xs.min(), xs.max()

        fig, ax = plt.subplots()
        hb = ax.hexbin(xs, ys, gridsize=75, mincnt=1, linewidth=0)
        ax.set_xlabel('Root value')
        ax.set_ylabel('Tip mean')
        ax.set_title(f'{feature_label}\nminimum length ≥ {min_length}')
        fig.colorbar(hb)
        ax.plot([xmin, xmax], [m * xmin + b, m * xmax + b], color='black', linewidth=1)
        ax.annotate(r'$\mathregular{R^2}$' + f' = {r2:.2f}', (0.85, 0.65), xycoords='axes fraction')
        fig.savefig(f'{prefix}/hexbin_mean-root_{feature_label}.png')
        plt.close()

    ys = np.arange(len(feature_labels))
    xs = []
    for feature_label in feature_labels:
        x = np.corrcoef(merge[f'{feature_label}_root'], merge[f'{feature_label}_mean'])[0, 1]
        xs.append(x)

    fig, ax = plt.subplots(figsize=(4.8, 8), layout='constrained')
    ax.invert_yaxis()
    ax.set_ymargin(0.01)
    ax.barh(ys, xs, color=color3)
    ax.set_yticks(ys, feature_labels, fontsize=6)
    ax.set_xlabel('Correlation between root and tip mean')
    ax.set_ylabel('Feature')
    ax.set_title('All regions')
    fig.savefig(f'{prefix}/bar_mean-root_corr.png')
    plt.close()

    # 4.2 Plot correlation of roots and rates
    nonmotif_labels_merge = ([f'{feature_label}_root' for feature_label in nonmotif_labels] +
                             [f'{feature_label}_rate' for feature_label in nonmotif_labels])
    merge = roots.merge(rates, how='inner', on=['OGid', 'start', 'stop', 'disorder'], suffixes=('_root', '_rate'))
    merge_nonmotif = merge[nonmotif_labels_merge]
    disorder = merge.loc[pdidx[:, :, :, True], :]
    order = merge.loc[pdidx[:, :, :, False], :]
    disorder_nonmotif = disorder[nonmotif_labels_merge]
    order_nonmotif = order[nonmotif_labels_merge]
    for feature_label in feature_labels:
        xs, ys = merge[f'{feature_label}_root'], merge[f'{feature_label}_rate']
        result = linregress(xs, ys)
        m, b = result.slope, result.intercept
        r2 = result.rvalue ** 2
        xmin, xmax = xs.min(), xs.max()
        xpos = xmax if m >= 0 else xmin
        ha = 'right' if m >= 0 else 'left'

        fig, ax = plt.subplots()
        hb = ax.hexbin(xs, ys, gridsize=75, mincnt=1, linewidth=0, cmap=cmap3)
        ax.set_xlabel('Root value')
        ax.set_ylabel('Rate')
        ax.set_title(f'{feature_label}\nminimum length ≥ {min_length}')
        fig.colorbar(hb)
        ax.plot([xmin, xmax], [m * xmin + b, m * xmax + b], color='black', linewidth=1)
        ax.annotate(r'$\mathregular{R^2}$' + f' = {r2:.2f}', (xpos, m * xpos + b), ha=ha, va='bottom')
        fig.savefig(f'{prefix}/hexbin_rate-root_{feature_label}1.png')
        plt.close()

        xmin, xmax = merge[feature_label + '_root'].min(), merge[feature_label + '_root'].max()
        ymin, ymax = merge[feature_label + '_rate'].min(), merge[feature_label + '_rate'].max()
        fig, axs = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)
        for ax, data, label, cmap in zip(axs, [disorder, order], ['disorder', 'order'], [cmap1, cmap2]):
            xs, ys = data[f'{feature_label}_root'], data[f'{feature_label}_rate']
            result = linregress(xs, ys)
            m, b = result.slope, result.intercept
            r2 = result.rvalue ** 2
            xpos = xmax if m >= 0 else xmin
            ha = 'right' if m >= 0 else 'left'

            hb = ax.hexbin(xs, ys, gridsize=50, mincnt=1, linewidth=0, cmap=cmap, extent=(xmin, xmax, ymin, ymax))
            ax.set_ylabel('Rate')
            handles = [Line2D([], [], label=label, marker='h', markerfacecolor=cmap(0.6),
                              markeredgecolor='None', markersize=8, linestyle='None')]
            ax.legend(handles=handles)
            fig.colorbar(hb, ax=ax)
            ax.plot([xmin, xmax], [m * xmin + b, m * xmax + b], color='black', linewidth=1)
            ax.annotate(r'$\mathregular{R^2}$' + f' = {r2:.2f}', (xpos, m * xpos + b), ha=ha, va='bottom')
        axs[1].set_xlabel('Root value')
        fig.suptitle(f'{feature_label}\nminimum length ≥ {min_length}')
        fig.savefig(f'{prefix}/hexbin_rate-root_{feature_label}2.png')
        plt.close()

    ys = np.arange(len(feature_labels))
    xs_all, xs_disorder, xs_order, xs_delta = [], [], [], []
    for feature_label in feature_labels:
        x = np.corrcoef(merge[f'{feature_label}_root'], merge[f'{feature_label}_rate'])[0, 1]
        x_disorder = np.corrcoef(disorder[f'{feature_label}_root'], disorder[f'{feature_label}_rate'])[0, 1]
        x_order = np.corrcoef(order[f'{feature_label}_root'], order[f'{feature_label}_rate'])[0, 1]
        x_delta = x_disorder - x_order
        xs_all.append(x)
        xs_disorder.append(x_disorder)
        xs_order.append(x_order)
        xs_delta.append(x_delta)

    plots = [(xs_all, 'all', 'All regions', color3),
             (xs_disorder, 'disorder', 'Disorder regions', color1),
             (xs_order, 'order', 'Order regions', color2),
             (xs_delta, 'delta', 'Difference of disorder and order regions', color3)]
    for xs, data_label, title_label, color in plots:
        fig, ax = plt.subplots(figsize=(4.8, 8), layout='constrained')
        ax.invert_yaxis()
        ax.set_ymargin(0.01)
        ax.barh(ys, xs, color=color)
        ax.set_yticks(ys, feature_labels, fontsize=6)
        ax.set_xlabel('Correlation between root and rate')
        ax.set_ylabel('Feature')
        ax.set_title(title_label)
        fig.savefig(f'{prefix}/bar_rate-root_corr_{data_label}.png')
        plt.close()

    # 4.3.1 Plot root-rate PCAs (combined)
    plots = [(merge, 'all', f'minimum length ≥ {min_length}, no norm, all features', 'nonorm_all'),
             (zscore(merge), 'all', f'minimum length ≥ {min_length}, z-score, all features', 'zscore_all'),
             (merge_nonmotif, 'all', f'minimum length ≥ {min_length}, no norm, non-motif features', 'nonorm_nonmotif'),
             (zscore(merge_nonmotif), 'all', f'minimum length ≥ {min_length}, z-score, non-motif features', 'zscore_nonmotif')]
    for data, data_label, title_label, file_label in plots:
        pca = PCA(n_components=pca_components)
        transform = pca.fit_transform(data.to_numpy())
        idx = data.index.get_level_values('disorder').array.astype(bool)
        cmap, color = cmap3, color3
        width_ratios = (0.76, 0.03, 0.03, 0.15, 0.03)

        # Feature variance bar chart
        var = data.var().sort_values(ascending=False)
        var = var / var.sum()
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        fig, ax = plt.subplots(gridspec_kw={'bottom': 0.35})
        ax.bar(range(len(truncate.index)), truncate.values)
        ax.set_xticks(range(len(truncate.index)), truncate.index,
                      rotation=60, rotation_mode='anchor', ha='right', va='center')
        ax.set_ylabel('Explained variance ratio')
        fig.savefig(f'{prefix}/bar_variance_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        fig, ax = plt.subplots()
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
               label=data_label, color=color)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(title_label)
        ax.legend()
        fig.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
        plt.close()

        # PCA scatters
        fig = plot_pca2(transform, 0, 1, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
                        hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor,
                        width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca2_arrows(pca, transform, data.columns, 0, 1, idx, ~idx, cmap1, cmap2, title_label,
                               hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                               width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png')
        plt.close()

        fig = plot_pca2(transform, 1, 2, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
                        hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor,
                        width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca2_arrows(pca, transform, data.columns, 1, 2, idx, ~idx, cmap1, cmap2, title_label,
                               hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                               width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png')
        plt.close()

    # 4.3.2 Plot root-rate PCAs (individual)
    plots = [(disorder, 'disorder', f'minimum length ≥ {min_length}, no norm, all features', 'nonorm_all'),
             (order, 'order', f'minimum length ≥ {min_length}, no norm, all features', 'nonorm_all'),
             (zscore(disorder), 'disorder', f'minimum length ≥ {min_length}, z-score, all features', 'zscore_all'),
             (zscore(order), 'order', f'minimum length ≥ {min_length}, z-score, all features', 'zscore_all'),
             (disorder_nonmotif, 'disorder', f'minimum length ≥ {min_length}, no norm, non-motif features', 'nonorm_nonmotif'),
             (order_nonmotif, 'order', f'minimum length ≥ {min_length}, no norm, non-motif features', 'nonorm_nonmotif'),
             (zscore(disorder_nonmotif), 'disorder', f'minimum length ≥ {min_length}, z-score, non-motif features', 'zscore_nonmotif'),
             (zscore(order_nonmotif), 'order', f'minimum length ≥ {min_length}, z-score, non-motif features', 'zscore_nonmotif')]
    for data, data_label, title_label, file_label in plots:
        pca = PCA(n_components=pca_components)
        transform = pca.fit_transform(data.to_numpy())
        cmap = cmap1 if data_label == 'disorder' else cmap2
        color = color1 if data_label == 'disorder' else color2
        width_ratios = (0.76, 0.03, 0.03, 0.18)

        # Feature variance bar chart
        var = data.var().sort_values(ascending=False)
        var = var / var.sum()
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        fig, ax = plt.subplots(gridspec_kw={'bottom': 0.35})
        ax.bar(range(len(truncate.index)), truncate.values)
        ax.set_xticks(range(len(truncate.index)), truncate.index,
                      rotation=60, rotation_mode='anchor', ha='right', va='center')
        ax.set_ylabel('Explained variance ratio')
        fig.savefig(f'{prefix}/bar_variance_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        fig, ax = plt.subplots()
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
               label=data_label, color=color)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(title_label)
        ax.legend()
        fig.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
        plt.close()

        # PCA scatters
        fig = plot_pca(transform, 0, 1, cmap, data_label, title_label,
                       hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor,
                       width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca_arrows(pca, transform, data.columns, 0, 1, cmap, title_label,
                              hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                              width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png')
        plt.close()

        fig = plot_pca(transform, 1, 2, cmap, data_label, title_label,
                       hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor,
                       width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca_arrows(pca, transform, data.columns, 1, 2, cmap, title_label,
                              hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                              width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png')
        plt.close()
