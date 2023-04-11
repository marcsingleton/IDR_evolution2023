"""Plot statistics associated with features."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace
from sklearn.decomposition import PCA
from src.brownian.pca import plot_pca, plot_pca_arrows, plot_pca2, plot_pca2_arrows


def zscore(df):
    return (df - df.mean()) / df.std()


pdidx = pd.IndexSlice
min_lengths = [30, 60, 90]

pca_components = 10
cmap1, cmap2, cmap3 = plt.colormaps['Blues'], plt.colormaps['Oranges'], plt.colormaps['Purples']
color1, color2, color3 = '#4e79a7', '#f28e2b', '#b07aa1'
hexbin_kwargs = {'gridsize': 75, 'mincnt': 1, 'linewidth': 0}
handle_markerfacecolor = 0.6
legend_kwargs = {'fontsize': 8, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
arrow_colors = ['#e15759', '#499894', '#59a14f', '#f1ce63', '#b07aa1', '#d37295', '#9d7660', '#bab0ac',
                '#ff9d9a', '#86bcb6', '#8cd17d', '#b6992d', '#d4a6c8', '#fabfd2', '#d7b5a6', '#79706e']

# Load features
all_features = pd.read_table('../get_features/out/features.tsv', header=[0, 1])
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

    features = all_segments.merge(all_features, how='left', on=['OGid', 'start', 'stop', 'ppid'])
    features = features.groupby(['OGid', 'start', 'stop', 'disorder']).mean()
    features = all_regions.merge(features, how='left', on=['OGid', 'start', 'stop', 'disorder'])
    features = features.set_index(['OGid', 'start', 'stop', 'disorder'])

    features_nonmotif = features[nonmotif_labels]
    disorder = features.loc[pdidx[:, :, :, True], :]
    order = features.loc[pdidx[:, :, :, False], :]
    disorder_nonmotif = disorder[nonmotif_labels]
    order_nonmotif = order[nonmotif_labels]

    # Feature histograms
    for feature_label in feature_labels:
        fig, axs = plt.subplots(2, 1, sharex=True)
        xmin, xmax = features[feature_label].min(), features[feature_label].max()
        axs[0].hist(disorder[feature_label], bins=linspace(xmin, xmax, 75), color=color1, label='disorder')
        axs[1].hist(order[feature_label], bins=linspace(xmin, xmax, 75), color=color2, label='order')
        axs[1].set_xlabel(f'Mean {feature_label}')
        axs[0].set_title(f'minimum length ≥ {min_length}')
        for ax in axs:
            ax.set_ylabel('Number of regions')
            ax.legend()
        fig.savefig(f'out/regions_{min_length}/hist_numregions-{feature_label}.png')
        plt.close()

    # Combined PCAs
    plots = [(features, 'all', f'minimum length ≥ {min_length}, no norm, all features', 'nonorm_all'),
             (zscore(features), 'all', f'minimum length ≥ {min_length}, z-score, all features', 'zscore_all'),
             (features_nonmotif, 'all', f'minimum length ≥ {min_length}, no norm, non-motif features', 'nonorm_nonmotif'),
             (zscore(features_nonmotif), 'all', f'minimum length ≥ {min_length}, z-score, non-motif features', 'zscore_nonmotif')]
    for data, data_label, title_label, file_label in plots:
        pca = PCA(n_components=pca_components)
        transform = pca.fit_transform(data.to_numpy())
        idx = data.index.get_level_values('disorder').array.astype(bool)
        cmap, color = cmap3, color3

        # Feature variance pie chart
        var = data.var().sort_values(ascending=False)
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        fig, ax = plt.subplots(gridspec_kw={'right': 0.65})
        ax.pie(truncate.values, labels=truncate.index, labeldistance=None)
        ax.set_title(f'Feature variance\n{title_label}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig(f'out/regions_{min_length}/pie_variance_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        fig, ax = plt.subplots()
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
               label=data_label, color=color3)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(title_label)
        ax.legend()
        fig.savefig(f'out/regions_{min_length}/bar_scree_{data_label}_{file_label}.png')
        plt.close()

        # PCA scatters
        fig = plot_pca2(transform, 0, 1, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
                        hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor)
        fig.savefig(f'out/regions_{min_length}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca2_arrows(pca, transform, data.columns, 0, 1, idx, ~idx, cmap1, cmap2, title_label,
                               hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)
        fig.savefig(f'out/regions_{min_length}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png')
        plt.close()

    # Individual PCAs
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

        # Feature variance pie chart
        var = data.var().sort_values(ascending=False)
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        fig, ax = plt.subplots(gridspec_kw={'right': 0.65})
        ax.pie(truncate.values, labels=truncate.index, labeldistance=None)
        ax.set_title(f'Feature variance\n{title_label}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig(f'out/regions_{min_length}/pie_variance_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        fig, ax = plt.subplots()
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
               label=data_label, color=color)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(title_label)
        ax.legend()
        fig.savefig(f'out/regions_{min_length}/bar_scree_{data_label}_{file_label}.png')
        plt.close()

        # PCA scatters
        fig = plot_pca(transform, 0, 1, cmap, data_label, title_label,
                       hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor)
        fig.savefig(f'out/regions_{min_length}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca_arrows(pca, transform, data.columns, 0, 1, cmap, title_label,
                              hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)
        fig.savefig(f'out/regions_{min_length}/hexbin_pc1-pc2_{data_label}_{file_label}_arrows.png')
        plt.close()
