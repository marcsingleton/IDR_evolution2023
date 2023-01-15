"""Plot statistics associated with features."""

import os
import re
from math import atan2, pi

import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace
from sklearn.decomposition import PCA
from src.brownian.features import motif_regexes
from src.brownian.pca_plots import plot_pca, plot_pca_arrows


def zscore(df):
    return (df - df.mean()) / df.std()


def get_angle(y, x):
    angle = atan2(y, x)
    if angle < 0:
        angle = 2 * pi + angle
    return angle


length_regex = r'regions_([0-9]+).tsv'
pdidx = pd.IndexSlice

cmap1, cmap2 = plt.colormaps['Blues'], plt.colormaps['Reds']
hexbin_kwargs = {'gridsize': 75, 'mincnt': 1, 'linewidth': 0}
handle_markerfacecolor = 0.6
legend_kwargs = {'fontsize': 8, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
pca_components = 10
arrow_colors = ['#e15759', '#499894', '#59a14f', '#f1ce63', '#b07aa1', '#d37295', '#9d7660', '#bab0ac',
                '#ff9d9a', '#86bcb6', '#8cd17d', '#b6992d', '#d4a6c8', '#fabfd2', '#d7b5a6', '#79706e']

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
features['length'] = features['length'] ** 0.6
features.rename(columns={'length': 'radius_gyration'}, inplace=True)

feature_labels = list(features.columns.drop(['OGid', 'ppid', 'start', 'stop']))
motif_labels = list(motif_regexes)

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
                             'ppid': ppid, 'min_length': min_length})
df = pd.DataFrame(rows)

for min_length in min_lengths:
    if not os.path.exists(f'out/regions_{min_length}/'):
        os.makedirs(f'out/regions_{min_length}/')

    segments = df[df['min_length'] == min_length].merge(features, how='left', on=['OGid', 'start', 'stop', 'ppid']).drop('min_length', axis=1)
    regions = segments.groupby(['OGid', 'start', 'stop', 'disorder'])

    means = regions.mean()
    disorder = means.loc[pdidx[:, :, :, True], :]
    order = means.loc[pdidx[:, :, :, False], :]
    disorder_motifs = disorder.drop(motif_labels, axis=1)
    order_motifs = order.drop(motif_labels, axis=1)

    # Feature histograms
    for feature_label in feature_labels:
        fig, axs = plt.subplots(2, 1, sharex=True)
        xmin, xmax = means[feature_label].min(), means[feature_label].max()
        axs[0].hist(disorder[feature_label], bins=linspace(xmin, xmax, 75), color='C0', label='disorder')
        axs[1].hist(order[feature_label], bins=linspace(xmin, xmax, 75), color='C1', label='order')
        axs[1].set_xlabel(f'Mean {feature_label}')
        for i in range(2):
            axs[i].set_ylabel('Number of regions')
            axs[i].legend()
        plt.savefig(f'out/regions_{min_length}/hist_numregions-{feature_label}.png')
        plt.close()

    # Individual PCAs
    plots = [(disorder, 'disorder', 'no norm', 'nonorm_all'),
             (order, 'order', 'no norm', 'nonorm_all'),
             (zscore(disorder), 'disorder', 'z-score', 'zscore_all'),
             (zscore(order), 'order', 'z-score', 'zscore_all'),
             (disorder_motifs, 'disorder', 'no norm', 'nonorm_motifs'),
             (order_motifs, 'order', 'no norm', 'nonorm_motifs'),
             (zscore(disorder_motifs), 'disorder', 'z-score', 'zscore_motifs'),
             (zscore(order_motifs), 'order', 'z-score', 'zscore_motifs')]
    for data, data_label, title_label, file_label in plots:
        pca = PCA(n_components=pca_components)
        transform = pca.fit_transform(data.to_numpy())
        cmap = plt.colormaps['Blues'] if data_label == 'disorder' else plt.colormaps['Reds']

        plot_pca(transform, 0, 1, cmap, data_label, title_label,
                 f'out/regions_{min_length}/hexbin_pc1-pc2_{data_label}_{file_label}.png',
                 hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor)
        plot_pca_arrows(pca, transform, data.columns, 0, 1, cmap, title_label,
                        f'out/regions_{min_length}/hexbin_pc1-pc2_{data_label}_{file_label}_arrows.png',
                        hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)

        # Scree plot
        plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, label=data_label, color=cmap(0.6))
        plt.xlabel('Principal component')
        plt.ylabel('Explained variance ratio')
        plt.title(title_label)
        plt.legend()
        plt.savefig(f'out/regions_{min_length}/bar_scree_{data_label}_{file_label}.png')
        plt.close()
