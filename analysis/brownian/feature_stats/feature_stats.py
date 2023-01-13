"""Plot statistics associated with features."""

import os
import re
from math import atan2, pi

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from numpy import linspace
from sklearn.decomposition import PCA
from src.brownian.features import motif_regexes


def zscore(df):
    return (df - df.mean()) / df.std()


def get_angle(y, x):
    angle = atan2(y, x)
    if angle < 0:
        angle = 2 * pi + angle
    return angle


length_regex = r'regions_([0-9]+).tsv'
pdidx = pd.IndexSlice

pca = PCA(n_components=10)
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
        cmap = plt.colormaps['Blues'] if data_label == 'disorder' else plt.colormaps['Reds']
        transform = pca.fit_transform(data.to_numpy())

        # PCA without arrows
        fig = plt.figure()
        gs = fig.add_gridspec(1, 4, width_ratios=(0.79, 0.06, 0.03, 0.12), wspace=0)
        ax = fig.add_subplot(gs[:, 0])
        hb = ax.hexbin(transform[:, 0], transform[:, 1], gridsize=75, cmap=cmap, linewidth=0, mincnt=1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(title_label)
        handles = [Line2D([], [], label=data_label, marker='h', markerfacecolor=cmap(0.6),
                          markeredgecolor='None', markersize=8, linestyle='None')]
        ax.legend(handles=handles)
        fig.colorbar(hb, cax=fig.add_subplot(gs[:, 2]))
        fig.savefig(f'out/regions_{min_length}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        # PCA with arrows
        fig = plt.figure()
        gs = fig.add_gridspec(1, 4, width_ratios=(0.79, 0.06, 0.03, 0.12), wspace=0)
        ax = fig.add_subplot(gs[:, 0])
        hb = ax.hexbin(transform[:, 0], transform[:, 1], gridsize=75, cmap=cmap, linewidth=0, mincnt=1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(title_label)

        projections = zip(data.columns, pca.components_[0], pca.components_[1])  # Match features to components in PC space
        projections = sorted(projections, key=lambda x: x[1] ** 2 + x[2] ** 2, reverse=True)[:len(arrow_colors)]  # Get features with largest magnitude
        projections = sorted(projections, key=lambda x: get_angle(x[2], x[1]))  # Re-order by angle from x-axis

        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        ratios = []
        for projection in projections:
            _, x, y = projection
            ratios.extend([x / xmin, x / xmax, y / ymin, y / ymax])
        scale = 0.9 / max(ratios)  # Scale the largest arrow within fraction of axes

        handles = []
        for arrow_color, projection in zip(arrow_colors, projections):
            feature_label, x, y = projection
            handles.append(Line2D([], [], color=arrow_color, linewidth=2, label=feature_label))
            ax.annotate('', xy=(scale * x, scale * y), xytext=(0, 0),
                        arrowprops={'headwidth': 6, 'headlength': 6, 'width': 1.75, 'color': arrow_color})
        ax.legend(handles=handles, fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig(f'out/regions_{min_length}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png')
        plt.close()

        # Scree plot
        plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, label=data_label, color=cmap(0.6))
        plt.xlabel('Principal component')
        plt.ylabel('Explained variance ratio')
        plt.title(title_label)
        plt.legend()
        plt.savefig(f'out/regions_{min_length}/bar_scree_{data_label}_{file_label}.png')
        plt.close()
