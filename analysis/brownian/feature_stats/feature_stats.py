"""Plot statistics associated with features."""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from numpy import linspace
from sklearn.decomposition import PCA
from src.brownian.features import motif_regexes


def zscore(df):
    return (df - df.mean()) / df.std()


length_regex = r'regions_([0-9]+).tsv'
pdidx = pd.IndexSlice

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
    pca = PCA(n_components=10)
    colors = ['#e15759', '#499894', '#59a14f', '#f1ce63', '#b07aa1', '#d37295', '#9d7660', '#bab0ac',
              '#ff9d9a', '#86bcb6', '#8cd17d', '#b6992d', '#d4a6c8', '#fabfd2', '#d7b5a6', '#79706e']

    plots = [(disorder, 'disorder', 'no norm', 'nonorm_all'),
             (order, 'order', 'no norm', 'nonorm_all'),
             (zscore(disorder), 'disorder', 'z-score', 'zscore_all'),
             (zscore(order), 'order', 'z-score', 'zscore_all'),
             (disorder_motifs, 'disorder', 'no norm', 'nonorm_motifs'),
             (order_motifs, 'order', 'no norm', 'nonorm_motifs'),
             (zscore(disorder_motifs), 'disorder', 'z-score', 'zscore_motifs'),
             (zscore(order_motifs), 'order', 'z-score', 'zscore_motifs')]
    for data, data_label, title_label, file_label in plots:
        color = 'C0' if data_label == 'disorder' else 'C1'
        transform = pca.fit_transform(data.to_numpy())

        # PCA without arrows
        plt.scatter(transform[:, 0], transform[:, 1], label=data_label, color=color, s=5, alpha=0.1, edgecolors='none')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(title_label)
        legend = plt.legend(markerscale=2)
        for lh in legend.legendHandles:
            lh.set_alpha(1)
        plt.savefig(f'out/regions_{min_length}/scatter_pca_{data_label}_{file_label}.png')
        plt.close()

        # PCA with arrows
        plt.scatter(transform[:, 0], transform[:, 1], label=data_label, color=color, s=5, alpha=0.1, edgecolors='none')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(title_label)

        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        scale = (xmax + ymax - xmin - ymin) / 3
        projections = sorted(zip(data.columns, pca.components_[:2].transpose()), key=lambda x: x[1][0]**2 + x[1][1]**2, reverse=True)

        handles = []
        for i in range(len(colors)):
            feature_label, (x, y) = projections[i]
            handles.append(Line2D([], [], color=colors[i % len(colors)], linewidth=2, label=feature_label))
            plt.annotate('', xy=(scale*x, scale*y), xytext=(0, 0),
                         arrowprops={'headwidth': 6, 'headlength': 6, 'width': 1.75, 'color': colors[i % len(colors)]})
        plt.legend(handles=handles, fontsize=8, loc='right', bbox_to_anchor=(1.05, 0.5))
        plt.savefig(f'out/regions_{min_length}/scatter_pca-arrow_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, label=data_label, color=color)
        plt.xlabel('Principal component')
        plt.ylabel('Explained variance ratio')
        plt.title(title_label)
        plt.legend()
        plt.savefig(f'out/regions_{min_length}/bar_scree_{data_label}_{file_label}.png')
        plt.close()
