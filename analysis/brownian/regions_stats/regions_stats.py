"""Plot statistics of filtered OGs."""

import os
import re

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from numpy import linspace
from sklearn.decomposition import PCA
from src.brownian.features import motif_regexes


def zscore(df):
    return (df - df.mean()) / df.std()


def plot_hexbin_pca(x1, y1, x2, y2, gridsize=None, bins=None, cmap1=None, cmap2=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    extent = (xmin, xmax, ymin, ymax)

    hb1 = ax.hexbin(x1, y1, gridsize=gridsize, extent=extent, cmap=cmap1, linewidth=0)
    hb2 = ax.hexbin(x2, y2, gridsize=gridsize, extent=extent, cmap=cmap2, linewidth=0)

    array1 = np.expand_dims(hb1.get_array().data, -1)
    array2 = np.expand_dims(hb2.get_array().data, -1)
    norm1 = colors.Normalize(array1.min(), array1.max())
    norm2 = colors.Normalize(array2.min(), array2.max())
    fc1 = np.array([cmap1(norm1(count)) for count in array1.squeeze()])
    fc2 = np.array([cmap2(norm2(count)) for count in array2.squeeze()])

    total = array1 + array2
    total[total == 0] = 1
    fc = (array1 * fc1 + array2 * fc2) / total
    ax.clear()

    z = ax.hexbin([], [], bins=bins, gridsize=gridsize, extent=extent, linewidth=0)
    z.set_array(None)
    z.set_facecolor(fc)

    return ax, hb1, hb2


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
features['radius_gyration'] = features['length'] ** 0.6

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

# Plots of combined segment sets
if not os.path.exists('out/'):
    os.mkdir('out/')

# Number of regions by length cutoff
disorder, order = [], []
for min_length in min_lengths:
    disorder.append(len(df.loc[df['disorder'] & (df['min_length'] == min_length), ['OGid', 'start', 'stop']].drop_duplicates()))
    order.append(len(df.loc[~df['disorder'] & (df['min_length'] == min_length), ['OGid', 'start', 'stop']].drop_duplicates()))
plt.plot(min_lengths, disorder, color='C0', label='disorder')
plt.plot(min_lengths, order, color='C1', label='order')
plt.xlabel('Length cutoff')
plt.ylabel('Number of regions')
plt.legend()
plt.savefig('out/line_numregions-minlength.png')
plt.close()

# Number of OGs by length cutoff
disorder, order = [], []
for min_length in min_lengths:
    disorder.append(len(df.loc[df['disorder'] & (df['min_length'] == min_length), 'OGid'].drop_duplicates()))
    order.append(len(df.loc[~df['disorder'] & (df['min_length'] == min_length), 'OGid'].drop_duplicates()))
plt.plot(min_lengths, disorder, color='C0', label='disorder')
plt.plot(min_lengths, order, color='C1', label='order')
plt.xlabel('Length cutoff')
plt.ylabel('Number of unique OGs')
plt.legend()
plt.savefig('out/line_numOGs-minlength.png')
plt.close()

# Plots of individual segment sets
for min_length in min_lengths:
    if not os.path.exists(f'out/regions_{min_length}/'):
        os.mkdir(f'out/regions_{min_length}/')

    segments = df[df['min_length'] == min_length].merge(features, how='left', on=['OGid', 'start', 'stop', 'ppid']).drop('min_length', axis=1)
    regions = segments.groupby(['OGid', 'start', 'stop', 'disorder'])

    means = regions.mean()
    disorder = means.loc[pdidx[:, :, :, True], :]
    order = means.loc[pdidx[:, :, :, False], :]

    # Mean region length histogram
    fig, axs = plt.subplots(2, 1, sharex=True)
    xmin, xmax = means['length'].min(), means['length'].max()
    axs[0].hist(disorder['length'], bins=linspace(xmin, xmax, 100), color='C0', label='disorder')
    axs[1].hist(order['length'], bins=linspace(xmin, xmax, 100), color='C1', label='order')
    axs[1].set_xlabel('Mean length of region')
    for i in range(2):
        axs[i].set_ylabel('Number of regions')
        axs[i].legend()
    plt.savefig(f'out/regions_{min_length}/hist_numregions-length.png')
    plt.close()

    # Number of sequences in region bar plot
    fig, ax = plt.subplots()
    counts1 = regions.size()[pdidx[:, :, :, True]].value_counts()
    counts2 = regions.size()[pdidx[:, :, :, False]].value_counts()
    ax.bar(counts1.index - 0.35/2, counts1.values, color='C0', label='disorder', width=0.35)
    ax.bar(counts2.index + 0.35/2, counts2.values, color='C1', label='order', width=0.35)
    ax.set_xlabel('Number of sequences in region')
    ax.set_ylabel('Number of regions')
    ax.legend()
    plt.savefig(f'out/regions_{min_length}/bar_numregions-numseqs.png')
    plt.close()

    # Counts of regions and unique OGs in each class
    disorder = segments[segments['disorder']]
    order = segments[~segments['disorder']]

    plt.bar([0, 1], [len(disorder[['OGid', 'start', 'stop']].drop_duplicates()), len(order[['OGid', 'start', 'stop']].drop_duplicates())],
            tick_label=['disorder', 'order'], color=['C0', 'C1'], width=0.35)
    plt.xlim((-0.5, 1.5))
    plt.ylabel('Number of regions')
    plt.savefig(f'out/regions_{min_length}/bar_numregions-DO.png')
    plt.close()

    plt.bar([0, 1], [len(disorder['OGid'].drop_duplicates()), len(order['OGid'].drop_duplicates())],
            tick_label=['disorder', 'order'], color=['C0', 'C1'], width=0.35)
    plt.xlim((-0.5, 1.5))
    plt.ylabel('Number of unique OGs')
    plt.savefig(f'out/regions_{min_length}/bar_numOGs-DO.png')
    plt.close()

    plots = [(means.drop(['length'], axis=1), 'no norm', 'nonorm_all'),
             (means.drop(['length'] + motif_labels, axis=1), 'no norm', 'nonorm_motifs'),
             (zscore(means.drop(['length'], axis=1)), 'z-score', 'zscore_all'),
             (zscore(means.drop(['length'] + motif_labels, axis=1)), 'z-score', 'zscore_motifs')]
    for data, title_label, file_label in plots:
        # Feature variance pie chart
        var = data.var().sort_values(ascending=False)
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        plt.pie(truncate.values, labels=truncate.index, labeldistance=None)
        plt.title(f'Feature variance\n{title_label}, length ≥ {min_length}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.65)
        plt.savefig(f'out/regions_{min_length}/pie_variance_{file_label}.png')
        plt.close()

        # Feature PCAs
        pca = PCA(n_components=5)
        idx = data.index.get_level_values('disorder').array.astype(bool)
        cmap1, cmap2 = plt.colormaps['Blues'], plt.colormaps['Reds']
        transform = pca.fit_transform(data.to_numpy())

        x1, x2 = transform[idx, 0], transform[~idx, 0]
        y1, y2 = transform[idx, 1], transform[~idx, 1]

        fig = plt.figure()
        gs = fig.add_gridspec(1, 5, width_ratios=(0.79, 0.03, 0.03, 0.12, 0.03), wspace=0)
        ax = fig.add_subplot(gs[:, 0])
        _, hb1, hb2 = plot_hexbin_pca(x1, y1, x2, y2, gridsize=75, cmap1=cmap1, cmap2=cmap2, ax=ax)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'{title_label}, length ≥ {min_length}')
        handles = [Line2D([], [], label='disorder', marker='h', markerfacecolor=cmap1(0.6),
                          markeredgecolor='None', markersize=8, linestyle='None'),
                   Line2D([], [], label='order', marker='h', markerfacecolor=cmap2(0.6),
                          markeredgecolor='None', markersize=8, linestyle='None')]
        ax.legend(handles=handles)
        fig.colorbar(hb1, cax=fig.add_subplot(gs[:, 2]))
        fig.colorbar(hb2, cax=fig.add_subplot(gs[:, 4]))
        fig.savefig(f'out/regions_{min_length}/hexbin_pc1-pc2_{file_label}.png')
        plt.close()
