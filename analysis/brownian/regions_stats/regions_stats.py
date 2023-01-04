"""Plot statistics of filtered OGs."""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
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

    plots = [(means.drop(['length'], axis=1), 'all'),
             (means.drop(['length'] + motif_labels, axis=1), 'motifs')]
    for data, file_label in plots:
        # Feature variance pie chart
        var = data.var().sort_values(ascending=False)
        truncate = pd.concat([var[:4], pd.Series({'other': var[4:].sum()})])
        plt.pie(truncate.values, labels=truncate.index, labeldistance=None)
        plt.title(f'Feature variance, length ≥ {min_length}')
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        plt.subplots_adjust(right=0.7)
        plt.savefig(f'out/regions_{min_length}/pie_variance_{file_label}.png')
        plt.close()

        # Feature PCAs
        pca = PCA(n_components=5)
        idx = data.index.get_level_values('disorder').array.astype(bool)

        transform = pca.fit_transform(data.to_numpy())
        plt.scatter(transform[idx, 0], transform[idx, 1], label='disorder', s=5, alpha=0.05, edgecolors='none')
        plt.scatter(transform[~idx, 0], transform[~idx, 1], label='order', s=5, alpha=0.05, edgecolors='none')
        plt.title(f'no norm, length ≥ {min_length}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        legend = plt.legend(markerscale=2)
        for lh in legend.legendHandles:
            lh.set_alpha(1)
        plt.savefig(f'out/regions_{min_length}/pca_nonorm_{file_label}.png')
        plt.close()

        norm = zscore(data)
        transform = pca.fit_transform(norm.to_numpy())
        plt.scatter(transform[idx, 0], transform[idx, 1], label='disorder', s=5, alpha=0.05, edgecolors='none')
        plt.scatter(transform[~idx, 0], transform[~idx, 1], label='order', s=5, alpha=0.05, edgecolors='none')
        plt.title(f'z-score, length ≥ {min_length}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        legend = plt.legend(markerscale=2)
        for lh in legend.legendHandles:
            lh.set_alpha(1)
        plt.savefig(f'out/regions_{min_length}/pca_zscore_{file_label}.png')
        plt.close()
