"""Plot statistics of filtered regions."""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace
from src.utils import read_fasta

pdidx = pd.IndexSlice
ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
length_regex = r'regions_([0-9]+).tsv'

# Get minimum lengths
min_lengths = []
for path in os.listdir('../regions_filter/out/'):
    match = re.search(length_regex, path)
    if match:
        min_lengths.append(int(match.group(1)))
min_lengths = sorted(min_lengths)

# Load regions as segments
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
all_segments = pd.DataFrame(rows)

# Get segment lengths
rows = []
for OGid, group in all_segments[['OGid', 'start', 'stop']].drop_duplicates().groupby('OGid'):
    msa = read_fasta(f'../../../data/alignments/fastas/{OGid}.afa')
    msa = [(re.search(ppid_regex, header).group(1), seq) for header, seq in msa]

    for row in group.itertuples():
        region_length = row.stop - row.start
        for ppid, seq in msa:
            segment = seq[row.start:row.stop]
            segment_length = region_length - segment.count('.') - segment.count('-')
            rows.append({'OGid': OGid, 'start': row.start, 'stop': row.stop, 'ppid': ppid, 'length': segment_length})
all_lengths = pd.DataFrame(rows)

# Plots of combined segment sets
if not os.path.exists('out/'):
    os.mkdir('out/')

# Number of regions by length cutoff
disorder, order = [], []
for min_length in min_lengths:
    df = all_segments[all_segments['min_length'] == min_length]
    disorder.append(len(df.loc[df['disorder'], ['OGid', 'start', 'stop']].drop_duplicates()))
    order.append(len(df.loc[~df['disorder'], ['OGid', 'start', 'stop']].drop_duplicates()))
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
    df = all_segments[all_segments['min_length'] == min_length]
    disorder.append(len(df.loc[df['disorder'], 'OGid'].drop_duplicates()))
    order.append(len(df.loc[~df['disorder'], 'OGid'].drop_duplicates()))
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

    segments = all_segments[all_segments['min_length'] == min_length].merge(all_lengths, how='left', on=['OGid', 'start', 'stop', 'ppid'])
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
    axs[0].set_title(f'minimum length ≥ {min_length}')
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
    ax.set_title(f'minimum length ≥ {min_length}')
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
    plt.title(f'minimum length ≥ {min_length}')
    plt.savefig(f'out/regions_{min_length}/bar_numregions-DO.png')
    plt.close()

    plt.bar([0, 1], [len(disorder['OGid'].drop_duplicates()), len(order['OGid'].drop_duplicates())],
            tick_label=['disorder', 'order'], color=['C0', 'C1'], width=0.35)
    plt.xlim((-0.5, 1.5))
    plt.ylabel('Number of unique OGs')
    plt.title(f'minimum length ≥ {min_length}')
    plt.savefig(f'out/regions_{min_length}/bar_numOGs-DO.png')
    plt.close()
