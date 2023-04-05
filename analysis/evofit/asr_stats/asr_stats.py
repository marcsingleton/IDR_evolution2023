"""Plot statistics for models fit for ASR."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

min_lengths = [30, 60, 90]

cmap1, cmap2 = plt.colormaps['Blues'], plt.colormaps['Reds']

for min_length in min_lengths:
    # Load regions
    OGid2regions = {}
    with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
            try:
                OGid2regions[OGid].append((start, stop, disorder))
            except KeyError:
                OGid2regions[OGid] = [(start, stop, disorder)]

    # Calculate rates
    rows = []
    for OGid, regions in OGid2regions.items():
        # Amino acid rates
        with open(f'../asr_root/out/{OGid}_aa_model.json') as file:
            aa_partitions = json.load(file)
        aa_dist = np.load(f'../asr_root/out/{OGid}_aa.npy')

        aa_rate_dist = aa_dist.sum(axis=1)
        aa_rate_values = np.zeros_like(aa_rate_dist)
        for partition in aa_partitions.values():
            partition_regions = partition['regions']
            partition_rates = partition['speed'] * np.array([[r] for r, _ in partition['rates']])
            for start, stop in partition_regions:
                aa_rate_values[:, start:stop] = partition_rates
        aa_rates = (aa_rate_dist * aa_rate_values).sum(axis=0)

        # Indel rates
        indel_partition = {'num_seqs': 0, 'num_columns': 0, 'num_categories': 0}  # Defaults for alignments with no indel model
        indel_rates = np.zeros(aa_rates.shape[-1])
        if os.path.exists(f'../asr_root/out/{OGid}_indel.npy'):
            with open(f'../asr_root/out/{OGid}_indel_model.json') as file:
                indel_partition = json.load(file)
            character_dist = np.load(f'../asr_root/out/{OGid}_indel.npy')

            character_rate_dist = character_dist.sum(axis=1)
            character_rate_values = indel_partition['speed'] * np.array([[r] for r, _ in indel_partition['rates']])
            character_rates = (character_rate_dist * character_rate_values).sum(axis=0)

            id2indel = {}
            with open(f'../asr_indel/out/{OGid}.tsv') as file:
                field_names = file.readline().rstrip('\n').split('\t')
                for line in file:
                    fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
                    character_id, start, stop = int(fields['character_id']), int(fields['start']), int(fields['stop'])
                    id2indel[character_id] = (start, stop)

            for character_id, indel in id2indel.items():
                start, stop = indel
                indel_rates[start] += character_rates[character_id] / 2
                indel_rates[stop-1] += character_rates[character_id] / 2

        for start, stop, disorder in regions:
            aa_partition = aa_partitions['disorder' if disorder else 'order']
            aa_rates_region = aa_rates[start:stop]
            indel_rates_region = indel_rates[start:stop]
            rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder,
                         'aa_num_categories': aa_partition['num_categories'],
                         'aa_num_seqs': aa_partition['num_seqs'], 'aa_num_columns': aa_partition['num_columns'],
                         'aa_rate_mean': aa_rates_region.mean(), 'aa_rate_std': aa_rates_region.std(),
                         'indel_num_categories': indel_partition['num_categories'],
                         'indel_num_seqs': indel_partition['num_seqs'], 'indel_num_columns': indel_partition['num_columns'],
                         'indel_rate_mean': indel_rates_region.mean(), 'indel_rate_std': indel_rates_region.std()})
    df = pd.DataFrame(rows)

    if not os.path.exists(f'out/regions_{min_length}/'):
        os.makedirs(f'out/regions_{min_length}/')

    df.drop('disorder', axis=1).to_csv(f'out/regions_{min_length}/rates.tsv', sep='\t', index=False)  # Keep convention that disorder labels are dropped

    # Column number distributions
    x = df[['disorder', 'aa_num_columns']]
    xmin, xmax = x['aa_num_columns'].min(), x['aa_num_columns'].max()
    bins = np.linspace(xmin, xmax, 200)
    fig, axs = plt.subplots(2, 1)
    axs[0].hist(x.loc[x['disorder'], 'aa_num_columns'], bins=bins, label='disorder', color=cmap1(0.6))
    axs[1].hist(x.loc[~x['disorder'], 'aa_num_columns'], bins=bins, label='order', color=cmap2(0.6))
    axs[1].set_xlabel('Number of columns in partition (amino acid)')
    for ax in axs:
        ax.set_ylabel('Number of partitions')
        ax.legend()
    fig.savefig(f'out/regions_{min_length}/hist_numaapartitions-numcolumns.png')
    plt.close()

    x = df[['disorder', 'indel_num_columns']]
    xmin, xmax = x['indel_num_columns'].min(), x['indel_num_columns'].max()
    bins = np.linspace(xmin, xmax, 100)
    fig, axs = plt.subplots(2, 1)
    axs[0].hist(x.loc[x['disorder'], 'indel_num_columns'], bins=bins, label='disorder', color=cmap1(0.6))
    axs[1].hist(x.loc[~x['disorder'], 'indel_num_columns'], bins=bins, label='order', color=cmap2(0.6))
    axs[1].set_xlabel('Number of columns in model (indel)')
    for ax in axs:
        ax.set_ylabel('Number of models')
        ax.legend()
    fig.savefig(f'out/regions_{min_length}/hist_numindelmodels-numcolumns.png')
    plt.close()

    counts = df.loc[df['indel_rate_mean'].isna(), 'indel_num_columns'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values, width=0.5)
    ax.set_xlabel('Number of columns in model (indel)')
    ax.set_ylabel('Number of models')
    ax.set_title('Indel models with undefined rates')
    fig.savefig(f'out/regions_{min_length}/hist_numindelmodels-numcolumns_nan.png')
    plt.close()

    # Rate mean distributions
    fig, axs = plt.subplots(2, 1)
    xmin, xmax = df['aa_rate_mean'].min(), df['aa_rate_mean'].max()
    bins = np.linspace(xmin, xmax, 100)
    for ax, label, color in zip(axs, ['disorder', 'order'], [cmap1(0.6), cmap2(0.6)]):
        ax.hist(df.loc[df['disorder'] if label == 'disorder' else ~df['disorder'], 'aa_rate_mean'],
                bins=bins, label=label, color=color)
        ax.set_ylabel('Number of regions')
        ax.legend()
    axs[1].set_xlabel('Average amino acid rate in region')
    fig.savefig(f'out/regions_{min_length}/hist_numregion-aarate.png')
    plt.close()

    fig, axs = plt.subplots(2, 1)
    xmin, xmax = df['indel_rate_mean'].min(), df['indel_rate_mean'].max()
    bins = np.linspace(xmin, xmax, 150)
    for ax, label, color in zip(axs, ['disorder', 'order'], [cmap1(0.6), cmap2(0.6)]):
        ax.hist(df.loc[df['disorder'] if label == 'disorder' else ~df['disorder'], 'indel_rate_mean'],
                bins=bins, label=label, color=color)
        ax.set_ylabel('Number of regions')
        ax.legend()
    axs[1].set_xlabel('Average indel rate in region')
    fig.savefig(f'out/regions_{min_length}/hist_numregion-indelrate1.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(df.loc[df['indel_num_columns'] < 5, 'indel_rate_mean'], bins=50)
    ax.set_xlabel('Average indel rate in region')
    ax.set_ylabel('Number of regions')
    ax.set_title('Indel models with fewer than five columns')
    fig.savefig(f'out/regions_{min_length}/hist_numregion-indelrate2.png')
    plt.close()

    fig, ax = plt.subplots()
    hb = ax.hexbin(df['aa_rate_mean'], df['indel_rate_mean'], gridsize=75, mincnt=1, linewidth=0, bins='log')
    ax.set_xlabel('Average amino acid rate in region')
    ax.set_ylabel('Average indel rate in region')
    fig.colorbar(hb)
    fig.savefig(f'out/regions_{min_length}/hexbin_indelrate-aarate.png')
    plt.close()

    x1 = (df.loc[df['disorder'] == True, 'aa_rate_mean'] + df.loc[df['disorder'] == True, 'indel_rate_mean']).dropna()
    x2 = (df.loc[df['disorder'] == False, 'aa_rate_mean'] + df.loc[df['disorder'] == False, 'indel_rate_mean']).dropna()
    fig, ax = plt.subplots()
    ax.boxplot([x1, x2], labels=['disorder', 'order'])
    ax.set_ylabel('Sum of average amino acid and indel rates in region')
    fig.savefig(f'out/regions_{min_length}/boxplot_rate-class.png')
    plt.close()

"""
NOTES
Indel models fit to alignments with a small number of indels should be considered poor quality. In the worst cases, the
rates are so large that they cause numerical overflow during matrix exponentiation, and the posterior probabilities of
the rate categories for those columns is NaN. This was only observed in alignments with fewer than three columns
(inclusive), however. Fortunately, the equilibrium distribution was concordant with the data, so it can be used to
generate ancestral sequences. For example, if an insertion was observed on a only small number of tips with very short
branch lengths, the equilibrium distribution of the gap state (1) was nearly one in all checked examples.

When the number of columns is fewer than five, the rates were often near zero, but there were some clear outliers.
Because these cases correspond to alignments with fewer than five distinct patterns of indels, their rates can be
considered as effectively zero for the purposes of filtering alignments based on their "gappiness."
"""