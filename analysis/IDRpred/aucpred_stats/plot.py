"""Plot statistics of AUCpreD scores."""

import os
import re
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
from src.draw import plot_msa_data
from src.phylo import get_brownian_weights, get_contrasts
from src.utils import read_fasta


def get_quantile(x, q):
    x = np.sort(x)
    return x[:ceil(q * len(x))]


def load_scores(path):
    with open(path) as file:
        scores = []
        for line in file:
            if not line.startswith('#'):
                score = line.split()[3]
                scores.append(score)
    return scores


ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'

plot_msa_kwargs = {'hspace': 0.2, 'left': 0.025, 'right': 0.925, 'top': 0.99, 'bottom': 0.05,
                   'data_min': -0.05, 'data_max': 1.05,
                   'msa_legend': True, 'legend_kwargs': {'bbox_to_anchor': (0.935, 0.5), 'loc': 'center left', 'fontsize': 10,
                                                         'handletextpad': 0.5, 'markerscale': 2, 'handlelength': 1}}

tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_template.tips())}

df1 = pd.read_table('out/stats.tsv')

df1['scores_fraction'] = df1['scores_sum'] / df1['length']
df1['binary_fraction'] = df1['binary_sum'] / df1['length']
groups = df1.groupby('OGid')

columns = ['scores_fraction', 'binary_fraction']
labels = ['Average AUCpreD score', 'Fraction disorder']
colors = ['C0', 'C1']

rows = []
for OGid, group in groups:
    tree = tree_template.shear(group['spid'])
    spid2tip = {tip.name: tip for tip in tree.tips()}
    for row in group.itertuples():
        tip = spid2tip[row.spid]
        tip.value = np.array([getattr(row, column) for column in columns])

    roots, contrasts = get_contrasts(tree)
    contrasts = np.stack(contrasts)
    rates = (contrasts ** 2).mean(axis=0)
    rows.append({'OGid': OGid,
                 **{f'{column}_root': root for column, root in zip(columns, roots)},
                 **{f'{column}_rate': rate for column, rate in zip(columns, rates)}})
df2 = pd.DataFrame(rows)

fig, axs = plt.subplots(2, 1, sharex=True)
for ax, column, label, color in zip(axs, columns, labels, colors):
    ax.hist(df1[column], bins=np.linspace(0, 1, 100), color=color)
    ax.set_xlabel(label)
    ax.set_ylabel('Number of sequences')
plt.savefig('out/hist_seqnum-root.png')
plt.close()

fig, axs = plt.subplots(2, 1, sharex=True)
for ax, column, label, color in zip(axs, columns, labels, colors):
    ax.hist(df2[f'{column}_root'], bins=np.linspace(0, 1, 100), color=color)
    ax.set_xlabel(f'{label} at root')
    ax.set_ylabel('Number of alignments')
plt.savefig('out/hist_alignmentnum-root.png')
plt.close()

xs = [get_quantile(df2[f'{column}_rate'].to_numpy(), 0.99) for column in columns]
xmin = min([x.min() for x in xs])
xmax = min([x.max() for x in xs])
xrange = np.linspace(xmin, xmax, 100)
fig, axs = plt.subplots(2, 1, sharex=True)
for ax, x, label, color in zip(axs, xs, labels, colors):
    ax.hist(x, bins=xrange, color=color)
    ax.set_xlabel(f'{label} rate')
    ax.set_ylabel('Number of alignments')
plt.savefig('out/hist_alignmentnum-rate.png')
plt.close()

for ax, column, label in zip(axs, columns, labels):
    plt.hexbin(df2[f'{column}_root'], df2[f'{column}_rate'], gridsize=50, mincnt=1, linewidth=0)
    plt.xlabel(f'{label} at root')
    plt.ylabel(f'{label} rate')
    plt.colorbar()
    plt.savefig(f'out/hexbin_rate-root_{column}.png')
    plt.close()

plt.hexbin(df2[f'{columns[0]}_root'], df2[f'{columns[1]}_root'], gridsize=50, mincnt=1, linewidth=0)
plt.xlabel(f'{labels[0]} at root')
plt.ylabel(f'{labels[1]} at root')
plt.colorbar()
plt.savefig(f'out/hexbin_{columns[1]}_root-{columns[0]}_root.png')
plt.close()

plt.hexbin(df2[f'{columns[0]}_rate'], df2[f'{columns[1]}_rate'], gridsize=50, mincnt=1, bins='log', linewidth=0)
plt.xlabel(f'{labels[0]} rate')
plt.ylabel(f'{labels[1]} rate')
plt.colorbar()
plt.savefig(f'out/hexbin_{columns[1]}_rate-{columns[0]}_rate.png')
plt.close()

if not os.path.exists('out/traces/'):
    os.mkdir('out/traces/')

sort = df2.sort_values(by='scores_fraction_rate', ascending=False, ignore_index=True)
examples = pd.concat([sort.iloc[:100],  # Pull out samples around quartiles
                      sort.iloc[(int(0.25*len(sort))-50):(int(0.25*len(sort))+50)],
                      sort.iloc[(int(0.5*len(sort))-50):(int(0.5*len(sort))+50)],
                      sort.iloc[(int(0.75*len(sort))-50):(int(0.75*len(sort))+50)]])
for row in examples.itertuples():
    # Load MSA
    msa = []
    for header, seq in read_fasta(f'../../../data/alignments/fastas/{row.OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        spid = re.search(spid_regex, header).group(1)
        msa.append({'ppid': ppid, 'spid': spid, 'seq': seq})
    msa = sorted(msa, key=lambda x: tip_order[x['spid']])

    # Get missing segments
    ppid2trims = {}
    with open(f'../../../data/alignments/missing/{row.OGid}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            trims = []
            for trim in fields['slices'].split(','):
                if trim:
                    start, stop = trim.split('-')
                    trims.append((int(start), int(stop)))
            ppid2trims[fields['ppid']] = trims

    # Align scores and interpolate between gaps that are not missing segments
    aligned_scores = np.full((len(msa), len(msa[0]['seq'])), np.nan)
    for i, record in enumerate(msa):
        ppid, seq = record['ppid'], record['seq']
        scores = load_scores(f'../aucpred_scores/out/{row.OGid}/{ppid}.diso_noprof')  # Remove anything after trailing .
        idx = 0
        for j, sym in enumerate(seq):
            if sym not in ['-', '.']:
                aligned_scores[i, j] = scores[idx]
                idx += 1

        nan_idx = np.isnan(aligned_scores[i])
        range_scores = np.arange(len(msa[0]['seq']))
        interp_scores = np.interp(range_scores[nan_idx], range_scores[~nan_idx], aligned_scores[i, ~nan_idx],
                                  left=np.nan, right=np.nan)
        aligned_scores[i, nan_idx] = interp_scores

        for start, stop in ppid2trims[ppid]:
            aligned_scores[i, start:stop] = np.nan
    aligned_scores = np.ma.masked_invalid(aligned_scores)

    # Plot all score traces
    plot_msa_data([record['seq'] for record in msa], aligned_scores, **plot_msa_kwargs)
    plt.savefig(f'out/traces/{row.Index:04}_{row.OGid}_all.png')
    plt.close()

    # Get Brownian weights and calculate root score
    spids = [record['spid'] for record in msa]
    tree = tree_template.shear(spids)
    tips, weights = get_brownian_weights(tree)
    weight_dict = {tip.name: weight for tip, weight in zip(tips, weights)}
    weight_array = np.zeros((len(msa), 1))
    for i, record in enumerate(msa):
        weight_array[i] = weight_dict[record['spid']]

    weight_sum = (weight_array * ~aligned_scores.mask).sum(axis=0)
    root_scores = (weight_array * aligned_scores).sum(axis=0) / weight_sum
    rate_scores = (weight_array * (aligned_scores - root_scores) ** 2).sum(axis=0) / weight_sum

    # Plot root score trace
    fig = plot_msa_data([record['seq'] for record in msa], root_scores, **plot_msa_kwargs)
    upper = root_scores + rate_scores ** 0.5
    lower = root_scores - rate_scores ** 0.5
    axs = [ax for i, ax in enumerate(fig.axes) if i % 2 == 1]
    for ax in axs:
        xmin, xmax = ax.get_xlim()
        xrange = np.arange(xmin, xmax)
        ax.fill_between(xrange, upper[int(xmin):int(xmax)], lower[int(xmin):int(xmax)], alpha=0.25)
    plt.savefig(f'out/traces/{row.Index:04}_{row.OGid}_root.png')
    plt.close()
