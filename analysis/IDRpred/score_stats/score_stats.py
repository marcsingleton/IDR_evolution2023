"""Plot statistics of AUCpreD Brownian motion parameters."""

import os
import re
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
from src.draw import plot_msa_data
from src.phylo import get_brownian_weights
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

plot_msa_kwargs = {'hspace': 0.2, 'left': 0.1, 'right': 0.925, 'top': 0.99, 'bottom': 0.05, 'anchor': (0, 0.5),
                   'data_min': -0.05, 'data_max': 1.05,
                   'tree_position': 0, 'tree_width': 0.1,
                   'msa_legend': True, 'legend_kwargs': {'bbox_to_anchor': (0.935, 0.5), 'loc': 'center left', 'fontsize': 10,
                                                         'handletextpad': 0.5, 'markerscale': 2, 'handlelength': 1}}

tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_template.tips())}

roots = pd.read_table('../get_contrasts/out/roots.tsv')
contrasts = pd.read_table('../get_contrasts/out/contrasts.tsv').set_index(['OGid', 'contrast_id'])
rates = (contrasts ** 2).groupby('OGid').mean()
df = roots.merge(rates, on='OGid', suffixes=('_root', '_rate'))

columns = ['score_fraction', 'binary_fraction']
labels = ['Average AUCpreD score', 'Fraction disorder']
colors = ['C0', 'C1']

fig, axs = plt.subplots(2, 1, sharex=True)
for ax, column, label, color in zip(axs, columns, labels, colors):
    ax.hist(df[f'{column}_root'], bins=np.linspace(0, 1, 100), color=color)
    ax.set_xlabel(f'{label} at root')
    ax.set_ylabel('Number of alignments')
fig.savefig('out/hist_alignmentnum-root.png')
plt.close()

xs = [get_quantile(df[f'{column}_rate'].to_numpy(), 0.99) for column in columns]
xmin = min([x.min() for x in xs])
xmax = min([x.max() for x in xs])
xrange = np.linspace(xmin, xmax, 100)
fig, axs = plt.subplots(2, 1, sharex=True)
for ax, x, label, color in zip(axs, xs, labels, colors):
    ax.hist(x, bins=xrange, color=color)
    ax.set_xlabel(f'{label} rate')
    ax.set_ylabel('Number of alignments')
fig.savefig('out/hist_alignmentnum-rate.png')
plt.close()

for ax, column, label in zip(axs, columns, labels):
    plt.hexbin(df[f'{column}_root'], df[f'{column}_rate'], gridsize=50, mincnt=1, linewidth=0)
    plt.xlabel(f'{label} at root')
    plt.ylabel(f'{label} rate')
    plt.colorbar()
    plt.savefig(f'out/hexbin_rate-root_{column}.png')
    plt.close()

plt.hexbin(df[f'{columns[0]}_root'], df[f'{columns[1]}_root'], gridsize=50, mincnt=1, linewidth=0)
plt.xlabel(f'{labels[0]} at root')
plt.ylabel(f'{labels[1]} at root')
plt.colorbar()
plt.savefig(f'out/hexbin_{columns[1]}_root-{columns[0]}_root.png')
plt.close()

plt.hexbin(df[f'{columns[0]}_rate'], df[f'{columns[1]}_rate'], gridsize=50, mincnt=1, linewidth=0, bins='log')
plt.xlabel(f'{labels[0]} rate')
plt.ylabel(f'{labels[1]} rate')
plt.colorbar()
plt.savefig(f'out/hexbin_{columns[1]}_rate-{columns[0]}_rate.png')
plt.close()

if not os.path.exists('out/traces/'):
    os.mkdir('out/traces/')

sort = df.sort_values(by='score_fraction_rate', ascending=False, ignore_index=True)
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
    ppid2missing = {}
    with open(f'../../../data/alignments/missing_trim/{row.OGid}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            missing = []
            for s in fields['slices'].split(','):
                if s:
                    start, stop = s.split('-')
                    missing.append((int(start), int(stop)))
            ppid2missing[fields['ppid']] = missing

    # Align scores and interpolate between gaps that are not missing segments
    aligned_scores = np.full((len(msa), len(msa[0]['seq'])), np.nan)
    for i, record in enumerate(msa):
        ppid, seq = record['ppid'], record['seq']
        scores = load_scores(f'../get_scores/out/{row.OGid}/{ppid}.diso_noprof')  # Remove anything after trailing .
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

        for start, stop in ppid2missing[ppid]:
            aligned_scores[i, start:stop] = np.nan
    aligned_scores = np.ma.masked_invalid(aligned_scores)

    # Get Brownian weights and calculate root score
    spids = [record['spid'] for record in msa]
    tree = tree_template.shear(spids)
    for node in tree.postorder():  # Ensure tree is ordered as in original
        if node.is_tip():
            node.value = tip_order[node.name]
        else:
            node.children = sorted(node.children, key=lambda x: x.value)
            node.value = sum([child.value for child in node.children])

    tips, weights = get_brownian_weights(tree)
    weight_dict = {tip.name: weight for tip, weight in zip(tips, weights)}
    weight_array = np.zeros((len(msa), 1))
    for i, record in enumerate(msa):
        weight_array[i] = weight_dict[record['spid']]

    weight_sum = (weight_array * ~aligned_scores.mask).sum(axis=0)
    root_scores = (weight_array * aligned_scores).sum(axis=0) / weight_sum
    rate_scores = (weight_array * (aligned_scores - root_scores) ** 2).sum(axis=0) / weight_sum
    upper = root_scores + rate_scores ** 0.5
    lower = root_scores - rate_scores ** 0.5

    # Format tree colors
    # Tip values are assigned by using a cumulative sum of Brownian weights in tip order to span the interval [0, 1]
    # Tip colors are assigned by using a colormap to map this value to a color
    # The values of internal nodes are calculated using a weighted average of the tip values
    # These values are then converted to colors with the colormap; this ensures all colors are "in" the colormap
    cmap = plt.colormaps['magma']
    get_color = lambda x: cmap(0.8 * x + 0.1)  # Shift range to [0.1, 0.9]
    weight_dict = {tip.name: weight for tip, weight in zip(tips, weights)}
    value_dict = {tip.name: i / len(tips) for i, tip in enumerate(tips)}

    node2color, node2names = {}, {}
    for node in tree.postorder():
        if node.is_tip():
            node2color[node] = get_color(value_dict[node.name])
            node2names[node] = [node.name]
        else:
            names = [name for child in node.children for name in node2names[child]]
            ws = [weight_dict[name] for name in names]
            vs = [value_dict[name] for name in names]
            value = sum([w * v for w, v in zip(ws, vs)]) / sum(ws)
            node2color[node] = get_color(value)
            node2names[node] = names

    # Plot all score traces
    fig = plot_msa_data([record['seq'] for record in msa], aligned_scores,
                        data_linewidths=1, data_colors=[node2color[tip] for tip in tree.tips()],
                        tree=tree, tree_kwargs={'linewidth': 1, 'linecolor': node2color,
                                                'tip_labels': False, 'xmin_pad': 0.025, 'xmax_pad': 0.025},
                        **plot_msa_kwargs)
    axs = [ax for i, ax in enumerate(fig.axes) if i % 2 == 1]
    for ax in axs:
        ax.set_ylabel('Disorder score')
    fig.savefig(f'out/traces/{row.Index:04}_{row.OGid}_all.png', dpi=300)
    plt.close()

    # Plot root score trace
    fig = plot_msa_data([record['seq'] for record in msa], root_scores,
                        tree=tree, tree_kwargs={'linewidth': 1, 'linecolor': 'black',
                                                'tip_labels': False, 'xmin_pad': 0.025, 'xmax_pad': 0.025},
                        **plot_msa_kwargs)
    axs = [ax for i, ax in enumerate(fig.axes) if i % 2 == 1]
    for ax in axs:
        xmin, xmax = ax.get_xlim()
        xrange = np.arange(xmin, xmax)
        ax.fill_between(xrange, upper[int(xmin):int(xmax)], lower[int(xmin):int(xmax)], alpha=0.25)
        ax.set_ylabel('Disorder score')
    fig.savefig(f'out/traces/{row.Index:04}_{row.OGid}_root.png', dpi=300)
    plt.close()
