"""Plot statistics from fitted evolutionary models."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from src.brownian.linkage import make_tree
from src.brownian.pca import plot_pca, plot_pca_arrows
from src.draw import plot_tree


pdidx = pd.IndexSlice
min_lengths = [30, 60, 90]

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 1
min_indel_rate = 0.1

with open('../simulate_stats/out/BM/critvals.json') as file:
    critvals = json.load(file)

pca_components = 10
cmap1, cmap2 = plt.colormaps['Blues'], plt.colormaps['Oranges']
color1, color2 = '#4e79a7', '#f28e2b'
hexbin_kwargs = {'gridsize': 75, 'mincnt': 1, 'linewidth': 0}
hexbin_kwargs_log = {'gridsize': 75, 'mincnt': 1, 'linewidth': 0}
handle_markerfacecolor = 0.6
legend_kwargs = {'fontsize': 8, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
arrow_colors = ['#e15759', '#499894', '#59a14f', '#f1ce63', '#b07aa1', '#d37295', '#9d7660', '#bab0ac',
                '#ff9d9a', '#86bcb6', '#8cd17d', '#b6992d', '#d4a6c8', '#fabfd2', '#d7b5a6', '#79706e']

for min_length in min_lengths:
    prefix = f'out/regions_{min_length}/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Load regions
    rows = []
    with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
            rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder})
    all_regions = pd.DataFrame(rows)

    # Filter by rates
    asr_rates = pd.read_table(f'../../evofit/asr_stats/out/regions_{min_length}/rates.tsv')
    asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
    row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
    asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

    row_idx = (asr_rates['aa_rate_mean'] > min_aa_rate) | (asr_rates['indel_rate_mean'] > min_indel_rate)
    column_idx = ['OGid', 'start', 'stop', 'disorder']
    region_keys = asr_rates.loc[row_idx, column_idx]

    # Load models
    models = pd.read_table(f'../model_compute/out/models_{min_length}.tsv', header=[0, 1])
    models = region_keys.merge(models.droplevel(1, axis=1), how='left', on=['OGid', 'start', 'stop'])
    models = models.set_index(['OGid', 'start', 'stop', 'disorder'])

    # Extract labels
    feature_groups = {}
    feature_labels = []
    nonmotif_labels = []
    with open(f'../model_compute/out/models_{min_length}.tsv') as file:
        column_labels = file.readline().rstrip('\n').split('\t')
        group_labels = file.readline().rstrip('\n').split('\t')
    for column_label, group_label in zip(column_labels, group_labels):
        if not column_label.endswith('_loglikelihood_BM') or group_label == 'ids_group':
            continue
        feature_label = column_label.removesuffix('_loglikelihood_BM')
        try:
            feature_groups[group_label].append(feature_label)
        except KeyError:
            feature_groups[group_label] = [feature_label]
        feature_labels.append(feature_label)
        if group_label != 'motifs_group':
            nonmotif_labels.append(feature_label)

    # Calculate delta loglikelihood
    columns = {}
    for feature_label in feature_labels:
        columns[f'{feature_label}_delta_loglikelihood'] = models[f'{feature_label}_loglikelihood_OU'] - models[f'{feature_label}_loglikelihood_BM']
    models = pd.concat([models, pd.DataFrame(columns)], axis=1)

    # ASR rate histogram with cutoff
    fig, axs = plt.subplots(2, 1, gridspec_kw={'right': 0.825, 'top': 0.99, 'bottom': 0.1, 'hspace': 0.25})

    ax = axs[0]
    xs = asr_rates.loc[asr_rates['disorder'], 'aa_rate_mean']
    ax.axvspan(min_aa_rate, xs.max(), color='#e6e6e6')
    ax.hist(xs, bins=100)
    ax.set_xlabel('Average amino acid rate in region')
    ax.set_ylabel('Number of regions')

    ax = axs[1]
    xs = asr_rates.loc[asr_rates['disorder'], 'indel_rate_mean']
    ax.axvspan(min_indel_rate, xs.max(), color='#e6e6e6')
    ax.hist(xs, bins=100)
    ax.set_xlabel('Average indel rate in region')
    ax.set_ylabel('Number of regions')

    fig.legend(handles=[Patch(facecolor=color1, label='disorder')], bbox_to_anchor=(0.825, 0.5), loc='center left')
    fig.savefig(f'{prefix}/hist_regionnum-rate.png')

    # Bar graph of fraction of regions with a significant feature
    column_labels = [f'{feature_label}_delta_loglikelihood' for feature_label in feature_labels]
    ys_95 = (models.loc[pdidx[:, :, :, True], column_labels] > critvals['q95']).mean()
    ys_99 = (models.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).mean()
    xs = list(range(len(column_labels)))
    xs_labels = [label.removesuffix('_delta_loglikelihood') for label in column_labels]

    fig, ax = plt.subplots(figsize=(7.5, 3.75),
                           gridspec_kw={'left': 0.08, 'right': 0.995, 'bottom': 0.325, 'top': 0.975})
    ax.bar(xs, ys_99, label='1%', color='C0')
    ax.bar(xs, ys_95 - ys_99, bottom=ys_99, label='5%', color='C1')
    ax.set_xmargin(0.005)
    ax.set_xticks(xs, xs_labels, fontsize=5.5,
                  rotation=60, rotation_mode='anchor', ha='right', va='center')
    ax.set_ylabel('Fraction of regions')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2,
              title='Type I error rate', fontsize=8, title_fontsize=8)
    fig.savefig(f'{prefix}/bar_regionfrac-feature.png', dpi=300)
    plt.close()

    # Bar graph of fraction of regions with a significant feature
    column_labels = [f'{feature_label}_delta_loglikelihood' for feature_label in feature_labels]
    ys_95 = (models.loc[pdidx[:, :, :, True], column_labels] > critvals['q95']).sum().transform(lambda x: x/x.sum())
    ys_99 = (models.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).sum().transform(lambda x: x/x.sum())
    xs = list(range(len(column_labels)))
    xs_labels = [label.removesuffix('_delta_loglikelihood') for label in column_labels]

    plots = [(ys_99, '1%', 'C0'),
             (ys_99, '5%', 'C1')]
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 4),
                            gridspec_kw={'left': 0.08, 'right': 0.995, 'bottom': 0.325, 'top': 0.975})
    for ax, (ys, data_label, color) in zip(axs, plots):
        ax.bar(xs, ys, label=data_label, color=color)
        ax.set_xmargin(0.005)
        ax.set_xticks(xs, ['' for _ in xs])
        ax.set_ylabel('Fraction of features')
    ax.set_xticks(xs, xs_labels, fontsize=5.5,
                  rotation=60, rotation_mode='anchor', ha='right', va='center')
    fig.legend(loc='upper center', bbox_to_anchor=(0.54, 0.1), ncol=2,
               title='Type I error rate', fontsize=8, title_fontsize=8)
    fig.savefig(f'{prefix}/bar_featurefrac-feature.png', dpi=300)
    plt.close()

    # Distribution of number of significant features in regions
    rng = np.random.default_rng(1)
    column_labels = [f'{feature_label}_delta_loglikelihood' for feature_label in feature_labels]

    counts_95 = (models.loc[pdidx[:, :, :, True], column_labels] > critvals['q95']).sum(axis=1).value_counts()
    shuffle_95 = rng.permuted((models.loc[pdidx[:, :, :, True], column_labels] > critvals['q95']).to_numpy(), axis=0)
    xs_95, ys_95 = np.unique(shuffle_95.sum(axis=1), return_counts=True)
    p_95 = shuffle_95.mean()

    counts_99 = (models.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).sum(axis=1).value_counts()
    shuffle_99 = rng.permuted((models.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).to_numpy(), axis=0)
    xs_99, ys_99 = np.unique(shuffle_99.sum(axis=1), return_counts=True)
    p_99 = shuffle_99.mean()

    fig, axs = plt.subplots(2, 1, layout='constrained')

    width = 0.35
    plots = [(counts_99, xs_99, ys_99, p_99, 'observed\n(1% type I error)', 'C0'),
             (counts_95, xs_95, ys_95, p_95, 'observed\n(5% type I error)', 'C1')]
    for ax, (counts, xs, ys, p, label, color) in zip(axs, plots):
        ax.bar(counts.index - width/2, counts.values, width=width, label=label, color=color)
        ax.bar(xs + width/2, ys, width=width, label='random\n($\hat{p} = $' + f'{p:.02})', color='C9')
        ax.set_xmargin(0.01)
        ax.set_ylabel('Number of regions')
        ax.legend(fontsize=8)
    ax.set_xlabel('Number of significant features')
    fig.savefig(f'{prefix}/bar_regionnum-numfeatures.png')
    plt.close()

    # Individual feature plots
    prefix = f'out/regions_{min_length}/features/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    for feature_label in feature_labels:
        # loglikelihood histograms
        fig, ax = plt.subplots()
        ax.hist(models[f'{feature_label}_delta_loglikelihood'], bins=50)
        ax.set_xlabel('$\mathregular{\log L_{OU} / L_{BM}}$' + f' ({feature_label})')
        ax.set_ylabel('Number of regions')
        fig.savefig(f'{prefix}/hist_regionnum-delta_loglikelihood_{feature_label}.png')
        plt.close()

    # PCAs
    prefix = f'out/regions_{min_length}/pcas/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    column_labels = [f'{feature_label}_delta_loglikelihood' for feature_label in feature_labels]
    column_labels_nonmotif = [f'{feature_label}_delta_loglikelihood' for feature_label in nonmotif_labels]
    plots = [(models.loc[pdidx[:, :, :, True], column_labels], 'disorder', 'all features', 'all'),
             (models.loc[pdidx[:, :, :, True], column_labels_nonmotif], 'disorder', 'no motifs', 'nonmotif'),
             (models.loc[pdidx[:, :, :, False], column_labels], 'order', 'all features', 'all'),
             (models.loc[pdidx[:, :, :, False], column_labels_nonmotif], 'order', 'no motifs', 'nonmotif')]
    for data, data_label, title_label, file_label in plots:
        pca = PCA(n_components=pca_components)
        transform = pca.fit_transform(np.nan_to_num(data.to_numpy(), nan=1))
        cmap = cmap1 if data_label == 'disorder' else cmap2
        color = color1 if data_label == 'disorder' else color2
        width_ratios = (0.79, 0.03, 0.03, 0.15)

        # Feature variance bar chart
        var = data.var().sort_values(ascending=False)
        var = var / var.sum()
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        xs_labels = [column_label.removesuffix('_delta_loglikelihood') for column_label in truncate.index]
        fig, ax = plt.subplots(gridspec_kw={'bottom': 0.3})
        ax.bar(range(len(xs_labels)), truncate.values)
        ax.set_xticks(range(len(xs_labels)), xs_labels,
                      rotation=60, rotation_mode='anchor', ha='right', va='center')
        ax.set_ylabel('Explained variance ratio')
        fig.savefig(f'{prefix}/bar_variance_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        fig, ax = plt.subplots()
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
               label=data_label, color=color)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(title_label)
        ax.legend()
        fig.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
        plt.close()

        # PCA scatters
        arrow_labels = [column_label.removesuffix('_delta_loglikelihood') for column_label in data.columns]
        fig = plot_pca(transform, 0, 1, cmap, data_label, title_label,
                       hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor,
                       width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca_arrows(pca, transform, arrow_labels, 0, 1, cmap, title_label,
                              hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                              width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png')
        plt.close()

        fig = plot_pca(transform, 1, 2, cmap, data_label, title_label,
                       hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor,
                       width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca_arrows(pca, transform, arrow_labels, 1, 2, cmap, title_label,
                              hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                              width_ratios=width_ratios)
        fig.savefig(f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png')
        plt.close()

    # Hierarchical heatmap
    prefix = f'out/regions_{min_length}/hierarchy/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    legend_args = {'aa_group': ('Amino acid content', 'grey', ''),
                   'charge_group': ('Charge properties', 'black', ''),
                   'physchem_group': ('Physiochemical properties', 'white', ''),
                   'complexity_group': ('Repeats and complexity', 'white', 4 * '.'),
                   'motifs_group': ('Motifs', 'white', 4 * '\\')}
    metagroups = {'all': ['aa_group', 'charge_group', 'physchem_group', 'complexity_group', 'motifs_group'],
                  'nonmotif': ['aa_group', 'charge_group', 'physchem_group', 'complexity_group']}
    gridspec_kw = {'width_ratios': [0.1, 0.9], 'wspace': 0,
                   'height_ratios': [0.975, 0.025], 'hspace': 0.01,
                   'left': 0.05, 'right': 0.95, 'top': 0.95, 'bottom': 0.125}

    cmap = plt.colormaps['RdBu']
    cmap.set_under((1.0, 0.0, 1.0))
    cmap.set_over((0.0, 1.0, 1.0))
    heatmap_args = {'delta_loglikelihood': {'cmap': plt.colormaps['inferno']},
                    'mu_OU': {'cmap': cmap, 'vmin': -3, 'vmax': 3}}
    colorbar_labels = {'delta_loglikelihood': '$\mathregular{\log L_{OU} / L_{BM}}$',
                       'mu_OU': '$z$-score of $\mathregular{\mu_{OU}}$'}
    colorbar_args = {'delta_loglikelihood': {},
                     'mu_OU': {'extend': 'both', 'extendrect': True, 'extendfrac': 0.03}}

    # Make data sets
    data_set_parameters = [('delta_loglikelihood', 'all', False),
                           ('delta_loglikelihood', 'nonmotif', False),
                           ('mu_OU', 'all', True)]
    data_sets = {}
    for statistic_label, metagroup_label, normalize in data_set_parameters:
        column_labels = []
        for group_label in metagroups[metagroup_label]:
            column_labels.extend([f'{feature_label}_delta_loglikelihood' for feature_label in feature_groups[group_label]])
        data = models.loc[pdidx[:, :, :, True], column_labels]  # Re-arrange columns

        if normalize:
            data = (data - data.mean()) / data.std()

        data_sets[f'{statistic_label}_{metagroup_label}'] = data

    # Make plots
    plots = [('euclidean', 'delta_loglikelihood', 'all'),
             ('euclidean', 'delta_loglikelihood', 'nonmotif'),
             ('correlation', 'delta_loglikelihood', 'all'),
             ('correlation', 'delta_loglikelihood', 'nonmotif'),
             ('euclidean', 'mu_OU', 'all'),
             ('correlation', 'mu_OU', 'all')]
    for metric, statistic_label, metagroup_label in plots:
        data = data_sets[f'{statistic_label}_{metagroup_label}']
        array = np.nan_to_num(data.to_numpy(), nan=1)

        # Convert to tree and calculate some useful data structures
        cdm = pdist(array, metric=metric)
        lm = linkage(cdm, method='average')
        tree = make_tree(lm)
        tip_order = [int(tip.name) for tip in tree.tips()]
        array = array[tip_order]

        # Save tree data
        ids2id = {}
        for tip in tree.tips():
            node_id = int(tip.name)
            OGid, start, stop, disorder = data.iloc[node_id].name
            ids2id[(OGid, start, stop, disorder)] = node_id
        with open(f'{prefix}/heatmap_{statistic_label}_{metagroup_label}_{metric}.tsv', 'w') as file:
            file.write('OGid\tstart\tstop\tdisorder\tnode_id\n')
            for (OGid, start, stop, disorder), node_id in sorted(ids2id.items()):
                file.write(f'{OGid}\t{start}\t{stop}\t{disorder}\t{node_id}\n')
        tree.write(f'{prefix}/heatmap_{statistic_label}_{metagroup_label}_{metric}.nwk')

        # Get branch colors
        node2color, node2tips = {}, {}
        for node in tree.postorder():
            if node.is_tip():
                tips = 1
            else:
                tips = sum([node2tips[child] for child in node.children])
            node2tips[node] = tips
            cmap = plt.colormaps['Greys_r']
            node2color[node] = cmap(max(0., (11 - tips) / 10))

        fig, axs = plt.subplots(2, 2, figsize=(7.5, 7.5), gridspec_kw=gridspec_kw)

        # Tree
        ax = axs[0, 0]
        plot_tree(tree, ax=ax, linecolor=node2color, linewidth=0.2, tip_labels=False,
                  xmin_pad=0.025, xmax_pad=0)
        ax.sharey(axs[0, 1])
        ax.set_ylabel('Disorder regions')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Heatmap
        ax = axs[0, 1]
        im = ax.imshow(array, aspect='auto', interpolation='none', **heatmap_args[statistic_label])
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('Features')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Corner axis
        ax = axs[1, 0]
        ax.set_visible(False)

        # Legend
        ax = axs[1, 1]
        x = 0
        handles = []
        for group_label in metagroups[metagroup_label]:
            label, color, hatch = legend_args[group_label]
            dx = len(feature_groups[group_label]) / len(data.columns)
            rect = Rectangle((x, 0), dx, 1, label=label, facecolor=color, hatch=hatch,
                             edgecolor='black', linewidth=0.75, clip_on=False)
            ax.add_patch(rect)
            handles.append(rect)
            x += dx
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.25, 0), fontsize=8)
        ax.set_axis_off()

        # Colorbar
        xcenter = gridspec_kw['width_ratios'][0] + gridspec_kw['width_ratios'][1] * 0.75
        width = 0.2
        ycenter = gridspec_kw['bottom'] / 2
        height = 0.015
        cax = fig.add_axes((xcenter - width / 2, ycenter - height / 2, width, height))
        cax.set_title(colorbar_labels[statistic_label], fontsize=10)
        fig.colorbar(im, cax=cax, orientation='horizontal', **colorbar_args[statistic_label])

        fig.savefig(f'{prefix}/heatmap_{statistic_label}_{metagroup_label}_{metric}.png', dpi=600)
        plt.close()
