"""Plot statistics from fitted evolutionary models."""

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
    if not os.path.exists(f'out/regions_{min_length}/'):
        os.makedirs(f'out/regions_{min_length}/')

    # Load regions
    rows = []
    with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
            rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder})
    all_regions = pd.DataFrame(rows)

    asr_rates = pd.read_table(f'../../evofit/asr_stats/out/regions_{min_length}/rates.tsv')
    asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
    row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
    asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

    row_idx = (asr_rates['aa_rate_mean'] > min_aa_rate) | (asr_rates['indel_rate_mean'] > min_indel_rate)
    column_idx = ['OGid', 'start', 'stop', 'disorder']
    region_keys = asr_rates.loc[row_idx, column_idx]

    models = pd.read_table(f'../get_models/out/models_{min_length}.tsv', header=[0, 1])
    models = region_keys.merge(models.droplevel(1, axis=1), how='left', on=['OGid', 'start', 'stop'])
    models = models.set_index(['OGid', 'start', 'stop', 'disorder'])

    feature_groups = {}
    feature_labels = []
    nonmotif_labels = []
    with open(f'../get_models/out/models_{min_length}.tsv') as file:
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

    columns = {}
    for feature_label in feature_labels:
        columns[f'{feature_label}_AIC_BM'] = 2*(2 - models[f'{feature_label}_loglikelihood_BM'])
        columns[f'{feature_label}_AIC_OU'] = 2*(3 - models[f'{feature_label}_loglikelihood_OU'])
        columns[f'{feature_label}_delta_AIC'] = columns[f'{feature_label}_AIC_BM'] - columns[f'{feature_label}_AIC_OU']
        columns[f'{feature_label}_sigma2_ratio'] = models[f'{feature_label}_sigma2_BM'] / models[f'{feature_label}_sigma2_OU']
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
    fig.savefig(f'out/regions_{min_length}/hist_regionnum-rate.png')

    # Individual feature plots
    for feature_label in feature_labels:
        # AIC histograms
        fig, ax = plt.subplots()
        ax.hist(models[f'{feature_label}_delta_AIC'], bins=50)
        ax.set_xlabel('$\mathregular{AIC_{BM} - AIC_{OU}}$' + f' ({feature_label})')
        ax.set_ylabel('Number of regions')
        fig.savefig(f'out/regions_{min_length}/hist_regionnum-delta_AIC_{feature_label}.png')
        plt.close()

        # sigma2 histograms
        fig, ax = plt.subplots()
        ax.hist(models[f'{feature_label}_sigma2_ratio'], bins=50)
        ax.set_xlabel('$\mathregular{\sigma_{BM}^2 / \sigma_{OU}^2}$' + f' ({feature_label})')
        ax.set_ylabel('Number of regions')
        fig.savefig(f'out/regions_{min_length}/hist_regionnum-sigma2_{feature_label}.png')
        plt.close()

        # AIC-sigma2 hexbins
        fig, ax = plt.subplots()
        hb = ax.hexbin(models[f'{feature_label}_delta_AIC'],
                       models[f'{feature_label}_sigma2_ratio'],
                       gridsize=75, mincnt=1, linewidth=0, bins='log')
        ax.set_xlabel('$\mathregular{AIC_{BM} - AIC_{OU}}$')
        ax.set_ylabel('$\mathregular{\sigma_{BM}^2 / \sigma_{OU}^2}$')
        ax.set_title(feature_label)
        fig.colorbar(hb)
        fig.savefig(f'out/regions_{min_length}/hexbin_sigma2-delta_AIC_{feature_label}.png')
        plt.close()

    # PCAs
    column_labels = [f'{feature_label}_delta_AIC' for feature_label in feature_labels]
    column_labels_nonmotif = [f'{feature_label}_delta_AIC' for feature_label in nonmotif_labels]
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

        # Feature variance pie chart
        var = data.var().sort_values(ascending=False)
        truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
        labels = [column_label.removesuffix('_delta_AIC') for column_label in truncate.index]
        fig, ax = plt.subplots(gridspec_kw={'right': 0.65})
        ax.pie(truncate.values, labels=labels, labeldistance=None)
        ax.set_title(f'Feature variance\n{title_label}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig(f'out/regions_{min_length}/pie_variance_{data_label}_{file_label}.png')
        plt.close()

        # Scree plot
        fig, ax = plt.subplots()
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
               label=data_label, color=color)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(title_label)
        ax.legend()
        fig.savefig(f'out/regions_{min_length}/bar_scree_{data_label}_{file_label}.png')
        plt.close()

        # PCA scatters
        arrow_labels = [column_label.removesuffix('_delta_AIC') for column_label in data.columns]
        fig = plot_pca(transform, 0, 1, cmap, data_label, title_label,
                       hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor,
                       width_ratios=width_ratios)
        fig.savefig(f'out/regions_{min_length}/hexbin_pc1-pc2_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca_arrows(pca, transform, arrow_labels, 0, 1, cmap, title_label,
                              hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                              width_ratios=width_ratios)
        fig.savefig(f'out/regions_{min_length}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png')
        plt.close()

        fig = plot_pca(transform, 1, 2, cmap, data_label, title_label,
                       hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor,
                       width_ratios=width_ratios)
        fig.savefig(f'out/regions_{min_length}/hexbin_pc2-pc3_{data_label}_{file_label}.png')
        plt.close()

        fig = plot_pca_arrows(pca, transform, arrow_labels, 1, 2, cmap, title_label,
                              hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                              width_ratios=width_ratios)
        fig.savefig(f'out/regions_{min_length}/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png')
        plt.close()

    # Hierarchical heatmap
    legend_args = {'aa_group': ('Amino acid content', 'grey', ''),
                   'charge_group': ('Charge properties', 'black', ''),
                   'physchem_group': ('Physiochemical properties', 'white', ''),
                   'complexity_group': ('Repeats and complexity', 'white', 4 * '.'),
                   'motifs_group': ('Motifs', 'white', 4 * '\\')}
    group_labels = ['aa_group', 'charge_group', 'motifs_group', 'physchem_group', 'complexity_group']
    group_labels_nonmotif = ['aa_group', 'charge_group', 'physchem_group', 'complexity_group']
    gridspec_kw = {'width_ratios': [0.1, 0.9], 'wspace': 0,
                   'height_ratios': [0.975, 0.025], 'hspace': 0.01,
                   'left': 0.05, 'right': 0.95, 'top': 0.95, 'bottom': 0.125}

    plots = [('euclidean', group_labels, 'all'),
             ('euclidean', group_labels_nonmotif, 'nonmotif'),
             ('correlation', group_labels, 'all'),
             ('correlation', group_labels_nonmotif, 'nonmotif')]
    for metric, group_labels, file_label in plots:
        column_labels = []
        for group_label in group_labels:
            column_labels.extend([f'{feature_label}_delta_AIC' for feature_label in feature_groups[group_label]])
        array = np.nan_to_num(models.loc[pdidx[:, :, :, True], column_labels].to_numpy(), nan=1)  # Re-arrange and convert to array

        cdm = pdist(array, metric=metric)
        lm = linkage(cdm, method='average')

        # Convert to tree and get branch colors
        tree = make_tree(lm)
        tip_order = [tip.name for tip in tree.tips()]
        node2color, node2tips = {}, {}
        for node in tree.postorder():
            if node.is_tip():
                tips = 1
            else:
                tips = sum([node2tips[child] for child in node.children])
            node2tips[node] = tips
            node2color[node] = str(max(0, (11 - tips) / 10))

        fig, axs = plt.subplots(2, 2, figsize=(7.5, 7.5), gridspec_kw=gridspec_kw)

        # Tree
        ax = axs[0, 0]
        plot_tree(tree, ax=ax, linecolor=node2color, linewidth=0.2, tip_labels=False,
                  xmin_pad=0.025, xmax_pad=0)
        ax.sharey(axs[0, 1])
        ax.set_ylabel('Disorder regions')
        ax.set_axis_off()

        # Heatmap
        ax = axs[0, 1]
        im = ax.imshow(array[tip_order], aspect='auto', cmap=plt.colormaps['inferno'], interpolation='none')
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('Features')
        ax.set_axis_off()

        # Corner axis
        ax = axs[1, 0]
        ax.set_visible(False)

        # Legend
        ax = axs[1, 1]
        x = 0
        handles = []
        for group_label in group_labels:
            label, color, hatch = legend_args[group_label]
            dx = len(feature_groups[group_label]) / len(column_labels)
            rect = Rectangle((x, 0), dx, 1, label=label, facecolor=color, hatch=hatch,
                             edgecolor='black', linewidth=0.75, clip_on=False)
            ax.add_patch(rect)
            handles.append(rect)
            x += dx
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.25, 0), fontsize=8)
        ax.set_axis_off()

        # Colorbar
        xcenter = 0.75
        width = 0.2
        ycenter = gridspec_kw['bottom'] / 2
        height = 0.015
        cax = fig.add_axes((xcenter - width / 2, ycenter - height / 2, width, height))
        cax.set_title('$\mathregular{AIC_{BM} - AIC_{OU}}$', fontdict={'fontsize': 10})
        fig.colorbar(im, cax=cax, orientation='horizontal')

        fig.savefig(f'out/regions_{min_length}/heatmap_{file_label}_{metric}.png', dpi=600)
        plt.close()
