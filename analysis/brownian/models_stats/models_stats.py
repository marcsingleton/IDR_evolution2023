"""Plot statistics from fit evolutionary models."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from src.brownian.pca_plots import plot_pca, plot_pca_arrows, plot_pca2, plot_pca2_arrows
from src.draw import plot_tree


def make_tree(lm):
    num_tips = len(lm) + 1
    nodes = {node_id: skbio.TreeNode(name=node_id, children=[]) for node_id in range(num_tips)}
    heights = {node_id: 0 for node_id in range(2*num_tips-1)}
    for idx in range(len(lm)):
        node_id = idx + num_tips
        child_id1, child_id2, distance, _ = lm[idx]
        child1, child2 = nodes[child_id1], nodes[child_id2]
        height1, height2 = heights[child_id1], heights[child_id2]
        child1.length = distance - height1
        child2.length = distance - height2
        parent = skbio.TreeNode(name=node_id, children=[child1, child2])
        nodes[node_id] = parent
        heights[node_id] = distance
    tree = nodes[2*(num_tips-1)]
    return tree


pdidx = pd.IndexSlice

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 1
min_indel_rate = 1

pca_components = 10
cmap1, cmap2, cmap3 = plt.colormaps['Blues'], plt.colormaps['Reds'], plt.colormaps['Purples']
hexbin_kwargs = {'gridsize': 75, 'mincnt': 1, 'linewidth': 0}
hexbin_kwargs_log = {'gridsize': 75, 'mincnt': 1, 'linewidth': 0}
handle_markerfacecolor = 0.6
legend_kwargs = {'fontsize': 8, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
arrow_colors = ['#e15759', '#499894', '#59a14f', '#f1ce63', '#b07aa1', '#d37295', '#9d7660', '#bab0ac',
                '#ff9d9a', '#86bcb6', '#8cd17d', '#b6992d', '#d4a6c8', '#fabfd2', '#d7b5a6', '#79706e']

# Load regions
rows = []
with open('../../IDRpred/regions_filter/out/regions_30.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        rows.append({'OGid': fields['OGid'], 'start': int(fields['start']), 'stop': int(fields['stop']),
                     'disorder': fields['disorder'] == 'True'})
regions = pd.DataFrame(rows)

models = pd.read_table('../get_models/out/models_30.tsv', header=[0, 1])
rates = pd.read_table('../../evosim/asr_stats/out/regions_30/rates.tsv')

df = regions.merge(models.droplevel(1, axis=1), how='left', on=['OGid', 'start', 'stop'])
df = df.merge(rates, how='left', on=['OGid', 'start', 'stop']).set_index(['OGid', 'start', 'stop', 'disorder'])

# Data filtering
df.loc[(df['indel_num_columns'] < min_indel_columns) | df['indel_rate_mean'].isna(), 'indel_rate_mean'] = 0
df = df[(df['aa_rate_mean'] > min_aa_rate) | (df['indel_rate_mean'] > min_indel_rate)]

feature_groups = {}
for column_label, group_label in models.columns:
    if not column_label.endswith('_loglikelihood_BM') or group_label == 'ids_group':
        continue
    feature_label = column_label.removesuffix('_loglikelihood_BM')
    try:
        feature_groups[group_label].append(feature_label)
    except KeyError:
        feature_groups[group_label] = [feature_label]
feature_labels = sum(feature_groups.values(), start=[])

columns = {}
for feature_label in feature_labels:
    columns[f'{feature_label}_AIC_BM'] = 2*(2 - df[f'{feature_label}_loglikelihood_BM'])
    columns[f'{feature_label}_AIC_OU'] = 2*(3 - df[f'{feature_label}_loglikelihood_OU'])
    columns[f'{feature_label}_delta_AIC'] = columns[f'{feature_label}_AIC_BM'] - columns[f'{feature_label}_AIC_OU']
    columns[f'{feature_label}_sigma2_ratio'] = df[f'{feature_label}_sigma2_BM'] / df[f'{feature_label}_sigma2_OU']
df = pd.concat([df, pd.DataFrame(columns)], axis=1)

if not os.path.exists('out/'):
    os.mkdir('out/')

for feature_label in feature_labels:
    fig, ax = plt.subplots()
    ax.hist(df[f'{feature_label}_delta_AIC'], bins=50)
    ax.set_xlabel('$\mathregular{AIC_{BM} - AIC_{OU}}$' + f' ({feature_label})')
    ax.set_ylabel('Number of regions')
    fig.savefig(f'out/hist_regionnum-delta_AIC_{feature_label}.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(df[f'{feature_label}_sigma2_ratio'], bins=50)
    ax.set_xlabel('$\mathregular{\sigma_{BM}^2 / \sigma_{OU}^2}$' + f' ({feature_label})')
    ax.set_ylabel('Number of regions')
    fig.savefig(f'out/hist_regionnum-sigma2_{feature_label}.png')
    plt.close()

    fig, ax = plt.subplots()
    hb = ax.hexbin(df[f'{feature_label}_delta_AIC'],
                   df[f'{feature_label}_sigma2_ratio'],
                   gridsize=75, mincnt=1, linewidth=0, bins='log')
    ax.set_xlabel('$\mathregular{AIC_{BM} - AIC_{OU}}$')
    ax.set_ylabel('$\mathregular{\sigma_{BM}^2 / \sigma_{OU}^2}$')
    ax.set_title(feature_label)
    fig.colorbar(hb)
    fig.savefig(f'out/hexbin_sigma2-delta_AIC_{feature_label}.png')
    plt.close()

column_labels = [f'{feature_label}_sigma2_ratio' for feature_label in feature_labels]
column_labels_motifs = [f'{feature_label}_sigma2_ratio' for feature_label in feature_labels
                        if feature_label not in set(feature_groups['motifs_group'])]
plots = [(df.loc[pdidx[:, :, :, True], column_labels], 'disorder', 'all features', 'all'),
         (df.loc[pdidx[:, :, :, True], column_labels_motifs], 'disorder', 'no motifs', 'motifs'),]
for data, data_label, title_label, file_label in plots:
    pca = PCA(n_components=pca_components)
    transform = pca.fit_transform(np.nan_to_num(data.to_numpy(), nan=1))
    arrow_labels = [column_label.removesuffix('_sigma2_ratio') for column_label in data.columns]
    cmap = cmap1 if data_label == 'disorder' else cmap2
    width_ratios = (0.79, 0.03, 0.03, 0.15)

    # Feature variance pie chart
    var = data.var().sort_values(ascending=False)
    truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
    plt.pie(truncate.values, labels=truncate.index, labeldistance=None)
    plt.title(f'Feature variance\n{title_label}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.65)
    plt.savefig(f'out/pie_variance_{data_label}_{file_label}.png')
    plt.close()

    # Scree plot
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, label=data_label,
            color=cmap(0.6))
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title(title_label)
    plt.legend()
    plt.savefig(f'out/bar_scree_{data_label}_{file_label}.png')
    plt.close()

    # PCA scatters
    plot_pca(transform, 0, 1, cmap, data_label, title_label,
             f'out/hexbin_pc1-pc2_{data_label}_{file_label}.png',
             hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor,
             width_ratios=width_ratios)
    plot_pca_arrows(pca, transform, arrow_labels, 0, 1, cmap, title_label,
                    f'out/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png',
                    hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                    width_ratios=width_ratios)

    plot_pca(transform, 1, 2, cmap, data_label, title_label,
             f'out/hexbin_pc2-pc3_{data_label}_{file_label}.png',
             hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor,
             width_ratios=width_ratios)
    plot_pca_arrows(pca, transform, arrow_labels, 1, 2, cmap, title_label,
                    f'out/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png',
                    hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                    width_ratios=width_ratios)

# Hierarchical heatmap
feature_group_records = [('aa_group', 'Amino acid content', 'grey', ''),
                         ('charge_group', 'Charge properties', 'black', ''),
                         ('motifs_group', 'Motifs', 'white', 4 * '\\'),
                         ('physchem_group', 'Physiochemical properties', 'white', ''),
                         ('complexity_group', 'Repeats and complexity', 'white', 4 * '.')]
gridspec_kw = {'width_ratios': [0.1, 0.9], 'wspace': 0,
               'height_ratios': [0.975, 0.025], 'hspace': 0.01,
               'left': 0.05, 'right': 0.95, 'top': 0.95, 'bottom': 0.1}

column_labels = []
for feature_group_record in feature_group_records:
    group_label = feature_group_record[0]
    column_labels.extend([f'{feature_label}_sigma2_ratio' for feature_label in feature_groups[group_label]])
array = np.nan_to_num(df.loc[pdidx[:, :, :, True], column_labels].to_numpy(), nan=1)  # Re-arrange and convert to array

for metric in ['euclidean', 'correlation']:
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
        alpha = min(1, (tips - 1) / 10)
        node2color[node] = (0, 0, 0, alpha)

    fig, axs = plt.subplots(2, 2, figsize=(6, 9), gridspec_kw=gridspec_kw)

    # Tree
    ax = axs[0, 0]
    plot_tree(tree, ax=ax, linecolor=node2color, linewidth=0.2, tip_labels=False,
              xmin_pad=0.025, xmax_pad=0, ymin_pad=1/(2*len(array)), ymax_pad=1/(2*len(array)))
    ax.set_ylabel('Disordered regions')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Heatmap
    ax = axs[0, 1]
    im = ax.imshow(array[tip_order], aspect='auto',
                   cmap=plt.colormaps['inferno_r'], vmin=0, vmax=1, interpolation='none')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Features')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Corner axis
    ax = axs[1, 0]
    ax.set_visible(False)

    # Legend
    ax = axs[1, 1]
    x = 0
    handles = []
    for group_label, legend_label, color, hatch in feature_group_records:
        dx = len(feature_groups[group_label]) / len(feature_labels)
        rectangle = mpatches.Rectangle((x, 0), dx, 1, label=legend_label, facecolor=color, hatch=hatch,
                                       edgecolor='black', linewidth=0.75, clip_on=False)
        ax.add_patch(rectangle)
        handles.append(rectangle)
        x += dx
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.25, 0), fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Colorbar
    xcenter = 0.75
    width = 0.2
    ycenter = gridspec_kw['bottom'] / 2
    height = 0.015
    cax = fig.add_axes((xcenter - width / 2, ycenter - height / 2, width, height))
    cax.set_title('$\mathregular{\sigma_{BM}^2 / \sigma_{OU}^2}$', fontdict={'fontsize': 10})
    fig.colorbar(im, cax=cax, orientation='horizontal')

    fig.savefig(f'out/cluster_{metric}.png', dpi=600)
    plt.close()
