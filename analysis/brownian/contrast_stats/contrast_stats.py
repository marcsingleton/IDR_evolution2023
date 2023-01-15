"""Plot statistics associated with contrasts."""

import os
import re
from math import exp, pi

import matplotlib.pyplot as plt
import pandas as pd
import skbio
from matplotlib.lines import Line2D
from numpy import linspace
from sklearn.decomposition import PCA
from src.brownian.features import motif_regexes
from src.brownian.pca_plots import plot_pca, plot_pca_arrows, plot_pca2, plot_pca2_arrows
from src.utils import read_fasta


def zscore(df):
    return (df - df.mean()) / df.std()


pdidx = pd.IndexSlice
ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'

cmap1, cmap2, cmap3 = plt.colormaps['Blues'], plt.colormaps['Reds'], plt.colormaps['Purples']
hexbin_kwargs = {'gridsize': 75, 'mincnt': 1, 'linewidth': 0}
hexbin_kwargs_log = {'gridsize': 75, 'mincnt': 1, 'linewidth': 0, 'bins': 'log'}
handle_markerfacecolor = 0.6
legend_kwargs = {'fontsize': 8, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
pca_components = 10
arrow_colors = ['#e15759', '#499894', '#59a14f', '#f1ce63', '#b07aa1', '#d37295', '#9d7660', '#bab0ac',
                '#ff9d9a', '#86bcb6', '#8cd17d', '#b6992d', '#d4a6c8', '#fabfd2', '#d7b5a6', '#79706e']

# Load contrasts and tree
features = pd.read_table('../get_features/out/features.tsv')
contrasts = pd.read_table('../get_contrasts/out/contrasts_100.tsv')
roots = pd.read_table('../get_contrasts/out/roots_100.tsv')
tree = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree.tips())}

features.loc[features['kappa'] == -1, 'kappa'] = 1
features.loc[features['omega'] == -1, 'omega'] = 1
features['length'] = features['length'] ** 0.6
features.rename(columns={'length': 'radius_gyration'}, inplace=True)

feature_labels = list(features.columns.drop(['OGid', 'ppid', 'start', 'stop']))
motif_labels = list(motif_regexes)

# Load sequence data
ppid2spid = {}
OGids = sorted([path.removesuffix('.afa') for path in os.listdir('../../../data/alignments/fastas/') if path.endswith('.afa')])
for OGid in OGids:
    for header, _ in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        spid = re.search(spid_regex, header).group(1)
        ppid2spid[ppid] = spid

# Load regions
rows, region2spids = [], {}
with open('../regions_filter/out/regions_100.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder})
        region2spids[(OGid, start, stop)] = [ppid2spid[ppid] for ppid in fields['ppids'].split(',')]
regions = pd.DataFrame(rows)

features = regions.merge(features, how='right', on=['OGid', 'start', 'stop']).set_index(['OGid', 'start', 'stop', 'disorder'])
contrasts = regions.merge(contrasts, how='right', on=['OGid', 'start', 'stop']).set_index(['OGid', 'start', 'stop', 'disorder', 'contrast_id'])
roots = regions.merge(roots, how='right', on=['OGid', 'start', 'stop']).set_index(['OGid', 'start', 'stop', 'disorder'])

# 1 CONTRASTS
if not os.path.exists('out/contrasts/'):
    os.makedirs('out/contrasts/')
prefix = 'out/contrasts/'

# 1.1 Plot contrast distributions
disorder = contrasts.loc[pdidx[:, :, :, True, :], :]
order = contrasts.loc[pdidx[:, :, :, False, :], :]
for feature_label in feature_labels:
    fig, axs = plt.subplots(2, 1, sharex=True)
    xmin, xmax = contrasts[feature_label].min(), contrasts[feature_label].max()
    axs[0].hist(disorder[feature_label], bins=linspace(xmin, xmax, 150), color='C0', label='disorder')
    axs[1].hist(order[feature_label], bins=linspace(xmin, xmax, 150), color='C1', label='order')
    axs[1].set_xlabel(f'Contrast value ({feature_label})')
    for i in range(2):
        axs[i].set_ylabel('Number of contrasts')
        axs[i].legend()
    plt.savefig(f'{prefix}/hist_numcontrasts-{feature_label}.png')
    for i in range(2):
        axs[i].set_yscale('log')
    plt.savefig(f'{prefix}/hist_numcontrasts-{feature_label}_log.png')
    plt.close()

# 2 CONTRAST MEANS
if not os.path.exists('out/means/'):
    os.makedirs('out/means/')
prefix = 'out/means/'

# 2.1 Plot contrast means distributions
means = contrasts.groupby(['OGid', 'start', 'stop', 'disorder']).mean()
disorder = means.loc[pdidx[:, :, :, True, :], :]
order = means.loc[pdidx[:, :, :, False, :], :]
for feature_label in feature_labels:
    fig, axs = plt.subplots(2, 1, sharex=True)
    xmin, xmax = means[feature_label].min(), means[feature_label].max()
    axs[0].hist(disorder[feature_label], bins=linspace(xmin, xmax, 150), color='C0', label='disorder')
    axs[1].hist(order[feature_label], bins=linspace(xmin, xmax, 150), color='C1', label='order')
    axs[1].set_xlabel(f'Mean contrast value ({feature_label})')
    for i in range(2):
        axs[i].set_ylabel('Number of regions')
        axs[i].legend()
    plt.savefig(f'{prefix}/hist_numregions-{feature_label}.png')
    plt.close()

# 2.2 Plot standardized contrast means distributions
# These are sample means which have a mean of 0 and variance sigma^2/n
# Estimate sigma^2 from contrasts by mean of contrast squares (since theoretical contrast mean is 0)
# Standardize sample means by dividing by sigma/sqrt(n)
# Regions with constant contrasts will have 0 variance, so the normalization will result in a NaN
# While it is possible for a tree with unequal tip values to yield constant (non-zero) contrasts, it is unlikely
# Thus constant contrasts are assumed to equal zero
variances = ((contrasts**2).groupby(['OGid', 'start', 'stop', 'disorder']).mean())
sizes = contrasts.groupby(['OGid', 'start', 'stop', 'disorder']).size()
stds = (means / (variances.div(sizes, axis=0))**0.5).fillna(0)
disorder = stds.loc[pdidx[:, :, :, True, :], :]
order = stds.loc[pdidx[:, :, :, False, :], :]
for feature_label in feature_labels:
    fig, axs = plt.subplots(2, 1, sharex=True)
    xmin, xmax = stds[feature_label].min(), stds[feature_label].max()
    axs[0].hist(disorder[feature_label], density=True, bins=linspace(xmin, xmax, 150), color='C0', label='disorder')
    axs[1].hist(order[feature_label], density=True, bins=linspace(xmin, xmax, 150), color='C1', label='order')
    axs[1].set_xlabel(f'Standardized mean contrast value ({feature_label})')
    for i in range(2):
        axs[i].plot(linspace(xmin, xmax), [1/(2*pi)**0.5 * exp(-x**2/2) for x in linspace(xmin, xmax)], color='black')
        axs[i].set_ylabel('Density of regions')
        axs[i].legend()
    plt.savefig(f'{prefix}/hist_numregions-{feature_label}_std.png')
    plt.close()

# 3 RATES
if not os.path.exists('out/rates/'):
    os.makedirs('out/rates/')
prefix = 'out/rates/'

# 3.1 Plot rate distributions
rates = ((contrasts**2).groupby(['OGid', 'start', 'stop', 'disorder']).mean())
rates_motifs = rates.drop(motif_labels, axis=1)
disorder = rates.loc[pdidx[:, :, :, True, :], :]
order = rates.loc[pdidx[:, :, :, False, :], :]
disorder_motifs = disorder.drop(motif_labels, axis=1)
order_motifs = order.drop(motif_labels, axis=1)
for feature_label in feature_labels:
    fig, axs = plt.subplots(2, 1, sharex=True)
    xmin, xmax = rates[feature_label].min(), rates[feature_label].max()
    axs[0].hist(disorder[feature_label], bins=linspace(xmin, xmax, 150), color='C0', label='disorder')
    axs[1].hist(order[feature_label], bins=linspace(xmin, xmax, 150), color='C1', label='order')
    axs[1].set_xlabel(f'Rate ({feature_label})')
    for i in range(2):
        axs[i].set_ylabel('Number of regions')
        axs[i].legend()
    plt.savefig(f'{prefix}/hist_numregions-{feature_label}.png')
    for i in range(2):
        axs[i].set_yscale('log')
    plt.savefig(f'{prefix}/hist_numregions-{feature_label}_log.png')
    plt.close()

# 3.2.1 Plot rate PCAs (combined)
plots = [(rates, 'merge', 'no norm', 'nonorm_all'),
         (zscore(rates), 'merge', 'z-score', 'zscore_all'),
         (rates_motifs, 'merge', 'no norm', 'nonorm_motifs'),
         (zscore(rates_motifs), 'merge', 'z-score', 'zscore_motifs')]
for data, data_label, title_label, file_label in plots:
    pca = PCA(n_components=pca_components)
    transform = pca.fit_transform(data.to_numpy())
    idx = data.index.get_level_values('disorder').array.astype(bool)
    cmap = cmap3

    # Feature variance pie chart
    var = data.var().sort_values(ascending=False)
    truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
    plt.pie(truncate.values, labels=truncate.index, labeldistance=None)
    plt.title(f'Feature variance\n{title_label}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.65)
    plt.savefig(f'{prefix}/pie_variance_{data_label}_{file_label}.png')
    plt.close()

    # Scree plot
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, label=data_label, color=cmap(0.6))
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title(title_label)
    plt.legend()
    plt.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
    plt.close()

    # PCA scatters
    plot_pca2(transform, 0, 1, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
              f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png',
              hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor)
    plot_pca2_arrows(pca, transform, data.columns, 0, 1, idx, ~idx, cmap1, cmap2, title_label,
                     f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png',
                     hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)

    plot_pca2(transform, 1, 2, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
              f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}.png',
              hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor)
    plot_pca2_arrows(pca, transform, data.columns, 1, 2, idx, ~idx, cmap1, cmap2, title_label,
                     f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png',
                     hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)

# 3.2.2 Plot rate PCAs (individual)
plots = [(disorder, 'disorder', 'no norm', 'nonorm_all'),
         (order, 'order', 'no norm', 'nonorm_all'),
         (zscore(disorder), 'disorder', 'z-score', 'zscore_all'),
         (zscore(order), 'order', 'z-score', 'zscore_all'),
         (disorder_motifs, 'disorder', 'no norm', 'nonorm_motifs'),
         (order_motifs, 'order', 'no norm', 'nonorm_motifs'),
         (zscore(disorder_motifs), 'disorder', 'z-score', 'zscore_motifs'),
         (zscore(order_motifs), 'order', 'z-score', 'zscore_motifs')]
for data, data_label, title_label, file_label in plots:
    pca = PCA(n_components=pca_components)
    transform = pca.fit_transform(data.to_numpy())
    cmap = cmap1 if data_label == 'disorder' else cmap2

    # Feature variance pie chart
    var = data.var().sort_values(ascending=False)
    truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
    plt.pie(truncate.values, labels=truncate.index, labeldistance=None)
    plt.title(f'Feature variance\n{title_label}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.65)
    plt.savefig(f'{prefix}/pie_variance_{data_label}_{file_label}.png')
    plt.close()

    # Scree plot
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, label=data_label, color=cmap(0.6))
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title(title_label)
    plt.legend()
    plt.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
    plt.close()

    # PCA scatters
    plot_pca(transform, 0, 1, cmap, data_label, title_label,
             f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png',
             hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor)
    plot_pca_arrows(pca, transform, data.columns, 0, 1, cmap, title_label,
                    f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png',
                    hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)

    plot_pca(transform, 1, 2, cmap, data_label, title_label,
             f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}.png',
             hexbin_kwargs=hexbin_kwargs_log, handle_markerfacecolor=handle_markerfacecolor)
    plot_pca_arrows(pca, transform, data.columns, 1, 2, cmap, title_label,
                    f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png',
                    hexbin_kwargs=hexbin_kwargs_log, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)

# 4 ROOTS
if not os.path.exists('out/roots/'):
    os.makedirs('out/roots/')
prefix = 'out/roots/'

# 4.1 Plot root distributions
roots_motifs = roots.drop(motif_labels, axis=1)
disorder = roots.loc[pdidx[:, :, :, True, :], :]
order = roots.loc[pdidx[:, :, :, False, :], :]
disorder_motifs = disorder.drop(motif_labels, axis=1)
order_motifs = order.drop(motif_labels, axis=1)
for feature_label in feature_labels:
    fig, axs = plt.subplots(2, 1, sharex=True)
    xmin, xmax = roots[feature_label].min(), roots[feature_label].max()
    axs[0].hist(disorder[feature_label], bins=linspace(xmin, xmax, 75), color='C0', label='disorder')
    axs[1].hist(order[feature_label], bins=linspace(xmin, xmax, 75), color='C1', label='order')
    axs[1].set_xlabel(f'Inferred root value ({feature_label})')
    for i in range(2):
        axs[i].set_ylabel('Number of regions')
        axs[i].legend()
    plt.savefig(f'{prefix}/hist_numregions-{feature_label}.png')
    plt.close()

# 4.2.1 Plot root PCAs (combined)
plots = [(roots, 'merge', 'no norm', 'nonorm_all'),
         (zscore(roots), 'merge', 'z-score', 'zscore_all'),
         (roots_motifs, 'merge', 'no norm', 'nonorm_motifs'),
         (zscore(roots_motifs), 'merge', 'z-score', 'zscore_motifs')]
for data, data_label, title_label, file_label in plots:
    pca = PCA(n_components=pca_components)
    transform = pca.fit_transform(data.to_numpy())
    idx = data.index.get_level_values('disorder').array.astype(bool)
    cmap = cmap3

    # Feature variance pie chart
    var = data.var().sort_values(ascending=False)
    truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
    plt.pie(truncate.values, labels=truncate.index, labeldistance=None)
    plt.title(f'Feature variance\n{title_label}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.65)
    plt.savefig(f'{prefix}/pie_variance_{data_label}_{file_label}.png')
    plt.close()

    # Scree plot
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, label=data_label, color=cmap(0.6))
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title(title_label)
    plt.legend()
    plt.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
    plt.close()

    # PCA scatters
    plot_pca2(transform, 0, 1, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
              f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png',
              hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor)
    plot_pca2_arrows(pca, transform, data.columns, 0, 1, idx, ~idx, cmap1, cmap2, title_label,
                     f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png',
                     hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)

# 4.2.2 Plot root PCAs (individual)
plots = [(disorder, 'disorder', 'no norm', 'nonorm_all'),
         (order, 'order', 'no norm', 'nonorm_all'),
         (zscore(disorder), 'disorder', 'z-score', 'zscore_all'),
         (zscore(order), 'order', 'z-score', 'zscore_all'),
         (disorder_motifs, 'disorder', 'no norm', 'nonorm_motifs'),
         (order_motifs, 'order', 'no norm', 'nonorm_motifs'),
         (zscore(disorder_motifs), 'disorder', 'z-score', 'zscore_motifs'),
         (zscore(order_motifs), 'order', 'z-score', 'zscore_motifs')]
for data, data_label, title_label, file_label in plots:
    pca = PCA(n_components=pca_components)
    transform = pca.fit_transform(data.to_numpy())
    cmap = cmap1 if data_label == 'disorder' else cmap2

    # Feature variance pie chart
    var = data.var().sort_values(ascending=False)
    truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
    plt.pie(truncate.values, labels=truncate.index, labeldistance=None)
    plt.title(f'Feature variance\n{title_label}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.65)
    plt.savefig(f'{prefix}/pie_variance_{data_label}_{file_label}.png')
    plt.close()

    # Scree plot
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, label=data_label, color=cmap(0.6))
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title(title_label)
    plt.legend()
    plt.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
    plt.close()

    # PCA scatters
    plot_pca(transform, 0, 1, cmap, data_label, title_label,
             f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png',
             hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor)
    plot_pca_arrows(pca, transform, data.columns, 0, 1, cmap, title_label,
                    f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png',
                    hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors)

# 5 MERGE
if not os.path.exists('out/merge/'):
    os.makedirs('out/merge/')
prefix = 'out/merge/'

# 5.1 Plot correlations of roots and feature means
merge = features.groupby(['OGid', 'start', 'stop', 'disorder']).mean().merge(roots, how='inner', on=['OGid', 'start', 'stop', 'disorder'])
for feature_label in feature_labels:
    plt.hexbin(merge[feature_label + '_x'], merge[feature_label + '_y'], gridsize=75, linewidth=0, mincnt=1)
    plt.xlabel('Tip mean')
    plt.ylabel('Inferred root value')
    plt.title(feature_label)
    plt.colorbar()

    x, y = merge[feature_label + '_x'], merge[feature_label + '_y']
    m = ((x - x.mean())*(y - y.mean())).sum() / ((x - x.mean())**2).sum()
    b = y.mean() - m * x.mean()
    r2 = 1 - ((y - m * x - b)**2).sum() / ((y - y.mean())**2).sum()

    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [m*xmin+b, m*xmax+b], color='black', linewidth=1)
    plt.annotate(r'$\mathregular{R^2}$' + f' = {round(r2, 2)}', (0.85, 0.65), xycoords='axes fraction')
    plt.savefig(f'{prefix}/hexbin_root-mean_{feature_label}.png')
    plt.close()

# 5.2 Plot correlation of roots and rates
motif_labels_merge = [f'{motif_label}_root' for motif_label in motif_labels] + [f'{motif_label}_rate' for motif_label in motif_labels]
merge = roots.merge(rates, how='inner', on=['OGid', 'start', 'stop', 'disorder'], suffixes=('_root', '_rate'))
merge_motifs = merge.drop(motif_labels_merge, axis=1)
disorder = merge.loc[pdidx[:, :, :, True, :], :]
order = merge.loc[pdidx[:, :, :, False, :], :]
disorder_motifs = disorder.drop(motif_labels_merge, axis=1)
order_motifs = order.drop(motif_labels_merge, axis=1)
for feature_label in feature_labels:
    plt.hexbin(merge[feature_label + '_root'], merge[feature_label + '_rate'],
               cmap=cmap3, gridsize=75, linewidth=0, mincnt=1)
    plt.xlabel('Inferred root value')
    plt.ylabel('Rate')
    plt.title(feature_label)
    plt.colorbar()
    plt.savefig(f'{prefix}/hexbin_rate-root_{feature_label}1.png')
    plt.close()

    xmin, xmax = merge[feature_label + '_root'].min(), merge[feature_label + '_root'].max()
    ymin, ymax = merge[feature_label + '_rate'].min(), merge[feature_label + '_rate'].max()
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)
    for ax, data, label, cmap in zip(axs, [disorder, order], ['disorder', 'order'], [cmap1, cmap2]):
        hb = ax.hexbin(data[feature_label + '_root'], data[feature_label + '_rate'],
                       cmap=cmap, gridsize=50, linewidth=0, mincnt=1, extent=(xmin, xmax, ymin, ymax))
        ax.set_ylabel('Rate')
        handles = [Line2D([], [], label=label, marker='h', markerfacecolor=cmap(0.6),
                          markeredgecolor='None', markersize=8, linestyle='None')]
        ax.legend(handles=handles)
        fig.colorbar(hb, ax=ax)
    axs[1].set_xlabel('Inferred root value')
    fig.suptitle(feature_label)
    plt.savefig(f'{prefix}/hexbin_rate-root_{feature_label}2.png')
    plt.close()

# 5.3.1 Plot root-rate PCAs (combined)
plots = [(merge, 'merge', 'no norm', 'nonorm_all'),
         (zscore(merge), 'merge', 'z-score', 'zscore_all'),
         (merge_motifs, 'merge', 'no norm', 'nonorm_motifs'),
         (zscore(merge_motifs), 'merge', 'z-score', 'zscore_motifs')]
for data, data_label, title_label, file_label in plots:
    pca = PCA(n_components=pca_components)
    transform = pca.fit_transform(data.to_numpy())
    idx = data.index.get_level_values('disorder').array.astype(bool)
    cmap = cmap3
    width_ratios = (0.76, 0.03, 0.03, 0.15, 0.03)

    # Feature variance pie chart
    var = data.var().sort_values(ascending=False)
    truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
    plt.pie(truncate.values, labels=truncate.index, labeldistance=None)
    plt.title(f'Feature variance\n{title_label}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.65)
    plt.savefig(f'{prefix}/pie_variance_{data_label}_{file_label}.png')
    plt.close()

    # Scree plot
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, label=data_label, color=cmap(0.6))
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title(title_label)
    plt.legend()
    plt.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
    plt.close()

    # PCA scatters
    plot_pca2(transform, 0, 1, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
              f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png',
              hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor,
              width_ratios=width_ratios)
    plot_pca2_arrows(pca, transform, data.columns, 0, 1, idx, ~idx, cmap1, cmap2, title_label,
                     f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png',
                     hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                     width_ratios=width_ratios)

    plot_pca2(transform, 1, 2, idx, ~idx, cmap1, cmap2, 'disorder', 'order', title_label,
              f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}.png',
              hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor,
              width_ratios=width_ratios)
    plot_pca2_arrows(pca, transform, data.columns, 1, 2, idx, ~idx, cmap1, cmap2, title_label,
                     f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png',
                     hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                     width_ratios=width_ratios)

# 5.3.2 Plot root-rate PCAs (individual)
plots = [(disorder, 'disorder', 'no norm', 'nonorm_all'),
         (order, 'order', 'no norm', 'nonorm_all'),
         (zscore(disorder), 'disorder', 'z-score', 'zscore_all'),
         (zscore(order), 'order', 'z-score', 'zscore_all'),
         (disorder_motifs, 'disorder', 'no norm', 'nonorm_motifs'),
         (order_motifs, 'order', 'no norm', 'nonorm_motifs'),
         (zscore(disorder_motifs), 'disorder', 'z-score', 'zscore_motifs'),
         (zscore(order_motifs), 'order', 'z-score', 'zscore_motifs')]
for data, data_label, title_label, file_label in plots:
    pca = PCA(n_components=pca_components)
    transform = pca.fit_transform(data.to_numpy())
    cmap = cmap1 if data_label == 'disorder' else cmap2
    width_ratios = (0.76, 0.03, 0.03, 0.18)

    # Feature variance pie chart
    var = data.var().sort_values(ascending=False)
    truncate = pd.concat([var[:9], pd.Series({'other': var[9:].sum()})])
    plt.pie(truncate.values, labels=truncate.index, labeldistance=None)
    plt.title(f'Feature variance\n{title_label}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.65)
    plt.savefig(f'{prefix}/pie_variance_{data_label}_{file_label}.png')
    plt.close()

    # Scree plot
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, label=data_label, color=cmap(0.6))
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title(title_label)
    plt.legend()
    plt.savefig(f'{prefix}/bar_scree_{data_label}_{file_label}.png')
    plt.close()

    # PCA scatters
    plot_pca(transform, 0, 1, cmap, data_label, title_label,
             f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}.png',
             hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor,
             width_ratios=width_ratios)
    plot_pca_arrows(pca, transform, data.columns, 0, 1, cmap, title_label,
                    f'{prefix}/hexbin_pc1-pc2_{data_label}_{file_label}_arrow.png',
                    hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                    width_ratios=width_ratios)

    plot_pca(transform, 1, 2, cmap, data_label, title_label,
             f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}.png',
             hexbin_kwargs=hexbin_kwargs, handle_markerfacecolor=handle_markerfacecolor,
             width_ratios=width_ratios)
    plot_pca_arrows(pca, transform, data.columns, 1, 2, cmap, title_label,
                    f'{prefix}/hexbin_pc2-pc3_{data_label}_{file_label}_arrow.png',
                    hexbin_kwargs=hexbin_kwargs, legend_kwargs=legend_kwargs, arrow_colors=arrow_colors,
                    width_ratios=width_ratios)
