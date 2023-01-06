"""Plot statistics associated with contrasts."""

import os
import re
from math import exp, pi

import matplotlib.pyplot as plt
import pandas as pd
import skbio
from matplotlib.lines import Line2D
from numpy import linspace, quantile
from src.draw import plot_msa
from sklearn.decomposition import PCA
from src.brownian.features import motif_regexes
from src.utils import read_fasta


def zscore(df):
    return (df - df.mean()) / df.std()


pdidx = pd.IndexSlice
ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'

# Load contrasts and tree
features = pd.read_table('../get_features/out/features.tsv')
contrasts = pd.read_table('../get_contrasts/out/contrasts_100.tsv')
roots = pd.read_table('../get_contrasts/out/roots_100.tsv')
tree = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree.tips())}

features.loc[features['kappa'] == -1, 'kappa'] = 1
features.loc[features['omega'] == -1, 'omega'] = 1
features['radius_gyration'] = features['length'] ** 0.6
features = features.drop('length', axis=1)

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
    plt.savefig(f'out/contrasts/hist_numcontrasts-{feature_label}.png')
    for i in range(2):
        axs[i].set_yscale('log')
    plt.savefig(f'out/contrasts/hist_numcontrasts-{feature_label}_log.png')
    plt.close()

# 1.2 Plot regions with extreme contrasts
df = contrasts.abs()
for feature_label in feature_labels:
    if not os.path.exists(f'out/contrasts/{feature_label}/'):
        os.makedirs(f'out/contrasts/{feature_label}/')

    regions = set()
    ranked = df[feature_label].sort_values(ascending=False).iteritems()  # Series have no itertuples method
    while len(regions) < 20:
        (OGid, start, stop, _, _), _ = next(ranked)
        if (OGid, start, stop) in regions:
            continue
        regions.add((OGid, start, stop))

        # Load MSA, filter seqs, and re-order
        msa1 = read_fasta(f'../../../data/alignments/fastas/{OGid}.afa')
        msa1 = {re.search(spid_regex, header).group(1): seq for header, seq in msa1}

        spids = region2spids[(OGid, start, stop)]
        msa2 = [msa1[spid][start:stop] for spid in sorted(spids, key=lambda x: tip_order[x])]
        fig = plot_msa(msa2, figsize=(8, 6), x_start=start)
        plt.savefig(f'out/contrasts/{feature_label}/{len(regions)-1}_{OGid}-{start}-{stop}.png', bbox_inches='tight')
        plt.close()

# 2 CONTRAST MEANS
if not os.path.exists('out/means/'):
    os.makedirs('out/means/')

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
    plt.savefig(f'out/means/hist_numregions-{feature_label}.png')
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
    plt.savefig(f'out/means/hist_numregions-{feature_label}_std.png')
    plt.close()

# 2.3 Plot contrast mean PCAs
pca = PCA(n_components=10)
idx = means.index.get_level_values('disorder').array.astype(bool)

transform = pca.fit_transform(means.to_numpy())
plt.scatter(transform[idx, 0], transform[idx, 1], label='disorder', s=5, color='C0', alpha=0.1, edgecolors='none')
plt.scatter(transform[~idx, 0], transform[~idx, 1], label='order', s=5, color='C1', alpha=0.1, edgecolors='none')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('raw means')
legend = plt.legend(markerscale=2)
for lh in legend.legendHandles:
    lh.set_alpha(1)
plt.savefig(f'out/means/scatter_pca_mean.png')
plt.close()

idx = stds.index.get_level_values('disorder').array.astype(bool)
transform = pca.fit_transform(stds.to_numpy())
plt.scatter(transform[idx, 0], transform[idx, 1], label='disorder', s=5, color='C0', alpha=0.1, edgecolors='none')
plt.scatter(transform[~idx, 0], transform[~idx, 1], label='order', s=5, color='C1', alpha=0.1, edgecolors='none')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('standardized means')
legend = plt.legend(markerscale=2)
for lh in legend.legendHandles:
    lh.set_alpha(1)
plt.savefig(f'out/means/scatter_pca_mean_std.png')
plt.close()

# 2.4 Plot regions with extreme means
df = means.abs()
for feature_label in feature_labels:
    if not os.path.exists(f'out/means/{feature_label}/'):
        os.makedirs(f'out/means/{feature_label}/')

    regions = set()
    ranked = df[feature_label].sort_values(ascending=False).iteritems()  # Series have no itertuples method
    while len(regions) < 20:
        (OGid, start, stop, _), _ = next(ranked)
        if (OGid, start, stop) in regions:
            continue
        regions.add((OGid, start, stop))

        # Load MSA, filter seqs, and re-order
        msa1 = read_fasta(f'../../../data/alignments/fastas/{OGid}.afa')
        msa1 = {re.search(spid_regex, header).group(1): seq for header, seq in msa1}

        spids = region2spids[(OGid, start, stop)]
        msa2 = [msa1[spid][start:stop] for spid in sorted(spids, key=lambda x: tip_order[x])]
        fig = plot_msa(msa2, figsize=(8, 6), x_start=start)
        plt.savefig(f'out/means/{feature_label}/{len(regions)-1}_{OGid}-{start}-{stop}.png', bbox_inches='tight')
        plt.close()

# 3 RATES
if not os.path.exists('out/rates/'):
    os.makedirs('out/rates/')

# 3.1 Plot rate distributions
rates = ((contrasts**2).groupby(['OGid', 'start', 'stop', 'disorder']).mean())
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
    plt.savefig(f'out/rates/hist_numregions-{feature_label}.png')
    for i in range(2):
        axs[i].set_yscale('log')
    plt.savefig(f'out/rates/hist_numregions-{feature_label}_log.png')
    plt.close()

# 3.2.1 Plot rate PCAs (combined)
pca = PCA(n_components=10)
colors = ['#e15759', '#499894', '#59a14f', '#f1ce63', '#b07aa1', '#d37295', '#9d7660', '#bab0ac',
          '#ff9d9a', '#86bcb6', '#8cd17d', '#b6992d', '#d4a6c8', '#fabfd2', '#d7b5a6', '#79706e']

plots = [(rates, 'no norm', 'nonorm_all'),
         (zscore(rates), 'z-score', 'zscore_all'),
         (rates.drop(motif_labels, axis=1), 'no norm', 'nonorm_motifs'),
         (zscore(rates.drop(motif_labels, axis=1)), 'z-score', 'z-score_motifs')]
for data, title_label, file_label in plots:
    idx1 = data.index.get_level_values('disorder').array.astype(bool)
    transform = pca.fit_transform(data.to_numpy())

    # PCs 1 and 2
    plt.scatter(transform[idx1, 0], transform[idx1, 1], label='disorder', s=5, color='C0', alpha=0.1, edgecolors='none')
    plt.scatter(transform[~idx1, 0], transform[~idx1, 1], label='order', s=5, color='C1', alpha=0.1, edgecolors='none')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title_label)
    legend = plt.legend(markerscale=2)
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    plt.savefig(f'out/rates/scatter_pca_{file_label}1.png')
    plt.close()

    # PCs 2 and 3
    plt.scatter(transform[idx1, 1], transform[idx1, 2], label='disorder', s=5, color='C0', alpha=0.1, edgecolors='none')
    plt.scatter(transform[~idx1, 1], transform[~idx1, 2], label='order', s=5, color='C1', alpha=0.1, edgecolors='none')
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.title(title_label)
    legend = plt.legend(markerscale=2)
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    plt.savefig(f'out/rates/scatter_pca_{file_label}2.png')
    plt.close()

    # PCs 2 and 3 with trimmed range
    r2 = transform[:, 1]**2 + transform[:, 2]**2  # radius**2 from center
    idx2 = r2 <= quantile(r2, 0.999, method='lower')  # Capture at least 99.9% of the data

    plt.scatter(transform[idx1 & idx2, 1], transform[idx1 & idx2, 2], label='disorder', s=5, color='C0', alpha=0.2, edgecolors='none')
    plt.scatter(transform[~idx1 & idx2, 1], transform[~idx1 & idx2, 2], label='order', s=5, color='C1', alpha=0.2, edgecolors='none')
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.title(title_label)
    legend = plt.legend(markerscale=2)
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    plt.savefig(f'out/rates/scatter_pca_{file_label}3.png')
    plt.close()

    # PCs 2 and 3 with trimmed range and arrows
    plt.scatter(transform[idx2, 1], transform[idx2, 2], label=title_label, color='C0', s=5, alpha=0.2, edgecolors='none')
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.title(title_label)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    scale = (xmax + ymax - xmin - ymin) / 2.5
    projections = sorted(zip(data.columns, pca.components_[1:3].transpose()), key=lambda x: x[1][0]**2 + x[1][1]**2, reverse=True)

    handles = []
    for i in range(len(colors)):
        feature_label, (x, y) = projections[i]
        handles.append(Line2D([], [], color=colors[i % len(colors)], linewidth=2, label=feature_label))
        plt.annotate('', xy=(scale*x, scale*y), xytext=(0, 0),
                     arrowprops={'headwidth': 6, 'headlength': 6, 'width': 1.75, 'color': colors[i % len(colors)]})
    plt.legend(handles=handles, fontsize=8, loc='right', bbox_to_anchor=(1.05, 0.5))
    plt.savefig(f'out/rates/scatter_pca_{file_label}3_arrow.png')
    plt.close()

    # Scree plot
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title(title_label)
    plt.savefig(f'out/rates/bar_scree_{file_label}.png')
    plt.close()

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
    color = 'C0' if data_label == 'disorder' else 'C1'
    transform = pca.fit_transform(data.to_numpy())

    # PCs 1 and 2
    plt.scatter(transform[:, 0], transform[:, 1], label=data_label, s=5, color=color, alpha=0.1, edgecolors='none')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title_label)
    legend = plt.legend(markerscale=2)
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    plt.savefig(f'out/rates/scatter_pca_{data_label}_{file_label}1.png')
    plt.close()

    # PCs 2 and 3
    plt.scatter(transform[:, 1], transform[:, 2], label=data_label, s=5, color=color, alpha=0.1, edgecolors='none')
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.title(title_label)
    legend = plt.legend(markerscale=2)
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    plt.savefig(f'out/rates/scatter_pca_{data_label}_{file_label}2.png')
    plt.close()

    # PCs 2 and 3 with trimmed range
    r2 = transform[:, 1]**2 + transform[:, 2]**2  # radius**2 from center
    idx = r2 <= quantile(r2, 0.999, method='lower')  # Capture at least 99.9% of the data

    plt.scatter(transform[idx, 1], transform[idx, 2], label=data_label, s=5, color=color, alpha=0.2, edgecolors='none')
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.title(title_label)
    legend = plt.legend(markerscale=2)
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    plt.savefig(f'out/rates/scatter_pca_{data_label}_{file_label}3.png')
    plt.close()

    # PCs 2 and 3 with trimmed range and arrows
    plt.scatter(transform[idx, 1], transform[idx, 2], label=data_label, color=color, s=5, alpha=0.2, edgecolors='none')
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.title(title_label)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    scale = (xmax + ymax - xmin - ymin) / 2.5
    projections = sorted(zip(data.columns, pca.components_[1:3].transpose()), key=lambda x: x[1][0]**2 + x[1][1]**2, reverse=True)

    handles = []
    for i in range(len(colors)):
        feature_label, (x, y) = projections[i]
        handles.append(Line2D([], [], color=colors[i % len(colors)], linewidth=2, label=feature_label))
        plt.annotate('', xy=(scale*x, scale*y), xytext=(0, 0),
                     arrowprops={'headwidth': 6, 'headlength': 6, 'width': 1.75, 'color': colors[i % len(colors)]})
    plt.legend(handles=handles, fontsize=8, loc='right', bbox_to_anchor=(1.05, 0.5))
    plt.savefig(f'out/rates/scatter_pca_{data_label}_{file_label}3_arrow.png')
    plt.close()

    # Scree plot
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, label=data_label, color=color)
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title(title_label)
    plt.legend()
    plt.savefig(f'out/rates/bar_scree_{data_label}_{file_label}.png')
    plt.close()

# 3.3 Plot regions with extreme rates
for feature_label in feature_labels:
    if not os.path.exists(f'out/rates/{feature_label}/'):
        os.makedirs(f'out/rates/{feature_label}/')

    regions = set()
    ranked = rates[feature_label].sort_values(ascending=False).iteritems()  # Series have no itertuples method
    while len(regions) < 20:
        (OGid, start, stop, _), _ = next(ranked)
        if (OGid, start, stop) in regions:
            continue
        regions.add((OGid, start, stop))

        # Load MSA, filter seqs, and re-order
        msa1 = read_fasta(f'../../../data/alignments/fastas/{OGid}.afa')
        msa1 = {re.search(spid_regex, header).group(1): seq for header, seq in msa1}

        spids = region2spids[(OGid, start, stop)]
        msa2 = [msa1[spid][start:stop] for spid in sorted(spids, key=lambda x: tip_order[x])]
        fig = plot_msa(msa2, figsize=(8, 6), x_start=start)
        plt.savefig(f'out/rates/{feature_label}/{len(regions)-1}_{OGid}-{start}-{stop}.png', bbox_inches='tight')
        plt.close()

# 4 ROOTS
if not os.path.exists('out/roots/'):
    os.makedirs('out/roots/')

# 4.1 Plot root distributions
disorder = roots.loc[pdidx[:, :, :, True, :], :]
order = roots.loc[pdidx[:, :, :, False, :], :]
for feature_label in feature_labels:
    fig, axs = plt.subplots(2, 1, sharex=True)
    xmin, xmax = roots[feature_label].min(), roots[feature_label].max()
    axs[0].hist(disorder[feature_label], bins=linspace(xmin, xmax, 75), color='C0', label='disorder')
    axs[1].hist(order[feature_label], bins=linspace(xmin, xmax, 75), color='C1', label='order')
    axs[1].set_xlabel(f'Inferred root value ({feature_label})')
    for i in range(2):
        axs[i].set_ylabel('Number of regions')
        axs[i].legend()
    plt.savefig(f'out/roots/hist_numregions-{feature_label}.png')
    plt.close()

# 4.2 Plot correlations of roots and feature means
df = features.groupby(['OGid', 'start', 'stop', 'disorder']).mean().merge(roots, how='inner', on=['OGid', 'start', 'stop', 'disorder'])
for feature_label in feature_labels:
    plt.scatter(df[feature_label + '_x'], df[feature_label + '_y'], s=5, alpha=0.1, edgecolor='none')
    plt.xlabel('Tip mean')
    plt.ylabel('Inferred root value')
    plt.title(feature_label)

    x, y = df[feature_label + '_x'], df[feature_label + '_y']
    m = ((x - x.mean())*(y - y.mean())).sum() / ((x - x.mean())**2).sum()
    b = y.mean() - m * x.mean()
    r2 = 1 - ((y - m * x - b)**2).sum() / ((y - y.mean())**2).sum()

    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [m*xmin+b, m*xmax+b], color='black', linewidth=1)
    plt.annotate(r'$\mathregular{R^2}$' + f' = {round(r2, 2)}', (0.85, 0.75), xycoords='axes fraction')
    plt.savefig(f'out/roots/scatter_root-mean_{feature_label}.png')
    plt.close()

# 4.3 Plot correlation of roots and rates
df = roots.merge(rates, how='inner', on=['OGid', 'start', 'stop', 'disorder'])
disorder = df.loc[pdidx[:, :, :, True, :], :]
order = df.loc[pdidx[:, :, :, False, :], :]
for feature_label in feature_labels:
    plt.scatter(df[feature_label + '_x'], df[feature_label + '_y'], s=5, alpha=0.15, edgecolor='none')
    plt.xlabel('Inferred root value')
    plt.ylabel('Rate')
    plt.title(feature_label)
    plt.savefig(f'out/roots/scatter_rate-root_{feature_label}1.png')
    plt.close()

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].scatter(disorder[feature_label + '_x'], disorder[feature_label + '_y'],
                   label='disorder', s=5, alpha=0.15, facecolor='C0', edgecolor='none')
    axs[1].scatter(order[feature_label + '_x'], order[feature_label + '_y'],
                   label='order', s=5, alpha=0.15, facecolor='C1', edgecolor='none')
    axs[1].set_xlabel('Inferred root value')
    for i in range(2):
        axs[i].set_ylabel('Rate')
        leg = axs[i].legend(markerscale=2)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
    fig.suptitle(feature_label)
    plt.savefig(f'out/roots/scatter_rate-root_{feature_label}2.png')
    plt.close()
