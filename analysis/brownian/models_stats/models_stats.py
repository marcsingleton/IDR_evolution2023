"""Plot statistics from fit evolutionary models."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.brownian.features import motif_regexes
from src.brownian.pca_plots import plot_pca, plot_pca_arrows, plot_pca2, plot_pca2_arrows

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

models = pd.read_table('../get_models/out/models_30.tsv')
rates = pd.read_table('../../evosim/asr_stats/out/regions_30/rates.tsv')
features = pd.read_table('../get_features/out/features.tsv').groupby(['OGid', 'start', 'stop']).mean()

df = regions.merge(models, how='left', on=['OGid', 'start', 'stop'])
df = df.merge(rates, how='left', on=['OGid', 'start', 'stop'])
df = df.merge(features, how='left', on=['OGid', 'start', 'stop']).set_index(['OGid', 'start', 'stop', 'disorder'])

# Data filtering
df.loc[(df['indel_num_columns'] < min_indel_columns) | df['indel_rate_mean'].isna(), 'indel_rate_mean'] = 0
df = df[(df['aa_rate_mean'] > min_aa_rate) | (df['indel_rate_mean'] > min_indel_rate)]

feature_labels = []
for column_label in df.columns:
    if column_label.endswith('_loglikelihood_BM'):
        feature_labels.append(column_label.removesuffix('_loglikelihood_BM'))
motif_labels = list(motif_regexes)

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
column_labels_motifs = [f'{feature_label}_sigma2_ratio' for feature_label in (set(feature_labels) - set(motif_labels))]
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
