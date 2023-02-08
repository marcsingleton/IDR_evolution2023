"""Plot statistics from fit evolutionary models."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

pdidx = pd.IndexSlice
pca_components = 10

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
models = models.merge(regions, how='right', on=['OGid', 'start', 'stop']).set_index(['OGid', 'start', 'stop', 'disorder'])

feature_labels = []
for column_label in models.columns:
    if column_label.endswith('_loglikelihood_BM'):
        feature_labels.append(column_label.removesuffix('_loglikelihood_BM'))

columns = {}
for feature_label in feature_labels:
    columns[f'{feature_label}_AIC_BM'] = 2*(2 - models[f'{feature_label}_loglikelihood_BM'])
    columns[f'{feature_label}_AIC_OU'] = 2*(3 - models[f'{feature_label}_loglikelihood_OU'])
models = pd.concat([models, pd.DataFrame(columns)], axis=1)

if not os.path.exists('out/'):
    os.mkdir('out/')

for feature_label in feature_labels:
    for param in ['mu', 'sigma2', 'loglikelihood', 'AIC']:
        xmin = min(models[f'{feature_label}_{param}_BM'].min(), models[f'{feature_label}_{param}_OU'].min())
        xmax = max(models[f'{feature_label}_{param}_BM'].max(), models[f'{feature_label}_{param}_OU'].max())
        bins = np.linspace(xmin, xmax, 50)
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].hist(models[f'{feature_label}_{param}_BM'], bins=bins, color='C2', label='BM')
        axs[1].hist(models[f'{feature_label}_{param}_OU'], bins=bins, color='C3', label='OU')
        axs[1].set_xlabel(f'{param} ({feature_label})')
        for ax in axs:
            ax.legend()
            ax.set_ylabel('Number of regions')
        fig.savefig(f'out/hist_regionnum-{param}_{feature_label}.png')
        plt.close()

        fig, ax = plt.subplots()
        hb = ax.hexbin(models[f'{feature_label}_{param}_BM'], models[f'{feature_label}_{param}_OU'], gridsize=50, mincnt=1, linewidth=0)
        ax.set_xlabel(f'BM {param}')
        ax.set_ylabel(f'OU {param}')
        ax.set_title(feature_label)
        fig.colorbar(hb)
        fig.savefig(f'out/hexbin_OU-BM_{param}_{feature_label}.png')
        plt.close()

    fig, ax = plt.subplots()
    ax.hist(models[f'{feature_label}_AIC_BM'] - models[f'{feature_label}_AIC_OU'], bins=50)
    ax.set_xlabel('$\mathregular{AIC_{BM} - AIC_{OU}}$' + f' ({feature_label})')
    ax.set_ylabel('Number of regions')
    fig.savefig(f'out/hist_regionnum-delta_AIC_{feature_label}.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(models[f'{feature_label}_sigma2_BM'] / models[f'{feature_label}_sigma2_OU'], bins=50)
    ax.set_xlabel('$\mathregular{\sigma_{BM}^2 / \sigma_{OU}^2}$' + f' ({feature_label})')
    ax.set_ylabel('Number of regions')
    fig.savefig(f'out/hist_regionnum-sigma2_{feature_label}.png')
    plt.close()

    fig, ax = plt.subplots()
    hb = ax.hexbin(models[f'{feature_label}_AIC_BM'] - models[f'{feature_label}_AIC_OU'],
                   models[f'{feature_label}_sigma2_BM'] / models[f'{feature_label}_sigma2_OU'],
                   gridsize=75, mincnt=1, linewidth=0, bins='log')
    ax.set_xlabel('$\mathregular{AIC_{BM} - AIC_{OU}}$')
    ax.set_ylabel('$\mathregular{\sigma_{BM}^2 / \sigma_{OU}^2}$')
    ax.set_title(feature_label)
    fig.colorbar(hb)
    fig.savefig(f'out/hexbin_sigma2-delta_AIC_{feature_label}.png')
    plt.close()

data = models.loc[pdidx[:, :, :, True], :]
array = np.zeros((len(data), len(feature_labels)))
for i, feature_label in enumerate(feature_labels):
    x = data[f'{feature_label}_sigma2_BM'] / data[f'{feature_label}_sigma2_OU']
    x[x.isna()] = 1
    array[:, i] = x
pca = PCA(n_components=pca_components)
transform = pca.fit_transform(array)

fig, ax = plt.subplots()
hb = ax.hexbin(transform[:, 0], transform[:, 1], gridsize=75, mincnt=1, linewidth=0, bins='log')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
fig.colorbar(hb)
fig.savefig('out/hexbin_pc1-pc2.png')
plt.close()

fig, ax = plt.subplots()
hb = ax.hexbin(transform[:, 1], transform[:, 2], gridsize=75, mincnt=1, linewidth=0, bins='log')
ax.set_xlabel('PC2')
ax.set_ylabel('PC3')
fig.colorbar(hb)
fig.savefig('out/hexbin_pc2-pc3.png')
plt.close()
