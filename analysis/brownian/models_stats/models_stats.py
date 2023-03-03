"""Plot statistics from fit evolutionary models."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

pdidx = pd.IndexSlice
pca_components = 10

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 1
min_indel_rate = 1

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

column_labels = [column_label for column_label in df.columns if column_label.endswith('sigma2_ratio')]
data = df.loc[pdidx[:, :, :, True], column_labels]
array = np.nan_to_num(data.to_numpy(), nan=1)
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
