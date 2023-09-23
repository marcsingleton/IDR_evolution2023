"""Plot statistics associated with BM and OU simulations."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from numpy import linspace
from src.brownian.simulate.sampling import sigma2_min, sigma2_max, sigma2_delta, sigma2_num
from src.brownian.simulate.sampling import alpha_min, alpha_max, alpha_delta, alpha_num

df_BM = pd.read_table('../simulate_compute/out/models_BM.tsv')
df_OU = pd.read_table('../simulate_compute/out/models_OU.tsv')
dfs = [df_BM, df_OU]

for df in dfs:
    df['AIC_BM'] = 2*(2 - df['loglikelihood_hat_BM'])
    df['AIC_OU'] = 2*(3 - df['loglikelihood_hat_OU'])
    df['delta_AIC'] = df['AIC_BM'] - df['AIC_OU']

groups_BM = df_BM.groupby('sigma2_id')
groups_OU = df_OU.groupby(['sigma2_id', 'alpha_id'])

# BM MODEL PLOTS
if not os.path.exists('out/BM/'):
    os.makedirs('out/BM/')
prefix = 'out/BM/'

# Violin plots of sigma2_BM
dataset = [group['sigma2_hat_BM'].to_list() for _, group in groups_BM]
positions = groups_BM['sigma2'].mean().apply(np.log10).to_list()

fig, ax = plt.subplots()
ax.violinplot(dataset, positions, widths=sigma2_delta/2, showmedians=True)
ax.set_xlabel('true $\mathregular{\log_{10}(\sigma^2_{BM})}$')
ax.set_ylabel('estimated $\mathregular{\sigma^2_{BM}}$')
ax.set_yscale('log')
fig.savefig(f'{prefix}/violin_sigma2_hat_BM-sigma2.png')
plt.close()

# Scatters with errors of sigma2_BM
xs = groups_BM['sigma2'].mean()
ys = groups_BM['sigma2_hat_BM'].mean()
yerrs = groups_BM['sigma2_hat_BM'].std()

fig, ax = plt.subplots()
ax.errorbar(xs, ys, yerr=yerrs, fmt='.')
ax.plot(xs, xs, color='black', linewidth=0.5)
ax.set_xlabel('true $\mathregular{\sigma^2_{BM}}$')
ax.set_ylabel('estimated $\mathregular{\sigma^2_{BM}}$')
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig(f'{prefix}/scatter_sigma2_BM_hat-sigma2.png')
plt.close()

# Violin plots of alpha
dataset = [group['alpha_hat_OU'].to_list() for _, group in groups_BM]
positions = groups_BM['sigma2'].mean().apply(np.log10).to_list()

fig, ax = plt.subplots()
ax.violinplot(dataset, positions, widths=sigma2_delta/2, showmedians=True)
ax.set_xlabel('true $\mathregular{\log_{10}(\sigma^2_{BM})}$')
ax.set_ylabel('estimated $\mathregular{\\alpha}$')
fig.savefig(f'{prefix}/violin_alpha_hat_OU-sigma2.png')
plt.close()

# Scatters with errors of alpha
xs = groups_BM['sigma2'].mean()
ys = groups_BM['alpha_hat_OU'].mean()
yerrs = groups_BM['alpha_hat_OU'].std()

fig, ax = plt.subplots()
ax.errorbar(xs, ys, yerr=yerrs, fmt='o')
ax.set_xlabel('true $\mathregular{\sigma^2_{BM}}$')
ax.set_ylabel('estimated $\mathregular{\\alpha}$')
ax.set_xscale('log')
fig.savefig(f'{prefix}/scatter_alpha_hat_OU-sigma2.png')
plt.close()

# Violin plots of sigma2_OU
dataset = [group['sigma2_hat_OU'].to_list() for _, group in groups_BM]
positions = groups_BM['sigma2'].mean().apply(np.log10).to_list()

fig, ax = plt.subplots()
ax.violinplot(dataset, positions, widths=sigma2_delta/2, showmedians=True)
ax.set_yscale('log')
ax.set_xlabel('true $\mathregular{\log_{10}(\sigma^2_{BM})}$')
ax.set_ylabel('estimated $\mathregular{\sigma^2_{OU}}$')
fig.savefig(f'{prefix}/violin_sigma2_hat_OU-sigma2.png')
plt.close()

# Scatters with errors of sigma2_OU
xs = groups_BM['sigma2'].mean()
ys = groups_BM['sigma2_hat_OU'].mean()
yerrs = groups_BM['sigma2_hat_OU'].std()

fig, ax = plt.subplots()
ax.errorbar(xs, ys, yerr=yerrs, fmt='.')
ax.plot(xs, xs, color='black', linewidth=0.5)
ax.set_xlabel('true $\mathregular{\sigma^2_{BM}}$')
ax.set_ylabel('estimated $\mathregular{\sigma^2_{OU}}$')
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig(f'{prefix}/scatter_sigma2_hat_OU-sigma2.png')
plt.close()

# Violin plots of delta AIC
dataset = [group['delta_AIC'].to_list() for _, group in groups_BM]
positions = groups_BM['sigma2'].mean().apply(np.log10).to_list()

fig, ax = plt.subplots()
ax.violinplot(dataset, positions, widths=sigma2_delta/2, showmedians=True)
ax.set_xlabel('true $\mathregular{\log_{10}(\sigma^2_{BM})}$')
ax.set_ylabel('$\mathregular{AIC_{BM} - AIC_{OU}}$')
fig.savefig(f'{prefix}/violin_delta_AIC.png')
plt.close()

# Type I error as function of cutoff
delta_AIC_min, delta_AIC_max = df_BM['delta_AIC'].min(), df_BM['delta_AIC'].max()
cutoffs = linspace(delta_AIC_min, delta_AIC_max, 50)

errors = []
for cutoff in cutoffs:
    errors.append(groups_BM['delta_AIC'].aggregate(lambda x: (x > cutoff).mean()))
errors = pd.DataFrame(errors).reset_index(drop=True)

fig, ax = plt.subplots()
id2value = groups_BM['sigma2'].mean().apply(np.log10).to_dict()
cmap = ListedColormap(plt.colormaps['viridis'].colors[:240])
norm = Normalize(sigma2_min, sigma2_max)
get_color = lambda x: cmap(norm(x))
for sigma2_id in errors:
    ax.plot(cutoffs, errors[sigma2_id], color=get_color(id2value[sigma2_id]), alpha=0.75)
ax.set_xlabel('$\mathregular{AIC_{BM}-AIC_{OU}}$ cutoff')
ax.set_ylabel('Type I error')
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), label='true $\mathregular{\log_{10}(\sigma^2_{BM})}$')
fig.savefig(f'{prefix}/line_typeI-sigma2.png')
plt.close()

errors = []
for cutoff in cutoffs:
    errors.append((df_BM['delta_AIC'] > cutoff).mean())
q95 = df_BM['delta_AIC'].quantile(0.95)
q99 = df_BM['delta_AIC'].quantile(0.99)

fig, ax = plt.subplots(gridspec_kw={'right': 0.745})  # To match dimensions w/ colorbar; unsure why exactly is 0.745 (0.9 - 15% for colorbar is 0.765)
ax.plot(cutoffs, errors)
ax.axvline(q95, color='C1', label='5% type I error')
ax.axvline(q99, color='C2', label='1% type I error')
ax.set_xlabel('$\mathregular{AIC_{BM}-AIC_{OU}}$ cutoff')
ax.set_ylabel('Type I error')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig(f'{prefix}/line_typeI-sigma2_merge.png')
plt.close()

# OU MODEL PLOTS
if not os.path.exists('out/OU/'):
    os.makedirs('out/OU/')
prefix = 'out/OU/'

df_parameter = groups_OU[['sigma2_hat_OU', 'alpha_hat_OU']].mean().sort_index(level=['sigma2_id', 'alpha_id'])
df_error = groups_OU['delta_AIC'].aggregate(lambda x: (x < q99).mean()).sort_index(level=['sigma2_id', 'alpha_id'])
extent = (alpha_min-alpha_delta/2, alpha_max+alpha_delta/2,
          sigma2_min-sigma2_delta/2, sigma2_max+sigma2_delta/2)

# Heatmap of sigma2
fig, ax = plt.subplots()
array = df_parameter['sigma2_hat_OU'].to_numpy().reshape((sigma2_num, alpha_num))
im = ax.imshow(np.log10(array), extent=extent, origin='lower', aspect='auto')
ax.set_xlabel('true $\mathregular{\log_{10}(\\alpha)}$')
ax.set_ylabel('true $\mathregular{\log_{10}(\sigma^2_{OU})}$')
ax.set_title('estimated $\mathregular{\log_{10}(\sigma^2_{OU})}$')
fig.colorbar(im)
fig.savefig(f'{prefix}/heatmap_sigma2_hat_OU.png')
plt.close()

# Heatmap of alpha
fig, ax = plt.subplots()
array = df_parameter['alpha_hat_OU'].to_numpy().reshape((sigma2_num, alpha_num))
im = ax.imshow(np.log10(array), extent=extent, origin='lower', aspect='auto')
ax.set_xlabel('true $\mathregular{\log_{10}(\\alpha)}$')
ax.set_ylabel('true $\mathregular{\log_{10}(\sigma^2_{OU})}$')
ax.set_title('estimated $\mathregular{\log_{10}(\\alpha)}$')
fig.colorbar(im)
fig.savefig(f'{prefix}/heatmap_alpha_hat_OU.png')
plt.close()

# Heatmap of type II errors
fig, ax = plt.subplots()
array = df_error.to_numpy().reshape((sigma2_num, alpha_num))
im = ax.imshow(array, extent=extent, origin='lower', aspect='auto')
ax.set_xlabel('true $\mathregular{\log_{10}(\\alpha)}$')
ax.set_ylabel('true $\mathregular{\log_{10}(\sigma^2_{OU})}$')
ax.set_title('Type II error at 1% type I error')
fig.colorbar(im)
fig.savefig(f'{prefix}/heatmap_typeII.png')
plt.close()
