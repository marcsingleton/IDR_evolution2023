"""Plot outputs of fitting GO term logit models."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

pdidx = pd.IndexSlice

color1, color2, color3 = '#4e79a7', '#f28e2b', '#b07aa1'

models = pd.read_table('out/models.tsv').set_index(['GOid', 'label'])
disorder = models.loc[pdidx[:, 'disorder'], :]
order = models.loc[pdidx[:, 'order'], :]
all = models.loc[pdidx[:, 'all'], :]

if not os.path.exists('out/'):
    os.mkdir('out/')

plots = [(disorder, 'disorder', color1),
         (order, 'order', color2),
         (all, 'all', color3)]
for data, data_label, color in plots:
    for feature_label in models.columns:
        fig, ax = plt.subplots()
        ax.hist(data[feature_label], bins=50, label=data_label, color=color)
        ax.set_xlabel(feature_label.capitalize())
        ax.set_ylabel('Number of models')
        ax.legend()
        fig.savefig(f'out/hist_modelnum-{feature_label}_{data_label}.png')
        plt.close()

    fig, ax = plt.subplots()
    ax.scatter(data['sensitivity'], data['specificity'],
               label=data_label, facecolor=color, edgecolor='none', alpha=0.25, s=12)
    ax.set_xlabel('Sensitivity')
    ax.set_ylabel('Specificity')
    ax.legend(handles=[Line2D([], [], label=data_label, marker='.', markerfacecolor=color, markeredgecolor='none', linestyle='none')])
    fig.savefig(f'out/scatter_spec-sens_{data_label}.png')
    plt.close()
