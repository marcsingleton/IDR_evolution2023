"""Functions to make specialize PCA plots."""

from math import atan2, pi

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import ArrowStyle, FancyArrowPatch


def get_angle(y, x):
    angle = atan2(y, x)
    if angle < 0:
        angle = 2 * pi + angle
    return angle


def plot_hexbin2(x1, y1, x2, y2, cmap1, cmap2, hexbin_kwargs=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    if hexbin_kwargs is None:
        hexbin_kwargs = {}
    if 'mincnt' in hexbin_kwargs:  # Remove mincnt since changes number of hexes
        hexbin_kwargs = hexbin_kwargs.copy()  # Copy to not change
        mincnt = hexbin_kwargs['mincnt']
        del hexbin_kwargs['mincnt']
    else:
        mincnt = 1

    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    extent = (xmin, xmax, ymin, ymax)

    hb1 = ax.hexbin(x1, y1, extent=extent, cmap=cmap1, **hexbin_kwargs)
    hb2 = ax.hexbin(x2, y2, extent=extent, cmap=cmap2, **hexbin_kwargs)

    array1 = np.expand_dims(hb1.get_array().data, -1)
    array2 = np.expand_dims(hb2.get_array().data, -1)
    norm1 = hb1.norm
    norm2 = hb2.norm
    fc1 = np.array([cmap1(norm1(count)) for count in array1.squeeze()])
    fc2 = np.array([cmap2(norm2(count)) for count in array2.squeeze()])

    total = array1 + array2
    total[total == 0] = 1  # Prevent divide by zero error
    fc = (array1 * fc1 + array2 * fc2) / total
    fc = (total >= mincnt) * fc  # Remove hexes below mincnt
    ax.clear()

    z = ax.hexbin([], [], extent=extent, **hexbin_kwargs)
    z.set_array(None)
    z.set_facecolor(fc)

    return hb1, hb2


def plot_pca(transform, pcx, pcy,
             cmap, label,
             title_label, file_label,
             hexbin_kwargs=None, handle_marker='h',
             handle_markersize=8, handle_markerfacecolor=0.5,
             figsize=None, width_ratios=None):
    if width_ratios is None:
        width_ratios = (0.79, 0.03, 0.03, 0.15)
    if hexbin_kwargs is None:
        hexbin_kwargs = {}

    x, y = transform[:, pcx], transform[:, pcy]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 4, width_ratios=width_ratios, wspace=0)
    ax = fig.add_subplot(gs[:, 0])
    hb = ax.hexbin(x, y, cmap=cmap, **hexbin_kwargs)
    ax.set_xlabel(f'PC{pcx + 1}')
    ax.set_ylabel(f'PC{pcy+1}')
    ax.set_title(title_label)
    handles = [Line2D([], [], label=label, marker=handle_marker, markerfacecolor=cmap(handle_markerfacecolor),
                      markersize=handle_markersize, markeredgecolor='none', linestyle='none')]
    ax.legend(handles=handles)
    fig.colorbar(hb, cax=fig.add_subplot(gs[:, 2]))

    fig.savefig(file_label)
    plt.close()


def plot_pca_arrows(pca, transform, feature_labels, pcx, pcy,
                    cmap,
                    title_label, file_label,
                    hexbin_kwargs=None,
                    legend_linewidth=2, legend_kwargs=None,
                    arrow_colors=None, arrow_scale=0.9, arrowstyle_kwargs=None,
                    figsize=None, width_ratios=None):
    if width_ratios is None:
        width_ratios = (0.79, 0.03, 0.03, 0.15)
    if hexbin_kwargs is None:
        hexbin_kwargs = {}

    x, y = transform[:, pcx], transform[:, pcy]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 4, width_ratios=width_ratios, wspace=0)
    ax = fig.add_subplot(gs[:, 0])
    ax.hexbin(x, y, cmap=cmap, **hexbin_kwargs)
    ax.set_xlabel(f'PC{pcx+1}')
    ax.set_ylabel(f'PC{pcy+1}')
    ax.set_title(title_label)

    add_pca_arrows(ax, pca, feature_labels, pcx, pcy,
                   legend_linewidth=legend_linewidth, legend_kwargs=legend_kwargs,
                   arrow_colors=arrow_colors, arrow_scale=arrow_scale, arrowstyle_kwargs=arrowstyle_kwargs)

    fig.savefig(file_label)
    plt.close()


def plot_pca2(transform, pcx, pcy,
              idx1, idx2, cmap1, cmap2, label1, label2,
              title_label, file_label,
              hexbin_kwargs=None,
              handle_marker='h', handle_markersize=8, handle_markerfacecolor=0.5,
              figsize=None, width_ratios=None):
    if width_ratios is None:
        width_ratios = (0.79, 0.03, 0.03, 0.12, 0.03)
    if hexbin_kwargs is None:
        hexbin_kwargs = {}

    x1, y1 = transform[idx1, pcx], transform[idx1, pcy]
    x2, y2 = transform[idx2, pcx], transform[idx2, pcy]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 5, width_ratios=width_ratios, wspace=0)
    ax = fig.add_subplot(gs[:, 0])
    hb1, hb2 = plot_hexbin2(x1, y1, x2, y2, cmap1=cmap1, cmap2=cmap2, hexbin_kwargs=hexbin_kwargs, ax=ax)
    ax.set_xlabel(f'PC{pcx+1}')
    ax.set_ylabel(f'PC{pcy+1}')
    ax.set_title(title_label)
    handles = [Line2D([], [], label=label1, marker=handle_marker, markerfacecolor=cmap1(handle_markerfacecolor),
                      markersize=handle_markersize, markeredgecolor='none', linestyle='none'),
               Line2D([], [], label=label2, marker=handle_marker, markerfacecolor=cmap2(handle_markerfacecolor),
                      markersize=handle_markersize, markeredgecolor='none', linestyle='none')]
    ax.legend(handles=handles)
    fig.colorbar(hb1, cax=fig.add_subplot(gs[:, 2]))
    fig.colorbar(hb2, cax=fig.add_subplot(gs[:, 4]))

    fig.savefig(file_label)
    plt.close()


def plot_pca2_arrows(pca, transform, feature_labels, pcx, pcy,
                     idx1, idx2, cmap1, cmap2,
                     title_label, file_label,
                     hexbin_kwargs=None,
                     legend_linewidth=2, legend_kwargs=None,
                     arrow_colors=None, arrow_scale=0.9, arrowstyle_kwargs=None,
                     figsize=None, width_ratios=None):
    if width_ratios is None:
        width_ratios = (0.79, 0.03, 0.03, 0.12, 0.03)
    if hexbin_kwargs is None:
        hexbin_kwargs = {}

    x1, y1 = transform[idx1, pcx], transform[idx1, pcy]
    x2, y2 = transform[idx2, pcx], transform[idx2, pcy]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 5, width_ratios=width_ratios, wspace=0)
    ax = fig.add_subplot(gs[:, 0])
    plot_hexbin2(x1, y1, x2, y2, cmap1=cmap1, cmap2=cmap2, hexbin_kwargs=hexbin_kwargs, ax=ax)
    ax.set_xlabel(f'PC{pcx+1}')
    ax.set_ylabel(f'PC{pcy+1}')
    ax.set_title(title_label)

    add_pca_arrows(ax, pca, feature_labels, pcx, pcy,
                   legend_linewidth=legend_linewidth, legend_kwargs=legend_kwargs,
                   arrow_colors=arrow_colors, arrow_scale=arrow_scale, arrowstyle_kwargs=arrowstyle_kwargs)

    fig.savefig(file_label)
    plt.close()


def add_pca_arrows(ax, pca, feature_labels, pcx, pcy,
                   legend_linewidth=2, legend_kwargs=None,
                   arrow_colors=None, arrow_scale=0.9, arrowstyle_kwargs=None):
    if legend_kwargs is None:
        legend_kwargs = {}
    if arrowstyle_kwargs is None:
        arrowstyle_kwargs = ArrowStyle('simple', head_length=8, head_width=8, tail_width=2)
    if arrow_colors is None:
        arrow_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    projections = zip(feature_labels, pca.components_[pcx], pca.components_[pcy])  # Match features to components in PC space
    projections = sorted(projections, key=lambda x: x[1] ** 2 + x[2] ** 2, reverse=True)[:len(arrow_colors)]  # Get features with largest magnitude
    projections = sorted(projections, key=lambda x: get_angle(x[2], x[1]))  # Re-order by angle from x-axis

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    ratios = []
    for projection in projections:
        _, x, y = projection
        ratios.extend([x / xmin, x / xmax, y / ymin, y / ymax])
    scale = arrow_scale / max(ratios)  # Scale the largest arrow within fraction of axes

    handles = []
    for color, projection in zip(arrow_colors, projections):
        feature_label, x, y = projection
        handles.append(Line2D([], [], color=color, linewidth=legend_linewidth, label=feature_label))
        arrow = FancyArrowPatch((0, 0), (scale*x, scale*y), facecolor=color, edgecolor='none',
                                arrowstyle=arrowstyle_kwargs)
        ax.add_patch(arrow)
    ax.legend(handles=handles, **legend_kwargs)
