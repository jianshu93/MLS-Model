#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
Created on  May 22 2019
Last Update May 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

This recreates the data and figure for figure 4
By default data is loaded unless parameters have changes, to rerun model set override_data to True

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mls_general_code as mlsg
from pathlib import Path
import matplotlib.colors as colr
import seaborn as sns

"""
# SET model settings
"""

# set to True to force recalculation of data
override_data = False

# set folder
data_folder = Path("Data_Paper/")
fig_Folder = Path("Figures_Paper/")
figureName = 'figure4.pdf'


"""
# SET figure settings
"""
# set figure settings
wFig = 8.7
hFig = 3
font = {'family': 'Helvetica',
        'weight': 'light',
        'size': 6}

axes = {'linewidth': 0.5,
        'titlesize': 7,
        'labelsize': 6,
        'labelpad': 2,
        'spines.top': False,
        'spines.right': False,
        }

ticks = {'major.width': 0.5,
         'minor.width': 0.3,
         'direction': 'in',
         'major.size': 2,
         'minor.size': 1.5,
         'labelsize': 6,
         'major.pad': 2}

legend = {'fontsize': 6,
          'handlelength': 1.5,
          'handletextpad': 0.5,
          'labelspacing': 0.2}

figure = {'dpi': 300}
savefigure = {'dpi': 300,
              'transparent': True}

mpl.style.use('seaborn-ticks')
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('xtick', **ticks)
mpl.rc('ytick', **ticks)
mpl.rc('legend', **legend)
mpl.rc('figure', **figure)
mpl.rc('savefig', **savefigure)


colors = ['777777', 'E24A33', '348ABD', '988ED5',
          'FBC15E', '8EBA42', 'FFB5B8']

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

"""
Main code
"""

# plot cell densities


def plot_ver_hor_cell(axs, n0, mig):
    # setup time vector
    maxT = 6
    minT = -3
    t = np.logspace(minT, maxT, int(1E5))

    # calculate cell density and fraction vertical transmitted
    n_t = mlsg.calc_tauHer_nt(t, n0, mig, 1, 1)
    f_t = mlsg.calc_tauHer_ft(t, n0, mig, 1, 1)
    v_t = f_t * n_t
    h_t = (1 - f_t) * n_t
    logt = np.log10(t)

    # plot
    axs.plot(logt, n_t, linewidth=1, label='tot')
    axs.plot(logt, v_t, '--', linewidth=1, label='vert')
    axs.plot(logt, h_t, ':', linewidth=1, label='hor')
    #axs.set_xlabel('$\log_{10}$ time [a.u.]')
    axs.set_ylabel("density")
    maxY = 1
    xStep = 4
    yStep = 3
    axs.set_ylim((0, maxY+0.05))
    axs.set_xlim((minT, maxT))
    axs.set_xticks(np.linspace(minT, maxT, xStep))
    axs.set_yticks(np.linspace(0, maxY, yStep))

    axs.tick_params(labelbottom=False)
    axs.legend(loc='upper left')

#    axs.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return None


# plot fraction vertical transmitted
def plot_verFracl(axs, n0, mig_rel_vec):
    # cerate time vector
    maxT = 6
    minT = -3
    t = np.logspace(minT, maxT, int(1E5))
    colors = sns.color_palette("Blues_d", n_colors=len(mig_rel_vec))

    # calculate fraction vertically transmitted for different migration rates
    for i, mig in enumerate(mig_rel_vec):
        f_t = mlsg.calc_tauHer_ft(t, n0, mig*n0, 1, 1)
        axs.plot(np.log10(t), f_t, linewidth=1, c=colors[i],
                 label='$%2.1f$' % mig)
    axs.plot([minT, maxT], [0.5, 0.5], 'k:', linewidth=0.5)
    axs.set_xlabel('$\log_{10}$ time [a.u.]')
    axs.set_ylabel("fraction vert")
    maxY = 1
    xStep = 4
    yStep = 3
    axs.set_xlim((minT, maxT))
    axs.set_xticks(np.linspace(minT, maxT, xStep))

    axs.set_ylim((0, maxY+0.05))
    #axs.set_xlim((0, maxT))
    #axs.set_xticks(np.linspace(0, maxT, xStep))
    axs.set_yticks(np.linspace(0, maxY, yStep))
#    axs.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#           ncol=4, mode="expand", borderaxespad=0.)
    axs.legend(bbox_to_anchor=(1.05, 0), loc='lower left',
               borderaxespad=0., borderpad=0.)

    return None


# plot heritability time
def plot_tauH_heatmap(fig, axs, cbaxes):
    # setup grid
    n0 = (-9, -1)
    mig = (-9, -1)
    n0Fine = np.logspace(*n0, 1000)
    migFine = np.logspace(*mig, 1000)
    n0Gr, migGr = np.meshgrid(n0Fine, migFine)

    # calc tau_Her
    tauHMat = mlsg.calc_tauHer_numeric(n0Gr, migGr)

    # plot Heatmap
    viridisBig = mpl.cm.get_cmap('coolwarm', 512)
    indexVec = np.hstack((np.linspace(0, 0.5, 100),
                          np.linspace(0.5, 1-1E-12, 150)))
    cmap = mpl.colors.ListedColormap(viridisBig(indexVec))

    colors = sns.color_palette("RdBu_r", 1024)
    idx = np.floor(indexVec*1024).astype(int)
    cmap = [colors[i] for i in idx]
    cmap = colr.ListedColormap(cmap)

    currData = np.log10(tauHMat)

    axl = axs.imshow(currData, cmap=cmap,
                     interpolation='nearest',
                     extent=[*n0, *mig],
                     origin='lower',
                     vmin=-6, vmax=9)

    axs.set_xticks([-9, -6, -3, -1])
    axs.set_yticks([-9, -6, -3, -1])

    axs.set_xlabel('$\\log_{10} \\frac{n_0}{k}$')
    axs.set_ylabel('$\\log_{10} \\frac{\\theta}{\\beta}$')
    axs.set_aspect('equal')

    cbaxes.set_axis_off()
    cb = fig.colorbar(axl, ax=cbaxes, orientation='horizontal',
                      label="$\\log_{10}\\tau_{her}$",
                      ticks=[-6, -3, 0, 3, 6, 9], anchor=(0.5, 0), aspect=30, shrink=0.5)

    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')


# create figure
def create_fig():
    # set settings
    n0 = 1E-4
    mig1 = 0.5 * n0
    mig_vec_rel = [0.1, 0.5, 2, 10]

    # setup manual axis for subplots
    bm = 0.22
    tm = 0.06
    cm = 0.05
    h = (1 - bm - cm - tm)/2
    lm = 0.08
    rm = 0
    cmh = 0.18
    w1 = 0.4
    w2 = 1 - w1 - cmh - rm - lm
    h3 = 0.1
    tm3 = 0.05
    h2 = 1 - h3 - cm - bm - tm3

    fig = plt.figure()
    mlsg.set_fig_size_cm(fig, wFig, hFig)
    ax = fig.add_axes([lm, bm+cm+h, w1, h])
    plot_ver_hor_cell(ax, n0, mig1)

    ax = fig.add_axes([lm, bm, w1, h])
    plot_verFracl(ax, n0, mig_vec_rel)
    ax.annotate('$\\frac{\\theta/\\beta}{n_0/k}=$',
                xy=(w1+lm+0.02, bm+h*1.2), xycoords='figure fraction',
                horizontalalignment='left',
                verticalalignment='top')

    ax = fig.add_axes([lm + cmh + w1, bm, w2, h2])
    cbaxes = fig.add_axes([lm + cmh + w1, h2 + bm + cm, w2, h3])
    plot_tauH_heatmap(fig, ax, cbaxes)

    #plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5)
    fig.savefig(fig_Folder / figureName,
                format="pdf", transparent=True)

    return None


if __name__ == "__main__":
    create_fig()
