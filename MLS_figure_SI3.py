#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
Created on  May 22 2019
Last Update May 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

This recreates the data and figure for figure SI3
By default data is loaded unless parameters have changes, to rerun model set override_data to True

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import MLS_static_fast as mlssf
import mls_general_code as mlsg
import numpy.lib.recfunctions as rf
from mpl_toolkits.mplot3d import axes3d
from joblib import Parallel, delayed
import datetime
from pathlib import Path
import itertools
import seaborn as sns
import matplotlib.colors as colr

"""
# SET model settings
"""

# set to True to force recalculation of data
override_data = False

# set folder
data_folder = Path("Data_Paper/")
fig_Folder = Path("Figures_Paper/")
figureName = 'figureSI3.pdf'
dataName = 'data_FigureSI3.npz'

# set model parameters
tau_H = 100
tauVRange = (-2, 2)
tauHRange = (-3, 6)
nStep = 30

model_par = {
    # selection strength settings
    "s": 1,
    "K_H": 500.,
    "D_H": 0.,
    # tau_var settings
    "TAU_H": tau_H,
    "sigmaBirth": 0.1,
    # tau_mig settings
    "n0": 1E-4,
    # init conditions
    "F0": 0.01,
    "N0init": 1.,
    "NUMGROUP": -1,
    # time settings
    "maxT": 150000,
    "dT": 5E-2,
    "sampleT": 10,
    "rms_err_treshold": 5E-3,
    "mav_window": 1000,
    "rms_window": 10000,
    "minTRun": 25000,
    # fixed model parameters
    "sampling": "fixedvar",
    "mu": 1E-9,
    "K": 1,
    "numTypeBins": 100
}


# calc other parameters
desiredTauV = np.logspace(*tauVRange, nStep) * tau_H
desiredTauH = np.logspace(*tauHRange, nStep)
B_H_vec = [0, 1]
mig_vec = model_par['n0'] / desiredTauH
cost_vec = 1 / desiredTauV


"""
# SET figure settings
"""
# set figure settings
wFig = 17.8
hFig = 4
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
         'direction': 'in',
         'major.size': 2,
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
#mpl.rc('ztick', **ticks)
mpl.rc('legend', **legend)
mpl.rc('figure', **figure)
mpl.rc('savefig', **savefigure)

colors = ['777777', 'E24A33', '348ABD', '988ED5',
          'FBC15E', '8EBA42', 'FFB5B8']

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)


def set_cost_mig_BH(cost, mig, B_H):
    model_par_local = model_par.copy()
    model_par_local['cost'] = cost
    model_par_local['mig'] = mig
    if B_H > 0:
        model_par_local['B_H'] = model_par['s']
        model_par_local['D_H'] = 0
    else:
        model_par_local['B_H'] = 0
        model_par_local['D_H'] = 0
    return model_par_local


"""
Main code
"""


def run_model():
    # set modelpar list to run
    modelParList = [set_cost_mig_BH(*x)
                    for x in itertools.product(*(cost_vec, mig_vec, B_H_vec))]

    # run model selection
    nJobs = min(len(modelParList), 4)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mlssf.single_run_finalstate)(par) for par in modelParList)

    # process and store output
    Output, InvPerHost = zip(*results)
    statData = np.vstack(Output)
    distData = np.vstack(InvPerHost)

    saveName = data_folder / dataName
    np.savez(saveName, statData=statData, distData=distData,
             modelParList=modelParList, date=datetime.datetime.now())

    return statData


def check_model_par(model_par_load, parToIgnore):
    rerun = False
    for key in model_par_load:
        if not (key in parToIgnore):
            if model_par_load[key] != model_par[key]:
                print('Parameter "%s" has changed, rerunning model!' % key)
                rerun = True
    return rerun


def load_model():
    # need not check these parameters
    parToIgnore = ('cost', 'mig', 'B_H', 'D_H')
    loadName = data_folder / dataName
    if loadName.is_file():
        # open file and load data
        data_file = np.load(loadName, allow_pickle=True)
        data1D = data_file['statData']
        rerun = check_model_par(data_file['modelParList'][0], parToIgnore)
        data_file.close()
    else:
        # cannot load, need to rerun model
        rerun = True
        print('Model data not found, running model')
    if rerun or override_data:
        # rerun model
        data1D = run_model()
    return data1D


def process_data(statData):
    # calculate heritability time
    tauHer = mlsg.calc_tauHer_numeric(
        statData['n0'], statData['mig'])
    tauVar = mlsg.calc_tauV(statData['cost'])
    tauHerRel = tauHer/statData['TAU_H']
    tauVar_rel = tauVar/statData['TAU_H']
    BH_cat = mlsg.make_categorial(statData['B_H'])
    dataToStore = (tauHer, tauVar, tauHerRel, tauVar_rel, BH_cat)
    nameToStore = ('tauHer', 'tauVar', 'tauHer_rel',
                   'tauVar_rel', 'BH_cat')

    statData = rf.append_fields(
        statData, nameToStore, dataToStore, usemask=False)

    return statData


def select_data(data1D, BHidx):
    # get subset of data to plot
    curBH = data1D['BH_cat'] == BHidx
    # remove nan and inf
    isFinite = np.logical_and.reduce(
        np.isfinite((data1D['tauVar_rel'], data1D['tauHer_rel'],
                     data1D['F_mav'], curBH)))
    currSubset = np.logical_and.reduce((curBH, isFinite))
    # extract data and log transform x,y
    x = np.log10(data1D['tauVar_rel'][currSubset])
    transMode = data1D['n0']/data1D['mig']
    y = np.log10(transMode[currSubset])
    z = data1D['F_mav'][currSubset]
    return (x, y, z)


def plot_3D(ax, data1D, BHidx):
    x, y, z = select_data(data1D, BHidx)
    ax.scatter(x, y, z,
               c=z,
               s=0.5, alpha=0.7,
               vmin=0, vmax=1, cmap='plasma')

    steps = (3, 4, 3)
    fRange = (0, 1)

    ax.set_xlim(tauVRange)
    ax.set_ylim(tauHRange)
    ax.set_zlim(fRange)
    ax.set_xticks(np.linspace(*tauVRange, steps[0]))
    ax.set_yticks(np.linspace(*tauHRange, steps[1]))
    ax.set_zticks(np.linspace(*fRange, steps[2]))

    # set labels
    ax.set_xlabel('$log_{10} \\frac{\\tau_{Var}}{\\tau_H}$')
    ax.set_ylabel('$log_{10} \\frac{n_0/k}{\\theta/\\beta}$')
    ax.set_zlabel('$\\langle f \\rangle$')

    ax.yaxis.labelpad = -10
    ax.xaxis.labelpad = -10
    ax.zaxis.labelpad = -10
    ax.tick_params(axis='z', which='major', pad=0)
    ax.tick_params(axis='both', which='major', pad=-5)

    ax.view_init(20, -115)

    return None


def bin_2Ddata(currXData, currYData, currZData, xbins, ybins):
    """[Bins x,y data into 2d bins]
    Arguments:
            currXData {np vector} -- xData to bin
            currYData {np vector} -- yData to bin
            currZData {np vector} -- zData to bin
            xbins {np vector} -- xBins to use
            ybins {np vector} -- yBins to use
    """
    # init output
    nX = xbins.size
    nY = ybins.size
    binnedData = np.full((nY, nX), np.nan)
    # loop over bins and calc mean
    for xx in range(nX - 1):
        for yy in range(nY - 1):
            # find data in bin
            inXBin = np.logical_and(
                (currXData >= xbins[xx]), (currXData < xbins[xx+1]))
            inYBin = np.logical_and(
                (currYData >= ybins[yy]), (currYData < ybins[yy+1]))
            inBin = np.logical_and(inXBin, inYBin)
            zInBin = currZData[inBin]
            # calc mean over bine
            binnedData[yy, xx] = np.nanmean(zInBin)
    return(binnedData)


def plot_heatmap(fig, ax, data1D):

    colors = sns.color_palette("RdBu_r", 265)

    indexVec = np.hstack((np.linspace(0, 0.5, 50),
                          np.linspace(0.5, 1-1E-12, 150)))
    colors = sns.color_palette("RdBu_r", 1024)
    idx = np.floor(indexVec*1024).astype(int)
    cmap = [colors[i] for i in idx]
    cmap = colr.ListedColormap(cmap)

    xStep = 0.25
    yStep = 0.5
    xbins = np.linspace(*tauVRange, int(
        np.ceil((tauVRange[1]-tauVRange[0])/xStep))+1)
    ybins = np.linspace(*tauHRange, int(
        np.ceil((tauHRange[1] - tauHRange[0]) / yStep)) + 1)

    # get data with selection
    xB, yB, zD = select_data(data1D, 1)
    binnedDataB = bin_2Ddata(xB, yB, zD, xbins, ybins)

    xD, yD, zD = select_data(data1D, 0)
    binnedDataD = bin_2Ddata(xD, yD, zD, xbins, ybins)

    BvsD = binnedDataB / binnedDataD

    im = ax.pcolormesh(xbins, ybins, np.log2(BvsD), cmap=cmap, vmin=-1, vmax=3)

    fig.colorbar(im, ax=ax,
                 label="$\\log_{2} \\frac{\\langle f \\rangle_{s_b=1}}{\\langle f \\rangle_{s_b=0}}$",
                 ticks=[-1, 0, 1, 2, 3], shrink=0.8)

    steps = (3, 4)

    ax.set_xlim(tauVRange)
    ax.set_ylim(tauHRange)
    ax.set_xticks(np.linspace(*tauVRange, steps[0]))
    ax.set_yticks(np.linspace(*tauHRange, steps[1]))

    # set labels
    ax.set_xlabel('$log_{10} \\frac{\\tau_{Var}}{\\tau_H}$')
    ax.set_ylabel('$log_{10} \\frac{n_0/k}{\\theta/\\beta}$')

    return None


def plot_line(axs, dataStruc, FieldName):
    # plot data
    axs.plot(dataStruc[0]['time'], dataStruc[0][FieldName])
    axs.plot(dataStruc[1]['time'], dataStruc[1][FieldName], '--')
    # make plot nice
    axs.set_xlabel('time [a.u.]')
    axs.set_ylabel("mean frac. helpers $\\langle f \\rangle$")
    maxY = 0.8
    maxX = 25000
    xStep = 3
    yStep = 3
    axs.set_ylim((0, maxY))
    axs.set_xlim((0, maxX))
    axs.set_xticks(np.linspace(0, maxX, xStep))
    axs.set_yticks(np.linspace(0, maxY, yStep))
    axs.legend(('$s_b$=%.0f' % model_par['s'], '$s_b$=0'), loc='center right')
    return


def calc_mav(data):
    dataMAV = data[-model_par['mav_window']:, :]
    dataMAV = np.nanmean(data, axis=0)
    return dataMAV


def plot_histogram_line(axs, data):
    # calc moving average
    dataMav1 = calc_mav(data[0])
    dataMav2 = calc_mav(data[1])
    # get bin centers
    bins = np.linspace(0, 1, dataMav1.size+1)
    x = (bins[1:] + bins[0:-1]) / 2
    # plot histogram
    axs.plot(x, dataMav1)
    axs.plot(x, dataMav2, '--')

    print(dataMav1.sum(), dataMav2.sum(),)
    # make plot nice
    maxY = 0.03
    maxX = 1
    xStep = 3
    yStep = 3
    axs.set_ylim((0, maxY))
    axs.set_xlim((0, maxX))
    axs.set_xticks(np.linspace(0, maxX, xStep))
    axs.set_yticks(np.linspace(0, maxY, yStep))
    axs.set_ylabel('frac. of hosts')
    axs.set_xlabel("frac. helpers in host $f_i$")
    sd = model_par['s']/(1+model_par['s'])
    axs.legend(('$s_b$=%.1f' %
                model_par['s'], '$s_d=%.1f$' % sd), loc='upper right')
    return None


def create_fig():
    # load data or compute model
    data1D = load_model()
    data1D = process_data(data1D)

    # set fonts
    fig = plt.figure()
    mlsg.set_fig_size_cm(fig, wFig, hFig)

    # plot average investment
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    plot_3D(ax, data1D, 1)
    ax.set_title('Brith rate effect $s_b=1$')
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title('No brith rate effect $s_b=0$')
    plot_3D(ax, data1D, 0)
    ax = fig.add_subplot(1, 3, 3)
    plot_heatmap(fig, ax, data1D)

    plt.tight_layout(pad=1, h_pad=2.5, w_pad=0.5)
    fig.savefig(fig_Folder / figureName,
                format="pdf", transparent=True)

    return None


if __name__ == "__main__":
    create_fig()
