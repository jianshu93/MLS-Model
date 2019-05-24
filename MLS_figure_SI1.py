#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
Created on  May 22 2019
Last Update May 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

This recreates the data and figure for figure SI1
By default data is loaded unless parameters have changes, to rerun model set override_data to True

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import MLS_static_fast as mlssf
import mls_general_code as mlsg
import numpy.lib.recfunctions as rf
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
figureName = 'figureSI1.pdf'
dataName = 'data_figureSI1.npz'

# set model parameters
# range of tauV and tauH to scan
tauVRange = (-2, 2)
tauHRange = (-2, 4)
nStep = 20

# model  parameters to change
# first value in tuple is default, second is list of alternative values to use
alt_model_par = {
    # selection strength settings
    "K_H": (500., [100., 2500.]),
    "B_H": (1., [0.2, 5.]),
    "TAU_H": (100., [10, 1000]),

    "sigmaBirth": (0.05, [0.01, 0.25]),
    "n0": (1E-4, [1E-5, 1E-3]),
    "mu": (1E-9, [1E-12, 1E-6])
}

# names of parameters (tarnslate between code and manuscript)
alt_model_par_names = {
    # selection strength settings
    "K_H": '$K_H$',
    "B_H": '$s_b$',
    "TAU_H": '$\\tau_H$',
    "sigmaBirth": '$\\sigma$',
    "n0": '$n_0$',
    "mu": '$\\mu$'
}

# fixed parameters
fixed_model_par = {
    # init conditions
    "F0": 0.01,
    "N0init": 1.,
    "NUMGROUP": -1,
    # time settings
    "D_H": 0.,
    "maxT": 150000,
    "dT": 5E-2,
    "sampleT": 10,
    "rms_err_treshold": 5E-2,
    "mav_window": 1000,
    "rms_window": 10000,
    "minTRun": 25000,
    # fixed model parameters
    "sampling": "fixedvar",
    "numTypeBins": 100,
    "K": 1
}

# calc other parameters
desiredTauV = np.logspace(*tauVRange, nStep)
desiredTauH = np.logspace(*tauHRange, nStep)

# create default model parameters:
# set fixed par
model_par_def = fixed_model_par.copy()
# set default values for variable par
for key, val in alt_model_par.items():
    model_par_def[key] = val[0]


"""
# SET figure settings
"""
# set figure settings
wFig = 17.8
hFig = 16
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
# mpl.rc('ztick', **ticks)
mpl.rc('legend', **legend)
mpl.rc('figure', **figure)
mpl.rc('savefig', **savefigure)


"""
Main code
"""

# set variable model parameters


def set_var_keys(model_par, cost, mig):
    model_par_local = model_par.copy()
    model_par_local['cost'] = cost
    model_par_local['mig'] = mig
    return model_par_local


# create model par list, varying one parameter at the time, keeping all others constant
def create_model_par_list(variable_key=None, variable_value=None):
    model_par_curr = model_par_def.copy()
    if variable_key != None:
        model_par_curr[variable_key] = variable_value

    TAU_H = model_par_curr['TAU_H']
    n0 = model_par_curr['n0']

    if model_par_curr['sigmaBirth'] > 0.05:
        model_par_curr['rms_err_treshold'] = 5E-3
    else:
        model_par_curr['rms_err_treshold'] = 1E-3

    # calc cost and migration vectors
    # migToN0 = mlsg.mig_from_tauH(desiredTauH * TAU_H, n0)
    cost_vec = 1 / (desiredTauV * TAU_H)
    mig_vec = n0 / desiredTauH

    modelParList = [set_var_keys(model_par_curr, *x)
                    for x in itertools.product(*(cost_vec, mig_vec))]
    return modelParList


# run model
def run_model():
    modelParList = create_model_par_list()
    for key, valueVec in alt_model_par.items():
        for val in valueVec[1]:
            modelParList.extend(create_model_par_list(key, val))

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

# load is possible run otherwise


def load_model():
    # need not check these parameters
    loadName = data_folder / dataName
    if loadName.is_file():
        # open file and load data
        data_file = np.load(loadName, allow_pickle=True)
        data1D = data_file['statData']
        data_file.close()
        rerun = False
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
    tauHer_Rel = tauHer/statData['TAU_H']
    tauVar_rel = tauVar/statData['TAU_H']
    dataToStore = (tauHer, tauVar, tauHer_Rel, tauVar_rel)
    nameToStore = ('tauHer', 'tauVar', 'tauHer_rel',
                   'tauVar_rel')
    statData = rf.append_fields(
        statData, nameToStore, dataToStore, usemask=False)

    return statData


def select_data(data1D, select_key=None, select_value=None):
    # get subset of data to plot
    isFinite = np.logical_and.reduce(
        np.isfinite((data1D['tauVar_rel'],
                     data1D['tauHer_rel'],
                     data1D['F_mav'])))

    currSubset = isFinite
    for key, valueVec in alt_model_par.items():
        if key != select_key:
            currVal = data1D[key] == valueVec[0]
        else:
            currVal = data1D[key] == select_value
        currSubset = np.logical_and(currVal, currSubset)
    # extract data and log transform x,y
    x = np.log10(data1D['tauVar_rel'][currSubset])

    transMode = data1D['n0']/data1D['mig']
    y = np.log10(transMode[currSubset])
    z = data1D['F_mav'][currSubset]
    return (x, y, z)


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
            # calc mean over bin
            binnedData[yy, xx] = np.nanmean(zInBin)
    return(binnedData)


def plot_parscan(fig, ax_list, key, data1D):
    xStep = 0.25
    yStep = 0.5
    xbins = np.linspace(*tauVRange, int(
        np.ceil((tauVRange[1]-tauVRange[0])/xStep))+1)
    ybins = np.linspace(*tauHRange, int(
        np.ceil((tauHRange[1] - tauHRange[0]) / yStep)) + 1)

    # get default data
    xD, yD, zD = select_data(data1D)
    binnedDataDef = bin_2Ddata(xD, yD, zD, xbins, ybins)

    # get low data
    xL, yL, zL = select_data(data1D, key, alt_model_par[key][1][0])
    binnedDataLow = bin_2Ddata(xL, yL, zL, xbins, ybins)
    relDataLow = binnedDataLow / binnedDataDef

    # get high data
    xH, yH, zH = select_data(data1D, key, alt_model_par[key][1][1])
    binnedDataHigh = bin_2Ddata(xH, yH, zH, xbins, ybins)
    relDataHigh = binnedDataHigh / binnedDataDef

    # plot data
    varName = alt_model_par_names[key]
    im0 = plot_heatmap_rel(fig, ax_list[0], xbins, ybins, relDataLow)
    ax_list[0].set_title('{}={}'.format(varName, alt_model_par[key][1][0]))
    im1 = plot_heatmap(fig, ax_list[1], xbins, ybins, binnedDataLow)
    ax_list[1].set_title('{}={}'.format(varName, alt_model_par[key][1][0]))
    im2 = plot_heatmap(fig, ax_list[2], xbins, ybins, binnedDataDef)
    ax_list[2].set_title('{}={}'.format(varName, alt_model_par[key][0]))
    im3 = plot_heatmap(fig, ax_list[3], xbins, ybins, binnedDataHigh)
    ax_list[3].set_title('{}={}'.format(varName, alt_model_par[key][1][1]))
    im4 = plot_heatmap_rel(fig, ax_list[4], xbins, ybins, relDataHigh)
    ax_list[4].set_title('{}={}'.format(varName, alt_model_par[key][1][1]))

    imVec = [im0, im1, im2, im3, im4]
    return imVec


def plot_heatmap(fig, ax, xbins, ybins, data):
    im = ax.pcolormesh(xbins, ybins, data, cmap='plasma', vmin=0, vmax=1)
#    fig.colorbar(im, ax=ax)
    steps = (3, 4)
    ax.set_xlim(tauVRange)
    ax.set_ylim(tauHRange)
    ax.set_xticks(np.linspace(*tauVRange, steps[0]))
    ax.set_yticks(np.linspace(*tauHRange, steps[1]))
    # set labels
    ax.set_xlabel('$log_{10} \\frac{\\tau_{Var}}{\\tau_H}$')
    ax.set_ylabel('$log_{10} \\frac{n_0/k}{\\theta/\\beta}$')
    return im


def plot_heatmap_rel(fig, ax, xbins, ybins, data):

    colors = sns.color_palette("RdBu_r", 265)

    indexVec = np.hstack((np.linspace(0, 0.5, 100),
                          np.linspace(0.5, 1-1E-12, 100)))
    colors = sns.color_palette("RdBu_r", 1024)
    idx = np.floor(indexVec*1024).astype(int)
    cmap = [colors[i] for i in idx]
    cmap = colr.ListedColormap(cmap)

    im = ax.pcolormesh(xbins, ybins, np.log2(data),
                       vmin=-5, vmax=5, cmap=cmap)
#    fig.colorbar(im, ax=ax)
    steps = (3, 4)
    ax.set_xlim(tauVRange)
    ax.set_ylim(tauHRange)
    ax.set_xticks(np.linspace(*tauVRange, steps[0]))
    ax.set_yticks(np.linspace(*tauHRange, steps[1]))
    # set labels
    ax.set_xlabel('$log_{10} \\frac{\\tau_{Var}}{\\tau_H}$')
    ax.set_ylabel('$log_{10} \\frac{n_0/k}{\\theta/\\beta}$')
    return im


def add_color_bars(fig, ax_list, im_list):
    absName = '$\\langle f \\rangle$'
    relName = '$\\log_2$ fold change'
    name_vec = [relName, absName, absName, absName, relName]

    for cbaxes, im, name in zip(ax_list, im_list, name_vec):
        cbaxes.set_axis_off()
        fig.colorbar(im, ax=cbaxes, orientation='vertical',
                          label=name, shrink=0.8)
    return None


def create_fig():
    # load data or compute model
    data1D = load_model()
    data1D = process_data(data1D)

    nr = 5
    nc = len(alt_model_par) + 1

    fig = plt.figure()
    mlsg.set_fig_size_cm(fig, wFig, hFig)

    idx = 0
    for key in alt_model_par:
        subplots = np.arange(nr)*nc + 1 + idx
        idx += 1
        ax_list = [fig.add_subplot(nr, nc, x) for x in subplots]
        imVec = plot_parscan(fig, ax_list, key, data1D)

    subplots = np.arange(nr)*nc + nc
    ax_list = [fig.add_subplot(nr, nc, x) for x in subplots]
    add_color_bars(fig, ax_list, imVec)

    plt.tight_layout(pad=1, h_pad=2.5, w_pad=1.5)
    fig.savefig(fig_Folder / figureName,
                format="pdf", transparent=True)

    return None


if __name__ == "__main__":
    statData = create_fig()
