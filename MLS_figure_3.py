#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
Created on  May 22 2019
Last Update May 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

This recreates the data and figure for figure 3
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

"""
# SET model settings
"""
# set to True to force recalculation of data
override_data = False

# set folder
data_folder = Path("Data_Paper/")
fig_Folder = Path("Figures_Paper/")
figureName = 'figure3.pdf'
dataName = 'data_Figure3.npz'

# set model parameters
tau_H = 100
tauVRange = (-2, 2)
tauHRange = (-2, 4)
nStep = 30
sigma_vec = [0.02, 0.1]

model_par = {
    # selection strength settings
    "s": 1,
    "K_H": 500.,
    "D_H": 0.,
    # tau_var settings
    "TAU_H": tau_H,
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
    "rms_err_treshold": 5E-2,
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
B_H_vec = [model_par['s'], 0]
cost_vec = 1 / desiredTauV
mig_vec = model_par['n0'] / desiredTauH


"""
# SET figure settings
"""
wFig = 8.7
hFig = 4.5
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

"""
Main code
"""
# set variable model parameters


def set_cost_mig_BH(cost, mig, B_H, sigma):
    model_par_local = model_par.copy()
    model_par_local['cost'] = cost
    model_par_local['mig'] = mig
    model_par_local['B_H'] = B_H
    model_par_local['sigmaBirth'] = sigma
    # change the error treshold to use to find if steady state is reached
    # higher sample variance gives more noise, needs lower treshold
    if sigma > 0.05:
        model_par_local['rms_err_treshold'] = 5E-3
    else:
        model_par_local['rms_err_treshold'] = 1E-3

    return model_par_local

# runs model


def run_model():
    # set modelpar list to run
    modelParList = [set_cost_mig_BH(*x)
                    for x in itertools.product(*(cost_vec, mig_vec, B_H_vec, sigma_vec))]

    # run model
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


# checks of model parmaters have changed
def check_model_par(model_par_load, parToIgnore):
    rerun = False
    for key in model_par_load:
        if not (key in parToIgnore):
            if model_par_load[key] != model_par[key]:
                print('Parameter "%s" has changed, rerunning model!' % 'load')
                rerun = True
    return rerun


# Load model is datafile found, run model if not found or if settings have changed
def load_model():
    # need not check these parameters
    parToIgnore = ('cost', 'mig', 'B_H', 'sigmaBirth', 'rms_err_treshold')
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


# Process data, calc timescale and other variables needed for plotting
def process_data(statData):
    # calculate heritability time
    tauHer = mlsg.calc_tauHer_numeric(
        statData['n0'], statData['mig'])
    tauVar = mlsg.calc_tauV(statData['cost'])
    tauHerRel = tauHer/statData['TAU_H']
    tauVar_rel = tauVar/statData['TAU_H']
    sigma_cat = mlsg.make_categorial(statData['sigmaBirth'])
    BH_cat = mlsg.make_categorial(statData['B_H'])
    dataToStore = (tauHer, tauVar, tauHerRel, tauVar_rel, sigma_cat, BH_cat)
    nameToStore = ('tauHer', 'tauVar', 'tauHer_rel',
                   'tauVar_rel', 'sigma_cat', 'BH_cat')

    statData = rf.append_fields(
        statData, nameToStore, dataToStore, usemask=False)

    return statData


# get subset of data to plot
def select_data(data1D, BHidx, sigmaidx):
    curSigma = data1D['sigma_cat'] == sigmaidx
    curBH = data1D['BH_cat'] == BHidx
    # remove nan and inf
    isFinite = np.logical_and.reduce(
        np.isfinite((data1D['tauVar_rel'], data1D['tauHer_rel'],
                     data1D['F_mav'], curSigma, curBH)))
    currSubset = np.logical_and.reduce((curSigma, curBH, isFinite))
    # extract data and log transform x,y
    x = np.log10(data1D['tauVar_rel'][currSubset])

    transMode = data1D['n0']/data1D['mig']
    y = np.log10(transMode[currSubset])
    z = data1D['F_mav'][currSubset]
    return (x, y, z)


# make 3D scatter plot
def plot_3D(ax, data1D, sigmaIndex):
    BHidx = 1
    x, y, z = select_data(data1D, BHidx, sigmaIndex)
    ax.scatter(x, y, z,
               c=z,
               s=0.3, alpha=1,
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
    ax.view_init(30, -115)

    return None


# bin scatter data to show as heatmap
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


# make heatmap
def plot_heatmap(fig, ax, data1D, sigmaIndex):
    BHidx = 1
    xStep = 0.25
    yStep = 0.5
    xbins = np.linspace(*tauVRange, int(
        np.ceil((tauVRange[1]-tauVRange[0])/xStep))+1)
    ybins = np.linspace(*tauHRange, int(
        np.ceil((tauHRange[1] - tauHRange[0]) / yStep)) + 1)

    # get data with selection
    xS, yS, zS = select_data(data1D, BHidx, sigmaIndex)
    binnedDataS = bin_2Ddata(xS, yS, zS, xbins, ybins)

    im = ax.pcolormesh(xbins, ybins, binnedDataS,
                       cmap='plasma', vmin=0, vmax=1)

    #cb = fig.colorbar(im, ax=ax)
    name = '$\\langle f \\rangle$'
    fig.colorbar(im, ax=ax, orientation='vertical',
                 label=name,
                 ticks=[0, 0.5, 1])

    steps = (3, 4)

    ax.set_xlim(tauVRange)
    ax.set_ylim(tauHRange)
    ax.set_xticks(np.linspace(*tauVRange, steps[0]))
    ax.set_yticks(np.linspace(*tauHRange, steps[1]))

    # set labels
    ax.set_xlabel('$log_{10} \\frac{\\tau_{Var}}{\\tau_H}$')
    ax.set_ylabel('$log_{10} \\frac{n_0/k}{\\theta/\\beta}$')

    return None


# main function to create figure
def create_fig():
    # load data or compute model
    data1D = load_model()
    data1D = process_data(data1D)

    fig = plt.figure()
    mlsg.set_fig_size_cm(fig, wFig, hFig)

    # setup manual axis for subplots
    bm = 0.15
    tm = 0.06
    cm = 0.13
    hf = 0.65
    h = [hf*(1 - bm - cm - tm), (1-hf)*(1 - bm - cm - tm)]
    lm = 0.05
    rm = 0.05
    cmh = 0.1
    w = (1 - cmh - rm - lm)/2
    wf = 0.85
    wo = (1-wf)*w
    left = [lm, lm+cmh+w]
    bot = [bm+h[1]+cm, bm]

    # plot for different sampling variance
    for ss in range(2):
        # plot scatter
        ax = fig.add_axes([left[ss], bot[0], w, h[0]], projection='3d')
        plot_3D(ax, data1D, ss)
        ax.set_title('$\\sigma={}$'.format(sigma_vec[ss]), fontsize=6)
        # plot heatmap
        ax = fig.add_axes([left[ss]+wo, bot[1], wf*w, h[1]])
        plot_heatmap(fig, ax, data1D, ss)

    #plt.tight_layout(pad=1, h_pad=2.5, w_pad=1.5)
    fig.savefig(fig_Folder / figureName,
                format="pdf", transparent=True)

    return None


if __name__ == "__main__":
    create_fig()
