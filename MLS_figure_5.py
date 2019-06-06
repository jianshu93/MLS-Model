#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
Created on  May 22 2019
Last Update May 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

This recreates the data and figure for figure 5
By default data is loaded unless parameters have changes, to rerun model set override_data to True

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mls_general_code as mlsg
from pathlib import Path
import matplotlib.colors as colr
import seaborn as sns
import numpy.lib.recfunctions as rf
import datetime
import itertools
import MLS_static_fast as mlssf
import math
from joblib import Parallel, delayed


"""
# SET model settings
"""
# set to True to force recalculation of data
override_data = False

# set folder
data_folder = Path("Data_Paper/")
fig_Folder = Path("Figures_Paper/")
figureName = 'figure5.pdf'

# set model parameters
tau_H = 100
tauVRange = (-2, 2)
nStep = 50
sigma_vec = [0.02, 0.03, 0.04, 0.1, 0.2, 0.5]

model_par = {
    # selection strength settings
    "s": 1,
    "K_H": 500.,
    "D_H": 0.,
    # tau_var settings
    "TAU_H": tau_H,
    # tau_mig settings
    "n0": 1E-4,
    "mig": 1E-6,
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
B_H_vec = [0, model_par['s']]
cost_vec = 1 / desiredTauV


"""
# SET figure settings
"""
# set figure settings
wFig = 8.7
hFig = 2.5
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
# set variable model parameters


def set_cost_sigma_BH(cost, B_H, sigma):
    model_par_local = model_par.copy()
    model_par_local['cost'] = cost
    model_par_local['B_H'] = B_H
    model_par_local['sigmaBirth'] = sigma
    if sigma > 0.05:
        model_par_local['rms_err_treshold'] = 5E-3
    else:
        model_par_local['rms_err_treshold'] = 1E-3

    return model_par_local

# runs model


def run_model():
    # set modelpar list to run
    modelParList = [set_cost_sigma_BH(*x)
                    for x in itertools.product(*(cost_vec, B_H_vec, sigma_vec))]

    # run model selection
    nJobs = min(len(modelParList), 4)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mlssf.single_run_finalstate)(par) for par in modelParList)

    # process and store output
    Output, InvPerHost = zip(*results)
    statData = np.vstack(Output)
    distData = np.vstack(InvPerHost)

    saveName = data_folder / 'data_Figure5.npz'
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
    parToIgnore = ('cost', 'B_H', 'sigmaBirth', 'rms_err_treshold')
    loadName = data_folder / 'data_Figure5.npz'
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


# plot fraction helper cells, from analytical calculation
def plot_helper_frac(axs, cost_vec):
    # set what to plot
    maxT = 1000
    minT = 0
    t = np.linspace(minT, maxT, int(1E5))
    f0_vec = [0.9, 0.3]

    colors = sns.color_palette("Blues_d", n_colors=2*len(cost_vec)+1)
    handle_list = []
    lineStyle = ['-', '--', '-.', ':']

    # calculate fraction helper analytically
    for i, cost in enumerate(cost_vec):
        for j, f0 in enumerate(f0_vec):
            f_t = mlsg.calc_tauVar_ft(t, f0, cost, 1)
            handle, = axs.plot(t, f_t, lineStyle[j], linewidth=1, c=colors[i*3],
                               label='$%.3f$' % cost)
            if j == 0:
                handle_list.append(handle)

    fHalf = f0_vec[0] / (f0_vec[0] * (1 - math.e) + math.e)
    tauH = 1 / cost_vec[1]

    axs.plot([0, tauH], [fHalf, fHalf], linewidth=0.5, color='black')
    axs.annotate('$\\tau_{var}$', (tauH/2, fHalf),
                 horizontalalignment='center',
                 verticalalignment='top')

    axs.set_xlabel('time [a.u.]')
    axs.set_ylabel("fraction helper")
    maxY = 1
    xStep = 3
    yStep = 3
    axs.set_xlim((minT, maxT))
    axs.set_xticks(np.linspace(minT, maxT, xStep))
    axs.set_ylim((0, maxY+0.05))
    axs.set_yticks(np.linspace(0, maxY, yStep))
    # axs.legend(bbox_to_anchor=(1.05, 0), loc='lower left',
    #            borderaxespad=0., borderpad=0.)
    axs.legend(handles=handle_list, loc='center right')

    return None


# slect data to plot
def select_data(data1D, BHidx, sigmaidx):
    # get subset of data to plot
    curSigma = data1D['sigma_cat'] == sigmaidx
    curBH = data1D['BH_cat'] == BHidx
    # remove nan and inf
    isFinite = np.logical_and.reduce(
        np.isfinite((data1D['tauVar_rel'], data1D['tauHer_rel'],
                     data1D['F_mav'], curSigma, curBH)))
    currSubset = np.logical_and.reduce((curSigma, curBH, isFinite))
    # extract data and log transform x,y
    x = np.log10(data1D['tauVar_rel'][currSubset])
    y = data1D['F_mav'][currSubset]
    return (x, y)


# plot fraction helper as function of tau_var computed from model
def plot_tau_var(axs, data1D):
    colors = sns.color_palette("Oranges_d", n_colors=len(sigma_vec))
    for i, sigma in enumerate(sigma_vec):
        x, yS = select_data(data1D, 1, i)
        axs.plot(x, yS, linewidth=1, c=colors[i],
                 label='$%.2f$' % sigma)

    axs.set_xlabel('$\\log_{10} \\frac{\\tau_{var}}{\\tau_H}$')
    axs.set_ylabel("mean frac. helper $\\langle f \\rangle$")
    maxY = 1
    xStep = 3
    yStep = 3
    axs.set_xlim(tauVRange)
    axs.set_xticks(np.linspace(*tauVRange, xStep))
    axs.set_ylim((0, maxY+0.05))
    axs.set_yticks(np.linspace(0, maxY, yStep))
    axs.legend(loc='upper left')


def create_fig():
    # load data or compute model
    data1D = load_model()
    data1D = process_data(data1D)

    # settings
    cost_vec = [0.001, 0.01, 0.1]
    # setup manual axis for subplots
    bm = 0.25
    tm = 0.02
    cm = 0.12
    #h = (1 - bm - tm - cm) / 2
    h = 1 - bm - tm
    lm = 0.08
    rm = 0.03
    cmh = 0.12
    w = (1 - cmh - rm - lm) / 2

    # set fonts
    fig = plt.figure()
    mlsg.set_fig_size_cm(fig, wFig, hFig)

    ax = fig.add_axes([lm, bm, w, h])
    plot_helper_frac(ax, cost_vec)
    ax.annotate('$\\gamma=$',
                xy=(w+lm-0.12, bm+h*0.7), xycoords='figure fraction',
                horizontalalignment='left',
                verticalalignment='top')

    ax = fig.add_axes([lm+cmh+w, bm, w, h])
    plot_tau_var(ax, data1D)
    ax.annotate('$\\sigma=$',
                xy=(w+lm+cmh+0.05, bm+h), xycoords='figure fraction',
                horizontalalignment='left',
                verticalalignment='top')

    fig.savefig(fig_Folder / figureName,
                format="pdf", transparent=True)

    return None


if __name__ == "__main__":
    create_fig()
