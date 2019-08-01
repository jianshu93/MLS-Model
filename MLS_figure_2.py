#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
Created on  May 22 2019
Last Update May 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

This recreates the data and figure for figure 2
By default data is loaded unless parameters have changes, to rerun model set override_data to True

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import MLS_static_fast as mlssf
import mls_general_code as mlsg
from joblib import Parallel, delayed
import datetime
from pathlib import Path

"""
# SET model settings
"""
# set to True to force recalculation of data
override_data = False

# set folder
data_folder = Path("Data_Paper/")
fig_Folder = Path("Figures_Paper/")
figureName = 'figure2b.pdf'
dataName = 'data_Figure2.npz'

# set model parameters
model_par = {
    # selection strength settings
    "s": 1,
    "K_H": 5000.,
    # tau_var settings
    "cost": 0.01,
    "TAU_H": 10,
    "sigmaBirth": 0.05,
    # tau_mig settings
    "n0": 1E-3,
    "mig": 1E-6,
    # init conditions
    "F0": 0.5,
    "N0init": 1.,
    "NUMGROUP": -1,
    # time settings
    "maxT": 5000,
    "dT": 5E-2,
    "sampleT": 1,
    "rms_err_treshold": 1E-10,
    "mav_window": 100,
    "rms_window": 1000,
    # fixed model parameters
    "sampling": "fixedvar",
    "mu": 1E-9,
    "K": 1E3,
    "numTypeBins": 100
}

# store timescales
tauHer, tauVar = mlsg.calc_timescale(model_par)
model_par['TAU_her'] = tauHer
model_par['TAU_var'] = tauVar

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
mpl.rc('legend', **legend)
mpl.rc('figure', **figure)
mpl.rc('savefig', **savefigure)

colors = ['777777', 'E24A33', '348ABD', '988ED5',
          'FBC15E', '8EBA42', 'FFB5B8']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

"""
Main code
"""

# set host birth/death effect in model par


def set_BD(bh, dh):
    model_par_local = model_par.copy()
    model_par_local['B_H'] = bh
    model_par_local['D_H'] = dh
    return model_par_local


# calcualte moving average over time
def calc_mav(data):
    dataMAV = data[-model_par['mav_window']:, :]
    dataMAV = np.nanmean(data, axis=0)
    return dataMAV


# run model
def run_model():
    # set modelpar list to run
    model_par_B = set_BD(model_par['s'], 0)
    model_par_NS = set_BD(0, 0)
    modelParList = (model_par_B, model_par_NS)

    # run model
    nJobs = min(len(modelParList), 4)
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mlssf.run_model_fixed_parameters)(par) for par in modelParList)

    # process and store output
    Output, InvPerHost, _, _ = zip(*results)
    saveName = data_folder / dataName
    
    #get end distribution investment
    InvPerHostB = calc_mav(InvPerHost[0])
    InvPerHostNS = calc_mav(InvPerHost[1])
    invPerHostEnd = (InvPerHostB, InvPerHostNS)
    
    np.savez(saveName, OutputB=Output[0], InvPerHostB=InvPerHostB,
             OutputNS=Output[1], InvPerHostNS=InvPerHostNS,
             modelParList=modelParList, date=datetime.datetime.now())

    return(Output, invPerHostEnd)


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
    parToIgnore = ('B_H', 'D_H')  # need not check these parameters
    loadName = data_folder / dataName
    if loadName.is_file():
        # open file and load data
        file = np.load(loadName, allow_pickle=True)
        OutputB = file['OutputB']
        InvPerHostB = file['InvPerHostB']
        OutputNS = file['OutputNS']
        InvPerHostNS = file['InvPerHostNS']
        modelParListLoad = file['modelParList']
        file.close()
        Output = (OutputB, OutputNS)
        InvPerHost = (InvPerHostB, InvPerHostNS)
        # check if parameters have changes, if yes rerun
        rerun = check_model_par(modelParListLoad[0], parToIgnore)
    else:
        # cannot load, need to rerun model
        rerun = True
        print('Model data not found, running model')
    if rerun or override_data:
        # rerun model
        Output, InvPerHost = run_model()
    return Output, InvPerHost


# plot line chart
def plot_line(axs, dataStruc, FieldName):
    # plot data
    axs.plot(dataStruc[0]['time'], dataStruc[0][FieldName])
    axs.plot(dataStruc[1]['time'], dataStruc[1][FieldName], '--')
    # make plot nice
    axs.set_xlabel('time [a.u.]')
    axs.set_ylabel("mean frac. helpers $\\langle f \\rangle$")
    maxY = 0.8
    maxX = model_par['maxT']
    xStep = 3
    yStep = 3
    axs.set_ylim((0, maxY))
    axs.set_xlim((0, maxX))
    axs.set_xticks(np.linspace(0, maxX, xStep))
    axs.set_yticks(np.linspace(0, maxY, yStep))
    axs.legend(('$s_b$=%.0f' % model_par['s'], '$s_b$=0'), loc='center right')
    return



# plot histogram chart
def plot_histogram_line(axs, data):
    # calc moving average
    dataMav1 = data[0]
    dataMav2 = data[1]
    # get bin centers
    bins = np.linspace(0, 1, dataMav1.size+1)
    x = (bins[1:] + bins[0:-1]) / 2
    # plot histogram
    axs.plot(x, dataMav1)
    axs.plot(x, dataMav2, '--')
    # make plot nice
    maxY = 0.06
    maxX = 1
    xStep = 3
    yStep = 3
    axs.set_ylim((0, maxY))
    axs.set_xlim((0, maxX))
    axs.set_xticks(np.linspace(0, maxX, xStep))
    axs.set_yticks(np.linspace(0, maxY, yStep))
    axs.set_ylabel('frac. of hosts')
    axs.set_xlabel("frac. helpers in host $f_i$")
    axs.legend(('$s_b$=%.0f' % model_par['s'], '$s_b$=0'), loc='upper right')
    return None


# main function to create figure
def create_fig():
    # load data or compute model
    Output, InvPerHost = load_model()
    # set fonts
    fig = plt.figure()
    mlsg.set_fig_size_cm(fig, wFig, hFig)

    # plot average investment
    axs = fig.add_subplot(1, 2, 1)
    plot_line(axs, Output, "F_mav")
    axs = fig.add_subplot(1, 2, 2)
    plot_histogram_line(axs, InvPerHost)
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5)

    fig.savefig(fig_Folder / figureName,
                format="pdf", transparent=True)
    return None


if __name__ == "__main__":
    create_fig()
