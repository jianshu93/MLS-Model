#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
Created on  May 22 2019
Last Update May 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

This recreates the data and figure for figure 6
By default data is loaded unless parameters have changes, to rerun model set override_data to True

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import MLS_evolveCoop_fast as mlse
import mls_general_code as mlsg
import datetime
from pathlib import Path

"""
# SET model settings
"""
# set to True to force recalculation of data
override_data = False

# set folder and file settings
data_folder = Path("Data_Paper/")
fig_Folder = Path("Figures_Paper/")
figureName = 'figure6.pdf'
dataName = 'data_Figure6.npz'

# set model settings
model_par = {
    # time step parameters
    "maxT": 200000.,
    "dT": 5E-2,
    "sampleT": 100,
    "rms_err_treshold": 1E-5,
    "mav_window": 1000,
    "rms_window": 5000,
    # fixed model parameters
    "sampling": "sample",
    "mu": 0.01,
    "B_H": 1.,
    "D_H": 0.,
    "K_H": 500.,
    # variable model parameters
    "cost": 0.01,
    "TAU_H": 100.,
    "n0": 1E-3,
    "mig": 1E-5,
    "r": 1.,
    "K": 10E3,
    # fixed intial condition
    "NUMGROUP": 500,
    "numTypeBins": 100,
    "meanGamma0": 0,
    "stdGamma0": 0.01,
    "N0init": 1.,
}

"""
# SET figure settings
"""
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

# runs model


def run_model():
    _, Output, InvestmentAll, InvestmentPerHost = \
        mlse.run_model_fixed_parameters(model_par)
    # process and store output
    saveName = data_folder / dataName
    np.savez(saveName, Output=Output,
             InvestmentAll=InvestmentAll, InvestmentPerHost=InvestmentPerHost,
             model_par=[model_par], date=datetime.datetime.now())

    return(Output, InvestmentAll, InvestmentPerHost)


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
    parToIgnore = ()
    # parToIgnore = ('B_H', 'D_H')  # need not check these parameters
    loadName = data_folder / dataName
    if loadName.is_file():
        # open file and load data
        file = np.load(loadName, allow_pickle=True)
        Output = file['Output']
        InvestmentAll = file['InvestmentAll']
        InvestmentPerHost = file['InvestmentPerHost']
        model_par_file = file['model_par']
        file.close()
        # check if parameters have changes, if yes rerun
        rerun = check_model_par(model_par_file[0], parToIgnore)
    else:
        # cannot load, need to rerun model
        rerun = True
        print('Model data not found, running model')
    if rerun or override_data:
        # rerun model
        Output, InvestmentAll, InvestmentPerHost = run_model()
    return (Output, InvestmentAll, InvestmentPerHost)


# plot line chart
def plot_line(axs, dataStruc, FieldName):
    # plot data
    axs.plot(dataStruc['time'], dataStruc[FieldName])
    # make plot nice
    axs.set_xlabel('time [a.u.]')
    axs.set_ylabel("mean investment")
    maxY = 1
    maxX = model_par['maxT']
    xStep = 3
    yStep = 3
    axs.set_ylim((0, maxY))
    axs.set_xlim((0, maxX))
    axs.set_xticks(np.linspace(0, maxX, xStep))
    axs.set_yticks(np.linspace(0, maxY, yStep))
    return


# calcualte moving average over time
def calc_mav(data):
    num_av = int(model_par['mav_window']/model_par['sampleT'])
    dataMAV = data[-num_av:, :]
    dataMAV = np.nanmean(data, axis=0)
    return dataMAV


# plot histogram chart
def plot_histogram_line(axs, data):
    # calc moving average
    dataMav1 = calc_mav(data)
    # get bin centers
    bins = np.linspace(0, 1, dataMav1.size+1)
    x = (bins[1:] + bins[0:-1]) / 2
    # plot histogram
    axs.plot(x, dataMav1)

    # make plot nice
    maxY = 0.1
    maxX = 1
    xStep = 3
    yStep = 3
    axs.set_ylim((0, maxY))
    axs.set_xlim((0, maxX))
    axs.set_xticks(np.linspace(0, maxX, xStep))
    axs.set_yticks(np.linspace(0, maxY, yStep))
    axs.set_ylabel('frac. of hosts')
    axs.set_xlabel("average investment in host")
    return None


# main function to create figure
def create_fig():
    # load data or compute model
    Output, _, InvestmentPerHost = load_model()
    # set fonts
    fig = plt.figure()
    mlsg.set_fig_size_cm(fig, wFig, hFig)

    # plot average investment
    axs = fig.add_subplot(1, 2, 1)
    plot_line(axs, Output, "F_mav")
    axs = fig.add_subplot(1, 2, 2)
    plot_histogram_line(axs, InvestmentPerHost)
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5)

    fig.savefig(fig_Folder / figureName,
                format="pdf", transparent=True)

    return None


if __name__ == "__main__":
    create_fig()
