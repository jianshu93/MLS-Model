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
figureName = 'figureSI5.pdf'
dataName = 'data_FigureSI5.npz'

# set model parameters
sb = [1, 0]
cost_vec = [-0.005, -0.001, 0, 0.001, 0.005]

maxT_vec = [5000, 10000, 50000, 10000, 5000]
#run each parset numRepeat times
numRepeat = 100

model_par = {
    # selection strength settings
    "s": 1,
    "K_H": 500.,
    "B_H": 1.,
    "D_H": 0,
    # tau_var settings
    "cost": -0.001,
    "TAU_H": 100,
    "sigmaBirth": 0.05,
    # tau_mig settings
    "n0": 1E-4,
    "mig": 1E-6,
    # init conditions
    "F0": 1E-6,
    "N0init": 1.,
    "NUMGROUP": -1,
    # time settings
    "maxT": 50000,
    "dT": 5E-2,
    "sampleT": 10,
    "rms_err_treshold": 1E-10,
    "mav_window": 100,
    "rms_window": 1000,
    # fixed model parameters
    "sampling": "fixedvar",
    "mu": 1E-9,
    "K": 1,
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
wFig = 17.8
hFig = 3.5
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

colors = ['777777', 'e24a33', '348ABD', '988ED5',
          'FBC15E', '8EBA42', 'FFB5B8']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

"""
Main code
"""

# set host birth/death effect and cost in model par
def set_BH_cost(BH, cost):
    model_par_local = model_par.copy()
    model_par_local['B_H'] = BH
    model_par_local['cost'] = cost
    return model_par_local


# run model
def run_model():
    
    #setup par list
    modelParList = []
    
    for cost in cost_vec:
        # set modelpar list to run
        model_par_wHostSel = set_BH_cost(sb[0], cost)
        model_par_woHostSel = set_BH_cost(sb[1], cost)
        #run each parset numRepeat times
        modelParListLoc = [model_par_wHostSel, model_par_woHostSel] * numRepeat
        modelParList.extend(modelParListLoc)

    # run model
    nJobs = min(len(modelParList), 4)
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mlssf.run_model_fixed_parameters)(par) for par in modelParList)

    # process and store output
    Output, _, _, _ = zip(*results)
    
    statData = np.vstack(Output)
    saveName = data_folder / dataName
    np.savez(saveName, statData=statData,
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
    parToIgnore = ('B_H', 'cost')  # need not check these parameters
    loadName = data_folder / dataName
    if loadName.is_file() and not override_data:
        # open file and load data
        data_file = np.load(loadName, allow_pickle=True)
        Output = data_file['statData']
        rerun = check_model_par(data_file['modelParList'][0], parToIgnore)
        data_file.close()
    else:
        # cannot load, need to rerun model
        rerun = True
        print('Model data not found, running model')
    if rerun or override_data:
        # rerun model
        Output = run_model()
    return Output


# plot line chart
def plot_line(axs, Output, FieldName, yAxis, maxX, legendLoc):
    
    handle_list = []
    
    for i in range(2):
        timeMat = Output[i::2]['time']    
        mavMat = Output[i::2][FieldName]
        
        timeAv = timeMat.mean(axis=0)
        mavAv = mavMat.mean(axis=0)
        
        # plot data
        alphaVal='30'
        axs.plot(timeMat.transpose(), mavMat.transpose(), 
                 linewidth=0.25, color='#'+colors[i]+alphaVal)
                 
        handle, = axs.plot(timeAv, mavAv, 
                 linewidth=1.5, color='#'+colors[i])         
                 
        handle_list.append(handle)
    
    
#    for i in range(numRepeat):
#        # plot data
#        axs.plot(dataStruc[2*i]['time'], dataStruc[2*i][FieldName], 
#                 linewidth=0.5, color=colors[0], alpha=0.5)
#        axs.plot(dataStruc[2*i+1]['time'], dataStruc[2*i+1][FieldName],
#                 '--', linewidth=0.5, color=colors[1], alpha=0.5)
    # make plot nice
    axs.set_xlabel('time [a.u.]')
    maxY = 1
    #maxX = model_par['maxT']
    xStep = 3
    yStep = 3
    axs.set_ylim((0, maxY))
    axs.set_xlim((0, maxX))
    axs.set_xticks(np.linspace(0, maxX, xStep))
    axs.set_yticks(np.linspace(0, maxY, yStep))
    #axs.legend(('$s_b$=%.0f' % model_par['s'], '$s_b$=0'), loc='center right')
    if legendLoc != 'none':
        axs.legend(handle_list, ['$s_b$=%.0f' % x for x in sb], loc=legendLoc)
    
    if yAxis:
        axs.set_ylabel("mean frac. helpers $\\langle f \\rangle$")
    else:
        axs.set_yticklabels([])
    
    return


# calcualte moving average over time
def calc_mav(data):
    dataMAV = data[-model_par['mav_window']:, :]
    dataMAV = np.nanmean(data, axis=0)
    return dataMAV

# plot histogram chart


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
    #axs.legend(('$G_H$=%.0f' % tauH, ), loc='upper right')
    axs.legend(['$s_b$=%.0f' % x for x in sb], loc='upper right')
    
    
    return None


# main function to create figure
def create_fig():
    
    # load data or compute model
    Output = load_model()
    # set fonts
    fig = plt.figure()
    mlsg.set_fig_size_cm(fig, wFig, hFig)
    
    
    numcost = len(cost_vec)
    
    numElTot = Output.shape[0]
    numElLoc = numElTot/numcost
    
    legendLoc = 'lower right'
    for i in range(numcost):
        
        if i == 0:
            yAxis = True
        else:
            yAxis = False
            
        if i>1:
            legendLoc = 'none'
        
        startP = int(i*numElLoc)
        endP = int((i+1)*numElLoc)
        currOutput = Output[startP:endP,:]
        # plot average investment
        axs = fig.add_subplot(1, 5, i+1)
        plot_line(axs, currOutput, "F_mav", yAxis, maxT_vec[i], legendLoc)
        axs.set_title('Cost $\\gamma=%.3f$' % cost_vec[i])
        
#    axs = fig.add_subplot(1, 2, 2)
#    plot_histogram_line(axs, InvPerHost)
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5)

    fig.savefig(fig_Folder / figureName,
                format="pdf", transparent=True)
    return None


if __name__ == "__main__":
    create_fig()
