#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 29 2018
@author:
Simon van Vliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ca
"""
# %%
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
import matplotlib as mpl
# %%

# calculate heritability time


def calc_tauH(n0, theta, cost):
    # define constants
    beta = 1
    k = 1
    # transition point between approximations
    theta_crit = (beta * n0) / (k - 3 * n0)
    # approximation for theta > theta_crit
    lown = np.log((k * theta) / ((k + n0) * theta -
                                 n0 * beta)) / (beta - theta)
    # approximation for n0 > n0_crit
    highn1 = np.log(k * (beta + theta) /
                    (2 * (n0 * (beta - theta) + k * theta)))
    highn2 = np.log((n0 * (beta - theta) + k * theta) /
                    (2 * (n0 * (beta + theta))))
    highn = highn1 / (beta - theta) - highn2 / theta
    # select right approximation tau_H
    lownMask = theta > theta_crit
    tauH = highn
    tauH[lownMask] = lown[lownMask]
    return tauH

# calculate varitaion maintainance time


def calc_tauV(cost):
    tauVar = 1. / cost
    return tauVar


# create tauH spacingVec
def mig_from_tauH(desiredTauH, n0):

    n0theta = np.zeros(desiredTauH.size)

    n0thetaTry = np.logspace(-6, 6, 1E6)
    tauHVec = calc_tauH(n0, n0*n0thetaTry, 0)

    for ii in range(desiredTauH.size):
        idx = np.argmin(np.abs(tauHVec-desiredTauH[ii]))
        n0theta[ii] = n0thetaTry[idx]

    return n0theta


# create spacing by subsequent division of left over interval
def divSpace(center, dev, minDist, factor=0.5):

    numstep = int(np.ceil(np.log(minDist/dev) / np.log(factor)))+1

    out = np.zeros(2*numstep+1)

    out[0] = center - dev
    out[-1] = center + dev
    out[numstep] = center

    for ii in range(1, numstep):
        out[ii] = center - dev*factor**ii
        out[-ii-1] = center + dev*factor**ii

    return out


# loads data and saves data and metadata in dictionary
def load_process_data(fileName):
    # load data
    file = np.load(fileName, allow_pickle=True)
    modelParList = file['modelParList']
    parRange = file['parRange']
    parOrder = file['parOrder']
    endStat_1D = file['data']
    if 'distData' in file:
        distData = file['distData']
    else:
        distData = None
    file.close()

    if 'cost' in endStat_1D.dtype.names:
        costName = 'cost'
    else:
        costName = 'gamma'

    # calculate heritability time
    tauHer = calc_tauH(endStat_1D['n0'], endStat_1D['mig'],
                       endStat_1D[costName])
    endStat_1D = rf.append_fields(endStat_1D, 'tauHer',
                                  np.squeeze(tauHer), usemask=False)

    # calculate variation maintainance time
    tauVar = calc_tauV(endStat_1D[costName])
    endStat_1D = rf.append_fields(endStat_1D, 'tauVar',
                                  np.squeeze(tauVar), usemask=False)

    # convert to N-D matrix
    ndSize = [x.size for x in parRange]
    endStat_ND = np.reshape(endStat_1D, ndSize)

    # get location of variables
    cost_idx = np.asscalar(np.nonzero(parOrder == costName)[0])
    tau_idx = np.asscalar(np.nonzero(parOrder == "tauH")[0])
    n0_idx = np.asscalar(np.nonzero(parOrder == "n0")[0])
    mig_idx = np.asscalar(np.nonzero(parOrder == "mig")[0])
    K_idx = np.asscalar(np.nonzero(parOrder == "K")[0])
    sigmaB_idx = np.asscalar(np.nonzero(parOrder == "sigma")[0])

    if 'B_H' in parOrder:
        BH_idx = np.asscalar(np.nonzero(parOrder == "B_H")[0])
        BH_n = ndSize[BH_idx]
    else:
        BH_idx = None
        BH_n = 0

    # save data and metadata
    data_file = {
        "cost_idx": cost_idx,
        "tau_idx": tau_idx,
        "n0_idx": n0_idx,
        "mig_idx": mig_idx,
        "K_idx": K_idx,
        "sigmaB_idx": sigmaB_idx,
        "BH_idx": BH_idx,
        "cost_n ": ndSize[cost_idx],
        "tau_n": ndSize[tau_idx],
        "K_N": ndSize[K_idx],
        "sigmaB_n": ndSize[sigmaB_idx],
        "BH_n": BH_n,
        "modelParList": modelParList,
        "parRange": parRange,
        "endStat_1D": endStat_1D,
        "endStat_ND": endStat_ND,
        "distData": distData
    }
    return data_file

# convert value with continuos categorial states to labeled categorial states


def make_categorial(vector):
    elements = np.unique(vector)
    indexVec = np.arange(elements.size)
    cat_vector = np.zeros(vector.size)
    for idx in range(vector.size):
        cat_vector[idx] = indexVec[elements == vector[idx]]

    return cat_vector

# extract data range of data


def data_range(data1D, precision=1):
    minData = np.floor(np.nanmin(data1D)/precision)*precision
    maxData = np.ceil(np.nanmax(data1D)/precision)*precision
    dataRange = (minData, maxData)
    return dataRange


# extract tick locations
def data_ticks(dataRange, step):
    nNeg = np.ceil((0-dataRange[0]) / step)
    nPos = np.ceil(dataRange[1] / step)
    ticks = np.linspace(-nNeg*step, nPos*step, (nNeg+nPos+1))

    return ticks


# %% Code below needs updating and checking
def create_fig(nRow, nCol):
    font = {'family': 'Arial',
            'weight': 'light',
            'size': 10}

    mpl.rc('font', **font)

    fig, axs = plt.subplots(nRow, nCol)
    w = 14
    h = 6
    fig.set_figwidth(w, forward=True)
    fig.set_figheight(h, forward=True)
    return fig, axs


def plot_heatmap(images, axs, data, xvec, yvec):
    cmap = "cool"

    currData = np.log10(data).transpose()

    axl = axs.imshow(currData, cmap=cmap,
                     interpolation='nearest',
                     extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
                     origin='lower',
                     vmin=-3, vmax=0)

    xticks = [xvec[0], xvec[-1]]
    xtickNames = ["%.0f" % np.log10(x) for x in xticks]

    yticks = [yvec[0], yvec[-1]]
    ytickNames = ["%.0f" % np.log10(x) for x in yticks]

    axs.set_xticks(xticks)
    axs.set_yticks(yticks)

    axs.set_xlabel('log10 $N_0/K$')
    axs.set_ylabel('log10 $\\theta$')

    axs.set_xticklabels(xtickNames)
    axs.set_yticklabels(ytickNames)
    axs.label_outer()

    axs.set_aspect('equal')
    return axl


def addAnnotation(curAx, labels):
    if len(labels) == 1:
        curAx.text(0.05, 0.5, labels[0],
                   horizontalalignment='left',
                   verticalalignment='center',
                   transform=curAx.transAxes)
    elif len(labels) == 2:
        curAx.text(0.7, 0.5, labels[0],
                   horizontalalignment='center',
                   verticalalignment='bottom',
                   transform=curAx.transAxes)
        curAx.text(0.7, 0, labels[1],
                   horizontalalignment='center',
                   verticalalignment='bottom',
                   transform=curAx.transAxes)

    curAx.set_axis_off()
    return


def create_fig_keynote(nr, nc):
    font = {'family': 'Arial',
            'weight': 'light',
            'size': 28}

    mpl.rc('font', **font)

    fig, axs = plt.subplots(nr, nc)
    mydpi = 150
    w = 1800
    h = 1000
    fig.set_size_inches(w/mydpi, h/mydpi)

    return (fig, axs)
