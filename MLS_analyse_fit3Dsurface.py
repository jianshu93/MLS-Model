#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 29 2018
@author:
Simon van Vliet
Department of Zoology
University of British Columbia
vanvliet@zoology.ca

Make 3D scatter plot of fraction of cooperators as function
of heritability time and time of variance maintenance

"""
# %%
import MLS_plot_general_code as mlspg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from scipy import interpolate
from scipy.interpolate import griddata
import datetime
from pathlib import Path
from matplotlib import collections as matcoll

# name of data to load

fileName = "parScan_fixedVar_timeScalePlot20190405_14h28"
#fileName = "parScan_fixedVar_timeScalePlot20190403_18h52"
# name of figure to save
saveNameMod = "3D_Plot_BH_sigma"
# set folders
data_folder = Path("Data/")
save_folder = Path("Figures/")

# create filenames to load and save
fileName = fileName + ".npz"
fileName = data_folder / fileName
now = datetime.datetime.now()
saveName = "3D_Surf_BH_sigma" + now.strftime("_%Y-%m-%d_") + ".pdf"
saveName2 = "3D_Scat_BH_sigma" + now.strftime("_%Y-%m-%d_") + ".pdf"
saveName3 = "3D_Heat_BH_sigma" + now.strftime("_%Y-%m-%d_") + ".pdf"


# get data
data_file = mlspg.load_process_data(fileName)
data1D = data_file['endStat_1D']

# set figure settings
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()


mydpi = 150

font = {'family': 'Arial',
        'weight': 'light',
        'size': 9}
mpl.rc('font', **font)



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


def fitsurface_2Ddata(currXData, currYData, currZData, xbins, ybins):
    xFit, yFit = np.meshgrid(xbins, ybins)
    zFit = griddata((currXData, currYData), currZData,
                    (xFit, yFit), method='linear', rescale=True)
    return(xFit, yFit, zFit)


def make3daxis(ax, xLim, yLim, zLim, stepsize):
    # set range
    xStep = stepsize[0]
    yStep = stepsize[1]
    zStep = stepsize[2]

    ax.set_xlim(xLim)
    ax.set_ylim(yLim)
    ax.set_zlim(zLim)
    # set labels
    ax.set_xlabel('$log_{10} \\tau_V / \\tau_R$')
    ax.set_ylabel('$log_{10} \\tau_H / \\tau_R$')
    ax.set_zlabel('f(helper)')
    # set ticks
    ax.set_xticks(mlspg.data_ticks(xLim, xStep))
    ax.set_yticks(mlspg.data_ticks(yLim, yStep))
    ax.set_zticks(mlspg.data_ticks(zLim, zStep))
    # set view options
    ax.view_init(20, -115)
    # add spacing between axis and labels
    ax.yaxis.labelpad = 0
    ax.xaxis.labelpad = 0
    ax.zaxis.labelpad = 0
    ax.tick_params(axis='z', which='major', pad=0)
    return None


def make2daxis(ax, xLim, yLim, stepsize):
    # set range
    xStep = stepsize[0]
    yStep = stepsize[1]
    ax.set_xlim(xLim)
    ax.set_ylim(yLim)
    # set labels
    ax.set_xlabel('$log_{10} \\tau_V / \\tau_R$')
    ax.set_ylabel('$log_{10} \\tau_H / \\tau_R$')
    # set ticks
    ax.set_xticks(mlspg.data_ticks(xLim, xStep))
    ax.set_yticks(mlspg.data_ticks(yLim, yStep))
    return None


# setup data
# plot relative time scale of variation maintenance and heritability to host time scale
xData = np.log10(data1D['tauVar'] / data1D['tau_H'])
yData = np.log10(data1D['tauHer'] / data1D['tau_H'])
zData = data1D['F_mav']

# setup bin vectors
xLim = (-2, 2)
yLim = (-4, 6)
zLim = (0, 1)
stepSize = (2, 4, 0.5)

xStep = 0.2
yStep = 0.5
xbins = np.linspace(xLim[0], xLim[1], int(
    np.ceil((xLim[1]-xLim[0])/xStep))+1)
ybins = np.linspace(yLim[0], yLim[1], int(
    np.ceil((yLim[1] - yLim[0]) / yStep)) + 1)

# variable to use to color data points
colData = mlspg.make_categorial(data1D['sigmaBirth'])
nCol = data_file['sigmaB_n']
colName = data_file['parRange'][data_file['sigmaB_idx']]
colName.sort()
colVar = '\\sigma_B'
# variable to use for different subplots
rowData = mlspg.make_categorial(data1D['B_H'])
nRow = data_file['BH_n']
rowName = data_file['parRange'][data_file['BH_idx']]
rowName.sort()
rowVar = 'B_H'

w = 2.5 *nCol
h = 2 * nRow
fig1.set_size_inches(w, h)
fig2.set_size_inches(w, h)
fig3.set_size_inches(w, h)

for rr in range(nRow):
    for cc in range(nCol):
        plotIdx = rr*nCol + cc + 1
        # get subset of data to plot
        currCol = colData == cc
        currRow = rowData == rr

        # remove nan and inf
        isFinite = np.logical_and.reduce(
            np.isfinite((xData, yData, zData, rowData, colData)))
        currSubset = np.logical_and.reduce((currCol, currRow, isFinite))

        # extract subset of data
        currXData = xData[currSubset]
        currYData = yData[currSubset]
        currZData = zData[currSubset]
        currCData = rowData[currSubset]

        binnedData = bin_2Ddata(currXData, currYData, currZData, xbins, ybins)
        xFit, yFit, zFit = fitsurface_2Ddata(
            currXData, currYData, currZData, xbins, ybins)

        # plot data, scatter + surface
        ax = fig1.add_subplot(nRow, nCol, plotIdx, projection='3d')
        ax.scatter(currXData, currYData, currZData,
                   c=currZData,
                   s=1, alpha=0.4,
                   vmin=0, vmax=1, cmap='plasma')
        ax.plot_surface(xFit, yFit, zFit,
                        rstride=1, cstride=1,
                        edgecolor=[0.15, 0.15, 0.15, 0.5], alpha=0.7,
                        vmin=0, vmax=1, cmap='plasma')

        ax.set_title("$%s=%.1f, %s=%.2f$" %
                     (rowVar, rowName[rr], colVar, colName[cc]), fontsize=9)

        make3daxis(ax, xLim, yLim, zLim, stepSize)

        # plot data, scatter only
        ax = fig2.add_subplot(nRow, nCol, plotIdx, projection='3d')
        ax.scatter(currXData, currYData, currZData,
                   c=currZData,
                   s=2, alpha=0.7,
                   vmin=0, vmax=1, cmap='plasma')

        ax.set_title("$%s=%.1f, %s=%.2f$" %
                     (rowVar, rowName[rr], colVar, colName[cc]), fontsize=9)

        make3daxis(ax, xLim, yLim, zLim, stepSize)
    #        for i in range(len(currZData)):
    #             ax.plot([currXData[i],currXData[i]], [currYData[i],currYData[i]], [0, currZData[i]], \
    #                     c=[0.15,0.15,0.15,0.3],linewidth=0.3)

        ax = fig3.add_subplot(nRow, nCol, plotIdx)
        ax.pcolormesh(xFit, yFit, binnedData, vmin=0, vmax=1, cmap='plasma')
        make2daxis(ax, xLim, yLim, stepSize)
        ax.set_title("$%s=%.1f, %s=%.2f$" %
                     (rowVar, rowName[rr], colVar, colName[cc]), fontsize=9)


# clean up plot and save
fig1.canvas.draw_idle()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
fig1.savefig(save_folder / saveName, dpi=mydpi,
             format="pdf", transparent=True)


# clean up plot and save
fig2.canvas.draw_idle()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
fig2.savefig(save_folder / saveName2, dpi=mydpi,
             format="pdf", transparent=True)

# clean up plot and save
fig3.canvas.draw_idle()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
fig3.savefig(save_folder / saveName3, dpi=mydpi,
             format="pdf", transparent=True)
