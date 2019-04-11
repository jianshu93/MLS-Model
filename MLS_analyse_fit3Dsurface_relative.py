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
import MLS_analyse_fit3Dsurface as mlsfit
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

fileNameB0 = "parScan_fixedVar_timeScalePlot20190405_14h28.npz"
fileName = "parScan_fixedVar_timeScalePlot20190403_18h52"
# name of figure to save
saveNameMod = "3D_Plot_BH_sigma"
# set folders
data_folder = Path("Data/")
save_folder = Path("Figures/")

# create filenames to load and save
fileName = fileName + ".npz"
fileName = data_folder / fileName
fileNameB0 = data_folder / fileNameB0

now = datetime.datetime.now()
saveName3 = "3D_Relative_Heat_BH_sigma" + now.strftime("_%Y-%m-%d_") + ".pdf"

# get data
data_file = mlspg.load_process_data(fileName)
data1D = data_file['endStat_1D']

# get data no community selection
data_file_B0 = mlspg.load_process_data(fileNameB0)
data1D_B0 = data_file_B0['endStat_1D']

# set figure settings
fig3 = plt.figure()

mydpi = 150

font = {'family': 'Arial',
        'weight': 'light',
        'size': 9}
mpl.rc('font', **font)


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


# process B0 data
# remove nan and inf
xDataB = np.log10(data1D_B0['tauVar'] / data1D_B0['tau_H'])
yDataB = np.log10(data1D_B0['tauHer'] / data1D_B0['tau_H'])
zDataB = data1D_B0['F_mav']

colDataB = mlspg.make_categorial(data1D_B0['sigmaBirth'])


# setup data
# plot relative time scale of variation maintenance and heritability to host time scale
xData = np.log10(data1D['tauVar'] / data1D['tau_H'])
yData = np.log10(data1D['tauHer'] / data1D['tau_H'])
zData = data1D['F_mav']

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

w = 2.5 * nCol
h = 2 * nRow
fig3.set_size_inches(w, h)

for rr in range(nRow):
    for cc in range(nCol):
        plotIdx = rr*nCol + cc + 1

        # process B=0 data
        currColB = colDataB == cc
        isFinite = np.logical_and.reduce(
            np.isfinite((xDataB, yDataB, zDataB)))
        currSubset = np.logical_and.reduce((currColB, isFinite))

        # extract subset of data
        currXDataB = xDataB[currSubset]
        currYDataB = yDataB[currSubset]
        currZDataB = zDataB[currSubset]

        binnedDataB0 = mlsfit.bin_2Ddata(
            currXDataB, currYDataB, currZDataB, xbins, ybins)

        # process B>0 data
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

        binnedData = mlsfit.bin_2Ddata(
            currXData, currYData, currZData, xbins, ybins)

        relData = binnedData / binnedDataB0

        # plot bins
        ax = fig3.add_subplot(nRow, nCol, plotIdx)

        im = ax.pcolormesh(xbins, ybins, np.log2(relData),
                           vmin=0, vmax=3, cmap='magma')
        mlsfit.make2daxis(ax, xLim, yLim, stepSize)
        ax.set_title("$%s=%.1f, %s=%.2f$" %
                     (rowVar, rowName[rr], colVar, colName[cc]), fontsize=9)

        fig3.colorbar(im, ax=ax)


# clean up plot and save
fig3.canvas.draw_idle()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
fig3.savefig(save_folder / saveName3, dpi=mydpi,
             format="pdf", transparent=True)
