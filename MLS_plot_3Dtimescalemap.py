#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 29 2018
@author: 
Simon van Vliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ca

Make 3D scatter plot of fraction of cooperators as function
of hertitability time and time of variance maintainance 

"""
#%%
import MLS_plot_general_code as mlspg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d    
import datetime
from pathlib import Path

#name of data to load
fileName = "parScan_fixedVar_scanAll20190329_23h52_"
fileName = "parScan_fixedVar_timeScalePlot20190331_17h27"
#name of figure to save
saveNameMod = "3D_Plot_HR"
#set folders
data_folder = Path("Data/")
save_folder = Path("Figures/")

#create filenames to load and save
fileName = fileName + ".npz"
fileName = data_folder / fileName
now = datetime.datetime.now()
saveName = saveNameMod + now.strftime("_%Y-%m-%d_") + ".pdf"
saveName = save_folder / saveName

#get data
data_file = mlspg.load_process_data(fileName)
data1D = data_file['endStat_1D']

#set figure settings
fig = plt.figure()
w = 4000
h = 1600
font = {'family' : 'Arial',
        'weight' : 'light',
        'size'   : 10}
mpl.rc('font', **font)
mydpi=150
fig.set_size_inches(w/mydpi,h/mydpi)

#setup data
#plot relative time scale of variation maintanance and heritability to host time scale
xData = np.log10(data1D['tauVar'] / data1D['tau_H'] )
yData = np.log10(data1D['tauHer'] / data1D['tau_H'] )
zData = data1D['F_mav']
##variable to use to color data points
#cData = mlspg.make_categorial(data1D['sigmaBirth'])
#numColor = data_file['sigmaB_n']
#cDataLabel = data_file['parRange'][data_file['sigmaB_idx']]
#cLabelName = "$\\sigma_f$"
##variable to use for different subplots
#subplotData = mlspg.make_categorial(data1D['B_H'])
#numSubplot = data_file['BH_n']
#titleLabel = data_file['parRange'][data_file['BH_idx']]


#variable to use to color data points
subplotData = mlspg.make_categorial(data1D['sigmaBirth'])
numSubplot = data_file['sigmaB_n']
titleLabel = data_file['parRange'][data_file['sigmaB_idx']]
titleVar = '\\sigma_B'
#variable to use for different subplots
cData = mlspg.make_categorial(data1D['B_H'])
numColor = data_file['BH_n']
cDataLabel = data_file['parRange'][data_file['BH_idx']]
cLabelName = "$B_H$"

cMap = mpl.cm.get_cmap('RdYlBu', numColor)


titleLabel.sort()
cDataLabel.sort()

for ss in range(numSubplot):
        #get subset of data to plot
        currSubset = subplotData == ss
        #extract subset of data
        currXData = xData[currSubset]
        currYData = yData[currSubset]
        currZData = zData[currSubset]
        currCData = cData[currSubset]
        #create 3d axis
        ax = fig.add_subplot(2, np.ceil(numSubplot/2), ss+1, projection='3d')
        #plot data
        ai = ax.scatter(currXData , currYData , currZData, c=currCData, \
                s=40, alpha=0.6, vmin=0, vmax=numColor-1, cmap=cMap)
        #set range
        xStep = 1.
        yStep = 4.
        zStep = 0.2
        
        xRange = mlspg.data_range(currXData, precision=xStep)
        yRange = mlspg.data_range(currYData, precision=yStep)
        zRange = mlspg.data_range(currZData, precision=zStep)
        ax.set_xlim(xRange)
        ax.set_ylim(yRange)
        ax.set_zlim(zRange)
        #set labels
        ax.set_xlabel('$log_{10} \\tau_V / \\tau_R$')
        ax.set_ylabel('$log_{10} \\tau_H / \\tau_R$')
        ax.set_zlabel('fraction cooperator')
        titleName = '$' + titleVar + '=$' + '%.3f' % titleLabel[ss]
        plt.title(titleName)
        #set ticks
        ax.set_xticks(mlspg.data_ticks(xRange, xStep))
        ax.set_yticks(mlspg.data_ticks(yRange, yStep))
        ax.set_zticks(mlspg.data_ticks(zRange, zStep))
        #set view options
        ax.view_init(20,-115)
        #add spacing between axis and labels
        ax.yaxis.labelpad=10
        ax.xaxis.labelpad=10
        ax.zaxis.labelpad=10
        ax.tick_params(axis='z', which='major', pad=10)
        #set colorbar options
        dC = 1 / (numColor + 1)
        dCSt = dC/2 * (numColor-1)
        dCEnd = (1-dC/2) * (numColor-1)
        cTick = np.linspace(dCSt,dCEnd,numColor)
        
        cbx = fig.colorbar(ai, ax=ax, orientation='vertical', \
                shrink=.5, label=cLabelName,\
                ticks=cTick)

        cbx.ax.set_yticklabels(cDataLabel)
        cbx.solids.set(alpha=1)
        



#clean up plot and save
plt.tight_layout()
plt.draw()
plt.savefig(saveName, dpi=mydpi, format="pdf", transparent=True)
