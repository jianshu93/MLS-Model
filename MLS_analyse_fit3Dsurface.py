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
from scipy import interpolate
from scipy.interpolate import griddata
import datetime
from pathlib import Path
from matplotlib import collections as matcoll

#name of data to load
fileName = "parScan_fixedVar_timeScalePlot20190402_15h14"
#name of figure to save
saveNameMod = "3D_Plot_BH_sigma_alt"
#set folders
data_folder = Path("Data/")
save_folder = Path("Figures/")

#create filenames to load and save
fileName = fileName + ".npz"
fileName = data_folder / fileName
now = datetime.datetime.now()
saveName = saveNameMod + now.strftime("_%Y-%m-%d_") + ".pdf"
saveName = save_folder / saveName

save2 = "temp.pdf"
save2 = save_folder / save2

#get data
data_file = mlspg.load_process_data(fileName)
data1D = data_file['endStat_1D']

#set figure settings
fig = plt.figure()
w = 2400
h = 600
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


def make3daxis(ax):
    #set range
    xStep = 1.
    yStep = 2.
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

for ss in range(1):
        #get subset of data to plot
        
        currSig = mlspg.make_categorial(data1D['sigmaBirth']) == 0
        currBH = mlspg.make_categorial(data1D['B_H']) == 0
        
        
        isFinite = np.logical_and.reduce(np.isfinite((xData, yData, zData, cData)))
        
        currSubset = np.logical_and.reduce((currSig, currBH, isFinite))
        
        
        
        #extract subset of data
        currXData = xData[currSubset]
        currYData = yData[currSubset]
        currZData = zData[currSubset]
        currCData = cData[currSubset]
        
        
#        #create 2d axis
#        ax = fig.add_subplot(2, 3, 2)
#        #plot data
#        ai = ax.scatter(currXData , currYData, c='b', \
#                s=10, alpha=0.5)
#        
#        
        
       
        
        xLim = (-2, 2)
        yLim = (-4, 4)
        xStep = 0.5
        yStep = 0.5
        
        xbins = np.linspace(xLim[0], xLim[1], int(np.ceil((xLim[1]-xLim[0])/xStep))+1)
        ybins = np.linspace(yLim[0], yLim[1], int(np.ceil((yLim[1]-yLim[0])/yStep))+1)
        
        nX = xbins.size
        nY = ybins.size
        
        
        xnew, ynew = np.meshgrid(xbins, ybins)
#        tck = interpolate.bisplrep(currXData, currYData, currZData, s=100)
#        znewS = interpolate.bisplev(xbins, ybins, tck)
#        
        
        znew = griddata((currXData, currYData), currZData, (xnew, ynew), method='linear', rescale=True)
        
                
        binnedData = np.full((nY,nX), np.nan)
        
        for xx in range(xbins.size - 1) :
            for  yy in range(ybins.size - 1) :
                inXBin = np.logical_and((currXData >=  xbins[xx]), (currXData <  xbins[xx+1]))
                inYBin = np.logical_and((currYData >=  ybins[yy]), (currYData <  ybins[yy+1]))
                inBin =  np.logical_and(inXBin, inYBin)
                
                zInBin = currZData[inBin]
                
                binnedData[yy, xx] = np.nanmean(zInBin)

        
        
        
        
        
        #create 3d axis
        ax = fig.add_subplot(1, 4, 1, projection='3d')
        #plot data
        ai = ax.scatter(currXData , currYData , currZData, c=currZData, \
                s=2, alpha=0.7, vmin=0, vmax=1, cmap='plasma')
        
#        lines = []
#        for i in range(len(currZData)):
##            pair=[(currXData[i],currYData[i],0), (currXData[i],currYData[i],currZData[i])]
##            lines.append(pair)
#             ax.plot([currXData[i],currXData[i]], [currYData[i],currYData[i]], [0, currZData[i]], \
#                     c=[0.15,0.15,0.15,0.3],linewidth=0.3)
#          
        
        make3daxis(ax)

        
        
        #create 3d axis
        ax = fig.add_subplot(1, 4, 2, projection='3d')
        #plot data
        ai = ax.scatter(currXData , currYData , currZData, c=[0.15,0.15,0.15,0.6], \
                s=3, alpha=0.6, vmin=0, vmax=1, cmap='plasma')
        
#        ax.plot_wireframe(xnew, ynew, znew)
        ax.plot_surface(xnew, ynew, znew, rstride=1, cstride=1,
                cmap='plasma', edgecolor=[0.15,0.15,0.15,0.5], vmin=0, vmax=1, alpha=0.5)
        
      
            
#        linecoll = matcoll.LineCollection(lines)
#        ax.add_collection(linecoll)   
#            
        
        make3daxis(ax)

#        #create 2d axis
#        ax = fig.add_subplot(2, 3, 2)
#        #plot data
#        ai = ax.scatter(currXData , currYData, c='b', \
#                s=10, alpha=0.5)
#        
       
#        ax = fig.add_subplot(2, 3, 4)
#        ax.pcolormesh(xnew, ynew, znewCubic)
#        plt.title('bicubic')
#        
        ax = fig.add_subplot(2, 4, 3)
        ax.pcolormesh(xnew, ynew, znew, cmap='plasma')
        plt.title('linear interp')
        
        ax = fig.add_subplot(2, 4, 4)
        ax.pcolormesh(xnew, ynew, np.log10(znew), cmap='plasma')
        plt.title('linear interp log10')


        ax = fig.add_subplot(2, 4, 7)
        ax.pcolormesh(xnew, ynew, binnedData, cmap='plasma')
        plt.title('binned')
        
        ax = fig.add_subplot(2, 4, 8)
        ax.pcolormesh(xnew, ynew, np.log10(binnedData), cmap='plasma')
        plt.title('binned')
        
       
        



#clean up plot and save
#plt.tight_layout()
plt.draw()
plt.savefig(saveName, dpi=mydpi, format="pdf", transparent=True)


#%%

fig = plt.figure()
w = 1200
h = 1200
font = {'family' : 'Arial',
        'weight' : 'light',
        'size'   : 10}
mpl.rc('font', **font)
mydpi=150
fig.set_size_inches(w/mydpi,h/mydpi)

ax = fig.add_subplot(2, 3, 2)
plt.scatter(currXData , currYData, c='b', s=2, alpha=0.5)

plt.savefig(save2, dpi=mydpi, format="pdf", transparent=True)
