#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:10:02 2018

@author: simonvanvliet
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import colors
from pathlib import Path
import matplotlib.patches as patches

now = datetime.datetime.now()

saveNameMod = "FixedVariance"

data_folder = Path("Data/")
fileName = "2018-11-15_13_35"
fileName = "parScan" + fileName + ".npz"
fileName = data_folder / fileName

save_folder = Path("Figures/")
saveName = "heatmap"+now.strftime("_%Y-%m-%d_")+saveNameMod + ".pdf"
saveName = save_folder / saveName

file = np.load(fileName)

modelParList = file['modelParList']
parRange = file['parRange']
parOrder = file['parOrder']

data1D = file['data']

#parRange = [gamma_vec, tauH_vec, n0_vec, mig_vec, rr_vec, K_vec] 

ndSize = [ x.size for x in parRange]
data = np.reshape(data1D, ndSize)

#parOrder = np.array(['gamma','tauH','n0','mig','r','K'])

gammaIndex = np.asscalar(np.nonzero(parOrder=="gamma")[0])
tauIndex = np.asscalar(np.nonzero(parOrder=="tauH")[0])
n0Index = np.asscalar(np.nonzero(parOrder=="n0")[0])
migIndex = np.asscalar(np.nonzero(parOrder=="mig")[0])
rIndex = np.asscalar(np.nonzero(parOrder=="r")[0])
KIndex = np.asscalar(np.nonzero(parOrder=="K")[0])
sigmaIndex = np.asscalar(np.nonzero(parOrder=="sigma")[0])




numGamma = ndSize[gammaIndex]
numTau = ndSize[tauIndex]
numR = ndSize[rIndex]
numK = ndSize[KIndex]
numSig = ndSize[sigmaIndex]

n0V = parRange[n0Index]
migV = parRange[migIndex]

plt.rcParams.update({'font.size': 8})

def create_fig(nRow,nCol):
    fig, axs = plt.subplots(nRow, nCol)
    w = 16
    h = 12
    fig.set_size_inches(w,h)
    return fig, axs

def plot_heatmap(images,axs,data,xvec,yvec):
    cmap = "cool"

    currData = np.log10(data).transpose()
                  
    images.append(axs.imshow(currData, cmap=cmap, \
                    interpolation='nearest', \
                    extent=[xvec[0],xvec[-1],yvec[0],yvec[-1]], \
                    origin='lower'))
    
    xticks = [xvec[0], xvec[-1]]
    xtickNames = ["%.0f" %np.log10(x) for x in xticks]
    
    yticks = [yvec[0], yvec[-1]]
    ytickNames = ["%.0f" %np.log10(x) for x in yticks]
    
    axs.set_xticks(xticks)
    axs.set_yticks(yticks)
    #axs.set_title(titlename)
    
    axs.set_xlabel('log10 $N_0/K$')
    axs.set_ylabel('log10 $\\theta$')
    
    axs.set_xticklabels(xtickNames)
    axs.set_yticklabels(ytickNames)
    axs.label_outer()
    return

def make_fig_nice(fig,images):
    norm = colors.Normalize(vmin=-3, vmax=0)
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, label="log10 fraction cooperator")
    plt.draw()
    return

def addAnnotation(curAx,labels):
    if len(labels)==1:
        curAx.text(0.05, 0.5, labels[0], \
                horizontalalignment='left', \
                verticalalignment='center', \
                transform=curAx.transAxes)
    elif len(labels)==2:
        curAx.text(0.5, 0.3, labels[0], \
                horizontalalignment='center', \
                verticalalignment='bottom', \
                transform=curAx.transAxes)
        curAx.text(0.5, 0.05, labels[1], \
                horizontalalignment='center', \
                verticalalignment='bottom', \
                transform=curAx.transAxes)
        
    curAx.set_axis_off() 
    return
    
    
#%%
fig, axs = create_fig(numSig+1 , numGamma*numTau+1)
images = []
kindex = 0
rindex = 0

axs[0,numGamma*numTau].set_axis_off() 

for ss in range(numSig):
    curText = ['$\\sigma_{trans}$=%g' % parRange[sigmaIndex][ss]]
    addAnnotation(axs[ss+1, numGamma*numTau],curText)
    
    for tt in range(numTau): 
        for gg in range(numGamma):
            curR = ss + 1
            curC = tt * numGamma + gg
            
            if ss==1:
                curText = ['$\\tau_{H}$=%g' % parRange[tauIndex][tt], \
                           '$\\gamma$=%g' % parRange[gammaIndex][gg]]
                addAnnotation(axs[00, curC],curText)
                
            currData = data['F_mav_ss'][gg, tt, :, :, rindex, kindex, ss].squeeze()
            plot_heatmap(images,axs[curR, curC],currData,n0V,migV)
                        
make_fig_nice(fig,images)
plt.savefig(saveName, dpi=300, format="pdf")

#%%
#curPar = (parRange[gammaIndex][gg], parRange[tauIndex][tt])
            #titlename  = '$\gamma$:%g, $\\tau  _{H}$:%.0d' % curPar
# kindex=0
# images = []
# for tt in range(numTau):
#     for gg in range(numGamma):
#         for rr in range(numR):
#             for ss in range(numSig):
                
#                 currR = rr * numTau + tt
#                 currC = ss * numGamma + gg

#                 currData = np.log10(data['F_mav_ss'][gg, tt, :, :, rr, kindex, ss] \
#                                     .squeeze()).transpose()
                  
#                 images.append(axs[currR, currC].imshow(currData, cmap=cmap, \
#                               interpolation='nearest', \
#                               extent=[n0V[0],n0V[-1],migV[0],migV[-1]], \
#                               origin='lower'))
#                 #axs[currR, currC].set_xticks([n0V[0], n0V[-1]])
#                 #axs[currR, currC].set_yticks([migV[0], migV[-1]])
#                 axs[currR, currC].set_xticks([])
#                 axs[currR, currC].set_yticks([])
                
#                 #axs[tt, gg].set_xlabel("n0")
#                 # axs[tt, gg].set_ylabel("theta")
#                 axs[currR, currC].label_outer()

# norm = colors.Normalize(vmin=-3, vmax=0)
# for im in images:
#     im.set_norm(norm)


# fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, label="fraction cooperator")

# plt.draw()

# #fig.set_dpi = 150

# plt.savefig(saveName, dpi=300, format="pdf")


#%%
