#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:10:02 2018

@author: simonvanvliet
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colr
import matplotlib as mpl
import itertools
from mpl_toolkits.mplot3d import axes3d    


import datetime
#from matplotlib import colors
from pathlib import Path

now = datetime.datetime.now()

saveNameMod = "FixedVariance"

data_folder = Path("Data/")
#fileName = "20181120_16h32_fixedcv"
#fileName = "20181115_20h15_fixedvar"
fileName1 = "20190319_10h40_fixedvar"
fileName2 = "20190319_15h14_fixedvar"


save_folder = Path("Figures/")
saveName = "heatmap"+now.strftime("_%Y-%m-%d_")+saveNameMod + ".pdf"
saveName = save_folder / saveName


#load file 1
fileName = "parScan_" + fileName1 + ".npz"
fileName = data_folder / fileName

file = np.load(fileName, allow_pickle=True)
modelParList1 = file['modelParList']
parRange = file['parRange']
parOrder = file['parOrder']

data1D1 = file['data']

index1 = 0*np.ones((data1D1.size,1))

#load file 2
fileName = "parScan_" + fileName2 + ".npz"
fileName = data_folder / fileName

file = np.load(fileName, allow_pickle=True)
modelParList2 = file['modelParList']
parRange2 = file['parRange']
parOrder2 = file['parOrder']
data1D2 = file['data']

index2 = 1*np.ones((data1D2.size,1))


data1D = np.concatenate((data1D1, data1D2), axis=0)
index  = np.concatenate((index1, index2), axis=0)
modelParList  = np.concatenate((modelParList1, modelParList2), axis=0)


#%%




#parRange = [gamma_vec, tauH_vec, n0_vec, mig_vec, rr_vec, K_vec] 

ndSize = [ x.size for x in parRange]
ndSize = np.concatenate((ndSize, [2]),axis=0)
data = np.reshape(data1D, ndSize)

#parOrder = np.array(['gamma','tauH','n0','mig','r','K'])

gammaIndex = np.asscalar(np.nonzero(parOrder=="gamma")[0])
tauIndex = np.asscalar(np.nonzero(parOrder=="tauH")[0])
n0Index = np.asscalar(np.nonzero(parOrder=="n0")[0])
migIndex = np.asscalar(np.nonzero(parOrder=="mig")[0])
rIndex = np.asscalar(np.nonzero(parOrder=="r")[0])
KIndex = np.asscalar(np.nonzero(parOrder=="K")[0])
sigmaIndex = np.asscalar(np.nonzero(parOrder=="sigma")[0])


sigmaL = [x[sigmaIndex] for x in itertools.product(*parRange)]
sigmaL = np.asarray(sigmaL)

numGamma = ndSize[gammaIndex]
numTau = ndSize[tauIndex]
numR = ndSize[rIndex]
numK = ndSize[KIndex]
numSig = ndSize[sigmaIndex]

n0V = parRange[n0Index]
migV = parRange[migIndex]

plt.rcParams.update({'font.size': 8})


#%%

def make_categorial(vector):
    elements =  np.unique(vector)
    index = np.arange(elements.size)
    cat_vector = np.zeros(vector.size)
    
    for idx in range(vector.size):
        cat_vector[idx]=index[elements==vector[idx]]
        
    return cat_vector 


def create_fig(nRow,nCol):
    fig, axs = plt.subplots(nRow, nCol)
    w = 14
    h = 6
    fig.set_figwidth(w, forward=True)
    #fig.set_size_inches(w,h)
    return fig, axs

def plot_heatmap(images,axs,data,xvec,yvec):
    cmap = "cool"

    currData = np.log10(data).transpose()
                  
    axl = axs.imshow(currData, cmap=cmap, \
                    interpolation='nearest', \
                    extent=[xvec[0],xvec[-1],yvec[0],yvec[-1]], \
                    origin='lower', \
                    vmin=-3, vmax=0)
    
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
    
    axs.set_aspect('equal')
    return axl


def addAnnotation(curAx,labels):
    if len(labels)==1:
        curAx.text(0.05, 0.5, labels[0], \
                horizontalalignment='left', \
                verticalalignment='center', \
                transform=curAx.transAxes)
    elif len(labels)==2:
        curAx.text(0.7, 0.5, labels[0], \
                horizontalalignment='center', \
                verticalalignment='bottom', \
                transform=curAx.transAxes)
        curAx.text(0.7, 0, labels[1], \
                horizontalalignment='center', \
                verticalalignment='bottom', \
                transform=curAx.transAxes)
        
    curAx.set_axis_off() 
    return
    
  
 

def create_fig_keynote(nr,nc):
    font = {'family' : 'Arial',
            'weight' : 'light',
            'size'   : 28}
    
    mpl.rc('font', **font)
    
    fig,axs = plt.subplots(nr,nc)
    mydpi=150
    w = 1800
    h = 1000
    fig.set_size_inches(w/mydpi,h/mydpi)


    return (fig,axs)

def calc_tauH(n0, theta, gamma, r):
    K = 1-gamma
    alpha = r*(1-gamma) - theta

    
    lown = np.log( K * theta / (K*theta - n0*alpha)) / alpha
    highn1 = np.log( K*(alpha+2*theta) / (2*(n0*alpha+K*theta)) ) / alpha
    highn2 = np.log( 2*n0*(alpha+2*theta) / (n0*alpha+K*theta) ) / theta
    
    #highn = (1 / theta) * np.log( 2*n0 / (theta * (1 - 2 * n0 - theta) + n0)) 
    highn = highn1 + highn2
    
    trn = K*theta / (alpha+4*theta)
    lownMask = n0 < trn
    
    tauH = highn 
    tauH[lownMask] = lown[lownMask]
    
    return tauH

#%%
    
indexCat = make_categorial(index)
sigmaCat = make_categorial(sigmaL)
tauVar = 1. / (data1D['gamma']*data1D['r'])
tauHer = calc_tauH(data1D['n0'], data1D['mig'], data1D['gamma'], data1D['r']) 

minTau = np.minimum(tauVar, tauHer)


#%% plot 3D 
saveName = "3D_plot2_tauH_TauV"+now.strftime("_%Y-%m-%d_")+saveNameMod + ".pdf"
saveName = save_folder / saveName

numSig =parRange[sigmaIndex].size
viridis7 = mpl.cm.get_cmap('RdYlBu', 2)


font = {'family' : 'Arial',
            'weight' : 'light',
            'size'   : 28}
    
mpl.rc('font', **font)

fig = plt.figure()
mydpi=150
w = 2200
h = 1600
fig.set_size_inches(w/mydpi,h/mydpi)



ax = fig.add_subplot(111, projection='3d')

tauVarRel = tauVar / data1D['tau_H'] 
tauHerRel = tauHer / data1D['tau_H']  

print(tauVarRel.min(), tauVarRel.max())


xData = np.log10(tauVarRel)
yData = np.log10(tauHerRel)
zData = data1D['F_mav_ss']

xRange = (np.floor(xData.min()), np.ceil(xData.max()))
yRange = (np.floor(yData.min()), np.ceil(yData.max()))

ai = ax.scatter(xData , yData ,zData , c=indexCat, s=60,\
                alpha=0.6, vmin=0, vmax=1, cmap=viridis7)
plt.xlim(xRange)
plt.ylim(yRange)

ax.set_xlabel('$log_{10} \\tau_V / \\tau_R$')
ax.set_ylabel('$log_{10} \\tau_H / \\tau_R$')

ax.set_xticks([-2, -1, 0, 1])
ax.set_yticks([-8, -4, 0, 4, 8])
ax.set_zticks([-0,0.2,0.4,0.6,0.8])
ax.set_zlim([0, 0.8])

ax.set_zlabel('fraction cooperator')
ax.view_init(20,-115)

ax.yaxis.labelpad=20
ax.xaxis.labelpad=20
ax.zaxis.labelpad=40
ax.tick_params(axis='z', which='major', pad=15)


numData = 2;

dC = (numData-1)/numData
dCSt = dC/2
dCEnd = (numData-1)-dC/2
cTick = np.linspace(dCSt,dCEnd,numData)
cbx = fig.colorbar(ai, ax=ax, orientation='vertical', \
             shrink=.5, label="Dat set",\
             ticks=cTick)

cbx.ax.set_yticklabels([1, 2])
cbx.solids.set(alpha=1)

plt.tight_layout()
plt.draw()


plt.savefig(saveName, dpi=mydpi, format="pdf", transparent=True)



