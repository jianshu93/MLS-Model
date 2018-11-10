#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:10:02 2018

@author: simonvanvliet
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.recfunctions import append_fields
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d    


fileName = "parScan181030.npz"

file = np.load(fileName)

metaData = file['metaData'][0]
data = file['data']

#metaData["defIdx"] = np.array([1, 1, 2, 3, 1])

numTau = metaData['TAU_H'].size
numGamma = metaData['gamma'].size

n0V = np.log10(metaData['n0'])
migV = np.log10(metaData['mig'])


datar1 = data[:,:,:,:,:,0]
data1d = np.reshape(datar1,datar1.size)


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

tauVar = 1. / (data1d['gamma']*data1d['r'])
tauHer = calc_tauH(data1d['n0'], data1d['mig'], data1d['gamma'], data1d['r']) 

minTau = np.minimum(tauVar, tauHer)
#minTau[minTau==0] = np.finfo(float).eps

relTau = data1d['tau_H'] / minTau

#data1d["tauV"] = 1 / data1d['gamma']
data1d = append_fields(data1d, 'tauVar', tauVar, 'f8', usemask=False)
data1d = append_fields(data1d, 'tauHer', tauHer, 'f8', usemask=False)
data1d = append_fields(data1d, 'minTau', minTau, 'f8', usemask=False)
data1d = append_fields(data1d, 'relTau', relTau, 'f8', usemask=False)


np.save('1dData.npy',data1d)

#gammaCat = np.ones(data1d.size)
#index = np.arange(numGamma)
#idx=0
#for x in data1d['gamma']:
#    gammaCat[idx]=index[metaData['gamma']==x]
#    idx += 1


def make_categorial(vector):
    elements =  np.unique(vector)
    index = np.arange(elements.size)
    cat_vector = np.zeros(vector.size)
    
    for idx in range(vector.size):
        cat_vector[idx]=index[elements==vector[idx]]
        
    return cat_vector    
    
gammaCat = make_categorial(data1d['gamma'])    

fig = plt.figure()

w = 1200 / 150
h = 600 /150
fig.set_size_inches(w,h)


#plt.subplot(1,3,1)
#plt.hist(tauVar)
#
#plt.subplot(1,3,2)
#plt.hist(tauHer)
#
#plt.subplot(1,3,3)
#plt.hist(minTau)

ax = plt.scatter((data1d['relTau']),data1d['F_mav'], c=[0.2, 0.2, 0.2], s=16, alpha=0.15)
plt.xscale('log')

plt.draw()
#
plt.tight_layout()

plt.savefig('fracVsTimeScale.png', dpi=150)




def plot_scat_cat(data1d,catName):
    
    cCat = make_categorial(data1d[catName])
    numCat = np.unique(cCat).size
    cmapNew = cm.get_cmap('Paired', numCat)
    ax = plt.scatter(data1d['relTau'],data1d['F_mav'], c=cCat, s=16, alpha=0.4, cmap=cmapNew)
    plt.xscale('log')
    cTick=np.linspace(0.5, numCat-1.5, numCat)
    cbar = plt.colorbar(label=catName, ticks=cTick, alpha=1)
    cbar.ax.set_yticklabels(np.unique(data1d[catName]))
    plt.xlim((1e-4, 1e12))
    
    return


def plot_scat_cont(xData,data1d,catName):
    ax = plt.scatter(xData,data1d['F_mav'], c=data1d[catName], s=16, alpha=0.4, cmap='Winter')
    plt.xscale('log')
    cbar = plt.colorbar(label=catName)
    plt.xlim((1e-2, 1e10))
    return

fig2 = plt.figure()

w = 3000 / 150
h = 1200 /150
fig2.set_size_inches(w,h)



alpha = 0.5
plt.subplot(2,3,1)
plot_scat_cat(data1d,'gamma')

plt.subplot(2,3,2)
plot_scat_cat(data1d,'n0')


plt.subplot(2,3,3)
plot_scat_cat(data1d,'mig')

plt.subplot(2,3,4)
plot_scat_cat(data1d,'tau_H')

plt.subplot(2,3,5)
plot_scat_cat(data1d,'r')

plt.tight_layout()
plt.draw()
plt.savefig('seperateFactors.png', dpi=150)



tauVRel = data1d['tau_H'] / tauVar
tauHRel = data1d['tau_H'] / tauHer


fig3 = plt.figure()

w = 1200 / 150
h = 600 /150
fig3.set_size_inches(w,h)

plt.subplot(1,2,1)
plt.scatter(tauVRel,data1d['F_mav'], c=np.log10(tauHer), s=16, alpha=0.4, \
            cmap='winter', vmin=-6, vmax=8)
plt.xscale('log')
plt.colorbar(label='log10 tauHer')
plt.xlim((0.01, 1000))


plt.subplot(1,2,2)
plt.scatter(tauHRel,data1d['F_mav'], c=np.log10(tauVar), s=16, alpha=0.4, \
            cmap='winter', vmin=0, vmax=2)
plt.xlim((1e-8, 1e10))
print(tauHRel.min(), tauHRel.max())
plt.xscale('log')
plt.colorbar(label='log10 tauVar')



#
plt.tight_layout()
plt.draw()
plt.savefig('seperateFrac.png', dpi=150)

#
#fig3 = plt.figure()
#
#w = 1200 / 150
#h = 600 /150
#fig3.set_size_inches(w,h)
#ax = fig3.add_subplot(111, projection='3d')
#
#
#tauVarRel = data1d['tau_H'] / tauVar
#tauHerRel = data1d['tau_H'] / tauHer
#
#
#ax.scatter(tauVarRel, tauHerRel, data1d['relTau'], c=[0.2, 0.2, 0.2], s=16, alpha=0.4)

fig3 = plt.figure()

w = 2400 / 150
h = 1800 /150
fig3.set_size_inches(w,h)

ax = fig3.add_subplot(111, projection='3d')

tauVarRel = data1d['tau_H'] / data1d['tauVar']
tauHerRel = data1d['tau_H'] / data1d['tauHer']

print(tauVarRel.min(), tauVarRel.max())


xData = np.log10(tauVarRel)
yData = np.log10(tauHerRel)

xRange = (np.floor(xData.min()), np.ceil(xData.max()))
yRange = (np.floor(yData.min()), np.ceil(yData.max()))

ax.scatter(xData , yData , data1d['F_mav'], c=data1d['F_mav'], s=60, alpha=0.6, vmin=0, vmax=0.6)
plt.xlim(xRange)
plt.ylim(yRange)
#plt.zlim((0, 0.6))

ax.set_xlabel('log tau_R / tau_V')
ax.set_ylabel('log tau_R / tau_H')
ax.set_zlabel('fraction cooperator')
ax.view_init(25,60)
plt.tight_layout()

plt.draw()

plt.savefig('3dFig.png', dpi=150)


fig4 = plt.figure()
w = 2400 / 150
h = 1400 /150
fig4.set_size_inches(w,h)


rangeHer = np.array([-4, -2, 0, 2, 4])
logHer = np.log10(data1d['tau_H'] / data1d['tauHer'])

tauV = data1d['tau_H'] / data1d['tauVar']
xRange = (tauV.min()/10., tauV.max()*10.)

for i in range(6):
    ax=plt.subplot(2,3,i+1)
    
    if i==0:
        minTauHer = -100
    else:
        minTauHer = rangeHer[i-1]
    if i==5:
        maxTauHer = 100
    else:    
        maxTauHer = rangeHer[i]
        
    inRange = (logHer>=minTauHer) & (logHer<maxTauHer)

    xData = tauV[inRange]
    yData = data1d['F_mav'][inRange]
    cData = data1d['F_mav'][inRange]

    plt.scatter(xData,yData, c=[0.2, 0.2, 0.2], s=30, alpha=0.3)
    plt.xscale('log')
    plt.xlim(xRange)
    plt.ylim((0, 0.6))
    titleName = "%d<log tau_R / tau_H<%d" % (minTauHer,maxTauHer,)
    plt.title(titleName)
    plt.ylabel("frac cooperator")
    plt.xlabel("tau_R / tau_V")
    ax.label_outer()
    

plt.tight_layout()
plt.draw()

plt.savefig('SlicedByTauHer.png', dpi=150)



fig4 = plt.figure()
w = 1600 / 150
h = 1400 /150
fig4.set_size_inches(w,h)


rangeVar = np.array([-1, 0, 1, 2])
logVar = np.log10(data1d['tau_H'] / data1d['tauVar'])

tauH = data1d['tau_H'] / data1d['tauHer']
xRange = (tauH.min()/10., tauH.max()*10.)

for i in range(4):
    ax=plt.subplot(2,2,i+1)
    
    if i==0:
        minTauVar = -100
    else:
        minTauVar = rangeVar[i-1]
    if i==5:
        maxTauVar = 100
    else:    
        maxTauVar = rangeVar[i]
        
    inRange = (logVar>=minTauVar) & (logVar<maxTauVar)

    xData = tauH[inRange]
    yData = data1d['F_mav'][inRange]
    cData = data1d['F_mav'][inRange]

    plt.scatter(xData,yData, c=[0.2, 0.2, 0.2], s=30, alpha=0.3)
    plt.xscale('log')
    plt.xlim(xRange)
    plt.ylim((0, 0.6))
    titleName = "%d<log tau_R / tau_V<%d" % (minTauVar,maxTauVar,)
    plt.title(titleName)
    plt.ylabel("frac cooperator")
    plt.xlabel("tau_R / tau_H")
    ax.label_outer()
    

plt.tight_layout()
plt.draw()

plt.savefig('SlicedByTauVar.png', dpi=150)

