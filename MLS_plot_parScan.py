#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:10:02 2018

@author: simonvanvliet
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import colors
from pathlib import Path


now = datetime.datetime.now()

saveNameMod = "HyperGeomSampling"



data_folder = Path("Data/")
fileName = "2018-11-13_15_00"
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


numGamma = ndSize[gammaIndex]
numTau = ndSize[tauIndex]
numR = ndSize[rIndex]
numK = ndSize[KIndex]

n0V = parRange[n0Index]
migV = parRange[migIndex]

plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(numTau*numR, numGamma*numK)

w = 12000 / 300
h = 8000 / 300


#fig.suptitle('Multiple images')
images = []
cmap = "cool"
for tt in range(numTau):
    for gg in range(numGamma):
        for rr in range(numR):
            for kk in range(numK):
                
                currR = rr * numTau + tt
                currC = kk * numGamma + gg

                currData = np.log10(data['F_mav'][gg, tt, :, :, rr, kk] \
                                    .squeeze()).transpose()
                  
                images.append(axs[currR, currC].imshow(currData, cmap=cmap, \
                              interpolation='nearest', \
                              extent=[n0V[0],n0V[-1],migV[0],migV[-1]], \
                              origin='lower'))
                #axs[currR, currC].set_xticks([n0V[0], n0V[-1]])
                #axs[currR, currC].set_yticks([migV[0], migV[-1]])
                axs[currR, currC].set_xticks([])
                axs[currR, currC].set_yticks([])
                
                #axs[tt, gg].set_xlabel("n0")
                # axs[tt, gg].set_ylabel("theta")
                axs[currR, currC].label_outer()

norm = colors.Normalize(vmin=-3, vmax=0)
for im in images:
    im.set_norm(norm)


fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, label="fraction cooperator")

plt.draw()
fig.set_size_inches(w,h)
#fig.set_dpi = 150

plt.savefig(saveName, dpi=300, format="pdf")

#plt.show()


