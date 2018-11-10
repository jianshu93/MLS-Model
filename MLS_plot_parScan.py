#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:10:02 2018

@author: simonvanvliet
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

fileName = "parScan181107.npz"

file = np.load(fileName)

metaData = file['metaData'][0]
data = file['data']

#metaData["defIdx"] = np.array([1, 1, 2, 3, 1])

numTau = metaData['TAU_H'].size
numGamma = metaData['gamma'].size

numR = metaData['r'].size
numK = metaData['K'].size

n0V = np.log10(metaData['n0'])
migV = np.log10(metaData['mig'])

plt.rcParams.update({'font.size': 18})
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

plt.savefig("heatMap", dpi=300, format="pdf")

#plt.show()


