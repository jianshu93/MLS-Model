#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:44:11 2019

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import MLS_plot_general_code as mlspg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math



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
        

#migToN0 = np.concatenate((  \
#                      np.logspace(-4, -1, 20),   \
#                      np.geomspace(0.1, 0.8, 12)[1:-1],   \
#                      divSpace(1, 0.2, 0.001, factor=0.7), \
#                      np.geomspace(1.2, 10, 8)[1:-1],   \
#                      np.logspace(1, 4, 10) ))


n0 = 1E-4
cost = 0
r = 0
tauR = 100


desiredTauH = np.logspace(-4.05, 4.05, 42) * tauR

migToN0 = mlspg.mig_from_tauH(desiredTauH, n0)

theta = migToN0 * n0 

tau_H = mlspg.calc_tauH(n0, theta, cost) / tauR



fig = plt.figure()
w = 400
h = 400
font = {'family' : 'Arial',
        'weight' : 'light',
        'size'   : 10}
mpl.rc('font', **font)
mydpi=150

xVec = np.arange(migToN0.size)


fig.set_size_inches(w/mydpi,h/mydpi)
plt.semilogy(xVec,tau_H,'bo', markersize=1)

plt.tight_layout()
plt.draw()



tauR = 100

tauVar_target = np.concatenate((  \
                      np.logspace(-2, -1, 10),   \
                      np.logspace(-1, 1, 30)[1:-1],   \
                      np.logspace(1, 2, 10))) \
                      * tauR
                                
                                
cost_vec = 1 / tauVar_target
                               
xVec = np.arange(cost_vec.size)

fig = plt.figure()
w = 400
h = 400
font = {'family' : 'Arial',
        'weight' : 'light',
        'size'   : 10}
mpl.rc('font', **font)
mydpi=150

fig.set_size_inches(w/mydpi,h/mydpi)
plt.semilogy(xVec,tauVar_target/tauR,'bo', markersize=1)

plt.tight_layout()
plt.draw()
                                
                                