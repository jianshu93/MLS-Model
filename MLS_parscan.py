#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:50:25 2018

@author: simonvanvliet
"""

import MLS_static as mlss
import numpy as np
#import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed 
import itertools
import datetime
from pathlib import Path

#%%

gamma_vec = np.array([0.01, 0.05, 0.1])
tauH_vec = np.logspace(1, 4, 4)
n0_vec = np.logspace(-12, -2, 11)
mig_vec = np.logspace(-12, -2, 11) 
rr_vec = np.array([1.])
K_vec = np.array([1.]) #, 1.E3, 1.E6, 1.E9, 1.E12])
sigma_vec = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01, 0.005])

parOrder = np.array(['gamma','tauH','n0','mig','r','K', 'sigma'])
parRange = [gamma_vec, tauH_vec, n0_vec, mig_vec, rr_vec, K_vec, sigma_vec] 
    
def createModelPar(input):
    modelPar = {
                #fixed model parameters
                "sampling" : "fixedvar",
                "maxT"  : 10000., 
                "sampleT": 1.,
                "mu"    : 1.E-5,
                "B_H"   : 3.,
                "D_H"   : 1./40.,
                #scanned model parameters
                "gamma" : input[0],
                "TAU_H" : input[1],
                "n0"    : input[2],  
                "mig"   : input[3],
                "r"     : input[4],
                "K"     : input[5],
                "sigmaBirth" : input[6],
                #fixed intial condition
                "NUMGROUP" : 100.,  
                "F0" : 0.5,
                "N0init" : 1., 
                }
    return modelPar

modelParList = [createModelPar(x) for x in itertools.product(*parRange)]

#%%
start = time.time()

results = Parallel(n_jobs=4, verbose=6, timeout=1.E9)(delayed(mlss.single_run_finalstate)(par) for par in modelParList)
#results = [mlss.single_run_finalstate(par) for par in modelParList]

OutputAll = np.asarray(results)
    
end = time.time()   
                 
print("Elapsed time run 1 = %s" % (end - start))

#%%
data_folder = Path("Data/")

now = datetime.datetime.now()
saveName = "parScan_" + now.strftime("%Y%m%d_%Hh%M_") + modelParList[0]['sampling'] + ".npz"
saveName = data_folder / saveName

np.savez(saveName, data=OutputAll, parOrder=parOrder, parRange=parRange, modelParList=modelParList)
                    

#Output = mlss.single_run_with_plot(MODEL_PAR, INIT_COND)
#parMatrix = np.array([x for x in itertools.product(*parRange)])

