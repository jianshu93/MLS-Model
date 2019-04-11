#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:59:53 2019

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""


import MLS_plot_general_code as mlspg
import MLS_static as mlss
import numpy as np
#import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed 
import itertools
import datetime
from pathlib import Path
import math

#%%
dataName = 'fixedVar_timeScalePlot'

tau_H = 100.
n0 = 1E-4


#tauVar_target = np.concatenate((  \
#                      np.logspace(-2, -1, 10),   \
#                      np.logspace(-1, 1, 40)[1:-1], \
#                      np.logspace(1, 2, 15) )) * tau_H
#                                
#migToN0 = np.concatenate((  \
#                      np.logspace(-4, -1, 20),   \
#                      np.geomspace(0.1, 0.8, 12)[1:-1],   \
#                      mlspg.divSpace(1, 0.2, 0.001, factor=0.7), \
#                      np.geomspace(1.2, 10, 8)[1:-1],   \
#                      np.logspace(1, 4, 10) ))



desiredTauV = np.logspace(-2, 2, 50) * tau_H
desiredTauH = np.logspace(-4, 6, 50) * tau_H

migToN0 = mlspg.mig_from_tauH(desiredTauH, n0)


cost_vec = 1 / desiredTauV
mig_vec = migToN0 * n0 

tauH_vec = np.array([tau_H])
n0_vec = np.array([n0])
K_vec = np.array([1.]) 

sigma_vec = np.array([0.2, 0.1, 0.05, 0.02, 0.01])
B_vec = np.array([0]) #[0.1, 0.5, 1, 2])

parOrder = np.array(['cost','tauH','n0','mig','K', 'sigma', 'B_H'])
parRange = [cost_vec, tauH_vec, n0_vec, mig_vec, K_vec, sigma_vec, B_vec] 
    
def createModelPar(input):
    modelPar = {
                #fixed model parameters
                "sampling" : "fixedvar",
                "maxT"  : 40000., 
                "sampleT": 1.,
                "rms_err_treshold": 1E-2,
                "mav_window": 1000,
                "rms_window": 5000,
                "mu"    : 1E-5,
                "D_H"   : 0.,
                "K_H"   : 30.,
                #scanned model parameters
                "cost" : input[0],
                "TAU_H" : input[1],
                "n0"    : input[2],  
                "mig"   : input[3],
                "K"     : input[4],
                "sigmaBirth" : input[5],
                "B_H"   : input[6],
                #fixed intial condition
                "NUMGROUP" : 30.,  
                "F0" : 0.5,
                "N0init" : 1., 
                "numTypeBins" : 100
                }
    return modelPar

modelParList = [createModelPar(x) for x in itertools.product(*parRange)]

#%%
start = time.time()
results = Parallel(n_jobs=4, verbose=9, timeout=1.E9) \
    (delayed(mlss.single_run_finalstate)(par) for par in modelParList)
end = time.time()   
                 
print("Elapsed time run 1 = %s" % (end - start))

#%%

#reformat data in np arrays
summaryStat, hostDistri = zip(*results)
statData = np.vstack(summaryStat)
distData = np.vstack(hostDistri)

data_folder = Path("Data/")

now = datetime.datetime.now()
saveName = "parScan_" + dataName + now.strftime("%Y%m%d_%Hh%M") + ".npz"
saveName = data_folder / saveName

np.savez(saveName, data=statData, distData=distData, \
         parOrder=parOrder, parRange=parRange, modelParList=modelParList)
       
message = "saved as: " + "parScan_" + dataName + now.strftime("%Y%m%d_%Hh%M") + ".npz"          
print(message)
#Output = mlss.single_run_with_plot(MODEL_PAR, INIT_COND)
#parMatrix = np.array([x for x in itertools.product(*parRange)])
