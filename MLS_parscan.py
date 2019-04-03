#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 29 2018
@author: 
Simon van Vliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ca
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
dataName = 'fixedVar_scanAll'

cost_vec = np.array([0.01, 0.02, 0.05, 0.1])
tauH_vec = np.array([10, 50, 100, 500, 1000])
n0_vec = np.logspace(-9, -3, 7)
mig_vec = np.logspace(-9, -3, 7) 
rr_vec = np.array([1.])
K_vec = np.array([1.]) #, 1.E3, 1.E6, 1.E9, 1.E12])
sigma_vec = np.array([0.2, 0.1, 0.05, 0.01, 0.005, 0.001])
B_vec = np.array([0.1, 0.5, 1, 3])

#sigma_vec = np.array([20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05])


parOrder = np.array(['cost','tauH','n0','mig','r','K', 'sigma', 'B_H'])
parRange = [cost_vec, tauH_vec, n0_vec, mig_vec, rr_vec, K_vec, sigma_vec, B_vec] 
    
def createModelPar(input):
    modelPar = {
                #fixed model parameters
                "sampling" : "fixedvar",
                "maxT"  : 30000., 
                "sampleT": 1.,
                "rms_err_treshold": 1E-2,
                "mav_window": 1000,
                "rms_window": 5000,
                "mu"    : 1E-5,
                "D_H"   : 0.,
                "K_H"   : 50.,
                #scanned model parameters
                "cost" : input[0],
                "TAU_H" : input[1],
                "n0"    : input[2],  
                "mig"   : input[3],
                "r"     : input[4],
                "K"     : input[5],
                "sigmaBirth" : input[6],
                "B_H"   : input[7],
                #fixed intial condition
                "NUMGROUP" : 50.,  
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
       
message = "saved as: " + "parScan_" + dataName + now.strftime("%Y%m%d_%Hh%M_") + ".npz"          
print(message)
#Output = mlss.single_run_with_plot(MODEL_PAR, INIT_COND)
#parMatrix = np.array([x for x in itertools.product(*parRange)])

