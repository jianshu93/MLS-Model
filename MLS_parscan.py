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

#%%

gamma_vec = np.array([0.01, 0.05, 0.1])
tauH_vec = np.logspace(2, 4, 3)
n0_vec = np.logspace(-10, -2, 9)
mig_vec = np.logspace(-10, -2, 9)
rr_vec = np.array([1.])
K_vec = np.array([1., 1.E3, 1.E6, 1.E9, 1.E12])

parOrder = np.array(['gamma','tauH','n0','mig','r','K'])
parRange = [gamma_vec, tauH_vec, n0_vec, mig_vec, rr_vec, K_vec] 
    
def createModelPar(input):
    modelPar = {
                #fixed model parameters
                "maxT"  : 10000., 
                "sampleT": 1.,
                "mu"    : 1.E-5,
                "B_H"   : 3.,
                "D_H"   : 1./40.,
                #scanned model parameters
                "gamma" : input[0],
                "TAU_H" : input[1],
                "n0"    : input[2]*input[5],  #n0*K = constant
                "mig"   : input[3],
                "r"     : input[4],
                "K"     : input[5],
                #fixed intial condition
                "NUMGROUP" : 100.,  
                "F0" : 0.5,
                "N0init" : 1., 
                }
    return modelPar

modelParList = [createModelPar(x) for x in itertools.product(*parRange)]


def runModel(index):
    gamma_vec = np.array([0.01, 0.05, 0.1])
    tauH_vec = np.logspace(2, 4, 3)
    n0_vec = np.logspace(-10, -2, 9)
    mig_vec = np.logspace(-10, -2, 9)
    rr_vec = np.array([1.])
    K_vec = np.array([1., 1.E3, 1.E6, 1.E9, 1.E12])
    
    parOrder = np.array(['gamma','tauH','n0','mig','r','K'])
    parRange = [gamma_vec, tauH_vec, n0_vec, mig_vec, rr_vec, K_vec] 
    
    input = [x for x in itertools.product(*parRange)][index]
    
    modelPar = {
                #fixed model parameters
                "maxT"  : 10000., 
                "sampleT": 1.,
                "mu"    : 1.E-5,
                "B_H"   : 3.,
                "D_H"   : 1./40.,
                #scanned model parameters
                "gamma" : input[0],
                "TAU_H" : input[1],
                "n0"    : input[2]*input[5],  #n0*K = constant
                "mig"   : input[3],
                "r"     : input[4],
                "K"     : input[5],
                #fixed intial condition
                "NUMGROUP" : 100.,  
                "F0" : 0.5,
                "N0init" : 1., 
                }
    
    output = mlss.single_run_finalstate(modelPar)

    
    return index
    
    
numLoop=len(modelParList)  

start = time.time()
results = Parallel(n_jobs=4, verbose=4, timeout=1.E9)(delayed(runModel)(i) for i in range(numLoop))
    
#results = Parallel(n_jobs=4, verbose=4, timeout=1.E9)(delayed(mlss.single_run_finalstate)(par) for par in modelParList)
OutputAll = np.asarray(results)
    
end = time.time()   
                 
print("Elapsed time run 1 = %s" % (end - start))


#np.savez("parScan181109.npz", data=OutputAll, parOrder=parOrder, parRange=parRange, modelParList=modelParList)
                    

#Output = mlss.single_run_with_plot(MODEL_PAR, INIT_COND)
#parMatrix = np.array([x for x in itertools.product(*parRange)])

