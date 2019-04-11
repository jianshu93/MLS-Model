#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:56:43 2018

@author: simonvanvliet
"""
#%% single run of model
import MLS_static as mlss
#import MLS_evolveCoop as mlse
#import MLS_coevolveCoop as mlsco
#import MLS_coevolveCoop_temp as mlsc

#import MLS_evolveHost as mlsh

#%%

model_par = {
                #time step parameters
                "maxT"  : 5000, 
                "sampleT": 1,
                "rms_err_treshold": 1E-2,
                "mav_window": 1000,
                "rms_window": 5000,
                #fixed model parameters
                #"sampling" : "sample",
                "sampling" : "fixedvar",
                "mu"    : 1E-5,
                "B_H"   : 1.,
                "D_H"   : 0,
                "K_H"   : 25.,
                #variable model parameters
                "cost" : 0.01,
                "TAU_H" : 1E12,
                "n0"    : 1E-3,
                "mig"   : 1E-9,
                "r"     : 1.,
                "K"     : 10E3,
                "hostInv0" : 0.,
                #fixed intial condition
                "NUMGROUP" : 50,  
                "numTypeBins" : 100,
                "meanGamma0" : 0.,
                "stdGamma0" : 0.01,
                "N0init" : 1.,
                #host evolution settings
                "HostEvolves" : False,
                "sigmaR" : 0,
                "sigmaMig" : 0,
                "sigmaN0" : 0,
                "sigmaHostInv" : 0.1,
                #static settings
                "sigmaBirth" : 0.2,
                "F0" : 1
        }





#Mat, Output, InvestmentAll, InvestmentPerHost = mlse.single_run_with_plot(model_par)
Output = mlss.single_run_with_plot(model_par)