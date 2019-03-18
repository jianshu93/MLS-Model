#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:56:43 2018

@author: simonvanvliet
"""
#%% single run of model
#import MLS_static as mlss
import MLS_evolveCoop as mlse
#import MLS_coevolveCoop as mlsco
import MLS_coevolveCoop_temp as mlsco

#import MLS_evolveHost as mlsh

MODEL_PAR = {
                #fixed model parameters
                "sampling" : "fixedvar",
                "maxT"  : 10000., 
                "sampleT": 10.,
                "mu"    : 1.E-3,
                "B_H"   : 3.,
                "D_H"   : 1./40.,
                #variable model parameters
                "gamma" : 0.01,
                "TAU_H" : 100.,
                "n0"    : 1E-3,
                "mig"   : 0,
                "r"     : 1.,
                "K"     : 5E3,
                "sigmaBirth" : 0.3,
                #fixed intial condition
                "NUMGROUP" : 100,  
                "F0" : 0.1,
                "N0init" : 1,
                #host evolution parameters
                "HostEvolves" : False,
                "sigmaR" : 0,
                "sigmaMig" : 0.2,
                "sigmaN0" : 0
        }


MODEL_PAR_EVOL = {
                #fixed model parameters
                "sampling" : "contsample",
                "maxT"  : 10000., 
                "sampleT": 100,
                "mu"    : 0.05,
                "B_H"   : 3.,
                "D_H"   : 1./30.,
                #variable model parameters
                "cost" : 0.1,
                "TAU_H" : 100.,
                "n0"    : 1E-3,
                "mig"   : 0;,
                "r"     : 1.,
                "K"     : 10E3,
                "hostInv0" : 0.,
                #fixed intial condition
                "NUMGROUP" : 40,  
                "numTypeBins" : 100,
                "meanGamma0" : 0,
                "stdGamma0" : 0.01,
                "N0init" : 1.,
                #host evolution settings
                "HostEvolves" : True,
                "sigmaR" : 0,
                "sigmaMig" : 0,
                "sigmaN0" : 0,
                "sigmaHostInv" : 0
        }



#Output = mlsh.single_run_with_plot(MODEL_PAR)


#mlse.run_model_fixed_parameters_testlocal(MODEL_PAR_EVOL)

#Output = mlse.single_run_with_plot(MODEL_PAR_EVOL)

Output = mlsco.single_run_with_plot(MODEL_PAR_EVOL)

#Output = mlss.single_run_with_plot(MODEL_PAR)