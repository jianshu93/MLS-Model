#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:56:43 2018

@author: simonvanvliet
"""
#%% single run of model
#import MLS_static as mlss
import MLS_evolveCoop as mlse

MODEL_PAR = {
                #fixed model parameters
                "sampling" : "fixedvar",
                "maxT"  : 50000., 
                "sampleT": 1.,
                "mu"    : 1.E-5,
                "B_H"   : 3.,
                "D_H"   : 1./40.,
                #variable model parameters
                "gamma" : 0.01,
                "TAU_H" : 100.,
                "n0"    : 1E-3,
                "mig"   : 1E-12,
                "r"     : 1.,
                "K"     : 1.,
                "sigmaBirth" : 0.1,
                #fixed intial condition
                "NUMGROUP" : 100,  
                "F0" : 0.5,
                "N0init" : 1
        }


MODEL_PAR_EVOL = {
                #fixed model parameters
                "sampling" : "sample",
                "maxT"  : 20., 
                "sampleT": 1.,
                "mu"    : 0.02,
                "B_H"   : 3.,
                "D_H"   : 1./40.,
                #variable model parameters
                "gamma" : 0.01,
                "TAU_H" : 100.,
                "n0"    : 1E-3,
                "mig"   : 1E-6,
                "r"     : 1.,
                "K"     : 1E4,
                "sigmaBirth" : 0.2,
                #fixed intial condition
                "NUMGROUP" : 100,  
                "numTypeBins" : 100,
                "meanGamma0" : 0.5,
                "stdGamma0" : 0.001,
                "N0init" : 1
        }



mlse.run_model_fixed_parameters_testlocal(MODEL_PAR_EVOL)

#Output = mlse.single_run_with_plot(MODEL_PAR_EVOL)

#Output = mlss.single_run_with_plot(MODEL_PAR)