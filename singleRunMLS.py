#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:56:43 2018

@author: simonvanvliet
"""
import MLS_static as mlss


MODEL_PAR = {
            "maxT"  : 100000, 
            "sampleT": 1.,
            "mu"    : 1.E-5,
            "B_H"   : 3.,
            "D_H"   : 1./40.,
            "gamma" : 0.01,
            "TAU_H" : 10,
            "n0"    : 1E-5,
            "mig"   : 1E-8,
            "r"     : 1,
            "K"     : 1E6,
            "NUMGROUP" : 100,  
            "F0" : 0.5,
            "N0init" : 1
        }

Output = mlss.single_run_with_plot(MODEL_PAR)