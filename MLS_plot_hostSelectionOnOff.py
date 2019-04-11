#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 8 2019
@author:
Simon van Vliet
Department of Zoology
University of British Columbia
vanvliet@zoology.ca

Compare result of MLS model with and without host dynamics 

"""
# %%
# import MLS_analyse_fit3Dsurface as mlsfit
# import MLS_plot_general_code as mlspg
import MLS_static as mlss
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import axes3d
# from scipy import interpolate
# from scipy.interpolate import griddata
# import datetime
# from pathlib import Path
# from matplotlib import collections as matcoll

modelPar = {
    # fixed model parameters
    "sampling": "fixedvar",
    "maxT": 100000.,
    "sampleT": 1.,
    "rms_err_treshold": 1E-3,
    "mav_window": 1000,
    "rms_window": 5000,
    "mu": 1E-5,
    "D_H": 0.,
    "K_H": 20.,
    # scanned model parameters
    "cost": 0.01,
    "TAU_H": 50.,
    "n0": 1E-4,
    "mig": 1E-7,
    "K": 1,
    "sigmaBirth": 0.05,
    "B_H": 1,
    # fixed initial condition
    "NUMGROUP": 20.,
    "F0": 0.5,
    "N0init": 1.,
    "numTypeBins": 100
}

Output, InvestmentPerHost = mlss.single_run_with_plot(modelPar)

# %%
