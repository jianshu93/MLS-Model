#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
CreateD on March 29 2018
Last Update March 29 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

"""
import MLS_coevolveCoop_temp as mlsco
import MLS_evolveCoop as mlse
import MLS_static as mlss
import mls_general_code as mlsg
import math
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib
import time
from numba import jit, void, f8, i8 
from numba.types import UniTuple
     
"""
Init functions - rewritten to accomadate different intial cooperation levels
"""
#initialize community - discrete model
def init_comm(model_par): 
    numGroup = int(model_par["NUMGROUP"])
    #setup inital vector of c,d
    cAct = model_par["N0init"] * np.linspace(0,1,numGroup)
    dAct = model_par["N0init"] - cAct

    #store in C-byte order 
    cAct = np.copy(cAct, order='C')
    dAct = np.copy(dAct, order='C')
    return (cAct, dAct)

#initialize community - continuous model
def init_comm_cont(model_par): 
    #get parameters
    numGammaBin = model_par['numTypeBins']
    stdGamma = model_par['stdGamma0']
    numGroup = int(model_par['NUMGROUP'])
    N0 = model_par['N0init']
    #create investment vector
    dGamma = 1 / numGammaBin    
    gammaVec = np.linspace(dGamma/2, 1-dGamma/2, numGammaBin)

    #init group matrix
    groupMatrix = np.zeros((numGroup, numGammaBin))
    
    #loop groups assingn different init fraction
    meanVec = np.linspace(0,1,numGroup)
    for ii in range(numGroup):    
        #calculate initial fraction in each investment bin based on normal distribution
        zVec = (gammaVec - meanVec[ii]) / stdGamma
        initDistri = st.norm.pdf(zVec)
        #normalize to total density of N0
        initDistri *= (N0 / initDistri.sum())
        #assign intital distribution to all initial groups
        groupMatrix[ii,:] = initDistri
    #store in c-order
    groupMatrix = np.copy(groupMatrix, order='C')
    return groupMatrix


##run discretemodel main code
def run_model_fixed_parameters(model_par):
    #possible dt to choose from
    dt = model_par['dt']
    #calculate max number of sampling points
    Num_t_sample = int(np.ceil(model_par['maxT'] / model_par['sampleT'])+1)
    #calc timesteps    
    samplingInterval = model_par['sampleT']
    #get bacterial rates
    r, mu, mig, cost = [float(model_par[x]) for x in ('r','mu','mig', 'cost')]
    numBins = int(model_par['numTypeBins']) 
    #init output
    binEdges = np.linspace(0, 1, numBins)
    Output = mlss.init_output_matrix(Num_t_sample)
    InvestmentPerHost = np.full((Num_t_sample, numBins-1), np.nan)
    # init groups
    CVec, DVec  = init_comm(model_par)
    #first sample
    currT = 0
    ti = 0
    sampleIndex = 0
    sampleIndex = mlss.sample_model(CVec, DVec, binEdges, \
                            Output, InvestmentPerHost, \
                            sampleIndex, currT, model_par)
    
    #run time    
    while currT <= model_par['maxT']:
        #update community
        mlss.update_comm(CVec, DVec, r, cost, mu, mig, dt)  
        #update time
        currT += dt
        ti += 1
        #sample model at intervals
        nextSampleT = samplingInterval * sampleIndex
        if currT >= nextSampleT:
            sampleIndex = mlss.sample_model(CVec, DVec, binEdges, \
                            Output, InvestmentPerHost, \
                            sampleIndex, currT, model_par)
            
            #check if steady state has been reached
            if Output['rms_err'][sampleIndex-1] < model_par['rms_err_treshold']:
                break
        
    #cut off non existing time points at end
    Output = Output[0:sampleIndex]
    InvestmentPerHost = InvestmentPerHost[0:sampleIndex,:]
    
    return (Output, InvestmentPerHost)

#run continuous model main code
##run model main code
def run_model_fixed_parameters_cont(model_par):
    #possible dt to choose from
    dt = model_par['dt']
    #calculate max number of sampling points
    Num_t_sample = int(np.ceil(model_par['maxT'] / model_par['sampleT'])+1)
    #calc timesteps    
    samplingInterval = model_par['sampleT']
    #get bacterial rates
    r, mu, cost = [float(model_par[x]) for x in ('r','mu','cost')]
    numBins = int(model_par['numTypeBins'])
    # init groups
    gMat = init_comm_cont(model_par)
    HVec = mlsco.init_host_trait_matrix(model_par)
    mutMat, birthVec, gammaVec = mlsco.create_local_update_matrix(r, mu, cost, numBins)
    #init output
    Output = mlse.init_output_matrix(Num_t_sample)
    Output = mlsco.init_host_output_matrix(Output, Num_t_sample)
    InvestmentAll = np.full((Num_t_sample, numBins), np.nan)
    InvestmentPerHost = np.full((Num_t_sample, numBins), np.nan)
    #first sample
    currT = 0
    ti = 0
    sampleIndex = 0
    sampleIndex = mlsco.sample_full_model(gMat, HVec, gammaVec, Output, \
                               InvestmentAll, InvestmentPerHost, \
                               sampleIndex, currT, model_par)
    
    #run time    
    while currT <= model_par['maxT']:
        #update community
        invPerCom = mlse.calc_community_performance(gMat, gammaVec)
        mlsco.update_comm_loc(gMat, mutMat, birthVec, invPerCom, HVec, dt)
        #update time
        currT += dt
        ti += 1  
        #sample model at intervals
        nextSampleT = samplingInterval * sampleIndex
        if currT >= nextSampleT:
            sampleIndex = mlsco.sample_full_model(gMat, HVec, gammaVec, Output, \
                               InvestmentAll, InvestmentPerHost, \
                               sampleIndex, currT, model_par)
            
            #check if steady state has been reached
            if Output['rms_err'][sampleIndex-1] < model_par['rms_err_treshold']:
                break
            
    #cut off non existing time points at end
    Output = Output[0:sampleIndex]
    InvestmentAll = InvestmentAll[0:sampleIndex,:]
    InvestmentPerHost = InvestmentPerHost[0:sampleIndex,:]

    return (Output, InvestmentAll, InvestmentPerHost)


"""
Run Full model 
"""
#run model, plot dynamics 
def single_run_with_plot(MODEL_PAR):
    
    #run code  
    start = time.time()

    if MODEL_PAR['modelType'] == 'discrete':
        Output, InvestmentPerHost = run_model_fixed_parameters(MODEL_PAR)
        nc = 2
    else:
        Output, InvestmentAll, InvestmentPerHost = run_model_fixed_parameters_cont(MODEL_PAR)
        nc = 3
    end = time.time()
    print("Elapsed time run 1 = %s" % (end - start))
    
    font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 9}
    matplotlib.rc('font', **font)
    
    fig = plt.figure()
    #plot average investment  
    plt.subplot(1,nc,1)  
    mlss.plot_data(Output,"F_T_av")  
    mlss.plot_data(Output,"F_mav")  
    plt.ylabel("investment") 
    plt.ylim((0, 1))

    #plot average investment per host
    axs= plt.subplot(1,nc,2)  
    currData = np.log10(InvestmentPerHost.transpose() + np.finfo(float).eps )
    im = axs.imshow(currData, cmap="viridis", \
                    interpolation='nearest', \
                    extent=[0,1,0,1], \
                    origin='lower', \
                    #vmin=-2, vmax=-1,\
                    aspect= 'auto')
    axs.set_xticks([0, 1])
    axs.set_yticks([0, 1])
    axs.set_ylabel('investment')
    axs.set_xlabel('time')
    cb = fig.colorbar(im, ax=axs, orientation='vertical', fraction=.1, label="log10 mean density")
    #cb.set_ticks([-2, -1])
    axs.set_yticklabels([0, 1])
    maxTData =Output['time'].max()
    axs.set_xticklabels([0, int(round(maxTData))])
    
    

    if MODEL_PAR['modelType'] == 'continuous':
        #plot average investment per host
        axs= plt.subplot(1,nc,3)  
        currData = np.log10(InvestmentAll.transpose() + np.finfo(float).eps )
        im = axs.imshow(currData, cmap="viridis", \
                        interpolation='nearest', \
                        extent=[0,1,0,1], \
                        origin='lower', \
                        #vmin=-2, vmax=-1,\
                        aspect= 'auto')
        axs.set_xticks([0, 1])
        axs.set_yticks([0, 1])
        axs.set_ylabel('investment')
        axs.set_xlabel('time')
        cb = fig.colorbar(im, ax=axs, orientation='vertical', fraction=.1, label="log10 mean density")
        #cb.set_ticks([-2, -1])
        axs.set_yticklabels([0, 1])
        maxTData =Output['time'].max()
        axs.set_xticklabels([0, int(round(maxTData))])


    fig.set_size_inches(10,4)
    plt.tight_layout()
    return Output 
