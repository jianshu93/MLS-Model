#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
CreateD on WeD OCt 17 12:25:24 2018
Last Update March 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

"""
import mls_general_code as mlsg
import MLS_evolveCoop as mlsec
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from numpy.lib import recfunctions as rfn
from numba import jit, void, f8, i8, from_dtype
from numba.types import UniTuple


"""
Init functions 
"""
#init host state output matrix
def init_host_output_matrix(Output, num_t_sample):
    #specify output fields of host traits 
    dNames = ['mig_t', 'r_t', 'n0_t', 'inv_H_t'] #change 2x M_T
    dType = ['f8','f8','f8','f8']
    dInitData = np.full(num_t_sample, np.nan)
    data = [dInitData, dInitData, dInitData, dInitData]

    #add fields to default output matrix
    Output = rfn.append_fields(Output, dNames, data, dtypes=dType)                 
    return Output

#initialize host trait matrix  
def init_host_trait_matrix(model_par): 
    #intialize host traits
    dtype = np.dtype([('mig', 'f8'), ('r', 'f8'),\
            ('hostInv', 'f8'), ('n0', 'f8')])
    host_traits = np.zeros(int(model_par['NUMGROUP']), dtype)
    
    #init all host with same intial trait
    host_traits['mig'] = model_par["mig"]
    host_traits['r'] = model_par["r"]
    host_traits['n0'] = model_par["n0"]
    host_traits['hostInv'] = model_par["hostInv0"]
    return host_traits

"""
Sample functions 
"""
#sampels state of 
def sample_host_state(output, hostTrait, sample_idx):  
    output['inv_H_t'][sample_idx] = hostTrait['hostInv'].mean()
    output['mig_t'][sample_idx] = hostTrait['mig'].mean()
    output['r_t'][sample_idx] = hostTrait['r'].mean()
    output['n0_t'][sample_idx] = hostTrait['n0'].mean()
    return 

#sample community and host state
def sample_full_model(gMat, HVec, gammaVec, Output, \
                               InvestmentAll, InvestmentPerHost, \
                               sampleIndex, currT, model_par):

    sample_host_state(Output, HVec, sampleIndex)
    sampleIndex = mlsec.sample_model(gMat, gammaVec, Output, \
                               InvestmentAll, InvestmentPerHost, \
                               sampleIndex, currT, model_par)

    return sampleIndex

"""
Host birth functions 
"""
#evolve host traits
def select_host_birth_trait(parTrait, model_parameter):
    #get traits parent
    r_par = parTrait['r']
    m_par = parTrait['mig'] 
    n0_par = parTrait['n0'] 
    hostInv_par = parTrait['hostInv'] 
    #get model parameters
    traitsStd = ['sigmaR', 'sigmaMig', 'sigmaN0', 'sigmaHostInv']
    stdR, stdM, stdN0, stdHostInv = [model_parameter[x] for x in traitsStd]

    #if sigma>0, draw new trait from truncated (log) normal distribution
    if stdR>0:
        rOff = mlsg.trunc_norm(r_par, stdR, min=0, max=10, type='lin')
    else:
        rOff = r_par   
    if stdM>0:
        mOff = mlsg.trunc_norm(m_par, stdM, min=1E-15, max=0.5, type='log')
    else:
        mOff = m_par  
    if stdN0>0:
        n0Off = mlsg.trunc_norm(n0_par, stdN0, min=1E-15, max=0.5, type='log')
    else:
        n0Off = n0_par  
    if stdHostInv>0:
        hostInvOff = mlsg.trunc_norm(hostInv_par, stdHostInv, \
                                 min=0, max=1, type='lin')
    else:
        hostInvOff = hostInv_par     
    
    #assign new traits
    newTraits = np.array(1, dtype=parTrait.dtype)
    newTraits['r'] = rOff
    newTraits['mig'] = mOff
    newTraits['n0'] = n0Off
    newTraits['hostInv'] = hostInvOff
    return newTraits

#create new host at birth
def host_birth_event(groupMat, host_trait, gammaVec, randNum, model_par):
    #extract fixed parameters
    B_H = model_par["B_H"]
    K = model_par['K']
    #select host to reproduce
    id_group = mlsec.select_host_to_reproduce(groupMat, gammaVec, randNum, B_H)
 
    #select parent group and  find offspring composition
    parComp = groupMat[id_group, :]
    n0 = host_trait['n0'][id_group]
    if model_par['sampling']=="sample":
         offComp = mlsec.host_birth_composition_sample(parComp, n0, K)
    else:
       raise Exception('Unknown sampling procedure')     
    #add new group to group matrix 
    groupMatNew = np.append(groupMat, np.atleast_2d(offComp), axis=0)

    #evolve host trait
    if model_par['HostEvolves']:
        newTraits = select_host_birth_trait(host_trait[id_group], model_par)
    else:
        newTraits = host_trait[id_group]
    host_traitNew = np.append(host_trait, newTraits)

    return (groupMatNew, host_traitNew)

"""
Host death functions 
"""        

def host_death_event(groupMatrix, host_trait, gammaVec, randNum, model_par): 
    #extract parameters
    D_H = model_par["D_H"]
    #select host to die
    id_group = mlsec.select_host_to_die(groupMatrix, gammaVec, randNum, D_H)
    #delete group
    groupMatNew = np.delete(groupMatrix, id_group, axis=0)
    host_traitNew = np.delete(host_trait, id_group)
    
    return (groupMatNew, host_traitNew)


"""
Community dynamics functions 
"""  
#create matrix for birth and mutation event 
def create_local_update_matrix(r, mu, cost, numBins):
    #setup investment vector
    dGamma = 1 / numBins    
    gammaVec = np.linspace(dGamma/2, 1-dGamma/2, numBins)
    #cost vector for each investment level
    costVec = gammaVec * cost
    #calculate rates
    noMutationRate = (1-mu)
    mutationRate = mu / 2
    #create sub matrices of size of single group
    mutationMat = np.diag(np.full(numBins,noMutationRate)) + \
               np.diag(np.full(numBins-1,mutationRate), -1) + \
               np.diag(np.full(numBins-1,mutationRate),  1)   
    basalBirthrate = (1 - costVec)
               
    return (mutationMat, basalBirthrate, gammaVec)


#get data type of strcutured array
dtype = np.dtype([('mig', 'f8'), ('r', 'f8'),\
                ('hostInv', 'f8'), ('n0', 'f8')])
ty = from_dtype(dtype)

#update community composition with Euler method
@jit(void(f8[:,::1], f8[:,::1], f8[::1], f8[::1], ty[::1], f8), nopython=True) 
def update_comm_loc(gMat, mutMat, basalBirthrate, invPerComm, hostTrait, dt):
    #get group properties
    nGroup = gMat.shape[0]
    migV = hostTrait['mig']
    rV = hostTrait['r']
    hostInv = hostTrait['hostInv']
    densPerGroup = gMat.sum(axis=1)
    globTypeMigPool = migV @ gMat 

    #effect of host of community birth rate
    hostBirthEffect = invPerComm * hostInv
    
    #calculate migration into host from global pool
    if nGroup > 1:
        migIn = globTypeMigPool / (nGroup - 1)
        migOutRate = (1 + 1 / (nGroup - 1)) * migV
    else:
        migIn = np.array([0.])
        migOutRate = migV
            
    #density dependent death rate per group    
    deathRatePerGroup = rV * densPerGroup
    #loop groups to calc change
    dx = np.zeros(gMat.shape)
    for i in range(nGroup):
        currGroup = gMat[i,:]
        births = rV[i] * (hostBirthEffect[i] + basalBirthrate) * currGroup
        birthMut = mutMat @ births
        migOut = migOutRate[i] * currGroup
        deaths =  deathRatePerGroup[i] * currGroup
        dx[i,:] =  birthMut - deaths - migOut + migIn
    
    gMat += dx * dt
    if np.any(gMat>2) | np.any(gMat<0):
        print("error")
    return  

"""
Full model 
"""

##run model main code
def run_model_fixed_parameters(model_par):
    #possible dt to choose from
    dtVec = np.logspace(-5, -2, 19)
    
    #calc timesteps    
    Num_t, Num_t_sample = mlsec.calc_max_num_timesteps(model_par, dtVec)     
    samplingInterval = model_par['sampleT']

    #get host rates
    B_H, D_H, TAU_H, K_H = [float(model_par[x]) for x in ('B_H','D_H','TAU_H','K_H')]
    #get bacterial rates
    r, mu, cost = [float(model_par[x]) for x in ('r','mu','cost')]
    numBins = int(model_par['numTypeBins'])

    #init randNum
    rndMat = mlsg.create_randMat(Num_t, 2)
    # init groups
    gMat = mlsec.init_comm(model_par)
    HVec = init_host_trait_matrix(model_par)
    mutMat, birthVec, gammaVec = create_local_update_matrix(r, mu, cost, numBins)

    #init output
    Output = mlsec.init_output_matrix(Num_t_sample)
    Output = init_host_output_matrix(Output, Num_t_sample)
    InvestmentAll = np.full((Num_t_sample, numBins), np.nan)
    InvestmentPerHost = np.full((Num_t_sample, numBins), np.nan)

    #first sample
    currT = 0
    ti = 0
    sampleIndex = 0
    sampleIndex = sample_full_model(gMat, HVec, gammaVec, Output, \
                               InvestmentAll, InvestmentPerHost, \
                               sampleIndex, currT, model_par)
    #get first time step
    dt = mlsec.calc_dynamic_timestep(model_par, gMat.shape[0], dtVec)

    #run time    
    while currT <= model_par['maxT']:
        #update community
        invPerCom = mlsec.calc_community_performance(gMat, gammaVec)
        update_comm_loc(gMat, mutMat, birthVec, invPerCom, HVec, dt)

        #check if there is host event
        birthProp, deathProp = mlsec.host_propensityLocal(invPerCom, dt, \
                                               B_H, D_H, TAU_H, K_H)

        #process host events
        if rndMat[ti,0] < birthProp:
            #process host birth event
            gMat, HVec = host_birth_event(gMat, HVec, gammaVec, rndMat[ti,1], model_par)
            #update time step to new number of groups
            dt = mlsec.calc_dynamic_timestep(model_par, gMat.shape[0], dtVec)
        elif rndMat[ti,0] < (birthProp + deathProp):
            #process host death event
            gMat, HVec = host_death_event(gMat, HVec, gammaVec, rndMat[ti,1], model_par)
            #update time step to new number of groups
            dt = mlsec.calc_dynamic_timestep(model_par, gMat.shape[0], dtVec)
   
        #stop run if all hosts die
        if  gMat.shape[0]==0:
            break

        #update time
        currT += dt
        ti += 1
        
        #sample model at intervals
        nextSampleT = samplingInterval * sampleIndex
        if currT >= nextSampleT:
            sampleIndex = sample_full_model(gMat, HVec, gammaVec, Output, \
                               InvestmentAll, InvestmentPerHost, \
                               sampleIndex, currT, model_par)
            
            #check if steady state has been reached
            if Output['rms_err'][sampleIndex-1] < model_par['rms_err_treshold']:
                break
            
    #cut off non existing time points at end
    Output = Output[0:sampleIndex]
    InvestmentAll = InvestmentAll[0:sampleIndex,:]
    InvestmentPerHost = InvestmentPerHost[0:sampleIndex,:]

    return (gMat, Output, InvestmentAll, InvestmentPerHost)

#run model, plot dynamics 
def single_run_with_plot(MODEL_PAR):
    #run code  
    start = time.time()
    gMat, Output, InvestmentAll, InvestmentPerHost = run_model_fixed_parameters(MODEL_PAR)
    end = time.time()
    
    print("Elapsed time run 1 = %s" % (end - start))
    
    font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 9}
    matplotlib.rc('font', **font)
    
    fig = plt.figure()
    nR=3
    nC=2
    
    #plot average investment  
    plt.subplot(nR,nC,1)  
    mlsec.plot_data(Output,"F_T_av")  
    mlsec.plot_data(Output,"F_mav")  
    plt.ylabel("investment") 
    plt.ylim((0, 1))
    
    #plot error
    plt.subplot(nR,nC,2)  
    mlsec.plot_data(Output,"rms_err",'log')  
    plt.ylabel("rms_err(t)") 


    #plot average host rates  
    plt.subplot(nR,nC,3)  
    mlsec.plot_data(Output,"mig_t",'log')  
    mlsec.plot_data(Output,"n0_t",'log')  
    plt.ylabel("rates") 
    plt.legend()

    #plot average host rates  
    plt.subplot(nR,nC,4)  
    mlsec.plot_data(Output,"inv_H_t")  
    mlsec.plot_data(Output,"r_t")  
    plt.ylabel("rates") 
    plt.legend()

    #plot investment distribution
    axs= plt.subplot(nR,nC,5)  
    currData = np.log10(InvestmentAll.transpose() + np.finfo(float).eps )
    im = axs.imshow(currData, cmap="viridis", \
                    interpolation='nearest', \
                    extent=[0,1,0,1], \
                    origin='lower', \
                    vmin=-4, vmax=-1,\
                    aspect= 'auto')
    axs.set_xticks([0, 1])
    axs.set_yticks([0, 1])    
    axs.set_ylabel('investment')
    axs.set_xlabel('time')
    cb = fig.colorbar(im, ax=axs, orientation='vertical', fraction=.1, label="log10 mean density")
    cb.set_ticks([-4, -3, -2, -1])
    axs.set_yticklabels([0, 1])
    maxTData =Output['time'].max()
    axs.set_xticklabels([0, int(round(maxTData))])
  
        
    #plot average investment per host
    axs= plt.subplot(nR,nC,6)  
    currData = np.log10(InvestmentPerHost.transpose() + np.finfo(float).eps )
    im = axs.imshow(currData, cmap="viridis", \
                    interpolation='nearest', \
                    extent=[0,1,0,1], \
                    origin='lower', \
                    vmin=-2, vmax=-1,\
                    aspect= 'auto')
    axs.set_xticks([0, 1])
    axs.set_yticks([0, 1])
    axs.set_ylabel('investment')
    axs.set_xlabel('time')
    cb = fig.colorbar(im, ax=axs, orientation='vertical', fraction=.1, label="log10 mean density")
    cb.set_ticks([-2, -1])
    axs.set_yticklabels([0, 1])
    maxTData =Output['time'].max()
    axs.set_xticklabels([0, int(round(maxTData))])

    fig.set_size_inches(4,4)
    plt.tight_layout()

    return (gMat, Output, InvestmentAll, InvestmentPerHost)

#run model with default parameters
def debug_code():
    model_par = {
                #time step parameters
                "maxT"  : 40000., 
                "sampleT": 10,
                "rms_err_treshold": 1E-5,
                "mav_window": 1000,
                "rms_window": 5000,
                #fixed model parameters
                "sampling" : "sample",
                "mu"    : 0.02,
                "B_H"   : 3.,
                "D_H"   : 0.,
                "K_H"   : 15.,
                #variable model parameters
                "cost" : 0.01,
                "TAU_H" : 10.,
                "n0"    : 1E-3,
                "mig"   : 1E-5,
                "r"     : 1.,
                "K"     : 5E3,
                "hostInv0" : 0.,
                #fixed intial condition
                "NUMGROUP" : 5,  
                "numTypeBins" : 100,
                "meanGamma0" : 0,
                "stdGamma0" : 0.01,
                "N0init" : 1.,
                #host evolution settings
                "HostEvolves" : True,
                "sigmaR" : 0,
                "sigmaMig" : 0.1,
                "sigmaN0" : 0.05,
                "sigmaHostInv" : 0
        }
    
    gMat, Output, InvestmentAll, InvestmentPerHost = single_run_with_plot(model_par)
    
    return Output, InvestmentAll, InvestmentPerHost

if __name__ == "__main__":
    print("running debug")
    Output, InvestmentAll, InvestmentPerHost = debug_code()