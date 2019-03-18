#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
CreateD on WeD OCt 17 12:25:24 2018

@author: simonvanvliet
"""
import math
import numpy as np
import scipy.stats as st
#import scipy.sparse as spar
import matplotlib.pyplot as plt
import time
from numba import jit, void, f8, i8 
from numba.types import UniTuple, Tuple


# calculates time step to use to keep max 1 host event per step
def calc_timestep(model_par):
    dtVec = np.logspace(-5, -2, 17)
    
    maxHostProp = 2 * (1 + model_par["B_H"])**2 \
        / (model_par["D_H"] * model_par["TAU_H"])
        
    p0 = np.exp(-maxHostProp * dtVec)  
    p1 = maxHostProp * dtVec * np.exp(-maxHostProp * dtVec) 
    p2 = 1 - p0 - p1
    dtMax = dtVec[p2<0.01].max()
    #print(f'Using dT {dtMax:.2e}\n')
    
    model_par['dt']=dtMax

    Num_t = int(np.ceil(model_par['maxT'] / model_par['dt']))
    Num_t_sample = int(np.ceil(model_par['maxT'] / model_par['sampleT'])+1)

    return (Num_t, Num_t_sample)


def calc_maxGroupNum(model_par):
    carCap = (1 + model_par["B_H"]) / model_par["D_H"]
    maxNum = int(2 * carCap)
    return maxNum
    

def init_comm(model_par): 
    
    numGammaBin = model_par['numTypeBins']
    meanGamma = model_par['meanGamma0']
    stdGamma = model_par['stdGamma0']
    numGroup = int(model_par['NUMGROUP'])
    N0 = model_par['N0init']

    dGamma = 1 / numGammaBin    
    gammaVec = np.linspace(dGamma/2, 1-dGamma/2, numGammaBin)
    zVec = (gammaVec - meanGamma) / stdGamma
    initDistri = st.norm.pdf(zVec)
    #initDistri /= initDistri.sum()
    initDistri *= (N0 / initDistri.sum())
    
    groupMatrix = np.broadcast_to(initDistri,(numGroup,numGammaBin))
    groupMatrix = np.copy(groupMatrix, order='C')
    
    return groupMatrix


def create_randMat(Num_t):
    notDone = True
    iter = 0 
    while notDone:
        iter += 1
        randMat = np.random.random((Num_t,2))
        
        containsNo0 = (~np.any(randMat == 0))
        containsNo1 = (~np.any(randMat == 1))
        if containsNo0 & containsNo1:
            notDone = False
    
    return randMat


@jit(UniTuple(f8,2)(f8[:,::1], f8[:], i8))    
def calc_mean_fraction(gMat, gammaVec, nGroup):
    invPerGroup = gMat @ gammaVec 
    F_av = invPerGroup.mean()
    N_av = gMat.sum() / nGroup

    return (F_av,N_av)


@jit(f8(f8[:], i8, i8))    
def calc_moving_av(f_t, curr_idx, windowLength):
    start_idx=max(0,curr_idx-windowLength+1)
    movingAv = f_t[start_idx:curr_idx].mean()
   
    return movingAv


@jit(f8(f8[:], i8, i8))    
def calc_rms_error(mav_t, curr_idx, windowLength):
    start_idx=max(0,curr_idx-windowLength+1)
    
    localsegment = mav_t[start_idx:curr_idx]
    av = localsegment.mean()
    errorSquared = (localsegment-av)**2
    meanErrorSquared = errorSquared.mean()
    rms = math.sqrt(meanErrorSquared)
   
    return rms

def sample_model(gMat, gammaVec, output, OutputState, sample_idx, currT, movingAvWind):  
    
    nGroup = gMat.shape[0]
    
    F_av, N_av = calc_mean_fraction(gMat, gammaVec, nGroup)
    
    output['F_T_av'][sample_idx] = F_av
    output['N_T_av'][sample_idx] = N_av

    output['H_T'][sample_idx] = nGroup
    
    if sample_idx >= movingAvWind:
        mav = calc_moving_av(output['F_T_av'], sample_idx, movingAvWind)
        output['F_mav'][sample_idx] = mav
        
    if sample_idx >= 2*movingAvWind:
        rms_err = calc_rms_error(output['F_mav'], \
                                         sample_idx, movingAvWind)
        output['rms_err'][sample_idx] = rms_err

    output['time'][sample_idx] = currT

    OutputState[sample_idx,:] = gMat.mean(axis=0)

    sample_idx += 1
    
    return sample_idx

def select_host_to_die(numGroup, randNum):
    id_group = int(np.floor(randNum*numGroup))
    return id_group

@jit(f8[:](f8[::1], f8))
def host_birth_composition_copy(parComp, n0):
    densPar = parComp.sum()

    #keep distribution reduce total density to n0
    dilution = n0 / densPar
    offComp = parComp * dilution

    return offComp

@jit(f8[:](f8[::1], f8, f8))
def host_birth_composition_sample(parComp, n0, K):
    N0 = max(int(np.ceil(n0*K)),1)
    randNum = np.random.rand(N0)

    #select group to reproduce
    cumPropensity = parComp.cumsum()
    randNumScaled = randNum * cumPropensity[-1]
    
    index = np.arange(cumPropensity.size)
    
    sample = [index[(cumPropensity>x)][0] for x in randNumScaled]
    
    offComp = np.zeros(parComp.shape)
    
    for idx in sample:
        offComp[idx] += 1
      
    offComp *= n0 / N0    
      
    return offComp


@jit(i8(f8[:,::1], f8, f8, f8, f8, i8), nopython=True)
def select_host_to_reproduce(groupMat, randNum, B_H, TAU_H, dt, numBins):
    
    dGamma = 1 / numBins    
    gammaVec = np.linspace(dGamma/2, 1-dGamma/2, numBins)
    gammaVec = np.reshape(gammaVec,(numBins,1))

    #calculate propensity for each group
    investmentPerGroup = groupMat @ gammaVec
    BirthProp = 1/TAU_H * (1 + B_H * investmentPerGroup) * dt
    
    #select group to reproduce
    cumPropensity = BirthProp.cumsum()
    randNumScaled = randNum * cumPropensity[-1]
    
    index = np.arange(cumPropensity.size)
    id_group = index[(cumPropensity>randNumScaled)][0]
    
    
    return id_group    

def host_birth_event(groupMat, randNum, model_par):
    #extract parameters
    B_H = model_par["B_H"]
    TAU_H = model_par["TAU_H"]
    dt = model_par["dt"]
    n0 = model_par['n0']
    K = model_par['K']
    numBins = model_par['numTypeBins']
    
    #select host to reproduce
    id_group = select_host_to_reproduce(groupMat, randNum, B_H, TAU_H, dt, numBins)
    
    #select parent group and  find offspring composition
    parComp = groupMat[id_group, :]
    if model_par['sampling']=="copy":
        offComp = host_birth_composition_copy(parComp, n0)
    elif model_par['sampling']=="sample":
        offComp = host_birth_composition_sample(parComp, n0, K)
    else:
       raise Exception('Unknown sampling procedure')     

    #add new group to group matrix 
    groupMatNew = np.append(groupMat, np.atleast_2d(offComp), axis=0)
    
    return groupMatNew
        

def host_death_event(groupMatrix, randNum): 
    #select host to die
    numGroup = groupMatrix.shape[0]
    id_group = select_host_to_die(numGroup, randNum)

    #delete group
    groupMatNew = np.delete(groupMatrix, id_group, axis=0)
    
    return groupMatNew
        
def create_local_update_matrix(r, mu, mig, cost, numBins):
    #extract model parameters
    dGamma = 1 / numBins    
    gammaVec = np.linspace(dGamma/2, 1-dGamma/2, numBins)
    costVec = gammaVec * cost
    #calculate rates
    birthmigRate = (1-mu) * (1  - costVec) * r - mig
    mutationRate = mu / 2  * (1  - costVec) * r

    #create sub matrices of size of single group
    locBirthMigOut = np.diag(birthmigRate) + \
               np.diag(mutationRate[0:-1], -1) + \
               np.diag(mutationRate[1:],  1)           
               
    return (locBirthMigOut, gammaVec)

@jit(UniTuple(f8,2)(f8[:,::1], f8[::1], f8, f8,  f8, f8), nopython=True)
def host_propensityLocal(gMat, gammaVec, dt, B_H, D_H, TAU_H): 
    NumHost = gMat.shape[0]
    invPerGroup = gMat @ gammaVec   
    totalInvest = invPerGroup.sum()
    totBirthProp = dt / TAU_H * (NumHost + B_H * totalInvest) 
    totDeathProp = dt / TAU_H * D_H * NumHost ** 2
    
    return (totBirthProp, totDeathProp)

def init_output_matrix(Num_t_sample, NumBins):
    #init output
    dType = np.dtype([('F_T_av', 'f8'), \
                      ('N_T_av', 'f8'), \
                      ('H_T', 'f8'), \
                      ('F_mav', 'f8'), \
                      ('rms_err', 'f8'), \
                      ('time', 'f8')])
    
    Output = np.full(Num_t_sample, np.nan, dType)
    Output['time'][0] = 0

    OutputState = np.full((Num_t_sample,NumBins), np.nan)

    return Output, OutputState

@jit(void(f8[:,::1], f8[:,::1], f8, f8, f8), nopython=True)   
def update_comm_loc(gMat, locBMMat, r, mig, dt):
    nGroup = gMat.shape[0]
    
    densPerGroup = gMat.sum(axis=1)
    globTypeFrac = gMat.sum(axis=0)
    
    migIn = globTypeFrac * mig / (nGroup - 1)
    deathRatePerGroup = r * densPerGroup
    
    dx = np.empty_like(gMat)
    for i in range(nGroup):
        currGroup = gMat[i,:]
        currBMout = locBMMat @ currGroup 
        currDeath = currGroup * deathRatePerGroup[i]
        dx[i,:] =  currBMout - currDeath + migIn
    
    gMat += dx * dt
    return  


def run_model_fixed_parameters(model_par):
   
    #calc timesteps    
    Num_t, Num_t_sample = calc_timestep(model_par)   
    Num_t += 1
    samplingInterval = model_par['sampleT']
    timeAvWindow = 100
    #maxGroupNum = calc_maxGroupNum(model_par)


    #host rates
    dt, B_H, D_H, TAU_H = [float(model_par[x]) for x in ('dt','B_H','D_H','TAU_H')]


    print(('using dt=%g' %dt))
    #bacterial rates
    r, mu, mig, cost = [float(model_par[x]) for x in ('r','mu','mig', 'cost')]
    numBins = int(model_par['numTypeBins'])
    #init randNum
    rndMat = create_randMat(Num_t)
    
    # init groups
    gMat = init_comm(model_par)
    locBMMat, gammaVec = create_local_update_matrix(r, mu, mig, cost, numBins)
    
    #init output
    Output, OutputState = init_output_matrix(Num_t_sample, numBins)

    #init sample
    currT = 0
    sampleIndex = 0
    sampleIndex = sample_model(gMat, gammaVec, Output, OutputState, \
                        sampleIndex, currT, timeAvWindow)
    
    #run time    
    for ti in range(Num_t):
        
        #update community
        update_comm_loc(gMat, locBMMat, r, mig, dt)
     
        #check if there is host event
        birthProp, deathProp = host_propensityLocal(gMat, gammaVec, dt, \
                                               B_H, D_H, TAU_H)
        
        #process host events
        if rndMat[ti,0] < birthProp:
            #process host birth event
            gMat = host_birth_event(gMat, rndMat[ti,1], model_par)
        elif rndMat[ti,0] < (birthProp + deathProp):
            #process host death event
            gMat = host_death_event(gMat, rndMat[ti,1])
            
        #stop run if all hosts die
        if  gMat.shape[0]==0:
            break

        #update time
        currT += model_par['dt']
        
        #sample model at intervals
        nextSampleT = samplingInterval * sampleIndex
        if currT >= nextSampleT:
            sampleIndex = sample_model(gMat, gammaVec, Output, OutputState, \
                        sampleIndex, currT, timeAvWindow)
            
#            #check if steady state has been reached
#            if Output['rms_err'][sampleIndex] < 5E-3:
#                Output = Output[0:sampleIndex]
#                OutputState = OutputState[0:sampleIndex,:]
#                break
            
    
    Output = Output[0:sampleIndex]
    OutputState = OutputState[0:sampleIndex,:]
    
    return (Output, OutputState, gMat)



def plot_data(dataStruc, FieldName):
    plt.plot(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    plt.xlabel("time")
    maxTData =dataStruc['time'].max()
    try:
        plt.xlim((0,maxTData))
    except:
        print(maxTData)
    return
  
def single_run_finalstate(MODEL_PAR):
    
    Output, OutputState, gMat = run_model_fixed_parameters(MODEL_PAR)
            
    dType = np.dtype([ \
              ('F_T_av', 'f8'), \
              ('F_mav', 'f8'), \
              ('F_mav_ss', 'f8'),\
              ('N_T_av', 'f8'), \
              ('H_T', 'f8'),   \
              ('gamma', 'f8'), \
              ('tau_H', 'f8'), \
              ('n0', 'f8'),  \
              ('mig', 'f8'), \
              ('r', 'f8'),   \
              ('K', 'f8')])     

    output_matrix = np.zeros(1, dType)
    
    output_matrix['F_T_av'] = Output['F_T_av'][-1]
    output_matrix['F_mav'] = Output['F_mav'][-1]
    
    #store nan if not reached steady state
    Num_t = Output.size
    Num_t_end = int(np.ceil(MODEL_PAR['maxT'] / MODEL_PAR['sampleT'])+1)
    
    if Num_t < Num_t_end:
        output_matrix['F_mav_ss'] = Output['F_mav'][-1]
    else:
        output_matrix['F_mav_ss'] = np.nan
        
    output_matrix['N_T_av'] = Output['N_T_av'][-1]
    output_matrix['H_T'] = Output['H_T'][-1]

    output_matrix['gamma'] = MODEL_PAR['gamma']
    output_matrix['tau_H'] = MODEL_PAR['TAU_H']
    output_matrix['n0'] = MODEL_PAR['n0']
    output_matrix['mig'] = MODEL_PAR['mig']
    output_matrix['r'] = MODEL_PAR['r']
    output_matrix['K'] = MODEL_PAR['K']   
    
    return output_matrix



def single_run_with_plot(MODEL_PAR):
    
    #run code  
    start = time.time()
    Output, OutputState, gMat = run_model_fixed_parameters(MODEL_PAR)
    end = time.time()
    print("Elapsed time run 1 = %s" % (end - start))
    
    
    
    fig = plt.figure()
    #plot data  
    nR=2
    nC=2
    plt.subplot(nR,nC,1)  
    plot_data(Output,"N_T_av")  
    plt.ylabel("N(t)") 
    
    plt.subplot(nR,nC,3)  
    plot_data(Output,"F_T_av")  
    #plot_data(Output,"F_mav")  
    plt.ylabel("investment") 
    #plt.legend()
    
    plt.subplot(nR,nC,2)  
    plot_data(Output,"H_T")  
    plt.ylabel("H(t)") 

    
    axs= plt.subplot(nR,nC,4)  
    
    currData = np.log10(OutputState.transpose() + np.finfo(float).eps )
    im = axs.imshow(currData, cmap="viridis", \
                    interpolation='nearest', \
                    extent=[0,1,0,1], \
                    origin='lower', \
                    vmin=-4, vmax=-1,\
                    aspect= 'auto')
    axs.set_xticks([0, 1])
    axs.set_yticks([0, 1])
    #axs.set_title(titlename)
    
    axs.set_ylabel('investment')
    axs.set_xlabel('time')
    
    cb = fig.colorbar(im, ax=axs, orientation='vertical', fraction=.1, label="log10 mean density")
    cb.set_ticks([-4, -3, -2, -1])
    
    axs.set_yticklabels([0, 1])
    
    maxTData =Output['time'].max()
    try:
        axs.set_xticklabels([0, int(round(maxTData))])
    except:
        print('oh no')
    
    
#    plot_data(Output,"rms_err") 
#    plt.ylabel("err of mav") 
#    
#    
    #fig.set_figwidth(10, forward=True)
    plt.tight_layout()

    return (Output, OutputState)
