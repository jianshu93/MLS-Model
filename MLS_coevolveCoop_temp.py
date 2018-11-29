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
from numba import jit, void, f8, i8, from_dtype
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
    Num_t += 1

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
    hostInv0 = model_par['hostInv0']

    dGamma = 1 / numGammaBin    
    gammaVec = np.linspace(dGamma/2, 1-dGamma/2, numGammaBin)
    zVec = (gammaVec - meanGamma) / stdGamma
    initDistri = st.norm.pdf(zVec)
    #initDistri /= initDistri.sum()
    initDistri *= (N0 / initDistri.sum())
    
    groupMatrix = np.broadcast_to(initDistri,(numGroup,numGammaBin))
    groupMatrix = np.copy(groupMatrix, order='C')


    dtype = np.dtype([('mig', 'f8'), ('r', 'f8'),\
            ('hostInv', 'f8'), ('n0', 'f8')])
    Host_traits = np.zeros(numGroup, dtype)
    
    Host_traits['mig'] = model_par["mig"]
    Host_traits['r'] = model_par["r"]
    Host_traits['n0'] = model_par["n0"]
    Host_traits['hostInv'] = model_par["hostInv0"]

    
    return (groupMatrix, Host_traits)


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

def sample_model(gMat, hostTrait, gammaVec, \
                output, OutputState, sample_idx, currT, movingAvWind):  
    
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

    output['HInv_T'][sample_idx] = hostTrait['hostInv'].mean()
    output['M_T'][sample_idx] = hostTrait['mig'].mean()
    output['R_T'][sample_idx] = hostTrait['r'].mean()
    output['N0_T'][sample_idx] = hostTrait['n0'].mean()

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

def trunc_norm(exp, std, min=-np.inf, max=np.inf, type='lin'):
    if type == 'log':
        exp = math.log10(exp)
        min = math.log10(min)
        max = math.log10(max)
    elif type != 'lin':
        raise Exception('problem with trunc_norm only support lin or log mode')

    minT = (min - exp) / std
    maxT = (max - exp) / std
    newT = st.truncnorm.rvs(minT, maxT) * std + exp

    if type == 'log':
        newT = 10 ** newT

    return newT

@jit(i8(f8[:,::1], f8, f8, f8, f8, i8), nopython=True)
def select_host_to_reproduce(groupMat, randNum, B_H, TAU_H, dt, numBins):
    
    nGroup = groupMat.shape[0]
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
    
    if (id_group<0) | (id_group >= nGroup):
        print(BirthProp)
        print(investmentPerGroup)
        print(randNum)
        print(id_group)
        Exception('problem with ID Group')
    
    
    return id_group    

def host_birth_trait(parTrait, model_parameter):
    r_par = parTrait['r']
    m_par = parTrait['mig'] 
    n0_par = parTrait['n0'] 
    hostInv_par = parTrait['hostInv'] 

    traitsStd = ['sigmaR', 'sigmaMig', 'sigmaN0', 'sigmaHostInv']
    stdR, stdM, stdN0, stdHostInv = [model_parameter[x] for x in traitsStd]

    if stdR>0:
        rOff = trunc_norm(r_par, stdR, min=0, max=10, type='lin')
    else:
        rOff = r_par   
    if stdM>0:
        mOff = trunc_norm(m_par, stdM, min=1E-15, max=0.5, type='log')
    else:
        mOff = m_par  
    if stdN0>0:
        n0Off = trunc_norm(n0_par, stdN0, min=1E-15, max=0.5, type='log')
    else:
        n0Off = n0_par  
    if stdHostInv>0:
        hostInvOff = trunc_norm(hostInv_par, stdHostInv, \
                                 min=10, max=1, type='lin')
    else:
        hostInvOff = hostInv_par     
    
    newTraits = np.array(1, dtype=parTrait.dtype)
    newTraits['r'] = rOff
    newTraits['mig'] = mOff
    newTraits['n0'] = n0Off
    newTraits['hostInv'] = hostInvOff

    return newTraits

def host_birth_event(groupMat, host_trait, randNum, model_par):
    #extract parameters
    B_H = model_par["B_H"]
    TAU_H = model_par["TAU_H"]
    dt = model_par["dt"]
    K = model_par['K']
    numBins = model_par['numTypeBins']
    sigma_inv = model_par['sigmaHostInv']

    #select host to reproduce
    id_group = select_host_to_reproduce(groupMat, randNum, B_H, TAU_H, dt, numBins)
    n0 = host_trait['n0'][id_group]
 
    #select parent group and  find offspring composition
    try:
        parComp = groupMat[id_group, :]
    except:
        print(id_group)
        print(groupMat.shape)

    if model_par['sampling']=="copy":
        offComp = host_birth_composition_copy(parComp, n0)
    elif model_par['sampling']=="sample":
        offComp = host_birth_composition_sample(parComp, n0, K)
    else:
       raise Exception('Unknown sampling procedure')     

    #add new group to group matrix 
    groupMatNew = np.append(groupMat, np.atleast_2d(offComp), axis=0)

    if model_par['HostEvolves']:
        #new host traits
        newTraits = host_birth_trait(host_trait[id_group], model_par)
        host_traitNew = np.append(host_trait, newTraits)
    else:
        host_traitNew = host_trait[id_group]

    return (groupMatNew, host_traitNew)
        

def host_death_event(groupMatrix, host_trait, randNum): 
    #select host to die
    numGroup = groupMatrix.shape[0]
    id_group = select_host_to_die(numGroup, randNum)

    #delete group
    groupMatNew = np.delete(groupMatrix, id_group, axis=0)
    host_traitNew = np.delete(host_trait, id_group)
    
    return (groupMatNew, host_traitNew)
        
def create_local_update_matrix(r, mu, mig, cost, numBins):
    #extract model parameters
    dGamma = 1 / numBins    
    gammaVec = np.linspace(dGamma/2, 1-dGamma/2, numBins)
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

@jit(UniTuple(f8,2)(f8[:,::1], f8[::1], f8, f8,  f8, f8), nopython=True)
def host_propensityLocal(gMat, invPerGroup, dt, B_H, D_H, TAU_H): 
    NumHost = gMat.shape[0]
    totalInvest = invPerGroup.sum()
    totBirthProp = dt / TAU_H * (NumHost + B_H * totalInvest) 
    totDeathProp = dt / TAU_H * D_H * NumHost ** 2
    
    return (totBirthProp, totDeathProp)

def init_output_matrix(Num_t_sample, NumBins):
    #init output
    dType = np.dtype([('F_T_av', 'f8'), \
                      ('N_T_av', 'f8'), \
                      ('H_T', 'f8'), \
                      ('HInv_T', 'f8'),
                      ('N0_T', 'f8'), \
                      ('R_T', 'f8'), \
                      ('M_T', 'f8'), \
                      ('F_mav', 'f8'), \
                      ('rms_err', 'f8'), \
                      ('time', 'f8')])
    
    Output = np.full(Num_t_sample, np.nan, dType)
    Output['time'][0] = 0

    OutputState = np.full((Num_t_sample,NumBins), np.nan)

    return Output, OutputState


dtype = np.dtype([('mig', 'f8'), ('r', 'f8'),\
                ('hostInv', 'f8'), ('n0', 'f8')])
ty = from_dtype(dtype)

@jit(void(f8[:,::1], f8[:,::1], f8[::1], f8[::1], ty[::1], f8), nopython=True) 
def update_comm_loc(gMat, mutMat, basalBirthrate, invPerComm,  hostTrait, dt):
    nGroup = gMat.shape[0]

    migV = hostTrait['mig']
    rV = hostTrait['r']
    hostInv = hostTrait['hostInv']

    hostBirthEffect = invPerComm * hostInv

    densPerGroup = gMat.sum(axis=1)

    globTypeMigPool = migV @ gMat 
    
    migInRate = globTypeMigPool / (nGroup - 1)
    migOutRate = (1 + 1 / (nGroup - 1)) * migV

    deathRatePerGroup = rV * densPerGroup

    dx = np.empty_like(gMat)
    for i in range(nGroup):
        currGroup = gMat[i,:]
        births = rV[i] * (hostBirthEffect[i] + basalBirthrate) * currGroup
        birthMut = mutMat @ births
        migOut = migOutRate[i] * currGroup
        deaths =  deathRatePerGroup[i] * currGroup
        dx[i,:] =  birthMut - deaths - migOut + migInRate
    
    gMat += dx * dt
    return  

@jit(UniTuple(f8[:],2)(f8[:,::1], f8[::1]))
def calc_community_perform(gMat, gammaVec):
    #calc community investment
    invPerComm = gMat @ gammaVec 

    return invPerComm



def run_model_fixed_parameters(model_par):
   
    #calc timesteps    
    Num_t, Num_t_sample = calc_timestep(model_par)   
    samplingInterval = model_par['sampleT']
    timeAvWindow = 100

    #host rates
    dt, B_H, D_H, TAU_H = [float(model_par[x]) for x in ('dt','B_H','D_H','TAU_H')]

    #bacterial rates
    r, mu, mig, cost = [float(model_par[x]) for x in ('r','mu','mig', 'cost')]
    numBins = int(model_par['numTypeBins'])

    #init randNum
    rndMat = create_randMat(Num_t)
    
    # init groups
    gMat, HVec = init_comm(model_par)
    mutMat, birthVec, gammaVec = create_local_update_matrix(r, mu, mig, cost, numBins)

    #init output
    Output, OutputState = init_output_matrix(Num_t_sample, numBins)

    #init sample
    currT = 0
    sampleIndex = 0
    sampleIndex = sample_model(gMat, HVec, gammaVec, Output, OutputState, \
                        sampleIndex, currT, timeAvWindow)
    
    #run time    
    for ti in range(Num_t):
        
        invPerComm = calc_community_perform(gMat, gammaVec)

        #update community
        update_comm_loc(gMat, mutMat, birthVec, invPerComm, HVec, dt)
     
        #check if there is host event
        birthProp, deathProp = host_propensityLocal(gMat, invPerComm, dt, \
                                               B_H, D_H, TAU_H)
        
        #process host events
        if rndMat[ti,0] < birthProp:
            #process host birth event
            gMat, HVec = host_birth_event(gMat, HVec, rndMat[ti,1], model_par)
        elif rndMat[ti,0] < (birthProp + deathProp):
            #process host death event
            gMat, HVec = host_death_event(gMat, HVec, rndMat[ti,1])
            
        #stop run if all hosts die
        if  gMat.shape[0]==0:
            break

        #update time
        currT += model_par['dt']
        
        #sample model at intervals
        nextSampleT = samplingInterval * sampleIndex
        if currT >= nextSampleT:
            sampleIndex = sample_model(gMat, HVec, gammaVec, Output, OutputState, \
                        sampleIndex, currT, timeAvWindow)
            
#            #check if steady state has been reached
#            if Output['rms_err'][sampleIndex] < 5E-3:
#                Output = Output[0:sampleIndex]
#                OutputState = OutputState[0:sampleIndex,:]
#                break
            
    Output = Output[0:sampleIndex]
    OutputState = OutputState[0:sampleIndex,:]
    
    return (Output, OutputState, gMat)

  
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

def plot_data(dataStruc, FieldName, type='lin'):
    if type == 'lin':
        plt.plot(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    elif type == 'log':
        plt.semilogy(dataStruc['time'], dataStruc[FieldName], label=FieldName)
        
        try:
            maxY = np.nanmax(dataStruc[FieldName])
            minY = np.nanmin(dataStruc[FieldName])
            
            minYAx = np.floor(np.log10(minY) - 0.01)
            maxYAx = np.ceil(np.log10(maxY) + 0.01)
            plt.ylim(10**minYAx, 10**maxYAx)
        except:
           a=1    
            
    else:
        Exception('only support lin or log plot type')

    plt.xlabel("time")
    maxTData = np.nanmax(dataStruc['time'])
    plt.xlim((0,maxTData))

    return

def single_run_with_plot(MODEL_PAR):
    
    #run code  
    start = time.time()
    Output, OutputState, gMat = run_model_fixed_parameters(MODEL_PAR)
    end = time.time()
    print("Elapsed time run 1 = %s" % (end - start))
    
    
    
    fig = plt.figure()
    #plot data  
    nR=2
    nC=4
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
    
    maxTData = np.nanmax(Output['time'])
    axs.set_xticklabels([0, int(round(maxTData))])
    
    plt.subplot(nR,nC,5)  
    plot_data(Output,"HInv_T", type='lin') 
    plt.ylabel("Host investment") 
    plt.ylim(0,1)

    plt.subplot(nR,nC,6)  
    plot_data(Output,"N0_T", type='log') 
    plt.ylabel("Inoculum N0") 
    
    plt.subplot(nR,nC,7)  
    plot_data(Output,"M_T", type='log') 
    plt.ylabel('Migration $\\theta$') 

    plt.subplot(nR,nC,8)  
    plot_data(Output,"R_T", type='lin') 
    plt.ylabel("Turnover r") 
    maxY = np.ceil(np.nanmax(Output['R_T'])+1)
    try:
        plt.ylim(0,maxY)
    except:
        a=1
    

    #fig.set_figwidth(10, forward=True)
    plt.tight_layout()
    
    return Output, OutputState
