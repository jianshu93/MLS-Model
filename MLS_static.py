#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
CreateD on WeD OCt 17 12:25:24 2018

@author: simonvanvliet
"""
import math
#import random
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time
from numba import jit, void, float64, int64 
from numba.types import UniTuple
#import gc
import traceback
#MODEL_PAR = {
#    "maxT"  : 2000, 
#    "sampleT": 1.,
#    "gamma" : 0.01,
#    "mu"    : 1.E-5,
#    "B_H"   : 3.,
#    "D_H"   : 1./40.,
#    "TAU_H" : 100.,
#    "mig"   : 1e-8,
#    "n0"    : 1e-4,
#    "r"     : 1.
#}
#
#INIT_COND = {
#    "NUMGROUP" : 100,  
#    "F0" : 0.01,
#    "N0" : 1,
#}        
 
# calculates time step to use to keep max 1 host event per step
def calc_timestep(model_par):
    dtVec = np.logspace(-5, 1, 17)
    
    maxHostProp = 2 * (1 + model_par["B_H"])**2 \
        / (model_par["D_H"] * model_par["TAU_H"])
        
    p0 = np.exp(-maxHostProp * dtVec)  
    p1 = maxHostProp * dtVec * np.exp(-maxHostProp * dtVec) 
    p2 = 1 - p0 - p1
    dtMax = dtVec[p2<0.01].max()
    #print(f'Using dT {dtMax:.2e}\n')
    
    model_par['dt']=dtMax
    return 


def calc_maxGroupNum(model_par):
    carCap = (1 + model_par["B_H"]) / model_par["D_H"]
    maxNum = int(2 * carCap)
    return maxNum
    

#initialize structured array of host properties
def init_comm(model_par): 
    
    numGroup = int(model_par["NUMGROUP"])
    cAct = np.full(numGroup, \
                   model_par["N0init"] * model_par["F0"])
    dAct = np.full(numGroup, \
                   model_par["N0init"] * (1 - model_par["F0"]))
    return (cAct, dAct)


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


@jit(UniTuple(float64,4)(float64[:],float64[:]))    
def calc_mean_fraction(c, d):
    TOT = c + d
    FRAC = c / TOT
    F_av = FRAC.mean()
    N_av = TOT.mean()
    F_std = FRAC.std()
    N_std = TOT.std()

    return (F_av,N_av,F_std,N_std)


@jit(UniTuple(float64,2)(float64[:],int64,int64))    
def calc_moving_av_and_cv(f_t, curr_idx, windowLength):
    start_idx=max(0,curr_idx-windowLength+1)
    
    avF_t = f_t[start_idx:curr_idx].mean()
    stdF_t = f_t[start_idx:curr_idx].std()
    cvF_t = stdF_t / avF_t
    
    return (avF_t,cvF_t) 


@jit(float64(float64[:],int64,int64))    
def calc_moving_cv(mav_t, curr_idx, windowLength):
    start_idx=max(0,curr_idx-windowLength+1)
    
    avF_t = mav_t[start_idx:curr_idx].mean()
    stdF_t = mav_t[start_idx:curr_idx].std()
    cvF_t = stdF_t / avF_t
    
    return cvF_t

@jit(float64(float64[:],int64,int64))    
def calc_rms_error(mav_t, curr_idx, windowLength):
    start_idx=max(0,curr_idx-windowLength+1)
    
    localsegment = mav_t[start_idx:curr_idx]
    av = localsegment.mean()
    errorSquared = (localsegment-av)**2
    meanErrorSquared = errorSquared.mean()
    rms = math.sqrt(meanErrorSquared)
   
    return rms


def sample_model(c, d, output, sample_idx, movingAvWind):  
    F_av, N_av, F_std, N_std = calc_mean_fraction(c, d)
    

    output['F_T_av'][sample_idx] = F_av
    output['N_T_av'][sample_idx] = N_av
    
    output['F_T_std'][sample_idx] = F_std
    output['N_T_std'][sample_idx] = N_std

    output['H_T'][sample_idx] = c.size
    
    if sample_idx >= movingAvWind:
        mav, mcv = calc_moving_av_and_cv(output['F_T_av'], \
                                         sample_idx, movingAvWind)
        output['F_mav'][sample_idx] = mav
        output['F_mcv'][sample_idx] = mcv
        
    if sample_idx >= 2*movingAvWind:
#        
#        cv_mav = calc_moving_cv(output['F_mav'], \
#                                         sample_idx, movingAvWind)
#        output['cv_mav'][sample_idx] = cv_mav
        rms_err = calc_rms_error(output['F_mav'], \
                                         sample_idx, movingAvWind)
        output['rms_err'][sample_idx] = rms_err
        
    
    return

def select_host_to_die(c, randNum):
    numHost=c.size
    id_group = int(np.floor(randNum*numHost))
    return id_group


def host_birth_composition_norm(cPar, dPar, n0, K):
    densPar = cPar + dPar
    fracPar = cPar / densPar
    densOff = n0
    #continous limit of binomial distribution
    expC = densOff * fracPar  #np
    stdC = math.sqrt(expC * (1-fracPar) / K) #np(1-p)
    minC = (0 - expC) / stdC
    maxC = (min(densPar,densOff) - expC) / stdC

    cOff = st.truncnorm.rvs(minC, maxC) * stdC + expC
    dOff = densOff - cOff
    
    return (cOff, dOff)


def host_birth_composition_binom(cPar, dPar, n0, K):
    densPar = np.asscalar(cPar + dPar)
    fracPar = np.asscalar(cPar / densPar)
    
    numParInt = int(np.round(densPar*K))
    
    if numParInt>0:
        cParInt = int(np.round(fracPar*densPar*K))
        fracParInt = cParInt / numParInt
    
        numOffTarget = int(np.round(n0*K))    
        numOff = min(numOffTarget,numParInt)
        
        try:
            cOffInt = st.binom.rvs(numOff, fracParInt)
        except:
            raise Exception('problem with binom')
        cOff = (cOffInt)/K
        dOff = (numOff - cOffInt)/K   
    else:
        cOff = 0
        dOff = 0
    return (cOff, dOff)


def host_birth_composition_hypgeo(cPar, dPar, n0, K):
    densPar = np.asscalar(cPar + dPar)
    fracPar = np.asscalar(cPar / densPar)
    
    numParInt = int(np.round(densPar*K))
    cParInt = int(np.round(fracPar*densPar*K))

    numOffTarget = int(np.round(n0*K))    
    numOff = min(numOffTarget,numParInt)
    
    if (numOff>0) & (numParInt>0):
        #numOff = max(int(np.round(n0*K)),1)  
        try:
            cOffInt = st.hypergeom.rvs(numParInt, cParInt, numOff)
        except:
            print("hi")
            print((numParInt, cParInt, numOff))
            raise Exception('problem with hypergeom')
        cOff = (cOffInt)/K
        dOff = (numOff - cOffInt)/K    
    else:
        cOff = 0
        dOff = 0
    
       
    return (cOff, dOff)

@jit(int64(float64[:],float64,float64,float64,float64))
def select_host_to_reproduce(c, randNum, B_H, TAU_H, dt):
    #calculate propensity for each group
    BirthProp = 1/TAU_H * (1 + B_H * c) * dt
    
    cumPropensity = BirthProp.cumsum()
    
    randNumScaled =randNum * cumPropensity[-1]
    
    index = np.arange(cumPropensity.size)
    LOWER_TR = np.hstack((np.zeros(1),cumPropensity[:-1]))
    id_group = index[(LOWER_TR<=randNumScaled) & (cumPropensity>randNumScaled)]
    
    return id_group    

def host_birth_event(c, d, randNum, model_par):
    B_H = model_par["B_H"]
    TAU_H = model_par["TAU_H"]
    dt = model_par["dt"]
    n0 = model_par['n0']
    K = model_par['K']
    
    id_group = select_host_to_reproduce(c, randNum, B_H, TAU_H, dt)
    
    
    if model_par['sampling']=="norm":
        cOff, dOff = host_birth_composition_norm(c[id_group],\
                                            d[id_group],\
                                            n0, K)
    elif model_par['sampling']=="binom":
        cOff, dOff = host_birth_composition_binom(c[id_group],\
                                                     d[id_group],\
                                                     n0, K)
    elif model_par['sampling']=="hypgeo":
        cOff, dOff = host_birth_composition_hypgeo(c[id_group],\
                                                     d[id_group],\
                                                     n0, K)  
    else:
       raise Exception('Unknown sampling procedure')     
     
        
    cPar = max(0,c[id_group]-cOff) 
    dPar = max(0,d[id_group]-dOff) 

    c[id_group] = cPar
    d[id_group] = dPar

    cNew = np.append(c,cOff)
    dNew = np.append(d,dOff)
    
    return (cNew, dNew)
        

def host_death_event(c, d, randNum, model_parameter):  
    id_group = select_host_to_die(c, randNum)
    cNew = np.delete(c,id_group)
    dNew = np.delete(d,id_group)
    
    return (cNew, dNew)
        
        
#updates communtity composition during timestep dt
@jit(void(float64[:],float64[:],float64,float64,float64,float64,float64))
def update_comm(c, d, r, gamma, mu, mig, dt):
     
    nGroup = c.size
    if nGroup>1:
        #calc derivatives
        DC =  r * (1 - mu) * (1 - gamma) *  c \
            + r * mu * d  \
            - r * (c + d) * c \
            - (1 + 1 / (nGroup - 1)) * mig * c \
            + 1 / (nGroup - 1) * mig * c.sum()
        DD =  r * (1 - mu) * d \
            + r * mu * (1 - gamma) * c \
            - r * (c + d) * d \
            - (1 + 1 / (nGroup - 1)) * mig * d \
            + 1 / (nGroup-1) * mig * d.sum()
    else:
       #calc derivatives
        DC =  r * (1 - mu) * (1 - gamma) *  c \
            + r * mu * d  \
            - r * (c+d) * c \
            - mig * c 
        DD =  r * (1 - mu) * d \
            + r * mu * (1 - gamma) * c \
            - r * (c+d) * d \
            - mig * d 
            
    c += DC*dt
    d += DD*dt

    return            

@jit(UniTuple(float64,2)(float64[:],float64[:],float64,float64,float64,float64))
def host_birth_death_propensity(c, d, dt, B_H, D_H, TAU_H):
    NumHost = c.size
    
    totBirthProp = dt / TAU_H * (NumHost + B_H * c.sum()) 
    totDeathProp = dt / TAU_H * D_H * NumHost ** 2
    
    return (totBirthProp, totDeathProp)


def run_model_fixed_parameters(model_par):
   
    #calc timesteps    
    calc_timestep(model_par)   
    Num_t = int(np.ceil(model_par['maxT'] / model_par['dt']))
    samplingInterval = model_par['sampleT']
    Num_t_sample = int(np.ceil(model_par['maxT'] / samplingInterval)+1)
    timeAvWindow = 100 #2 * int(np.ceil(model_par['TAU_H']))

    #init randNum
    RAND_T = create_randMat(Num_t)
        
    #init output
    dType = np.dtype([('F_T_av', 'f8'), ('F_T_std', 'f8'), \
                      ('N_T_av', 'f8'), ('N_T_std', 'f8'), \
                      ('H_T', 'f8'), \
                      ('F_mav', 'f8'), ('F_mcv', 'f8'), \
                      ('cv_mav', 'f8'),('rms_err', 'f8'), \
                      ('time', 'f8')])
    
    Output = np.full(Num_t_sample, np.nan, dType)
    
    # init groups
    CVec, DVec = init_comm(model_par)
    
    #init sample
    sample_model(CVec, DVec, Output, 0, timeAvWindow)
    Output['time'][0] = 0
    currT = 0
    nextSampleT = samplingInterval
    sampleIndex = 1
    indexBelowTr = 0
    
    #community rates
    r = model_par['r']
    gamma = model_par['gamma']
    mu = model_par['mu']
    mig = model_par['mig']
    dt = model_par['dt']
    
    B_H = model_par['B_H']
    D_H = model_par['D_H']
    TAU_H = model_par['TAU_H']
    
    #run time    
    for ti in range(Num_t):
        update_comm(CVec, DVec, r, gamma, mu, mig, dt)
        birthProp, deathProp = host_birth_death_propensity(CVec, DVec,\
                                                           dt, B_H, D_H, TAU_H)
        
        if RAND_T[ti,0] < birthProp:
            CVec, DVec = host_birth_event(CVec, DVec, RAND_T[ti,1], model_par)
        elif RAND_T[ti,0] < (birthProp+deathProp):
            CVec, DVec = host_death_event(CVec, DVec, RAND_T[ti,1], model_par)
            
        if  CVec.size==0:
            Output = Output[0:sampleIndex]
            break

        currT += model_par['dt']
        
        if currT >= nextSampleT:
            sample_model(CVec, DVec, Output, sampleIndex, timeAvWindow)
            Output['time'][sampleIndex] = currT
            
            if Output['rms_err'][sampleIndex] < 1E-5:
                indexBelowTr += 1
                
            if  indexBelowTr >= 1:  
                Output = Output[0:sampleIndex]
                break
        
            nextSampleT += samplingInterval
            sampleIndex += 1
    
    return Output




def plot_data(dataStruc, FieldName):
    plt.plot(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    plt.xlabel("time")
    maxTData =dataStruc['time'].max()
    plt.xlim((0,maxTData))
    return
  
def single_run_finalstate(MODEL_PAR):
    
    Output = run_model_fixed_parameters(MODEL_PAR)
            
    dType = np.dtype([ \
              ('F_T_av', 'f8'), ('F_T_std', 'f8'), ('F_mav', 'f8'), \
              ('N_T_av', 'f8'), ('N_T_std', 'f8'), \
              ('H_T', 'f8'),   \
              ('gamma', 'f8'), ('tau_H', 'f8'), \
              ('n0', 'f8'),    ('mig', 'f8'), \
              ('r', 'f8'),     ('K', 'f8')])     

    output_matrix = np.zeros(1, dType)
    
    output_matrix['F_T_av'] = Output['F_T_av'][-1]
    output_matrix['F_T_std'] = Output['F_T_std'][-1]
    output_matrix['F_mav'] = Output['F_mav'][-1]

    output_matrix['N_T_av'] = Output['N_T_av'][-1]
    output_matrix['N_T_std'] = Output['N_T_std'][-1]
    output_matrix['H_T'] = Output['H_T'][-1]

    output_matrix['gamma'] = MODEL_PAR['gamma']
    output_matrix['tau_H'] = MODEL_PAR['TAU_H']
    output_matrix['n0'] = MODEL_PAR['n0']
    output_matrix['mig'] = MODEL_PAR['mig']
    output_matrix['r'] = MODEL_PAR['r']
    output_matrix['K'] = MODEL_PAR['K']   
    
#    output_matrix2= np.array([Output['F_mav'][-1], Output['F_T_std'][-1], \
#                              Output['N_T_av'][-1], Output['N_T_std'][-1],\
#                              Output['H_T'][-1]])
#    
#    output_list= (Output['F_mav'][-1], Output['F_T_std'][-1], \
#                              Output['N_T_av'][-1], Output['N_T_std'][-1],\
#                              Output['H_T'][-1])
#    
#    #gc.collect()
    
    return output_matrix



def single_run_with_plot(MODEL_PAR):
    
    #run code  
    start = time.time()
    Output = run_model_fixed_parameters(MODEL_PAR)
    end = time.time()
    print("Elapsed time run 1 = %s" % (end - start))
    
    
    
    fig = plt.figure()
    #plot data  
    nR=2
    plt.subplot(nR,2,1)  
    plot_data(Output,"N_T_av")  
    plt.ylabel("N(t)") 
    
    plt.subplot(nR,2,3)  
    plot_data(Output,"F_T_av")  
    plot_data(Output,"F_mav")  
    plt.ylabel("F(t)") 
    #plt.legend()
    
    plt.subplot(nR,2,2)  
    plot_data(Output,"H_T")  
    plt.ylabel("H(t)") 

    
    plt.subplot(nR,2,4)  
    plot_data(Output,"rms_err") 
    plt.ylabel("err of mav") 
    
    
    fig.set_size_inches(10,10)
    plt.tight_layout()

    return Output
