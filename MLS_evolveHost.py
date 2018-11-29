#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
CreateD on WeD OCt 17 12:25:24 2018

@author: simonvanvliet
"""
import math
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time
from numba import jit, void, f8, i8, from_dtype 
from numba.types import UniTuple
 
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
    

#initialize structured array of host properties
def init_comm(model_par): 
    
    numGroup = int(model_par["NUMGROUP"])
    cAct = np.full(numGroup, \
                   model_par["N0init"] * model_par["F0"])
    dAct = np.full(numGroup, \
                   model_par["N0init"] * (1 - model_par["F0"]))

    if model_par['HostEvolves']: 
        Host_traits = np.zeros(numGroup, \
                        dtype = [('mig', 'f8'), ('r', 'f8'), ('n0', 'f8')])
        
        Host_traits['mig'] = model_par["mig"]
        Host_traits['r'] = model_par["r"]
        Host_traits['n0'] = model_par["n0"]
    else:
        Host_traits = 0

    return (cAct, dAct, Host_traits)


def create_randMat(Num_t):
    notDone = True
    while notDone:
        randMat = np.random.random((Num_t,2))
        
        containsNo0 = (~np.any(randMat == 0))
        containsNo1 = (~np.any(randMat == 1))
        if containsNo0 & containsNo1:
            notDone = False
    
    return randMat


@jit(UniTuple(f8,4)(f8[:],f8[:]))    
def calc_mean_fraction(c, d):
    TOT = c + d
    FRAC = c / TOT
    F_av = FRAC.mean()
    N_av = TOT.mean()
    F_std = FRAC.std()
    N_std = TOT.std()

    return (F_av,N_av,F_std,N_std)


@jit(UniTuple(f8,2)(f8[:],i8,i8))    
def calc_moving_av_and_cv(f_t, curr_idx, windowLength):
    start_idx=max(0,curr_idx-windowLength+1)
    avF_t = f_t[start_idx:curr_idx].mean()
    
    return avF_t 


@jit(f8(f8[:], i8, i8))    
def calc_moving_cv(mav_t, curr_idx, windowLength):
    start_idx=max(0,curr_idx-windowLength+1)
    
    avF_t = mav_t[start_idx:curr_idx].mean()
    stdF_t = mav_t[start_idx:curr_idx].std()
    cvF_t = stdF_t / avF_t
    
    return cvF_t

@jit(f8(f8[:],i8,i8))    
def calc_rms_error(mav_t, curr_idx, windowLength):
    start_idx=max(0,curr_idx-windowLength+1)
    
    localsegment = mav_t[start_idx:curr_idx]
    av = localsegment.mean()
    errorSquared = (localsegment-av)**2
    meanErrorSquared = errorSquared.mean()
    rms = math.sqrt(meanErrorSquared)
   
    return rms


def sample_model(c, d, host_traits, output, sample_idx, currT, movingAvWind):  
    F_av, N_av, F_std, N_std = calc_mean_fraction(c, d)
    
    output['F_T_av'][sample_idx] = F_av
    output['N_T_av'][sample_idx] = N_av
    
    output['F_T_std'][sample_idx] = F_std
    output['N_T_std'][sample_idx] = N_std

    output['H_T'][sample_idx] = c.size

    if not np.isscalar(host_traits):
        output['N0_T'][sample_idx] = (host_traits['n0']).mean()
        output['R_T'][sample_idx] = (host_traits['r']).mean()
        output['M_T'][sample_idx] = (host_traits['mig']).mean()
    
    if sample_idx >= movingAvWind:
        mav = calc_moving_av_and_cv(output['F_T_av'], \
                                         sample_idx, movingAvWind)
        output['F_mav'][sample_idx] = mav
        
    if sample_idx >= 2*movingAvWind:
        rms_err = calc_rms_error(output['F_mav'], \
                                         sample_idx, movingAvWind)
        output['rms_err'][sample_idx] = rms_err
        
    output['time'][sample_idx] = currT
    sample_idx += 1
    
    return sample_idx

def select_host_to_die(c, randNum):
    numHost=c.size
    id_group = int(np.floor(randNum*numHost))
    return id_group


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

def host_birth_trait(parTrait, model_parameter):
    r_par = parTrait['r']
    m_par = parTrait['mig'] 
    n0_par = parTrait['n0'] 

    traitsStd = ['sigmaR', 'sigmaMig', 'sigmaN0']
    stdR, stdM, stdN0 = [model_parameter[x] for x in traitsStd]

    if stdR>0:
        rOff = trunc_norm(r_par, stdR, min=0, max=10, type='lin')
    else:
        rOff = r_par   
    if stdM>0:
        mOff = trunc_norm(m_par, stdM, min=1E-10, max=0.1, type='log')
    else:
        mOff = m_par  
    if stdN0>0:
        n0Off = trunc_norm(n0_par, stdN0, min=1E-10, max=0.1, type='log')
    else:
        n0Off = n0_par  
    
    newTraits = np.array(1, dtype=parTrait.dtype)
    newTraits['r'] = rOff
    newTraits['mig'] = mOff
    newTraits['n0'] = n0Off

    return newTraits


# birth composition draw fraction from normal distribution with fixed variance
def host_birth_composition_fixedvar(cPar, dPar, n0, sigma):
    densPar = np.asscalar(cPar + dPar)
    fracPar = np.asscalar(cPar / densPar)
    #constant variance
    minFrac = (0 - fracPar) / sigma
    maxFrac = (1 - fracPar) / sigma

    fracOff = st.truncnorm.rvs(minFrac, maxFrac) * sigma + fracPar

    cOff = n0 * fracOff
    dOff = n0 * (1-fracOff)
    
    return (cOff, dOff)

# birth composition draw fraction from normal distribution with fixed variance
def host_birth_composition_fixedcv(cPar, dPar, n0, cv):
    densPar = np.asscalar(cPar + dPar)
    fracPar = np.asscalar(cPar / densPar)
    
    sigma = fracPar * cv
    
    #constant variance
    minFrac = (0 - fracPar) / sigma
    maxFrac = (1 - fracPar) / sigma

    fracOff = st.truncnorm.rvs(minFrac, maxFrac) * sigma + fracPar

    cOff = n0 * fracOff
    dOff = n0 * (1-fracOff)
    
    return (cOff, dOff)


#% birth composition continuous approximation to hypergeometric dirstribution
def host_birth_composition_norm(cPar, dPar, n0, K):
    densPar = np.asscalar(cPar + dPar)
    fracPar = np.asscalar(cPar / densPar)
    densOff = n0
    
    expC = densOff * fracPar  #np
    stdC = math.sqrt(expC * (1-fracPar) / K) #np(1-p)
    minC = (0 - expC) / stdC
    maxC = (min(densPar,densOff) - expC) / stdC

    cOff = st.truncnorm.rvs(minC, maxC) * stdC + expC
    dOff = densOff - cOff
    
    return (cOff, dOff)

#% birth composition discrete binomial distribution
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

#% birth composition discrete hypergeometric distribution
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

@jit(i8(f8[::1], f8, f8, f8, f8))
def select_host_to_reproduce(c, randNum, B_H, TAU_H, dt):
    #calculate propensity for each group
    BirthProp = 1/TAU_H * (1 + B_H * c) * dt
    
    cumPropensity = BirthProp.cumsum()
    
    randNumScaled =randNum * cumPropensity[-1]
    
    index = np.arange(cumPropensity.size)
    LOWER_TR = np.hstack((np.zeros(1),cumPropensity[:-1]))
    id_group = index[(LOWER_TR<=randNumScaled) & (cumPropensity>randNumScaled)]
    
    return id_group    

def host_birth_event(c, d, host_trait, randNum, model_par):
    B_H = model_par["B_H"]
    TAU_H = model_par["TAU_H"]
    dt = model_par["dt"]
    K = model_par['K']
    sigma = model_par['sigmaBirth']
    
    id_group = select_host_to_reproduce(c, randNum, B_H, TAU_H, dt)

    if model_par['HostEvolves']:
        n0 = host_trait['n0'][id_group]
    else:
        n0 = model_par['n0']

    
    if model_par['sampling']=="norm":
        cOff, dOff = host_birth_composition_norm(c[id_group],\
                                            d[id_group],\
                                            n0, K)
    elif model_par['sampling']=="fixedvar":
        cOff, dOff = host_birth_composition_fixedvar(c[id_group],\
                                                     d[id_group],\
                                                     n0, sigma)
    elif model_par['sampling']=="binom":
        cOff, dOff = host_birth_composition_binom(c[id_group],\
                                                     d[id_group],\
                                                     n0, K)
    elif model_par['sampling']=="hypgeo":
        cOff, dOff = host_birth_composition_hypgeo(c[id_group],\
                                                     d[id_group],\
                                                     n0, K)  
    elif model_par['sampling']=="fixedcv":
        cOff, dOff = host_birth_composition_fixedcv(c[id_group],\
                                                     d[id_group],\
                                                     n0, sigma)   
    else:
       raise Exception('Unknown sampling procedure')     
     
        
    cPar = max(0,c[id_group]-cOff) 
    dPar = max(0,d[id_group]-dOff) 

    c[id_group] = cPar
    d[id_group] = dPar

    cNew = np.append(c,cOff)
    dNew = np.append(d,dOff)

    if model_par['HostEvolves']:
        #new host traits
        newTraits = host_birth_trait(host_trait[id_group], model_par)
        host_traitNew = np.append(host_trait, newTraits)
    else:
        host_traitNew = 0

    return (cNew, dNew, host_traitNew)
        

def host_death_event(c, d, host_trait, randNum, model_parameter):  
    id_group = select_host_to_die(c, randNum)
    cNew = np.delete(c,id_group)
    dNew = np.delete(d,id_group)
    
    if model_parameter['HostEvolves']:
        #new host traits
        host_traitNew = np.delete(host_trait, id_group)
    else:
        host_traitNew = 0
    
    return (cNew, dNew, host_traitNew)
        
        
#updates communtity composition during timestep dt
@jit(void(f8[::1], f8[::1], f8, f8, f8, f8, f8))
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

#updates communtity composition during timestep dt
dtype = np.dtype([('mig', '<f8'), ('r', '<f8'), ('n0', '<f8')])
ty = from_dtype(dtype)

@jit(void(f8[::1], f8[::1], ty[::1], f8, f8, f8))
def update_comm_evolvhost(c, d, hostProp, gamma, mu, dt):
     
    migV = hostProp['mig']
    rVec = hostProp['r']      
    
    nGroup = c.size
    
    bRateC = rVec * (1 - gamma) 
    bRateD = rVec 
    
    bCtoC = (1 - mu) * bRateC
    bCtoD = mu * bRateC
    
    bDtoD = (1 - mu) * bRateD
    bDtoC = mu * bRateD
    
    deaRate = rVec * (c + d)
    outRate = (1 + 1 / (nGroup - 1)) * migV
    
    if nGroup > 1:
        inRateC = 1 / (nGroup-1) * migV.dot(c)
        inRateD = 1 / (nGroup-1) * migV.dot(d) 
    else:
        inRateC = 0
        inRateD = 0
        
    DC = (bCtoC - deaRate - outRate) * c + bDtoC * d + inRateC 
    DD = (bDtoD - deaRate - outRate) * d + bCtoD * c + inRateD
        
    c += DC*dt
    d += DD*dt

    return 

@jit(UniTuple(f8,2)(f8[::1],f8[::1],f8,f8,f8,f8))
def host_birth_death_propensity(c, d, dt, B_H, D_H, TAU_H):
    NumHost = c.size
    
    totBirthProp = dt / TAU_H * (NumHost + B_H * c.sum()) 
    totDeathProp = dt / TAU_H * D_H * NumHost ** 2
    
    return (totBirthProp, totDeathProp)

def init_output_matrix(Num_t_sample):
    #init output
    dType = np.dtype([('F_T_av', 'f8'), ('F_T_std', 'f8'), \
                      ('N_T_av', 'f8'), ('N_T_std', 'f8'), \
                      ('H_T', 'f8'), ('N0_T', 'f8'), \
                      ('R_T', 'f8'), ('M_T', 'f8'), \
                      ('F_mav', 'f8'), ('rms_err', 'f8'), \
                      ('time', 'f8')])
    
    Output = np.full(Num_t_sample, np.nan, dType)

    return Output

def run_model_fixed_parameters(model_par):
   
    #calc timesteps    
    Num_t, Num_t_sample = calc_timestep(model_par)   
    samplingInterval = model_par['sampleT']
    timeAvWindow = 1000 
    hostEvolves = model_par['HostEvolves']

    #init randNum and output matrix
    RAND_T = create_randMat(Num_t)
    Output = init_output_matrix(Num_t_sample)
    
    # init groups
    CVec, DVec, HVec = init_comm(model_par)
    
    #extract community rates
    r, gamma, mu, mig, dt = [float(model_par[x]) for x \
                                in ('r','gamma', 'mu','mig', 'dt')]

    #extract host rates
    B_H, D_H, TAU_H = [float(model_par[x]) for x \
                                in ('B_H','D_H', 'TAU_H')]
    #first sample
    sampleIndex = 0
    currT = 0
    sampleIndex = sample_model(CVec, DVec, HVec, Output, sampleIndex, \
                                currT, timeAvWindow)
    


    #run time    
    for ti in range(Num_t):
        #update community
        if hostEvolves:
            update_comm_evolvhost(CVec, DVec, HVec, gamma, mu, dt)
        else:
            update_comm(CVec, DVec, r, gamma, mu, mig, dt)
        #check if there is host event
        birthProp, deathProp = host_birth_death_propensity(CVec, DVec,\
                                                           dt, B_H, D_H, TAU_H)
        #process host events
        if RAND_T[ti,0] < birthProp:
            CVec, DVec, HVec = host_birth_event(CVec, DVec, HVec, RAND_T[ti,1], model_par)
        elif RAND_T[ti,0] < (birthProp+deathProp):
            CVec, DVec, HVec = host_death_event(CVec, DVec, HVec, RAND_T[ti,1], model_par)
            
        if  CVec.size==0:
            Output = Output[0:sampleIndex]
            break

        #update time
        currT += model_par['dt']    

        #sample state at intervals
        nextSampleT = samplingInterval * sampleIndex
        if currT >= nextSampleT:
            sampleIndex = sample_model(CVec, DVec, HVec, Output,\
                                 sampleIndex, currT, timeAvWindow)
            
            isStabalized = Output['rms_err'][sampleIndex-1] < 5E-3
            
            if isStabalized & (not hostEvolves):
                break
                
    Output = Output[0:sampleIndex]

    return Output


def single_run_finalstate(MODEL_PAR):
    
    Output = run_model_fixed_parameters(MODEL_PAR)
            
    dType = np.dtype([ \
              ('F_T_av', 'f8'), ('F_T_std', 'f8'), ('F_mav', 'f8'), \
              ('F_mav_ss', 'f8'), ('N_T_av', 'f8'), ('N_T_std', 'f8'), \
              ('H_T', 'f8'),   \
              ('gamma', 'f8'), ('tau_H', 'f8'), \
              ('n0', 'f8'),    ('mig', 'f8'), \
              ('r', 'f8'),     ('K', 'f8'), \
              ('sigmaBirth', 'f8')])     

    output_matrix = np.zeros(1, dType)
    
    output_matrix['F_T_av'] = Output['F_T_av'][-1]
    output_matrix['F_T_std'] = Output['F_T_std'][-1]
    output_matrix['F_mav'] = Output['F_mav'][-1]
    
    #store nan if not reached steady state
    Num_t = Output.size
    Num_t_end = int(np.ceil(MODEL_PAR['maxT'] / MODEL_PAR['sampleT'])+1)
    
    if Num_t < Num_t_end:
        output_matrix['F_mav_ss'] = Output['F_mav'][-1]
    else:
        output_matrix['F_mav_ss'] = np.nan
        
    output_matrix['N_T_av'] = Output['N_T_av'][-1]
    output_matrix['N_T_std'] = Output['N_T_std'][-1]
    output_matrix['H_T'] = Output['H_T'][-1]

    output_matrix['gamma'] = MODEL_PAR['gamma']
    output_matrix['tau_H'] = MODEL_PAR['TAU_H']
    output_matrix['n0'] = MODEL_PAR['n0']
    output_matrix['mig'] = MODEL_PAR['mig']
    output_matrix['r'] = MODEL_PAR['r']
    output_matrix['K'] = MODEL_PAR['K']  
    output_matrix['sigmaBirth'] = MODEL_PAR['sigmaBirth']   

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
    Output = run_model_fixed_parameters(MODEL_PAR)
    end = time.time()
    print("Elapsed time run 1 = %s" % (end - start))
        
    fig = plt.figure()
    nR=2
    nC=3

    plt.subplot(nR,nC,1)  
    plot_data(Output,"N_T_av")  
    plt.ylabel("N(t)") 
    plt.ylim(-0.05,1.05)
    plt.yticks([0, 0.5, 1])
    
    plt.subplot(nR,nC,2)  
    plot_data(Output,"H_T")  
    plt.ylabel("H(t)") 
    
    plt.subplot(nR,nC,3)  
    plot_data(Output,"F_T_av")  
    plot_data(Output,"F_mav")  
    plt.ylabel("F(t)") 
    plt.ylim(-0.05,1.05)
    plt.yticks([0, 0.5, 1])

    #plt.legend()
    
    plt.subplot(nR,nC,4)  
    plot_data(Output,"N0_T", type='log') 
    plt.ylabel("Inoculum N0") 
    
    plt.subplot(nR,nC,5)  
    plot_data(Output,"M_T", type='log') 
    plt.ylabel('Migration $\\theta$') 

    plt.subplot(nR,nC,6)  
    plot_data(Output,"R_T", type='lin') 
    plt.ylabel("Turnover r") 
    maxY = np.ceil(np.nanmax(Output['R_T'])+1)
    try:
        plt.ylim(0,maxY)
    except:
        a=1
    
    #fig.set_figwidth(10, forward=True)
    #fig.set_size_inches(10,10)
    plt.tight_layout()

    return Output
