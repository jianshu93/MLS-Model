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
import math
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib
import time
from numba import jit, void, f8, i8 
from numba.types import UniTuple
     
"""
Time step functions 
"""
#calculate maximum number of hosts at carrying capacity 
def calc_max_group_num(model_par):
    carCap = model_par["K_H"]*(1 + model_par["B_H"]) / (1-model_par["D_H"])
    return carCap

#calculate maximum birth and death rates 
def calc_max_host_prop(model_par, host_nr):
    #max birth rate at c=1
    maxBirthRate = (1 + model_par["B_H"])
    #max death rate at c=0
    maxDeathRate = (1) * host_nr / model_par["K_H"]  
    #calc max total propensity 
    #using prop per host < (maxBirthRate + maxDeathRate)
    maxHostProp = (maxBirthRate + maxDeathRate) * host_nr \
                     / model_par["TAU_H"]

    return maxHostProp

# calculates maximum time step to use to keep max 1 host event per step
def calc_max_num_timesteps(model_par, dtVec):
    #get highest number of host in population
    maxHostNum = 2 * calc_max_group_num(model_par)
    #calculate highest ever host propensity
    maxHostProp = calc_max_host_prop(model_par, maxHostNum)
    #calculate smallest time step ever needed
    dt = mlsg.calc_max_time_step(maxHostProp, dtVec)
    #calculate max number of time steps
    Num_t = int(np.ceil(model_par['maxT'] / dt)) + 1
    #calculate max number of sampling points
    Num_t_sample = int(np.ceil(model_par['maxT'] / model_par['sampleT'])+1)

    return (Num_t, Num_t_sample)

# calculates time step to use to keep max 1 host event per step for current nr of hosts
def calc_dynamic_timestep(model_par, curr_host_num, dtVec):
    #calculate highest ever host propensity
    maxHostProp = calc_max_host_prop(model_par, curr_host_num)
    #calculate smallest time step ever needed
    dt = mlsg.calc_max_time_step(maxHostProp, dtVec)
    return dt

"""
Init functions 
"""
#initialize community  
def init_comm(model_par): 
    numGroup = int(model_par["NUMGROUP"])
    #setup initial vector of c,d

    if model_par["F0"] == 'uniform':
        #setup initial vector of c,d
        cAct = model_par["N0init"] * np.linspace(0, 1, numGroup)
        dAct = model_par["N0init"] - cAct
    else:
        cAct = np.full(numGroup, \
                    model_par["N0init"] * model_par["F0"])
        dAct = np.full(numGroup, \
                    model_par["N0init"] * (1 - model_par["F0"]))

    #store in C-byte order 
    cAct = np.copy(cAct, order='C')
    dAct = np.copy(dAct, order='C')
    return (cAct, dAct)


#initialize output matrix  
def init_output_matrix(Num_t_sample):
    #specify output fields 
    dType = np.dtype([('F_T_av', 'f8'), \
                      ('N_T_av', 'f8'), \
                      ('H_T', 'f8'), \
                      ('F_mav', 'f8'), \
                      ('H_mav', 'f8'), \
                      ('F_mstd', 'f8'), \
                      ('H_mstd', 'f8'), \
                      ('rms_err', 'f8'), \
                      ('time', 'f8')])
    
    Output = np.full(Num_t_sample, np.nan, dType)
    Output['time'][0] = 0
    return Output

"""
Sample community functions 
"""
#calculate statistics of current community composition
@jit(UniTuple(f8,2)(f8[::1],f8[::1]), nopython=True)    
def calc_mean_fraction(c, d):
    TOT = c + d #total density per host
    FRAC = c[TOT>0] / TOT[TOT>0] #cooperator fraction per host
    F_av = FRAC.mean()
    N_av = TOT.mean()
    return (F_av,N_av)


#calculate distribution of investments over host population
#@jit(f8[::1](f8[::1], f8[::1], f8[::1]), nopython=True)    
def calc_perhost_inv_distri(c, d, binEdges):
    TOT = c + d #total density per host
    FRAC = c[TOT>0] / TOT[TOT>0] #cooperator fraction per host
    
    #get distribution of average cooperator fraction per host
    avHostDens, _ = np.histogram(FRAC, bins=binEdges)
    avHostDens = avHostDens / np.count_nonzero([TOT>0])
    return avHostDens

#sample model
def sample_model(c, d, binEdges, output, InvestmentPerHost, sample_idx, currT, model_par):  
    #store time
    output['time'][sample_idx] = currT    
    #calc host population properties
    nGroup = c.size
    output['H_T'][sample_idx] = nGroup
    InvestmentPerHost[sample_idx,:] = calc_perhost_inv_distri(c, d, binEdges)
    #calc population average properties
    F_av, N_av = calc_mean_fraction(c, d)
    output['F_T_av'][sample_idx] = F_av
    output['N_T_av'][sample_idx] = N_av

    #calc time windows to average over
    timeAvWindow = int(np.ceil(model_par['mav_window'] / model_par['sampleT']))
    rmsAvWindow = int(np.ceil(model_par['rms_window'] / model_par['sampleT']))
    #calc moving average fraction 
    if sample_idx >= timeAvWindow:
        mav, mstd = mlsg.calc_moving_av(output['F_T_av'], sample_idx, timeAvWindow)
        output['F_mav'][sample_idx] = mav
        output['F_mstd'][sample_idx] = mstd

        mavH, mstdH = mlsg.calc_moving_av(output['H_T'], sample_idx, timeAvWindow)
        output['H_mav'][sample_idx] = mavH
        output['H_mstd'][sample_idx] = mstdH
    #calc rms error    
    if sample_idx >= rmsAvWindow:
        rms_err = mlsg.calc_rms_error(output['F_mav'],sample_idx, rmsAvWindow)
        output['rms_err'][sample_idx] = rms_err    

    sample_idx += 1
    return sample_idx

"""
Host birth functions 
"""

# birth composition draw fraction from normal distribution with fixed variance
def host_birth_composition_fixedvar(cPar, dPar, n0, sigma):
    fracPar = np.asscalar(cPar / (cPar + dPar))
    #draw fraction from truncated normal distribution
    fracOff = mlsg.trunc_norm(fracPar, sigma, min=0, max=1)
    #assign new densities to offspring
    cOff = n0 * fracOff
    dOff = n0 * (1-fracOff)
    return (cOff, dOff)

# birth composition draw fraction from normal distribution with fixed cv
def host_birth_composition_fixedcv(cPar, dPar, n0, cv):
    fracPar = np.asscalar(cPar / (cPar + dPar))
    sigma = fracPar * cv
    #draw fraction from truncated normal distribution
    fracOff = mlsg.trunc_norm(fracPar, sigma, min=0, max=1)
    #assign new densities to offspring
    cOff = n0 * fracOff
    dOff = n0 * (1-fracOff)
    return (cOff, dOff)   

# birth composition draw from continuous approximation to hypergeometric dirstribution
def host_birth_composition_conthypogeo(cPar, dPar, n0, K):
    #hypergeometric distribution approximated as Normal(Np,Np(1-p))
    #after rescaling N -> n0*K, we get Normal(n0p,n0p(1-p)/k)
    #expected density cooperator
    fracPar = np.asscalar(cPar / (cPar + dPar))
    expC = n0 * fracPar  #np
    #std of density cooperator
    stdC = math.sqrt(expC * (1-fracPar) / K) #np(1-p)
    #draw density cooperator from truncated normal distribution
    cOff = mlsg.trunc_norm(expC, stdC, min=0, max=n0)
    #assign new densities to offspring
    dOff = n0 - cOff
    return (cOff, dOff)   

# birth composition discrete hypergeometric distribution
def host_birth_composition_hypgeo(cPar, dPar, n0, K):
    fracPar = np.asscalar(cPar / (cPar + dPar))
    densPar = np.asscalar(cPar + dPar)

    #convert continuos denisty in expected integer nr.     
    numParInt = int(np.round(densPar*K))
    cParInt = int(np.round(fracPar*densPar*K))
    #draw number of bacteria to sample from Poisson distribution with mean n0*K
    N0Exp = int(np.round(n0*K))    
    N0int = np.random.poisson(N0Exp)
    N0int = max(N0int, numParInt)
    
    try:
        #draw offspring composition from hypergiometric distribution  
        if (N0int>0) & (numParInt>0) & (numParInt>N0int):
            cOffInt = st.hypergeom.rvs(numParInt, cParInt, N0int)
            cOff = (cOffInt)/K
            dOff = (N0int - cOffInt)/K    
        else:
            cOff = 0
            dOff = 0   
    except:
        print('what?!')
    return (cOff, dOff)

#choose which host will reproduce
@jit(i8(f8[::1],f8,f8), nopython=True)
def select_host_to_reproduce(c, randNum, B_H):
    #calculate propensity for each group
    BirthProp = (1 + B_H * c)
    #randomly select group to reproduce, P proportional to propensity
    id_group = mlsg.select_random_event(BirthProp, randNum)
    return id_group    

#create new host at birth
def host_birth_event(c, d, randNum, model_par):
    #extract parameters
    B_H = model_par["B_H"]
    n0 = model_par['n0']
    K = model_par['K']
    sigma = model_par['sigmaBirth']
    
    #select host to reproduce
    id_group = select_host_to_reproduce(c, randNum, B_H)
    #draw offspring composition
    if model_par['sampling']=="conthypogeo":
        cOff, dOff = host_birth_composition_conthypogeo( \
            c[id_group], d[id_group], n0, K)
    elif model_par['sampling']=="fixedvar":
        cOff, dOff = host_birth_composition_fixedvar(\
            c[id_group], d[id_group], n0, sigma)
    elif model_par['sampling']=="hypgeo":
        cOff, dOff = host_birth_composition_hypgeo(\
            c[id_group], d[id_group], n0, K)  
    elif model_par['sampling']=="fixedcv":
        cOff, dOff = host_birth_composition_fixedcv(\
            c[id_group],  d[id_group], n0, sigma) 
    else:
       raise Exception('Unknown sampling procedure')     

#    #make sure we don't transfer more than parent has    
#    if cOff > c[id_group]:
#        cOff = c[id_group]
#    if dOff > d[id_group]:
#        dOff = d[id_group]   
#    #take sample away from parent
#    c[id_group] -= cOff
#    d[id_group] -= dOff
    #add new group to group matrix 
    cNew = np.append(c,cOff)
    dNew = np.append(d,dOff)
    return (cNew, dNew)
 
"""
Host death functions 
"""      

#choose which host will die
@jit(i8(f8[::1],f8,f8), nopython=True)
def select_host_to_die(c, randNum, D_H):
    #calculate propensity for each group
    DeathProp = (1 - D_H * c)
    #randomly select group to reproduce, P proportional to propensity
    id_group = mlsg.select_random_event(DeathProp, randNum)
    return id_group   

#delete host at death
def host_death_event(c, d, randNum, model_parameter):  
    D_H = model_parameter["D_H"]    
    #select host to die
    id_group = select_host_to_die(c, randNum, D_H)
    #delete group
    cNew = np.delete(c,id_group)
    dNew = np.delete(d,id_group)
    return (cNew, dNew)
        
#calculate total host birth and death propensities 
@jit(UniTuple(f8,2)(f8[::1],f8[::1],f8,f8,f8,f8,f8), nopython=True)
def host_birth_death_propensity(c, d, dt, B_H, D_H, TAU_H, K_H):
    NumHost = c.size
    #calc birth rate per group 
    birtRatePerGroup = (1 + B_H * c)
    #calc death rate per group 
    deathRatePerGroup = (NumHost / K_H) * (1 - D_H * c)
    #calc total birth and death propensity 
    totBirthProp = (dt / TAU_H) * birtRatePerGroup.sum()
    totDeathProp = (dt / TAU_H) * deathRatePerGroup.sum()
    
    return (totBirthProp, totDeathProp)

"""
Community dynamics functions 
"""     
#updates community composition during timestep dt
@jit(void(f8[::1],f8[::1],f8,f8,f8,f8), nopython=True)
def update_comm(c, d, cost, mu, mig, dt):     
    nGroup = c.size
    if nGroup>1:
        #calc derivatives
        DC =  (1 - mu) * (1 - cost) *  c \
            + mu * d  \
            - (c + d) * c \
            - (1 + 1 / (nGroup - 1)) * mig * c \
            + 1 / (nGroup - 1) * mig * c.sum()
        DD =  (1 - mu) * d \
            + mu * (1 - cost) * c \
            - (c + d) * d \
            - (1 + 1 / (nGroup - 1)) * mig * d \
            + 1 / (nGroup-1) * mig * d.sum()
    else: #no migration
       #calc derivatives
        DC =  (1 - mu) * (1 - cost) *  c \
            + mu * d  \
            - (c+d) * c \
            - mig * c 
        DD =  (1 - mu) * d \
            + mu * (1 - cost) * c \
            - (c+d) * d \
            - mig * d 
    #update state        
    c += DC*dt
    d += DD*dt
    return            

def test_host_prop(birthProp, deathProp, dt):
    max_host_prop = birthProp + deathProp
    p0 = np.exp(-max_host_prop * dt)
    p1 = max_host_prop * dt * np.exp(-max_host_prop * dt)
    p2 = 1 - p0 - p1
    
    if p2 > 0.01:
        print('error, p2=%.2f is too high' %p2)
        
    return None 

"""
Full model 
"""

##run model main code
def run_model_fixed_parameters(model_par):
    #possible dt to choose from
    dtVec = np.logspace(-5, -2, 19)
    
    #calc timesteps    
    Num_t, Num_t_sample = calc_max_num_timesteps(model_par, dtVec)   
    samplingInterval = model_par['sampleT']
    
    #get host rates
    B_H, D_H, TAU_H, K_H = [float(model_par[x]) for x in ('B_H','D_H','TAU_H','K_H')]
    #get bacterial rates
    mu, mig, cost = [float(model_par[x]) for x in ('mu','mig', 'cost')]
    numBins = int(model_par['numTypeBins'])
    
    #init randNum
    rndMat = mlsg.create_randMat(Num_t, 2)
  
    #init output
    binEdges = np.linspace(0, 1, numBins)
    Output = init_output_matrix(Num_t_sample)
    InvestmentPerHost = np.full((Num_t_sample, numBins-1), np.nan)
    
    # init groups
    CVec, DVec  = init_comm(model_par)

    #first sample
    currT = 0
    ti = 0
    sampleIndex = 0
    sampleIndex = sample_model(CVec, DVec, binEdges, \
                            Output, InvestmentPerHost, \
                            sampleIndex, currT, model_par)
    #get first time step
    dt = calc_dynamic_timestep(model_par, CVec.size, dtVec)

    #run time    
    while currT <= model_par['maxT']:
        #update community
        update_comm(CVec, DVec, cost, mu, mig, dt)
        
        #check if there is host event
        birthProp, deathProp = host_birth_death_propensity(CVec, DVec,\
                                                           dt, B_H, D_H, TAU_H, K_H)
        
        #test_host_prop(birthProp, deathProp, dt)
        
        if rndMat[ti,0] < birthProp:
            #process host birth event
            CVec, DVec = host_birth_event(CVec, DVec, rndMat[ti,1], model_par)
            #update time step to new number of groups
            dt = calc_dynamic_timestep(model_par, CVec.size, dtVec)
        elif rndMat[ti,0] < (birthProp+deathProp):
            #process host death event
            CVec, DVec = host_death_event(CVec, DVec, rndMat[ti,1], model_par)
            #update time step to new number of groups
            dt = calc_dynamic_timestep(model_par, CVec.size, dtVec)
            
        #stop run if all hosts die
        if  CVec.size==0:
            Output = Output[0:sampleIndex]
            break

        #update time
        currT += dt
        ti += 1
        
        #sample model at intervals
        nextSampleT = samplingInterval * sampleIndex
        if currT >= nextSampleT:
            sampleIndex = sample_model(CVec, DVec, binEdges, \
                            Output, InvestmentPerHost, \
                            sampleIndex, currT, model_par)
            
            #check if steady state has been reached
            if Output['rms_err'][sampleIndex-1] < model_par['rms_err_treshold']:
                break
        
    #cut off non existing time points at end
    Output = Output[0:sampleIndex]
    InvestmentPerHost = InvestmentPerHost[0:sampleIndex,:]
    
    return (Output, InvestmentPerHost)

"""
Run Full model 
"""
  
##run model, store final state only                      
def single_run_finalstate(MODEL_PAR):
    
    #rund model
    Output, InvestmentPerHost = run_model_fixed_parameters(MODEL_PAR)

    #init output        
    dType = np.dtype([ \
              ('F_T_av', 'f8'), ('F_mav', 'f8'), ('F_mstd', 'f8'), \
              ('F_mav_ss', 'f8'), ('N_T_av', 'f8'), \
              ('H_T', 'f8'),('H_mav', 'f8'), ('H_mstd', 'f8'),\
              ('cost', 'f8'), ('K', 'f8'), \
              ('n0', 'f8'), ('mig', 'f8'), ('sigmaBirth', 'f8'), \
              ('tau_H', 'f8'), ('B_H', 'f8'),('D_H', 'f8'),('K_H', 'f8')])     
    output_matrix = np.zeros(1, dType)
    
    #store final state
    output_matrix['F_T_av'] = Output['F_T_av'][-1]
    output_matrix['F_mav'] = Output['F_mav'][-1]
    output_matrix['F_mstd'] = Output['F_mstd'][-1]
    output_matrix['H_T'] = Output['H_T'][-1]
    output_matrix['H_mav'] = Output['H_mav'][-1]
    output_matrix['H_mstd'] = Output['H_mstd'][-1]
    output_matrix['N_T_av'] = Output['N_T_av'][-1]
    
    #store nan in F_mav_ss if not reached steady state
    if Output['rms_err'][-1] < MODEL_PAR['rms_err_treshold']:
        output_matrix['F_mav_ss'] = Output['F_mav'][-1]
    else:
        output_matrix['F_mav_ss'] = np.nan
        
    #store model settings
    output_matrix['cost'] = MODEL_PAR['cost']
    output_matrix['n0'] = MODEL_PAR['n0']
    output_matrix['mig'] = MODEL_PAR['mig']
    output_matrix['K'] = MODEL_PAR['K']  
    output_matrix['sigmaBirth'] = MODEL_PAR['sigmaBirth']   
    output_matrix['tau_H'] = MODEL_PAR['TAU_H']
    output_matrix['B_H'] = MODEL_PAR['B_H']   
    output_matrix['D_H'] = MODEL_PAR['D_H']   
    output_matrix['K_H'] = MODEL_PAR['K_H']    
    
    InvestmentPerHostEnd = InvestmentPerHost[-1,:]


    return (output_matrix, InvestmentPerHostEnd)

#run model, do not plot dynamics 
def single_run_noplot(MODEL_PAR):
    #run code  
    start = time.time()
    Output, InvestmentPerHost = run_model_fixed_parameters(MODEL_PAR)
    end = time.time()
    print("Elapsed time run 1 = %s" % (end - start))

    return (Output, InvestmentPerHost)

def plot_data(dataStruc, FieldName, type='lin'):
    if type == 'lin':
        plt.plot(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    elif type == 'log':
        plt.semilogy(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    plt.xlabel("time")
    maxTData =dataStruc['time'].max()
    try:
        plt.xlim((0,maxTData))
    except:
        print(maxTData)
    return

#run model, plot dynamics 
def single_run_with_plot(MODEL_PAR):
    
    #run code  
    start = time.time()
    Output, InvestmentPerHost = run_model_fixed_parameters(MODEL_PAR)
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
    plot_data(Output,"F_T_av")  
    plot_data(Output,"F_mav")  
    plt.ylabel("investment") 
    plt.ylim((0, 1))

    #plot host number investment  
    plt.subplot(nR,nC,2)  
    plot_data(Output,"H_mav")  
    plt.ylabel("H(t)") 
    
    
    plt.subplot(nR,nC,5)  
    plot_data(Output,"N_T_av")  
    plt.ylabel("pop size") 
    
    #plot error
    plt.subplot(nR,nC,3)  
    plot_data(Output,"rms_err",'log')  
    plt.ylabel("rms_err(t)") 


    #plot average investment per host
    axs= plt.subplot(nR,nC,4)  
    currData = np.log10(InvestmentPerHost.transpose() + np.finfo(float).eps )
    im = axs.imshow(currData, cmap="viridis", \
                    interpolation='nearest', \
                    extent=[0,1,0,1], \
                    origin='lower', \
                    vmin = -3,
                    aspect= 'auto')
    axs.set_xticks([0, 1])
    axs.set_yticks([0, 1])
    axs.set_ylabel('investment')
    axs.set_xlabel('time')
    cb = fig.colorbar(im, ax=axs, orientation='vertical', fraction=.1, label="log10 mean density")
   # cb.set_ticks([-2, -1])
    axs.set_yticklabels([0, 1])
    maxTData =Output['time'].max()
    axs.set_xticklabels([0, int(round(maxTData))])
    
    
    fig.set_size_inches(4,4)
    plt.tight_layout()

    return (Output, InvestmentPerHost)

#run model with default parameters
def debug_code():
    model_par = {
                #time step parameters
                "maxT"  : 10000., 
                "sampleT": 1,
                "rms_err_treshold": 1E-5,
                "mav_window": 1000,
                "rms_window": 5000,
                #fixed model parameters
                "sampling" : "hypgeo",
                "sigmaBirth" : 0.05,
                "mu"    : 1E-5,
                "B_H"   : 1.,
                "D_H"   : 0.,
                "K_H"   : 20.,
                #variable model parameters
                "cost" : 0.01,
                "TAU_H" : 10.,
                "n0"    : 1E-3,
                "mig"   : 1E-5,
                "K"     : 10E3,
                #fixed intial condition
                "NUMGROUP" : 25,  
                "numTypeBins" : 100,
                "F0" : 0.5,
                "N0init" : 1.
        }
    
    Output, InvestmentPerHost = single_run_with_plot(model_par)
    
    return Output, InvestmentPerHost

if __name__ == "__main__":
    print("running debug")
    Output, InvestmentPerHost = debug_code()
 
