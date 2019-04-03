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
import numpy as np
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
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
    #get parameters
    numGammaBin = model_par['numTypeBins']
    meanGamma = model_par['meanGamma0']
    stdGamma = model_par['stdGamma0']
    numGroup = int(model_par['NUMGROUP'])
    N0 = model_par['N0init']
    #create investment vector
    dGamma = 1 / numGammaBin    
    gammaVec = np.linspace(dGamma/2, 1-dGamma/2, numGammaBin)
    #calculate initial fraction in each investment bin based on normal distribution
    zVec = (gammaVec - meanGamma) / stdGamma
    initDistri = st.norm.pdf(zVec)
    #normalize to total density of N0
    initDistri *= (N0 / initDistri.sum())
    #assign intital distribution to all initial groups
    groupMatrix = np.broadcast_to(initDistri,(numGroup,numGammaBin))
    groupMatrix = np.copy(groupMatrix, order='C')
    
    return groupMatrix

#initialize output matrix  
def init_output_matrix(Num_t_sample):
    #specify output fileds 
    dType = np.dtype([('F_T_av', 'f8'), \
                      ('N_T_av', 'f8'), \
                      ('H_T', 'f8'), \
                      ('F_mav', 'f8'), \
                      ('F_mstd', 'f8'), \
                      ('rms_err', 'f8'), \
                      ('time', 'f8')])
    
    Output = np.full(Num_t_sample, np.nan, dType)
    Output['time'][0] = 0
    return Output

"""
Sample community functions 
"""
#calculate mean investment and density over all groups
@jit(UniTuple(f8,2)(f8[:,::1], f8[:], i8), nopython=True)    
def calc_mean_fraction(gMat, gammaVec, nGroup):
    #calc total investment in each host
    invPerGroup = gMat @ gammaVec 
    #calc total investment in full population
    totInv = invPerGroup.sum()
    #calc average investment 
    totDensity = gMat.sum()
    F_av = totInv / totDensity
    #calc average density
    N_av = totDensity / nGroup
    return (F_av,N_av)

#calculate dirstribution of investments over host population
@jit(f8[::1](f8[:,::1], f8[::1]))    
def calc_perhost_inv_distri(gMat,gammaVec):
    nGroup = gMat.shape[0]
    binEdges = np.append(gammaVec-gammaVec[0],gammaVec[-1]+gammaVec[0])
    #convert denisty matrix to probability matrix by dividing density in each bin
    # by total density in that host
    densPerHost = gMat.sum(axis=1)
    probDens = gMat / densPerHost[:,None]
    #calc average investment per host
    avInvPerHost = probDens @ gammaVec 
    #get distribution of average investment per host
    avHostDens, bin_edges = np.histogram(avInvPerHost, bins=binEdges)
    avHostDens = avHostDens / nGroup
    return avHostDens

#sample model
def sample_model(gMat, gammaVec, output, InvestmentAll, InvestmentPerHost, sample_idx, currT, model_par):  
    #store time
    output['time'][sample_idx] = currT    
    #calc host population properties
    nGroup = gMat.shape[0]
    output['H_T'][sample_idx] = nGroup
    InvestmentPerHost[sample_idx,:] = calc_perhost_inv_distri(gMat,gammaVec)
    #calc population average properties
    F_av, N_av = calc_mean_fraction(gMat, gammaVec, nGroup)
    output['F_T_av'][sample_idx] = F_av
    output['N_T_av'][sample_idx] = N_av
    InvestmentAll[sample_idx,:] = gMat.mean(axis=0)
    #calc time windows to average over
    timeAvWindow = int(np.ceil(model_par['mav_window'] / model_par['sampleT']))
    rmsAvWindow = int(np.ceil(model_par['rms_window'] / model_par['sampleT']))
    #calc moving average fraction 
    if sample_idx >= timeAvWindow:
        mav, mstd = mlsg.calc_moving_av(output['F_T_av'], sample_idx, timeAvWindow)
        output['F_mav'][sample_idx] = mav
        output['F_mstd'][sample_idx] = mstd
    #calc rms error    
    if sample_idx >= rmsAvWindow:
        rms_err = mlsg.calc_rms_error(output['F_mav'],sample_idx, rmsAvWindow)
        output['rms_err'][sample_idx] = rms_err    

    sample_idx += 1
    return sample_idx

"""
Host birth functions 
"""
#take discrete sample from parent community with fixed size
@jit(f8[:](f8[::1], f8, f8), nopython=True)
def host_birth_composition_assorted_sample(parComp, n0, numSample):
    #divide inocculum over samples
    N0perSample = n0 / numSample
   
    #init offspring
    offComp = np.zeros(parComp.shape)
    #sample offspring, pick numSample individuals from parent distribution
    #calc cumulative propensities and sample with uniform rand numbers
    cumPropensity = parComp.cumsum() 
    randNum = np.random.rand(numSample)    
    randNumScaled = randNum * cumPropensity[-1]
    index = np.arange(cumPropensity.size)
    for x in randNumScaled:
        idx = index[(cumPropensity>x)][0] 
        offComp[idx] += N0perSample
     
    return offComp

#take discrete sample from parent community
@jit(f8[:](f8[::1], f8, f8), nopython=True)
def host_birth_composition_sample(parComp, n0, K):
    #draw number of bacteria to sample from Poisson distribution with mean n0*K
    N0Exp = n0 * K
    N0int = np.random.poisson(N0Exp)
    if N0int > 0:
        #init offspring
        offComp = np.zeros(parComp.shape)
        #sample offspring, pick N0int individuals from parent distribution
        #calc cumulative propensities and sample with uniform rand numbers
        cumPropensity = parComp.cumsum() 
        randNum = np.random.rand(N0int)    
        randNumScaled = randNum * cumPropensity[-1]
        index = np.arange(cumPropensity.size)
        for x in randNumScaled:
            idx = index[(cumPropensity>x)][0] 
            offComp[idx] += 1
        #normalize total density to n0
        offComp *= n0 / N0int   
    else:
        offComp = np.zeros(parComp.shape)
    return offComp

#choose which host will reproduce
@jit(i8(f8[:,::1], f8[::1], f8, f8), nopython=True)
def select_host_to_reproduce(groupMat, gammaVec, randNum, B_H):
    #calculate propensity for each group
    investmentPerGroup = groupMat @ gammaVec
    BirthProp = (1 + B_H * investmentPerGroup)
    #randomly select group to reproduce, P proprtional to propensity
    id_group = mlsg.select_random_event(BirthProp, randNum)
    return id_group    

#create new host at birth
def host_birth_event(groupMat, gammaVec, randNum, model_par):
    #extract parameters
    B_H = model_par["B_H"]
    n0 = model_par['n0']
    K = model_par['K']    
    numSample = model_par['numSample']    
    #select host to reproduce
    id_group = select_host_to_reproduce(groupMat, gammaVec, randNum, B_H)
    #select parent group and find offspring composition
    parComp = groupMat[id_group, :]
    if model_par['sampling']=="sample":
        offComp = host_birth_composition_sample(parComp, n0, K)
    elif model_par['sampling']=="assortsample":
        offComp = host_birth_composition_assorted_sample(parComp, n0, numSample)
    else:
       raise Exception('Unknown sampling procedure')     
    #add new group to group matrix 
    groupMatNew = np.append(groupMat, np.atleast_2d(offComp), axis=0)
    return groupMatNew

"""
Host death functions 
"""

#choose which host will die
@jit(i8(f8[:,::1], f8[::1], f8, f8), nopython=True)
def select_host_to_die(groupMat, gammaVec, randNum, D_H):
    #calculate propensity for each group
    investmentPerGroup = groupMat @ gammaVec
    DeathProp = (1 - D_H * investmentPerGroup)
    #randomly select group to reproduce, P proprtional to propensity
    id_group = mlsg.select_random_event(DeathProp, randNum)
    return id_group      

#delete host at death
def host_death_event(groupMatrix, gammaVec, randNum, model_par): 
    #extract parameters
    D_H = model_par["D_H"]
    #select host to die
    id_group = select_host_to_die(groupMatrix, gammaVec, randNum, D_H)
    #delete group
    groupMatNew = np.delete(groupMatrix, id_group, axis=0)
    return groupMatNew

"""
Host event functions 
"""    
#calulate host investment
@jit(UniTuple(f8[:],2)(f8[:,::1], f8[::1]))
def calc_community_performance(gMat, gammaVec):
    #calc community investment
    invPerCom = gMat @ gammaVec 
    return invPerCom

#calulate total host birth and death propensities
@jit(UniTuple(f8,2)(f8[::1], f8, f8,  f8, f8, f8), nopython=True)
def host_propensityLocal(invPerCom, dt, B_H, D_H, TAU_H, K_H): 
    numHost = invPerCom.size
    #calc birth rate per group 
    birtRatePerGroup = (B_H * invPerCom + 1)
    #calc death rate per group 
    deathRatePerGroup = (numHost / K_H) * (1 - D_H * invPerCom)
    #calc total birth and death propensity 
    totBirthProp = (dt / TAU_H) * birtRatePerGroup.sum()
    totDeathProp = (dt / TAU_H) * deathRatePerGroup.sum()
    return (totBirthProp, totDeathProp)

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
    birthRate = (1 - costVec) * r
    noMutationRate = (1-mu) * birthRate
    mutationRate = (mu/2) * birthRate 
    #create sub matrices of size of single group
    locBirthMut = np.diag(noMutationRate) + \
               np.diag(mutationRate[0:-1], -1) + \
               np.diag(mutationRate[1:],  1)           
               
    return (locBirthMut, gammaVec)

#update community composition with Euler method
@jit(void(f8[:,::1], f8[:,::1], f8, f8, f8), nopython=True)   
def update_comm_loc(gMat, locBirthMut, r, mig, dt):
    #get group properties
    nGroup = gMat.shape[0]
    densPerGroup = gMat.sum(axis=1)
    globTypeFrac = gMat.sum(axis=0)
    
    #calculate migration into host from global pool
    if nGroup>1:
        migIn = globTypeFrac * mig / (nGroup - 1)
        migOutRate = (1 + 1 / (nGroup - 1)) * mig
    else:
        migIn = globTypeFrac * 0
        migOutRate = mig
    #density dependent death rate per group    
    deathRatePerGroup = r * densPerGroup
    #loop groups to calc change
    dx = np.zeros(gMat.shape)
    for i in range(nGroup):
        currGroup = gMat[i,:]
        birthMut = locBirthMut @ currGroup 
        migOut = migOutRate * currGroup
        deaths =  deathRatePerGroup[i] * currGroup
        dx[i,:] =  birthMut - deaths - migOut + migIn
    
    gMat += dx * dt
    return  

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
    r, mu, mig, cost = [float(model_par[x]) for x in ('r','mu','mig', 'cost')]
    numBins = int(model_par['numTypeBins'])
   
    #init randNum
    rndMat = mlsg.create_randMat(Num_t, 2)
    # init groups
    gMat = init_comm(model_par)
    locBMMat, gammaVec = create_local_update_matrix(r, mu, cost, numBins)
    #init output
    Output = init_output_matrix(Num_t_sample)
    InvestmentAll = np.full((Num_t_sample, numBins), np.nan)
    InvestmentPerHost = np.full((Num_t_sample, numBins), np.nan)
    
    #first sample
    currT = 0
    ti = 0
    sampleIndex = 0
    sampleIndex = sample_model(gMat, gammaVec, Output, \
                               InvestmentAll, InvestmentPerHost, \
                               sampleIndex, currT, model_par)
    #get first time step
    dt = calc_dynamic_timestep(model_par, gMat.shape[0], dtVec)
    
    #run time    
    while currT <= model_par['maxT']:
        #update community
        update_comm_loc(gMat, locBMMat, r, mig, dt)
        #check if there is host event
        invPerCom = calc_community_performance(gMat, gammaVec)
        birthProp, deathProp = host_propensityLocal(invPerCom, dt, \
                                               B_H, D_H, TAU_H, K_H)
        #process host events
        if rndMat[ti,0] < birthProp:
            #process host birth event
            gMat = host_birth_event(gMat, gammaVec, rndMat[ti,1], model_par)
            #update time step to new number of groups
            dt = calc_dynamic_timestep(model_par, gMat.shape[0], dtVec)
        elif rndMat[ti,0] < (birthProp + deathProp):
            #process host death event
            gMat = host_death_event(gMat, gammaVec, rndMat[ti,1], model_par)
            #update time step to new number of groups
            dt = calc_dynamic_timestep(model_par, gMat.shape[0], dtVec)
            
        #stop run if all hosts die
        if  gMat.shape[0]==0:
            break

        #update time
        currT += dt
        ti += 1
        
        #sample model at intervals
        nextSampleT = samplingInterval * sampleIndex
        if currT >= nextSampleT:
            sampleIndex = sample_model(gMat, gammaVec, Output, \
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

"""
Run Full model 
"""

##run model, store final state only
def single_run_finalstate(MODEL_PAR):
    
    gMat, Output, InvestmentAll, InvestmentPerHost = run_model_fixed_parameters(MODEL_PAR)
            
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
    gMat, Output, InvestmentAll, InvestmentPerHost = run_model_fixed_parameters(MODEL_PAR)
    end = time.time()
    
    print("Elapsed time run 1 = %s" % (end - start))
    
    font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 9}
    matplotlib.rc('font', **font)
    
    fig = plt.figure()
    nR=2
    nC=2
    
    #plot average investment  
    plt.subplot(nR,nC,1)  
    plot_data(Output,"F_T_av")  
    plot_data(Output,"F_mav")  
    plt.ylabel("investment") 
    plt.ylim((0, 1))
    
    #plot error
    plt.subplot(nR,nC,2)  
    plot_data(Output,"rms_err",'log')  
    plt.ylabel("rms_err(t)") 

    #plot investment distribution
    axs= plt.subplot(nR,nC,3)  
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
    axs= plt.subplot(nR,nC,4)  
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
 

 # #take contiouns sample from parent community
# def host_birth_composition_contsample(parComp, gammaVec, n0, K):
#     Num0 = max(int(np.ceil(n0*K)),2)
#     probDens = parComp / parComp.sum()
#     sampMeanExp = probDens.dot(gammaVec)
#     centralDev = (gammaVec - sampMeanExp)**2
#     sampMeanVar = probDens.dot(centralDev)
#     sampMeanStd = math.sqrt(sampMeanVar / Num0)
#     sampMean = trunc_norm(sampMeanExp, sampMeanStd, min=0, max=1, type='lin')
#     isDone = False
#     offComp = parComp
#     while not isDone:
#         sampStd = math.sqrt((st.chi2.rvs(1) * sampMeanVar) / (Num0-1))
#         zVec = (gammaVec - sampMean) / sampStd    
#         offComp = st.norm.pdf(zVec)
#         if not offComp.sum()==0:
#             offComp *= (n0 / offComp.sum())
#             if not np.any(np.isnan(offComp)):
#                 isDone = True
#     return offComp