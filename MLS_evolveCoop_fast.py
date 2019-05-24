#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
CreateD on WeD OCt 17 12:25:24 2018
Last Update May 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

This code implements the continuos investment model. 


"""
import mls_general_code as mlsg
import numpy as np
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import time
from numba import jit, void, f8, i8 
from numba.types import UniTuple, Tuple
import datetime
from pathlib import Path

"""
Time step functions 
"""

# calc dt for host dynamics, creating integer number of steps within dt of bacterial dynamics
@jit(Tuple((f8, i8))(f8, f8[::1], f8), nopython=True)
def calc_dynamic_timestep(hostProp, dtVec, dtBac):
    # calculate smallest time step ever needed
    dt = mlsg.calc_max_time_step(hostProp, dtVec)
    if dt == dtVec.min():
        print("warning: min dt reached")
    # make sure that there is integer number of time steps
    numSubStep = int(np.ceil(dtBac / dt))
    dt = dtBac/numSubStep
    return (dt, numSubStep)

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

#calculate distribution of investments over host population
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


"""
Host event functions 
"""    
# calc host propensity vectors for birth and death events
@jit(Tuple((f8[::1], i8))(f8[:, ::1], f8[::1], f8, f8, f8, f8, f8, f8[::1]), nopython=True)
def calc_host_propensity_vectors(gMat, gammaVec, TAU_H, B_H, D_H, K_H, dtBac, dtVec):
    invPerCom = gMat @ gammaVec
    nGroup = gMat.shape[0]
    birthPropVec = (1 + B_H * invPerCom) / TAU_H
    deathPropVec = (1 - D_H * invPerCom) * nGroup / (K_H * TAU_H)
    totPropVec = np.concatenate((birthPropVec, deathPropVec))
    cumulPropVec = totPropVec.cumsum()
    # calc number of required time steps
    (dtHost, numSubStep) = calc_dynamic_timestep(
        cumulPropVec[-1], dtVec, dtBac)
    # calc probVec
    cumulPropVec *= dtHost
    return (cumulPropVec, numSubStep)


# update host dynamics while keeping microbial dynamics fixed
@jit(Tuple((f8[:, ::1], i8))(f8[:, ::1], f8[::1], i8, f8, f8, f8[:, ::1], i8), nopython=True)
def update_host(gMat, cumulPropVec, numSubStep, n0, K, rndMat, ridx):
    # init  vectors that keep track of changes in host
    numGroup = gMat.shape[0]
    numBins = gMat.shape[1]
    gMatTemp = np.zeros((numSubStep, numBins))
    numNewBorn = 0
    hasDied = np.zeros(numGroup)
    indexVec = np.arange(numGroup * 2)

    # first process host events at fine time resolution
    for tt in range(numSubStep):
        # select random event based on prop, returns -1 if there is no event
        if rndMat[ridx, 0] < cumulPropVec[-1]:
            # there is a host event, select group
            id_group = indexVec[(cumulPropVec > rndMat[ridx, 0])][0]
            birthEvent = (id_group >= 0) and (id_group < numGroup)
            deathEvent = id_group >= numGroup
            hostAlive = hasDied[(id_group % numGroup)] == 0
            if birthEvent and hostAlive:
                 # there is birth event
                parComp = gMat[id_group, :]
                offComp = host_birth_composition_sample(parComp, n0, K)
                gMatTemp[numNewBorn, :] = offComp
                numNewBorn += 1
            elif deathEvent and hostAlive:
                # process host death event
                id_group -= numGroup
                # store which hosts have died
                hasDied[id_group] = 1
        ridx += 1

    # now process host birth and death event
    # delete hosts that have died and add newborn hosts
    gMat=np.concatenate((gMat[hasDied == 0, :], gMatTemp[0:numNewBorn, :]))
 
    return (gMat, ridx)


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
    dtVec = np.logspace(-7, -2, 19)

    # get time settings
    dtBac, maxT, samplingInterval = [
        float(model_par[x]) for x in ('dT', 'maxT', 'sampleT')]
    if 'minTRun' in model_par:
        minTRun = min(model_par['minTRun'], maxT)
    else:
        minTRun = 0

    # calc timesteps
    Num_t_sample = int(np.ceil(maxT / samplingInterval)+1)
    
    #get host rates
    B_H, D_H, TAU_H, K_H = [float(model_par[x]) for x in ('B_H','D_H','TAU_H','K_H')]
    #get bacterial rates
    r, mu, mig, cost = [float(model_par[x]) for x in ('r','mu','mig', 'cost')]
    # get reproduction rates
    n0, K = [float(model_par[x]) for x in ('n0', 'K')]
    numBins = int(model_par['numTypeBins'])
   
    # set limit on nr of rand number to preload
    maxRandMatSize = int(1E6)
    rndMat=mlsg.create_randMat(maxRandMatSize, 2)
    
    # init groups
    gMat = init_comm(model_par)
    locBMMat, gammaVec = create_local_update_matrix(r, mu, cost, numBins)
    #init output
    Output = init_output_matrix(Num_t_sample)
    InvestmentAll = np.full((Num_t_sample, numBins), np.nan)
    InvestmentPerHost = np.full((Num_t_sample, numBins), np.nan)
    
    #first sample
    currT = 0
    tiR = 0
    sampleIndex = 0
    sampleIndex = sample_model(gMat, gammaVec, Output, \
                               InvestmentAll, InvestmentPerHost, \
                               sampleIndex, currT, model_par)
   
    #run time    
    while currT <= model_par['maxT']:
         # calc fixed propensities for each host
        cumulPropVec, numSubStep = calc_host_propensity_vectors(
            gMat, gammaVec, TAU_H, B_H, D_H, K_H, dtBac, dtVec)

        # reset rand matrix when used up
        if tiR >= (maxRandMatSize - numSubStep):
            rndMat = mlsg.create_randMat(maxRandMatSize, 2)
            tiR = 0

        # update hosts
        gMat, tiR = update_host(
            gMat, cumulPropVec, numSubStep, n0, K, rndMat, tiR)

        #update community
        update_comm_loc(gMat, locBMMat, r, mig, dtBac)
    
        #stop run if all hosts die
        if  gMat.shape[0]==0:
            break

        #update time
        currT += dtBac
        
        #sample model at intervals
        nextSampleT = samplingInterval * sampleIndex
        if currT >= nextSampleT:
            sampleIndex = sample_model(gMat, gammaVec, Output, \
                                       InvestmentAll, InvestmentPerHost, \
                                       sampleIndex, currT, model_par)
            
            # check if steady state has been reached
            lowError = Output['rms_err'][sampleIndex -
                                         1] < model_par['rms_err_treshold']
            longEnough = currT > minTRun

            if lowError & longEnough:
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
              ('cost', 'f8'), \
              ('TAU_H', 'f8'), \
              ('n0', 'f8'),  \
              ('mig', 'f8'), \
              ('r', 'f8'),   \
              ('B_H', 'f8'),   \
              ('D_H', 'f8'),   \
              ('K_H', 'f8'),   \
              ('mu', 'f8'),   \
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

    output_matrix['cost'] = MODEL_PAR['cost']
    output_matrix['TAU_H'] = MODEL_PAR['TAU_H']
    output_matrix['n0'] = MODEL_PAR['n0']
    output_matrix['mig'] = MODEL_PAR['mig']
    output_matrix['r'] = MODEL_PAR['r']
    output_matrix['K'] = MODEL_PAR['K']   
    output_matrix['B_H']=MODEL_PAR['B_H']
    output_matrix['D_H']=MODEL_PAR['D_H']
    output_matrix['K_H']=MODEL_PAR['K_H']
    output_matrix['mu']=MODEL_PAR['mu']
    
    return (output_matrix, InvestmentPerHost)

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


#run model, plot dynamics, save output to disk
def single_run_plot_save(MODEL_PAR):
    gMat, Output, InvestmentAll, InvestmentPerHost = single_run_with_plot(
        MODEL_PAR)
    
    varToGet = ('maxT','mu','B_H','D_H','K_H','cost','TAU_H','n0','mig','K','sampling')
    varList = ['-{}_{}'.format(x, MODEL_PAR[x]) for x in varToGet]
    
    savename = 'mlsef' + ''.join(varList) + '.npz'

    data_folder = Path("Data_Runs/")
    loc = data_folder / savename

    np.savez(loc, gMat=gMat, InvestmentAll=InvestmentAll,
             InvestmentPerHost=InvestmentPerHost, Output=Output,
             MODEL_PAR=[MODEL_PAR], date=datetime.datetime.now())
    
    print('saved as {}'.format(loc))
    
    return None
    
#run model with default parameters
def debug_code():
    model_par = {
                #time step parameters
                "maxT"  : 1000., 
                "dT" : 1E-2,
                "sampleT": 1,
                "rms_err_treshold": 1E-5,
                "mav_window": 1000,
                "rms_window": 5000,
                #fixed model parameters
                "sampling" : "sample",
                "mu"    : 0.02,
                "B_H"   : 1.,
                "D_H"   : 0.,
                "K_H"   : 20.,
                #variable model parameters
                "cost" : 0.01,
                "TAU_H" : 10.,
                "n0"    : 1E-3,
                "mig"   : 1E-5,
                "r"     : 1.,
                "K"     : 5E3,
                #fixed intial condition
                "NUMGROUP" : 20,  
                "numTypeBins" : 100,
                "meanGamma0" : 0.01,
                "stdGamma0" : 0.01,
                "N0init" : 1.,
        }
    
    gMat, Output, InvestmentAll, InvestmentPerHost = single_run_with_plot(model_par)
    
    return Output, InvestmentAll, InvestmentPerHost

if __name__ == "__main__":
    print("running debug")
    Output, InvestmentAll, InvestmentPerHost = debug_code()