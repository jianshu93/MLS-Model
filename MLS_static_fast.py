#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
CreateD on WeD OCt 17 12:25:24 2018
Last Update May 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

This code implements the two species (helper and neutral cells) model
"""
import mls_general_code as mlsg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from numba import jit, void, f8, i8
from numba.types import UniTuple, Tuple

"""
Init functions 
"""

# initialize community
def init_comm(model_par):
    numGroup = int(model_par["NUMGROUP"])

    if numGroup == -1:
        numGroup = int(model_par["K_H"])
    # setup initial vector of c,d

    if model_par["F0"] == 'uniform':
        # setup initial vector of c,d
        cAct = model_par["N0init"] * np.linspace(0, 1, numGroup)
        dAct = model_par["N0init"] - cAct
    else:
        cAct = np.full(numGroup,
                       model_par["N0init"] * model_par["F0"])
        dAct = np.full(numGroup,
                       model_par["N0init"] * (1 - model_par["F0"]))

    # store in C-byte order
    cAct = np.copy(cAct, order='C')
    dAct = np.copy(dAct, order='C')
    return (cAct, dAct)


# initialize output matrix
def init_output_matrix(Num_t_sample):
    # specify output fields
    dType = np.dtype([('F_T_av', 'f8'),
                      ('N_T_av', 'f8'),
                      ('H_T', 'f8'),
                      ('F_mav', 'f8'),
                      ('H_mav', 'f8'),
                      ('F_mmed', 'f8'),
                      ('F_T_med', 'f8'),
                      ('F_mstd', 'f8'),
                      ('H_mstd', 'f8'),
                      ('rms_err', 'f8'),
                      ('time', 'f8')])

    Output = np.full(Num_t_sample, np.nan, dType)
    Output['time'][0] = 0
    return Output


"""
Sample community functions 
"""
# calculate statistics of current community composition
@jit(UniTuple(f8, 3)(f8[::1], f8[::1]), nopython=True)
def calc_mean_fraction(c, d):
    TOT = c + d  # total density per host
    FRAC = c[TOT > 0] / TOT[TOT > 0]  # cooperator fraction per host
    F_av = FRAC.mean()
    F_med = np.median(FRAC)
    N_av = TOT.mean()
    return (F_av, N_av, F_med)


# calculate distribution of investments over host population
def calc_perhost_inv_distri(c, d, binEdges):
    TOT = c + d  # total density per host
    FRAC = c[TOT > 0] / TOT[TOT > 0]  # cooperator fraction per host

    # get distribution of average cooperator fraction per host
    avHostDens, _ = np.histogram(FRAC, bins=binEdges)
    avHostDens = avHostDens / np.count_nonzero([TOT > 0])
    return avHostDens


# sample model
def sample_model(c, d, binEdges, output, InvestmentPerHost, sample_idx, currT, model_par):
    # store time
    output['time'][sample_idx] = currT
    # calc host population properties
    nGroup = c.size
    output['H_T'][sample_idx] = nGroup
    InvestmentPerHost[sample_idx, :] = calc_perhost_inv_distri(c, d, binEdges)
    # calc population average properties
    F_av, N_av, F_med = calc_mean_fraction(c, d)
    output['F_T_av'][sample_idx] = F_av
    output['F_T_med'][sample_idx] = F_med
    output['N_T_av'][sample_idx] = N_av

    # calc time windows to average over
    timeAvWindow = int(np.ceil(model_par['mav_window'] / model_par['sampleT']))
    rmsAvWindow = int(np.ceil(model_par['rms_window'] / model_par['sampleT']))
    # calc moving average fraction
    if sample_idx >= 1:
        mav, mstd = mlsg.calc_moving_av(
            output['F_T_av'], sample_idx, timeAvWindow)
        output['F_mav'][sample_idx] = mav
        output['F_mstd'][sample_idx] = mstd
        output['F_mmed'][sample_idx] = mlsg.calc_moving_med(
            output['F_T_med'], sample_idx, timeAvWindow)

        mavH, mstdH = mlsg.calc_moving_av(
            output['H_T'], sample_idx, timeAvWindow)
        output['H_mav'][sample_idx] = mavH
        output['H_mstd'][sample_idx] = mstdH
    # calc rms error
    if sample_idx >= rmsAvWindow:
        rms_err = mlsg.calc_rms_error(output['F_mav'], sample_idx, rmsAvWindow)
        output['rms_err'][sample_idx] = rms_err

    sample_idx += 1
    return sample_idx


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
Host rate functions 
"""

# calc host propensity vectors for birth and death events
@jit(Tuple((f8[::1], i8))(f8, f8, f8, f8, f8, f8[::1], f8[::1]), nopython=True)
def calc_host_propensity_vectors(TAU_H, B_H, D_H, K_H, dtBac, c, dtVec):
    birthPropVec = (1 + B_H * c) / TAU_H
    deathPropVec = (1 - D_H * c) * c.size / (K_H * TAU_H)
    totPropVec = np.concatenate((birthPropVec, deathPropVec))
    cumulPropVec = totPropVec.cumsum()
    # calc number of required time steps
    (dtHost, numSubStep) = calc_dynamic_timestep(
        cumulPropVec[-1], dtVec, dtBac)
    # calc probVec
    cumulPropVec *= dtHost
    return (cumulPropVec, numSubStep)

# get composition of host parent that will give birth
@jit(f8(f8[::1], f8[::1], i8), nopython=True)
def host_create_offspring(c, d, id_group):
    # draw offspring composition
    fracPar = c[id_group] / (c[id_group] + d[id_group])
    return fracPar


# update host dynamics while keeping microbial dynamics fixed
@jit(Tuple((f8[::1], f8[::1], f8[::1], i8))(f8[::1], f8[::1], f8[::1], f8[::1], i8, f8, f8, f8[:, ::1], i8))
def update_host(CVec, DVec, AgeVec, cumulPropVec, numSubStep, n0, sigma, rndMat, ridx):
    # init  vectors that keep track of changes in host
    numGroup = CVec.size
    cNewTemp = np.zeros(numSubStep)
    dNewTemp = np.zeros(numSubStep)
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
                fracPar = host_create_offspring(CVec, DVec, id_group)
                fracOff = mlsg.trunc_norm_fast(
                    fracPar, sigma, 0., 1., rndMat[ridx, 1])
                cOff = n0 * fracOff
                dOff = n0 * (1 - fracOff)
                cNewTemp[numNewBorn] = cOff
                dNewTemp[numNewBorn] = dOff
                numNewBorn += 1
            elif deathEvent and hostAlive:
                # process host death event
                id_group -= numGroup
                # store which hosts have died
                hasDied[id_group] = 1
        ridx += 1

    # now process host birth and death event
    # delete hosts that have died and add newborn hosts
    CVec = np.concatenate((CVec[hasDied == 0], cNewTemp[0:numNewBorn]))
    DVec = np.concatenate((DVec[hasDied == 0], dNewTemp[0:numNewBorn]))
    AgeVec = np.concatenate((AgeVec[hasDied == 0], np.zeros(numNewBorn)))

    return(CVec, DVec, AgeVec, ridx)


"""
Community dynamics functions 
"""
# updates community composition during timestep dt
@jit(void(f8[::1], f8[::1], f8, f8, f8, f8), nopython=True)
def update_comm(c, d, cost, mu, mig, dt):
    nGroup = c.size
    fh = 1 / (nGroup - 1) #fraction of migrants per host
    n = c + d #tot pop size
    if nGroup > 1:
        # calc derivatives
        c += dt * (
            c * ((1 - mu) * (1 - cost) - n - (1 + fh) * mig)
            + mu * d
            + fh * mig * c.sum())
        d += dt * (
            d * ((1 - mu) - n - (1 + fh) * mig)
            + mu * (1 - cost) * c
            + fh * mig * d.sum())
    else:  # no migration
        # calc derivatives
        c += dt * (
            c * ((1 - mu) * (1 - cost) - n - mig)
            + mu * d)
        d += dt * (
            d * ((1 - mu) - n - mig)
            + mu * (1 - cost) * c)
        
    return


"""
Full model
"""
# run model main code


def run_model_fixed_parameters(model_par):
    # possible dt to choose from
    dtVec = np.logspace(-7, -2, 29)

    # get time settings
    dtBac, maxT, samplingInterval = [
        float(model_par[x]) for x in ('dT', 'maxT', 'sampleT')]

    if 'minTRun' in model_par:
        minTRun = max(model_par['minTRun'], model_par['rms_window']+1)
    else:
        minTRun = model_par['rms_window']+1

    # calc timesteps
    Num_t_sample = int(np.ceil(maxT / samplingInterval)+1)
    # get host rates
    B_H, D_H, TAU_H, K_H = [float(model_par[x])
                            for x in ('B_H', 'D_H', 'TAU_H', 'K_H')]
    # get bacterial rates
    mu, mig, cost = [float(model_par[x]) for x in ('mu', 'mig', 'cost')]
    # get reproduction rates
    n0, sigma = [float(model_par[x]) for x in ('n0', 'sigmaBirth')]

    # set limit on nr of rand number to preload
    maxRandMatSize = int(1E6)
    rndMat = mlsg.create_randMat(maxRandMatSize, 2)

    # init output
    numBins = int(model_par['numTypeBins'])
    binEdges = np.linspace(0, 1, numBins)
    Output = init_output_matrix(Num_t_sample)
    InvestmentPerHost = np.full((Num_t_sample, numBins-1), np.nan)

    # init groups
    CVec, DVec = init_comm(model_par)
    AgeVec = np.zeros_like(CVec)

    # first sample
    currT = 0
    sampleIndex = 0
    tiR = 0
    sampleIndex = sample_model(CVec, DVec, binEdges,
                               Output, InvestmentPerHost,
                               sampleIndex, currT, model_par)

    # run time
    while currT <= maxT:
        # calc fixed propensities for each host
        cumulPropVec, numSubStep = calc_host_propensity_vectors(
            TAU_H, B_H, D_H, K_H, dtBac, CVec, dtVec)

        # reset rand matrix when used up
        if tiR >= (maxRandMatSize - numSubStep):
            rndMat = mlsg.create_randMat(maxRandMatSize, 2)
            tiR = 0

        # update hosts
        CVec, DVec, AgeVec, tiR = update_host(
            CVec, DVec, AgeVec, cumulPropVec, numSubStep, n0, sigma, rndMat, tiR)

        # update age
        AgeVec += dtBac

        # next update community
        update_comm(CVec, DVec, cost, mu, mig, dtBac)

        # stop run if all hosts die
        if CVec.size == 0:
            Output = Output[0:sampleIndex]
            break
        # update time
        currT += dtBac
        # sample model at intervals
        nextSampleT = samplingInterval * sampleIndex
        if currT >= nextSampleT:
            sampleIndex = sample_model(CVec, DVec, binEdges,
                                       Output, InvestmentPerHost,
                                       sampleIndex, currT, model_par)
            # check if steady state has been reached
            if currT > minTRun:
                lowError = Output['rms_err'][sampleIndex -
                                             1] < model_par['rms_err_treshold']
                dIdx = int(model_par['rms_window'] / samplingInterval)

                delta = np.abs(Output['F_mav'][sampleIndex - 1] -
                               Output['F_mav'][sampleIndex - dIdx])
                lowDelta = delta < 0.05
                if lowError & lowDelta:
                    break

    # cut off non existing time points at end
    Output = Output[0:sampleIndex]
    InvestmentPerHost = InvestmentPerHost[0:sampleIndex, :]

    FracVec = CVec / (CVec + DVec)

    return (Output, InvestmentPerHost, FracVec, AgeVec)


"""
Run Full model
"""

# run model, output final state only


def single_run_finalstate(MODEL_PAR):
    # run model
    Output, InvestmentPerHost, FracVec, AgeVec = run_model_fixed_parameters(
        MODEL_PAR)

    # init output
    dType = np.dtype([
        ('F_T_av', 'f8'), ('F_mav', 'f8'), ('F_mmed', 'f8'), ('F_mstd', 'f8'),
        ('F_mav_ss', 'f8'), ('N_T_av', 'f8'),
        ('H_T', 'f8'), ('H_mav', 'f8'), ('H_mstd', 'f8'),
        ('cost', 'f8'), ('K', 'f8'),
        ('n0', 'f8'), ('mig', 'f8'), ('sigmaBirth', 'f8'),
        ('TAU_H', 'f8'), ('B_H', 'f8'), ('D_H', 'f8'), ('K_H', 'f8'), ('mu', 'f8')])
    output_matrix = np.zeros(1, dType)

    # store final state
    output_matrix['F_mmed'] = Output['F_mmed'][-1]
    output_matrix['F_T_av'] = Output['F_T_av'][-1]
    output_matrix['F_mav'] = Output['F_mav'][-1]
    output_matrix['F_mstd'] = Output['F_mstd'][-1]
    output_matrix['H_T'] = Output['H_T'][-1]
    output_matrix['H_mav'] = Output['H_mav'][-1]
    output_matrix['H_mstd'] = Output['H_mstd'][-1]
    output_matrix['N_T_av'] = Output['N_T_av'][-1]

    # store nan in F_mav_ss if not reached steady state
    if Output['rms_err'][-1] < MODEL_PAR['rms_err_treshold']:
        output_matrix['F_mav_ss'] = Output['F_mav'][-1]
    else:
        output_matrix['F_mav_ss'] = np.nan

    # store model settings
    output_matrix['cost'] = MODEL_PAR['cost']
    output_matrix['n0'] = MODEL_PAR['n0']
    output_matrix['mig'] = MODEL_PAR['mig']
    output_matrix['K'] = MODEL_PAR['K']
    output_matrix['sigmaBirth'] = MODEL_PAR['sigmaBirth']
    output_matrix['TAU_H'] = MODEL_PAR['TAU_H']
    output_matrix['B_H'] = MODEL_PAR['B_H']
    output_matrix['D_H'] = MODEL_PAR['D_H']
    output_matrix['K_H'] = MODEL_PAR['K_H']
    output_matrix['mu'] = MODEL_PAR['mu']
    InvestmentPerHostEnd = InvestmentPerHost[-1, :]
    return (output_matrix, InvestmentPerHostEnd)


# run model, do not plot dynamics
def single_run_noplot(MODEL_PAR):
    # run code
    start = time.time()
    Output, InvestmentPerHost, _, _ = run_model_fixed_parameters(MODEL_PAR)
    end = time.time()
    print("Elapsed time run 1 = %s" % (end - start))

    return (Output, InvestmentPerHost)


def plot_data(dataStruc, FieldName, type='lin'):
    if type == 'lin':
        plt.plot(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    elif type == 'log':
        plt.semilogy(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    plt.xlabel("time")
    maxTData = dataStruc['time'].max()
    try:
        plt.xlim((0, maxTData))
    except:
        print(maxTData)
    return


# run model, plot dynamics
def single_run_with_plot(MODEL_PAR):
    # run code
    start = time.time()
    Output, InvestmentPerHost, FracVec, AgeVec = run_model_fixed_parameters(
        MODEL_PAR)
    end = time.time()
    print("Elapsed time run 1 = %s" % (end - start))

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 9}

    matplotlib.rc('font', **font)

    fig = plt.figure()
    nR = 3
    nC = 2

    # plot average investment
    plt.subplot(nR, nC, 1)
    plot_data(Output, "F_T_av")
    plot_data(Output, "F_mav")
    plt.ylabel("investment")
    plt.ylim((0, 1))

    # plot host number investment
    plt.subplot(nR, nC, 2)
    plot_data(Output, "H_mav")
    plt.ylabel("H(t)")

    plt.subplot(nR, nC, 5)
    plot_data(Output, "N_T_av")
    plt.ylabel("pop size")

    plt.subplot(nR, nC, 6)
    plt.scatter(AgeVec, FracVec, s=0.2, c=[0.15, 0.15, 0.15, 0.5])
    plt.ylabel("frac helper")
    plt.xlabel("host age")

    # plot error
    plt.subplot(nR, nC, 3)
    plot_data(Output, "rms_err", 'log')
    plt.ylabel("rms_err(t)")

    # plot average investment per host
    axs = plt.subplot(nR, nC, 4)
    currData = np.log10(InvestmentPerHost.transpose() + np.finfo(float).eps)
    im = axs.imshow(currData, cmap="viridis",
                    interpolation='nearest',
                    extent=[0, 1, 0, 1],
                    origin='lower',
                    vmin=-3,
                    aspect='auto')
    axs.set_xticks([0, 1])
    axs.set_yticks([0, 1])
    axs.set_ylabel('investment')
    axs.set_xlabel('time')
    fig.colorbar(im, ax=axs, orientation='vertical',
                      fraction=.1, label="log10 mean density")
    axs.set_yticklabels([0, 1])
    maxTData = Output['time'].max()
    axs.set_xticklabels([0, int(round(maxTData))])

    fig.set_size_inches(4, 4)
    plt.tight_layout()

    return (Output, InvestmentPerHost)


# run model with default parameters
def debug_code():
    model_par = {
        # time step parameters
        "dT": 1E-2,
        "maxT": 100.,
        "sampleT": 1,
        "rms_err_treshold": 1E-5,
        "mav_window": 1000,
        "rms_window": 5000,
        # fixed model parameters
        "sampling": "fixedvar",
        "sigmaBirth": 0.1,
        "mu": 1E-5,
        "B_H": 1.,
        "D_H": 0.,
        "K_H": 100.,
        # variable model parameters
        "cost": 0.01,
        "TAU_H": 10.,
        "n0": 1E-3,
        "mig": 1E-5,
        "K": 10E3,
        # fixed intial condition
        "NUMGROUP": 100,
        "numTypeBins": 100,
        "F0": 0.5,
        "N0init": 1.
    }

    Output, InvestmentPerHost = single_run_with_plot(model_par)

    return Output, InvestmentPerHost


if __name__ == "__main__":
    print("running debug")
    Output, InvestmentPerHost = debug_code()
