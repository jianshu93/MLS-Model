#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
CreateD on March 22 2019
Last Update March 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

"""
import math
import numpy as np
import scipy.stats as st
from numba import jit, f8, i8
from numba.types import UniTuple

"""
 General functions
"""

# create matrix with random numbers, excluding 0 and 1


def create_randMat(num_t, num_rand):
    notDone = True
    while notDone:
        randMat = np.random.random((num_t, num_rand))
        containsNo0 = (~np.any(randMat == 0))
        containsNo1 = (~np.any(randMat == 1))
        if containsNo0 & containsNo1:
            notDone = False

    return randMat

# calculate timestep to have max 1 host level event per step
#@jit(f8(f8, f8[:]), nopython=True)
@jit    
def calc_max_time_step(max_host_prop, dtVec, max_p2=0.01):
    # calc P(2 events)
    p0 = np.exp(-max_host_prop * dtVec)
    p1 = max_host_prop * dtVec * np.exp(-max_host_prop * dtVec)
    p2 = 1 - p0 - p1
    # choose biggest dt with constrained that P(2 events) < maxP2
    dtMax = dtVec[p2 < max_p2].max()
    return dtMax

# draw from truncated normal distribution


def trunc_norm(exp, std, min=-np.inf, max=np.inf, type='lin'):
    # log transform data if needed
    if type == 'log':
        exp = math.log10(exp)
        min = math.log10(min)
        max = math.log10(max)
    elif type != 'lin':
        raise Exception('problem with trunc_norm only support lin or log mode')
    # convert bounds to to standard normal distribution
    minT = (min - exp) / std
    maxT = (max - exp) / std
    # draw from truncated normal distribution and transfrom to desired mean and var
    newT = st.truncnorm.rvs(minT, maxT) * std + exp
    # log transfrom if needed
    if type == 'log':
        newT = 10 ** newT
    return newT

# random sample based on propensity
@jit(i8(f8[::1], f8), nopython=True)
def select_random_event(propensity_vec, randNum):
    # calculate cumulative propensities
    cumPropensity = propensity_vec.cumsum()
    # rescale uniform random number [0,1] to total propensity
    randNumScaled = randNum * cumPropensity[-1]
    # create index vector
    index = np.arange(cumPropensity.size)
    # select group
    id_group = index[(cumPropensity > randNumScaled)][0]

    return id_group

# calculate moving average of time vector
@jit(UniTuple(f8, 2)(f8[:], i8, i8), nopython=True)
def calc_moving_av(f_t, curr_idx, windowLength):
    # get first time point
    start_idx = max(0, curr_idx - windowLength + 1)
    movingAv = f_t[start_idx:curr_idx].mean()
    movindStd = f_t[start_idx:curr_idx].std()

    return (movingAv, movindStd)

# calculate rms error of time vector
@jit(f8(f8[:], i8, i8), nopython=True)
def calc_rms_error(mav_t, curr_idx, windowLength):
    # get first time point
    start_idx = max(0, curr_idx-windowLength+1)
    # get time points to process
    localsegment = mav_t[start_idx:curr_idx]
    # calc rms error
    av = localsegment.mean()
    errorSquared = (localsegment-av)**2
    meanErrorSquared = errorSquared.mean()
    rms_err = math.sqrt(meanErrorSquared)

    return rms_err
