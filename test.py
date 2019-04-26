#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:44:11 2019

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import math
import scipy.stats as st
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import mls_general_code as mlsg


def cdf_sp(x):
    'Cumulative distribution function for the standard normal distribution'
    return st.norm.cdf(x)


#newT = st.truncnorm.rvs(minT, maxT) * std + exp


def trunc_n(exp, std):
    minX = (0 - exp) / std
    maxX = (1 - exp) / std

    return st.truncnorm.rvs(minX, maxX) * std + exp


def inf_cdf1(x):
    return st.norm.ppf(x)


def norm_cdf(x):
    'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def norm_inv_cdf(x):
    return - math.sqrt(2.0) * special.erfcinv(2.0 * x)


def trunc_n_fast(exp, std, rnd):
    minX = norm_cdf((0 - exp) / std)
    maxX = norm_cdf((1 - exp) / std)

    rndRescaled = rnd * (maxX - minX) + minX

    rndTrunc = norm_inv_cdf(rndRescaled) * std + exp

    if rndTrunc < 0:
        rndTrunc = 0
    elif rndTrunc > 1:
        rndTrunc = 1
    return rndTrunc


def t1(cumPropensity, randNumScaled):
    # create index vector
    index = np.arange(cumPropensity.size)
    # select group
    id_group = index[(cumPropensity > randNumScaled)][0]
    return id_group


def t2(cumPropensity, randNumScaled):    
    # select group
    id_group = (np.nonzero((cumPropensity > randNumScaled)))[0].min()
    return id_group


def t3(cumPropensity, randNumScaled):    
    # select group
    id_group = np.argwhere((cumPropensity > randNumScaled)).min()
    return id_group



CVec = np.arange(20)
hasDied = np.zeros(20)
hasDied[[2, 5, 7, 11, 19]] = 1
CVecNew = np.arange(5) + 20
def t4():
    CVec1 = np.delete(CVec, np.nonzero(hasDied))
    CVec1 = np.append(CVec1, CVecNew)
    return CVec1

def t5():
    CVec = np.concatenate((CVec[hasDied==0], CVecNew))
    return CVec1





def testCode(exp, std):

    N = int(1E5)
    n_bins = 100

    rnd = mlsg.create_randMat(int(N), 1)

    y1 = np.zeros(N)
    y2 = np.zeros(N)

    for ii in range(N):
        y1[ii] = trunc_n_fast(exp, std, rnd[ii])
        y2[ii] = trunc_n(exp, std)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    fig.set_size_inches(4, 3)

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(y1, bins=n_bins)
    axs[1].hist(y2, bins=n_bins)
