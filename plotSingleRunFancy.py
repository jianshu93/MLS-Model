#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:36:34 2018

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colr
import matplotlib as mpl
import itertools
import datetime
from pathlib import Path
import MLS_static as mlss


now = datetime.datetime.now()
save_folder = Path("Figures/")

saveName = "singleRun_noHD"+now.strftime("_%Y-%m-%d_")+".pdf"
saveName = save_folder / saveName

MODEL_PAR = {
                #fixed model parameters
                "sampling" : "fixedvar",
                "maxT"  : 8000., 
                "sampleT": 20.,
                "mu"    : 1E-3,
                "B_H"   : 3.,
                "D_H"   : 1./60.,
                #variable model parameters
                "gamma" : 0.05,
                "TAU_H" : 50E6,
                "n0"    : 1E-3,
                "mig"   : 1E-6,
                "r"     : 1.,
                "K"     : 1E3,
                "sigmaBirth" : 0.3,
                #fixed intial condition
                "NUMGROUP" : 100,  
                "F0" : 1.,
                "N0init" : 1,
                #host evolution parameters
                "HostEvolves" : False,
                "sigmaR" : 0,
                "sigmaMig" : 0.2,
                "sigmaN0" : 0
        }


Output = mlss.single_run_noplot(MODEL_PAR)


#%% plot gamma-curve
   
def create_fig_keynote(nr,nc):
    font = {'family' : 'Arial',
            'weight' : 'light',
            'size'   : 28}
    
    mpl.rc('font', **font)
    
    fig,axs = plt.subplots(nr,nc)
    mydpi=150
    w = 2400
    h = 800
    fig.set_size_inches(w/mydpi,h/mydpi)


    return (fig,axs)


def plot_data_fancy(axs,dataStruc, FieldName):
    axs.plot(dataStruc['time'], dataStruc[FieldName], \
             label=FieldName, linewidth=3)
    axs.set_xlabel("time")
    maxTData = np.nanmax(dataStruc['time'])
    
    plt.xlim((0,maxTData))
    axs.set_xticks([0, maxTData/2, maxTData])
   
    return


fig, axs = create_fig_keynote(1,2)

plot_data_fancy(axs[0],Output,"F_T_av")  
axs[0].set_ylabel('fraction of cooperators $f_c$')
axs[0].set_yticks([0, 0.5, 1])

plot_data_fancy(axs[1],Output,"H_T")  
axs[1].set_ylabel('number of host')
axs[1].set_yticks([0, 60, 120])

axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)

axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

plt.tight_layout()
plt.draw()

plt.savefig(saveName, dpi=150, format="pdf", transparent=True)
