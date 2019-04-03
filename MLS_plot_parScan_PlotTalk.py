#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:10:02 2018

@author: simonvanvliet
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colr
import matplotlib as mpl
import itertools
from mpl_toolkits.mplot3d import axes3d    


import datetime
#from matplotlib import colors
from pathlib import Path

now = datetime.datetime.now()

saveNameMod = "FixedVariance"

data_folder = Path("Data/")
#fileName = "20181120_16h32_fixedcv"
#fileName = "20181115_20h15_fixedvar"
fileName = "20190321_14h34_fixedvar"

fileName = "parScan_" + fileName + ".npz"
fileName = data_folder / fileName

save_folder = Path("Figures/")
saveName = "heatmap"+now.strftime("_%Y-%m-%d_")+saveNameMod + ".pdf"
saveName = save_folder / saveName

file = np.load(fileName, allow_pickle=True)

modelParList = file['modelParList']
parRange = file['parRange']
parOrder = file['parOrder']

data1D = file['data']





#parRange = [gamma_vec, tauH_vec, n0_vec, mig_vec, rr_vec, K_vec] 

ndSize = [ x.size for x in parRange]
data = np.reshape(data1D, ndSize)

#parOrder = np.array(['gamma','tauH','n0','mig','r','K'])

gammaIndex = np.asscalar(np.nonzero(parOrder=="gamma")[0])
tauIndex = np.asscalar(np.nonzero(parOrder=="tauH")[0])
n0Index = np.asscalar(np.nonzero(parOrder=="n0")[0])
migIndex = np.asscalar(np.nonzero(parOrder=="mig")[0])
rIndex = np.asscalar(np.nonzero(parOrder=="r")[0])
KIndex = np.asscalar(np.nonzero(parOrder=="K")[0])
sigmaIndex = np.asscalar(np.nonzero(parOrder=="sigma")[0])


sigmaL = [x[sigmaIndex] for x in itertools.product(*parRange)]
sigmaL =np.asarray(sigmaL)

numGamma = ndSize[gammaIndex]
numTau = ndSize[tauIndex]
numR = ndSize[rIndex]
numK = ndSize[KIndex]
numSig = ndSize[sigmaIndex]

n0V = parRange[n0Index]
migV = parRange[migIndex]

plt.rcParams.update({'font.size': 8})


#%%

def make_categorial(vector):
    elements =  np.unique(vector)
    index = np.arange(elements.size)
    cat_vector = np.zeros(vector.size)
    
    for idx in range(vector.size):
        cat_vector[idx]=index[elements==vector[idx]]
        
    return cat_vector 


def create_fig(nRow,nCol):
    fig, axs = plt.subplots(nRow, nCol)
    w = 14
    h = 6
    fig.set_figwidth(w, forward=True)
    #fig.set_size_inches(w,h)
    return fig, axs

def plot_heatmap(images,axs,data,xvec,yvec):
    cmap = "cool"

    currData = np.log10(data).transpose()
                  
    axl = axs.imshow(currData, cmap=cmap, \
                    interpolation='nearest', \
                    extent=[xvec[0],xvec[-1],yvec[0],yvec[-1]], \
                    origin='lower', \
                    vmin=-3, vmax=0)
    
    xticks = [xvec[0], xvec[-1]]
    xtickNames = ["%.0f" %np.log10(x) for x in xticks]
    
    yticks = [yvec[0], yvec[-1]]
    ytickNames = ["%.0f" %np.log10(x) for x in yticks]
    
    axs.set_xticks(xticks)
    axs.set_yticks(yticks)
    #axs.set_title(titlename)
    
    axs.set_xlabel('log10 $N_0/K$')
    axs.set_ylabel('log10 $\\theta$')
    
    axs.set_xticklabels(xtickNames)
    axs.set_yticklabels(ytickNames)
    axs.label_outer()
    
    axs.set_aspect('equal')
    return axl


def addAnnotation(curAx,labels):
    if len(labels)==1:
        curAx.text(0.05, 0.5, labels[0], \
                horizontalalignment='left', \
                verticalalignment='center', \
                transform=curAx.transAxes)
    elif len(labels)==2:
        curAx.text(0.7, 0.5, labels[0], \
                horizontalalignment='center', \
                verticalalignment='bottom', \
                transform=curAx.transAxes)
        curAx.text(0.7, 0, labels[1], \
                horizontalalignment='center', \
                verticalalignment='bottom', \
                transform=curAx.transAxes)
        
    curAx.set_axis_off() 
    return
    
  
 

def create_fig_keynote(nr,nc):
    font = {'family' : 'Arial',
            'weight' : 'light',
            'size'   : 28}
    
    mpl.rc('font', **font)
    
    fig,axs = plt.subplots(nr,nc)
    mydpi=150
    w = 1800
    h = 1000
    fig.set_size_inches(w/mydpi,h/mydpi)


    return (fig,axs)

def calc_tauH(n0, theta, gamma, r):
    K = 1-gamma
    alpha = r*(1-gamma) - theta

    
    lown = np.log( K * theta / (K*theta - n0*alpha)) / alpha
    highn1 = np.log( K*(alpha+2*theta) / (2*(n0*alpha+K*theta)) ) / alpha
    highn2 = np.log( 2*n0*(alpha+2*theta) / (n0*alpha+K*theta) ) / theta
    
    #highn = (1 / theta) * np.log( 2*n0 / (theta * (1 - 2 * n0 - theta) + n0)) 
    highn = highn1 + highn2
    
    trn = K*theta / (alpha+4*theta)
    lownMask = n0 < trn
    
    tauH = highn 
    tauH[lownMask] = lown[lownMask]
    
    return tauH

#%%
    
sigmaCat = make_categorial(sigmaL)
tauVar = 1. / (data1D['gamma']*data1D['r'])
tauHer = calc_tauH(data1D['n0'], data1D['mig'], data1D['gamma'], data1D['r']) 

minTau = np.minimum(tauVar, tauHer)


#%% plot 3D 
saveName = "3D_plot2_tauH_TauV"+now.strftime("_%Y-%m-%d_")+saveNameMod + ".pdf"
saveName = save_folder / saveName

numSig =parRange[sigmaIndex].size
viridis7 = mpl.cm.get_cmap('RdYlBu', numSig)


font = {'family' : 'Arial',
            'weight' : 'light',
            'size'   : 28}
    
mpl.rc('font', **font)

fig = plt.figure()
mydpi=150
w = 2200
h = 1600
fig.set_size_inches(w/mydpi,h/mydpi)



ax = fig.add_subplot(111, projection='3d')

tauVarRel = tauVar / data1D['tau_H'] 
tauHerRel = tauHer / data1D['tau_H']  

print(tauVarRel.min(), tauVarRel.max())


xData = np.log10(tauVarRel)
yData = np.log10(tauHerRel)

xRange = (np.floor(xData.min()), np.ceil(xData.max()))
yRange = (np.floor(yData.min()), np.ceil(yData.max()))

ai = ax.scatter(xData , yData , data1D['F_mav_ss'], c=sigmaCat, s=60,\
                alpha=0.6, vmin=0, vmax=numSig-1, cmap=viridis7)
plt.xlim(xRange)
plt.ylim(yRange)

ax.set_xlabel('$log_{10} \\tau_V / \\tau_R$')
ax.set_ylabel('$log_{10} \\tau_H / \\tau_R$')

ax.set_xticks([-2, -1, 0, 1])
ax.set_yticks([-8, -4, 0, 4, 8])
ax.set_zticks([-0,0.2,0.4,0.6,0.8])
ax.set_zlim([0, 0.8])

ax.set_zlabel('fraction cooperator')
ax.view_init(20,-115)

ax.yaxis.labelpad=20
ax.xaxis.labelpad=20
ax.zaxis.labelpad=40
ax.tick_params(axis='z', which='major', pad=15)


dC = (numSig-1)/numSig
dCSt = dC/2
dCEnd = (numSig-1)-dC/2
cTick = np.linspace(dCSt,dCEnd,numSig)
cbx = fig.colorbar(ai, ax=ax, orientation='vertical', \
             shrink=.5, label="$\\sigma_f$",\
             ticks=cTick)

cbx.ax.set_yticklabels(np.flip(parRange[sigmaIndex],0))
cbx.solids.set(alpha=1)

plt.tight_layout()
plt.draw()


plt.savefig(saveName, dpi=mydpi, format="pdf", transparent=True)




 #%% plot tau_her  plot 
w=1800
h=1200
mydpi=150


saveName = "tau_her_heatmap"+now.strftime("_%Y-%m-%d_")+saveNameMod + ".pdf"
saveName = save_folder / saveName

fig,axs = create_fig_keynote(1,1)


tauPlot = 1
rPlot = 0
kPlot =0
gammaToPlot =0
tauToPlot = 0  
sigToPlot = 0

n0 = np.log10(parRange[n0Index])
mig = np.log10(parRange[migIndex])


fracN0Mig = data['F_mav_ss'][gammaToPlot, tauToPlot, :, :, rPlot, kPlot, sigToPlot].squeeze()

n0Mat = data['n0'][gammaToPlot, tauToPlot, :, :, rPlot, kPlot, sigToPlot].squeeze()
migMat = data['mig'][gammaToPlot, tauToPlot, :, :, rPlot, kPlot, sigToPlot].squeeze()




n0Fine = np.linspace(n0.min(), n0.max(), 1000)
migFine = np.linspace(mig.min(), mig.max(), 1000)

n0Fine = 10 ** n0Fine
migFine = 10 ** migFine

n0Gr, migGr = np.meshgrid(migFine,n0Fine)

gamma = parRange[gammaIndex][gammaToPlot]

tauHMat=calc_tauH(n0Gr, migGr, gamma, 1)

cmap = "RdYlBu"

viridisBig = mpl.cm.get_cmap('RdYlBu', 512)

indexVec= np.hstack( ( np.linspace(0, 0.5, 51), \
                       np.linspace(0.5, 1, 205)))
#np.linspace(0.375, 1, 256)

cmap = mpl.colors.ListedColormap(viridisBig(indexVec))

currData = np.log10((tauHMat))
              
axl = axs.imshow(currData, cmap=cmap, \
                interpolation='nearest', \
                extent=[n0[0],n0[-1],mig[0],mig[-1]], \
                origin='lower', \
                vmin=-3, vmax=12)

xticks = [n0[0], n0[5], n0[-1]]
xtickNames = ["%.0f" %(x) for x in xticks]

yticks = [mig[0], mig[5], mig[-1]]
ytickNames = ["%.0f" %(x) for x in yticks]

axs.set_xticks(xticks)
axs.set_yticks(yticks)
#axs.set_title(titlename)


axs.set_xticklabels(xtickNames)
axs.set_yticklabels(ytickNames)

axs.set_xlabel('$\log_{10} N_0/K$')
axs.set_ylabel('$\log_{10} \\theta$')

axs.set_aspect('equal')


fig.colorbar(axl, ax=axs, orientation='vertical', \
             shrink=.6, label="$\log_{10}\\tau_H$",\
             ticks=[-3, 0, 3, 6, 9, 12])

fig.set_size_inches(w/mydpi,h/mydpi)
plt.tight_layout()
plt.draw()


plt.savefig(saveName, dpi=mydpi, format="pdf", transparent=True)




  


#%% plot N0 - Mig  plot 
w=1800
h=1200
mydpi=150


saveName = "n0_mig_heatmap"+now.strftime("_%Y-%m-%d_")+saveNameMod + ".pdf"
saveName = save_folder / saveName

fig,axs = create_fig_keynote(1,1)


tauPlot = 1
rPlot = 0
kPlot =0
gammaToPlot =0
tauToPlot = 0  
sigToPlot = 0

n0 = np.log10(parRange[n0Index])
mig = np.log10(parRange[migIndex])


fracN0Mig = data['F_mav_ss'][gammaToPlot, tauToPlot, :, :, rPlot, kPlot, sigToPlot].squeeze()


cmap = "viridis"

currData = (fracN0Mig).transpose()
              
axl = axs.imshow(currData, cmap=cmap, \
                interpolation='nearest', \
                extent=[n0[0],n0[-1],mig[0],mig[-1]], \
                origin='lower', \
                vmin=0, vmax=0.6)

xticks = [n0[0], n0[5], n0[-1]]
xtickNames = ["%.0f" %(x) for x in xticks]

yticks = [mig[0], mig[5], mig[-1]]
ytickNames = ["%.0f" %(x) for x in yticks]

axs.set_xticks(xticks)
axs.set_yticks(yticks)
#axs.set_title(titlename)


axs.set_xticklabels(xtickNames)
axs.set_yticklabels(ytickNames)

axs.set_xlabel('$\log_{10} N_0/K$')
axs.set_ylabel('$\log_{10} \\theta$')

axs.set_aspect('equal')


fig.colorbar(axl, ax=axs, orientation='vertical', \
             shrink=.6, label="fraction cooperator",\
             ticks=[0, 0.3, 0.6])

fig.set_size_inches(w/mydpi,h/mydpi)
plt.tight_layout()
plt.draw()


plt.savefig(saveName, dpi=mydpi, format="pdf", transparent=True)





#%% plot cost vs host time scale curve 
w=1800
h=1100
mydpi=150


saveName = "costVsTauCurve"+now.strftime("_%Y-%m-%d_")+saveNameMod + ".pdf"
saveName = save_folder / saveName

fig,axs = create_fig_keynote(1,1)

n0Plot = -2
migPlot = -3
tauPlot = 2
rPlot = 0
kPlot =0

sigToPlot = 0

gamma = parRange[gammaIndex]
tau = parRange[tauIndex]

lab = ['$\\tau_R$= %.0f' %x for x in tau]

fracGammaTau = data['F_mav_ss'][:, :, n0Plot, migPlot, rPlot, kPlot, sigToPlot].squeeze()
lin = axs.plot(gamma,fracGammaTau, '-o',linewidth=4, markersize=10)

lineColors = mpl.cm.get_cmap('viridis', len(lin)+1).colors
for (i,c) in zip(lin, lineColors):
    i.set_color(c)

xtickVec = [0.01, 0.05, 0.1]

axs.set_xticks(xtickVec)
axs.set_yticks([0, 0.4, 0.8])
axs.set_xlabel('cost $\\gamma$')
axs.set_ylabel('fraction of cooperators $f_c$')



new_tick_locations = np.array([.2, .5, .9])


xl = ['%.0f' %(1/x) for x in xtickVec]
ax2 = axs.twiny()

ax2.set_xlim(axs.get_xlim())
ax2.set_xticks(xtickVec)
ax2.set_xticklabels(xl)
ax2.set_xlabel("$\\tau_V$")

axs.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
#plt.ylim((0,1))

plt.legend(lin, lab, loc='upper right', fancybox=True, framealpha=0.2)
fig.set_size_inches(w/mydpi,h/mydpi)
plt.tight_layout()
plt.draw()


plt.savefig(saveName, dpi=mydpi, format="pdf", transparent=True)



#%% plot gamma curve
w=1800
h=1000
mydpi=150


saveName = "gammaCurve"+now.strftime("_%Y-%m-%d_")+saveNameMod + ".pdf"
saveName = save_folder / saveName

fig,axs = create_fig_keynote(1,1)

n0Plot = -2
migPlot = -3
tauPlot = 2
rPlot = 0
kPlot =0


gamma = parRange[gammaIndex]
sigma = parRange[sigmaIndex]

lab = ['$\\sigma_f$= %.2g' %x for x in sigma]

fracGammaSigma = data['F_mav_ss'][:, tauPlot, n0Plot, migPlot, rPlot, kPlot, :].squeeze()
lin = axs.plot(gamma,fracGammaSigma, '-o',linewidth=4, markersize=10)

lineColors = mpl.cm.get_cmap('viridis', len(lin)+1).colors
for (i,c) in zip(lin, lineColors):
    i.set_color(c)

axs.set_xticks([0.01, 0.05, 0.1])
axs.set_yticks([0, 0.2, 0.4])
axs.set_xlabel('cost $\\gamma$')
axs.set_ylabel('fraction of cooperators $f_c$')

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)


plt.ylim((0,0.4))

plt.legend(lin, lab, loc='upper right', fancybox=True, framealpha=0.2)
fig.set_size_inches(w/mydpi,h/mydpi)
plt.tight_layout()
plt.draw()


plt.savefig(saveName, dpi=mydpi, format="pdf", transparent=True)


#%%
#fig, axs = create_fig(numSig+1 , numGamma*numTau+1)
#images = []
#kindex = 0
#rindex = 0
#
#axs[0,numGamma*numTau].set_axis_off() 
#
#for ss in range(numSig):
#    curText = ['$\\sigma_{trans}$=%g' % parRange[sigmaIndex][ss]]
#    addAnnotation(axs[ss+1, numGamma*numTau],curText)
#    
#    for tt in range(numTau): 
#        for gg in range(numGamma):
#            curR = ss + 1
#            curC = tt * numGamma + gg
#            
#            if ss==1:
#                curText = ['$\\tau_{H}$=%g' % parRange[tauIndex][tt], \
#                           '$\\gamma$=%g' % parRange[gammaIndex][gg]]
#                addAnnotation(axs[00, curC],curText)
#                
#            currData = data['F_mav_ss'][gg, tt, :, :, rindex, kindex, ss].squeeze()
#            axl = plot_heatmap(images,axs[curR, curC],currData,n0V,migV)
# 
#
#plt.subplots_adjust(left=0.05, bottom=0.1, right=1, top=1, wspace=0.12, hspace=0.18)
#fig.colorbar(axl, ax=axs, orientation='vertical', shrink=.6, label="log10 fraction cooperator")
#plt.draw()
#                       
#plt.savefig(saveName, dpi=300, format="pdf")

#%%
#curPar = (parRange[gammaIndex][gg], parRange[tauIndex][tt])
            #titlename  = '$\gamma$:%g, $\\tau  _{H}$:%.0d' % curPar
# kindex=0
# images = []
# for tt in range(numTau):
#     for gg in range(numGamma):
#         for rr in range(numR):
#             for ss in range(numSig):
                
#                 currR = rr * numTau + tt
#                 currC = ss * numGamma + gg

#                 currData = np.log10(data['F_mav_ss'][gg, tt, :, :, rr, kindex, ss] \
#                                     .squeeze()).transpose()
                  
#                 images.append(axs[currR, currC].imshow(currData, cmap=cmap, \
#                               interpolation='nearest', \
#                               extent=[n0V[0],n0V[-1],migV[0],migV[-1]], \
#                               origin='lower'))
#                 #axs[currR, currC].set_xticks([n0V[0], n0V[-1]])
#                 #axs[currR, currC].set_yticks([migV[0], migV[-1]])
#                 axs[currR, currC].set_xticks([])
#                 axs[currR, currC].set_yticks([])
                
#                 #axs[tt, gg].set_xlabel("n0")
#                 # axs[tt, gg].set_ylabel("theta")
#                 axs[currR, currC].label_outer()

# norm = colors.Normalize(vmin=-3, vmax=0)
# for im in images:
#     im.set_norm(norm)


# fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, label="fraction cooperator")

# plt.draw()

# #fig.set_dpi = 150

# plt.savefig(saveName, dpi=300, format="pdf")


#%%
