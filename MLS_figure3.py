import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import MLS_static_fast as mlssf
import mls_general_code as mlsg
import numpy.lib.recfunctions as rf
from mpl_toolkits.mplot3d import axes3d
from joblib import Parallel, delayed
import datetime
from pathlib import Path
import itertools

# set to True to force recalculation of data
override_data = False

# set folder
data_folder = Path("Data_Paper/")
fig_Folder = Path("Figures_Paper/")
figureName = 'figure3.pdf'

# set figure settings
wFig = 8.7
hFig = 5
font = {'family': 'Helvetica',
        'weight': 'light',
        'size': 6}

axes = {'linewidth': 0.5,
        'titlesize': 7,
        'labelsize': 6,
        'labelpad': 2,
        'spines.top': False,
        'spines.right': False,
        }

ticks = {'major.width': 0.5,
         'direction': 'in',
         'major.size': 2,
         'labelsize': 6,
         'major.pad': 2}

legend = {'fontsize': 6,
          'handlelength': 1.5,
          'handletextpad': 0.5,
          'labelspacing': 0.2}

figure = {'dpi': 300}
savefigure = {'dpi': 300,
              'transparent': True}

mpl.style.use('seaborn-ticks')
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('xtick', **ticks)
mpl.rc('ytick', **ticks)
#mpl.rc('ztick', **ticks)
mpl.rc('legend', **legend)
mpl.rc('figure', **figure)
mpl.rc('savefig', **savefigure)

# set model parameters
tau_H = 1000
tauVRange = (-2, 2)
tauHRange = (-3, 6)
nStep = 40
sigma_vec = [0.02, 0.1]

model_par = {
    # selection strength settings
    "s": 1,
    "K_H": 1000.,
    "B_H": 1.,
    "D_H": 0.,
    # tau_var settings
    "cost": 0.01,
    "TAU_H": tau_H,
    "sigmaBirth": 0.1,
    # tau_mig settings
    "n0": 1E-3,
    "mig": 1E-5,
    # init conditions
    "F0": 0.01,
    "N0init": 1.,
    "NUMGROUP": -1,
    # time settings
    "maxT": 150000,
    "dT": 5E-2,
    "sampleT": 10,
    "rms_err_treshold": 5E-2,
    "mav_window": 1000,
    "rms_window": 10000,
    # fixed model parameters
    "sampling": "fixedvar",
    "mu": 1E-9,
    "K": 1E3,
    "numTypeBins": 100
}

# calc other parameters
desiredTauV = np.logspace(*tauVRange, nStep) * tau_H
desiredTauH = np.logspace(*tauHRange, nStep) * tau_H
B_H_vec = [0, model_par['s']]
migToN0 = mlsg.mig_from_tauH(desiredTauH, model_par['n0'])
cost_vec = 1 / desiredTauV
mig_vec = migToN0 * model_par['n0']


def set_cost_mig_BH(cost, mig, B_H, sigma):
    model_par_local = model_par.copy()
    model_par_local['cost'] = cost
    model_par_local['mig'] = mig
    model_par_local['B_H'] = B_H
    model_par_local['sigmaBirth'] = sigma
    
    if sigma > 0.05:
        model_par_local['rms_err_treshold'] = 5E-3
    else:
        model_par_local['rms_err_treshold'] = 1E-3
    
    return model_par_local


def run_model():
    # set modelpar list to run
    modelParList = [set_cost_mig_BH(*x)
                    for x in itertools.product(*(cost_vec, mig_vec, B_H_vec, sigma_vec))]

    # run model selection
    nJobs = min(len(modelParList), 4)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mlssf.single_run_finalstate)(par) for par in modelParList)

    # process and store output
    Output, InvPerHost = zip(*results)
    statData = np.vstack(Output)
    distData = np.vstack(InvPerHost)

    saveName = data_folder / 'data_Figure3.npz'
    np.savez(saveName, statData=statData, distData=distData,
             modelParList=modelParList, date=datetime.datetime.now())

    return statData


def check_model_par(model_par_load, parToIgnore):
    rerun = False
    for key in model_par_load:
        if not (key in parToIgnore):
            if model_par_load[key] != model_par[key]:
                print('Parameter "%s" has changed, rerunning model!' % 'load')
                rerun = True
    return rerun


def load_model():
    # need not check these parameters
    parToIgnore = ('cost', 'mig', 'B_H', 'sigmaBirth', 'rms_err_treshold')
    loadName = data_folder / 'data_Figure3.npz'
    if loadName.is_file():
        # open file and load data
        data_file = np.load(loadName, allow_pickle=True)
        data1D = data_file['statData']
        rerun = check_model_par(data_file['modelParList'][0], parToIgnore)
        data_file.close()
    else:
        # cannot load, need to rerun model
        rerun = True
        print('Model data not found, running model')
    if rerun or override_data:
        # rerun model
        data1D = run_model()
    return data1D


def process_data(statData):
    # calculate heritability time
    tauHer = mlsg.calc_tauHer_numeric(
        statData['n0'], statData['mig'])
    tauVar = mlsg.calc_tauV(statData['cost'])
    tauHerRel = tauHer/statData['tau_H']
    tauVar_rel = tauVar/statData['tau_H']
    sigma_cat = mlsg.make_categorial(statData['sigmaBirth'])
    BH_cat = mlsg.make_categorial(statData['B_H'])
    dataToStore = (tauHer, tauVar, tauHerRel, tauVar_rel, sigma_cat, BH_cat)
    nameToStore = ('tauHer', 'tauVar', 'tauHer_rel',
                   'tauVar_rel', 'sigma_cat', 'BH_cat')

    statData = rf.append_fields(
        statData, nameToStore, dataToStore, usemask=False)

    return statData


def select_data(data1D, BHidx, sigmaidx):
    # get subset of data to plot
    curSigma = data1D['sigma_cat'] == sigmaidx
    curBH = data1D['BH_cat'] == BHidx
    # remove nan and inf
    isFinite = np.logical_and.reduce(
        np.isfinite((data1D['tauVar_rel'], data1D['tauHer_rel'],
                     data1D['F_mav'], curSigma, curBH)))
    currSubset = np.logical_and.reduce((curSigma, curBH, isFinite))
    # extract data and log transform x,y
    x = np.log10(data1D['tauVar_rel'][currSubset])
    y = np.log10(data1D['tauHer_rel'][currSubset])
    z = data1D['F_mav'][currSubset]
    return (x, y, z)


def plot_3D(ax, data1D, sigmaIndex):
    x, y, z = select_data(data1D, 1, sigmaIndex)
    ax.scatter(x, y, z,
               c=z,
               s=0.5, alpha=0.7,
               vmin=0, vmax=1, cmap='plasma')

    steps = (3, 4, 3)
    fRange = (0, 1)

    ax.set_xlim(tauVRange)
    ax.set_ylim(tauHRange)
    ax.set_zlim(fRange)
    ax.set_xticks(np.linspace(*tauVRange, steps[0]))
    ax.set_yticks(np.linspace(*tauHRange, steps[1]))
    ax.set_zticks(np.linspace(*fRange, steps[2]))

    # set labels
    ax.set_xlabel('$log_{10} \\tau_{Var} / \\tau_H$')
    ax.set_ylabel('$log_{10} \\tau_{Her} / \\tau_H$')
    ax.set_zlabel('$\\langle f \\rangle$')

    ax.yaxis.labelpad = -10
    ax.xaxis.labelpad = -10
    ax.zaxis.labelpad = -10
    ax.tick_params(axis='z', which='major', pad=0)
    ax.tick_params(axis='both', which='major', pad=-5)
    
    ax.view_init(30, -115)

    return None


def bin_2Ddata(currXData, currYData, currZData, xbins, ybins):
    """[Bins x,y data into 2d bins]
    Arguments:
            currXData {np vector} -- xData to bin
            currYData {np vector} -- yData to bin
            currZData {np vector} -- zData to bin
            xbins {np vector} -- xBins to use
            ybins {np vector} -- yBins to use
    """
    # init output
    nX = xbins.size
    nY = ybins.size
    binnedData = np.full((nY, nX), np.nan)
    # loop over bins and calc mean
    for xx in range(nX - 1):
        for yy in range(nY - 1):
            # find data in bin
            inXBin = np.logical_and(
                (currXData >= xbins[xx]), (currXData < xbins[xx+1]))
            inYBin = np.logical_and(
                (currYData >= ybins[yy]), (currYData < ybins[yy+1]))
            inBin = np.logical_and(inXBin, inYBin)
            zInBin = currZData[inBin]
            # calc mean over bine
            binnedData[yy, xx] = np.nanmean(zInBin)
    return(binnedData)


def plot_heatmap(fig, ax, data1D, sigmaIndex):
    xStep = 0.5
    yStep = 1
    xbins = np.linspace(*tauVRange, int(
        np.ceil((tauVRange[1]-tauVRange[0])/xStep))+1)
    ybins = np.linspace(*tauHRange, int(
        np.ceil((tauHRange[1] - tauHRange[0]) / yStep)) + 1)

    # get data with selection
    xS, yS, zS = select_data(data1D, 1, sigmaIndex)
    binnedDataS = bin_2Ddata(xS, yS, zS, xbins, ybins)
    # get data without selection
    xNS, yNS, zNS = select_data(data1D, 0, sigmaIndex)
    binnedDataNS = bin_2Ddata(xNS, yNS, zNS, xbins, ybins)

    relData = binnedDataS / binnedDataNS

    im = ax.pcolormesh(xbins, ybins, relData, cmap='plasma')
    
    fig.colorbar(im, ax=ax)

    steps = (3, 4)

    ax.set_xlim(tauVRange)
    ax.set_ylim(tauHRange)
    ax.set_xticks(np.linspace(*tauVRange, steps[0]))
    ax.set_yticks(np.linspace(*tauHRange, steps[1]))

    # set labels
    ax.set_xlabel('$log_{10} \\tau_{Var} / \\tau_H$')
    ax.set_ylabel('$log_{10} \\tau_{Her} / \\tau_H$')

    return None


def create_fig():
    # load data or compute model
    data1D = load_model()
    data1D = process_data(data1D)

    # set fonts
    fig = plt.figure()
    mlsg.set_fig_size_cm(fig, wFig, hFig)

    for ss in range(2):
        # plot average investment
        ax = fig.add_subplot(2, 2, ss+1, projection='3d')
        plot_3D(ax, data1D, ss)

        ax = fig.add_subplot(2, 2, ss+3)
        plot_heatmap(fig, ax, data1D, ss)

    plt.tight_layout(pad=1, h_pad=2.5, w_pad=1.5)
    fig.savefig(fig_Folder / figureName,
                format="pdf", transparent=True)

    return None


if __name__ == "__main__":
    create_fig()
