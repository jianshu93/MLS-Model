import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import MLS_static_fast as mlssf
import time
import seaborn as sns


def plot_heatmap_distri(fig, axs, data, model_par):
    currData = np.log10(data.transpose() + np.finfo(float).eps)
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
    axs.set_xticklabels([0, model_par['maxT']])
    return None


def plot_histogram(fig, axs, data, model_par, c):
    data = data[-model_par['mav_window']:, :]
    data = np.nanmean(data, axis=0)
    bins = np.linspace(0, 1, data.size+1)
    x = (bins[1:] + bins[0:-1])/2
    axs.hist(x, bins, weights=data, color=c)
    axs.set_xlim((0, 1))
    axs.set_ylabel('freq')
    axs.set_xlabel('investment')
    return None


def plot_histogram_line(fig, axs, data, model_par):
    data = data[-model_par['mav_window']:, :]
    data = np.nanmean(data, axis=0)
    bins = np.linspace(0, 1, data.size+1)
    x = (bins[1:] + bins[0:-1])/2
    axs.plot(x, data)
    axs.set_xlim((0, 1))
    axs.set_ylabel('freq')
    axs.set_xlabel('investment')
    return None


def compare_birth_death_ns(model_par, selectionCoef):
    start = time.time()

    # run birth effect
    model_par['B_H'] = selectionCoef
    model_par['D_H'] = 0.
    OutputB, InvestmentPerHostB, _, _ = mlssf.run_model_fixed_parameters(
        model_par)

    # run death effect
    sd = selectionCoef / (1 + selectionCoef)
    model_par['B_H'] = 0
    model_par['D_H'] = sd
    OutputD, InvestmentPerHostD, _, _ = mlssf.run_model_fixed_parameters(
        model_par)

    # run birth+death
    sbd = selectionCoef / (2 + selectionCoef)
    model_par['B_H'] = sbd
    model_par['D_H'] = sbd
    OutputBD, InvestmentPerHostBD, _, _ = mlssf.run_model_fixed_parameters(
        model_par)

    # run no selection
    model_par['B_H'] = 0
    model_par['D_H'] = 0
    OutputN, InvestmentPerHostN, _, _ = mlssf.run_model_fixed_parameters(
        model_par)

    end = time.time()
    print("Elapsed time run 1 = %s" % (end - start))

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 7}

    matplotlib.rc('font', **font)

    fig = plt.figure()
    nR = 2
    nC = 2

    legendName = (('$s_b=%.1f,s_d=%.1f$' % (selectionCoef, 0)),
                  ('$s_b=%.1f,s_d=%.1f$' % (0, sd)),
                  ('$s_b=%.1f,s_d=%.1f$' % (sbd, sbd)),
                  ('$s_b=%.1f,s_d=%.1f$' % (0, 0)))

    # plot average investment
    plt.subplot(nR, nC, 1)
    mlssf.plot_data(OutputB, "F_mav")
    mlssf.plot_data(OutputD, "F_mav")
    mlssf.plot_data(OutputBD, "F_mav")
    mlssf.plot_data(OutputN, "F_mav")
    plt.ylabel("investment")
    plt.ylim((0, 1))
    plt.xlim((0, model_par['maxT']))
    plt.legend(legendName)

    # plot average host num
    plt.subplot(nR, nC, 2)
    mlssf.plot_data(OutputB, "H_mav")
    mlssf.plot_data(OutputD, "H_mav")
    mlssf.plot_data(OutputBD, "H_mav")
    mlssf.plot_data(OutputN, "H_mav")

    plt.ylabel("host ne")

    if np.any(~np.isnan(OutputB['H_mav'])) or np.any(~np.isnan(OutputD['H_mav'])):
        maxY = np.ceil(
            max(np.nanmax(OutputB['H_mav']), np.nanmax(OutputD['H_mav']))/100)*100
        plt.ylim((0, maxY))
    plt.xlim((0, model_par['maxT']))
    plt.legend(legendName)

    # plot average investment per host
    # axs = plt.subplot(nR, nC, 3)
    # plot_heatmap_distri(fig, axs, InvestmentPerHostB, model_par)
    # axs.set_title('birth')
    # axs = plt.subplot(nR, nC, 4)
    # plot_heatmap_distri(fig, axs, InvestmentPerHostD, model_par)
    # axs.set_title('death')

    c1 = (0.122, 0.467, 0.706, 0.5)
    c2 = (1, 0.498, 0.05, 0.5)
    c3 = (0.173, 0.627, 0.173, 0.5)
    c4 = (0.839, 0.153, 0.157, 0.5)

    axs = plt.subplot(nR, nC, 3)
    plot_histogram(fig, axs, InvestmentPerHostB, model_par, c1)
    plot_histogram(fig, axs, InvestmentPerHostD, model_par, c2)
    plot_histogram(fig, axs, InvestmentPerHostBD, model_par, c3)
    plot_histogram(fig, axs, InvestmentPerHostN, model_par, c4)

    axs = plt.subplot(nR, nC, 4)
    plot_histogram_line(fig, axs, InvestmentPerHostB, model_par)
    plot_histogram_line(fig, axs, InvestmentPerHostD, model_par)
    plot_histogram_line(fig, axs, InvestmentPerHostBD, model_par)
    plot_histogram_line(fig, axs, InvestmentPerHostN, model_par)

    plt.legend(legendName)

    fig.set_size_inches(4, 4)
    plt.tight_layout()

    return None


if __name__ == "__main__":
    print("running debug")
    model_par = {
        # time step parameters
        "dT": 5E-2,
        "maxT": 200.,
        "sampleT": 1,
        "rms_err_treshold": 1E-5,
        "mav_window": 100,
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
        "NUMGROUP": -1,
        "numTypeBins": 50,
        "F0": 0.5,
        "N0init": 1.
    }
    compare_birth_death(model_par, 1)
