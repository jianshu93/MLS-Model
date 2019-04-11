import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


cmapList = ('viridis', 'plasma', 'inferno', 'magma', 'cividis')
x = np.linspace(0.0, 1.0, 256)


for cmapName in cmapList:
    print(cmapName)
    rgb = np.squeeze(cm.get_cmap(plt.get_cmap(cmapName))(x)[np.newaxis, :, :3])
    fileName = cmapName + '.csv'
    np.savetxt(fileName, rgb, delimiter=',')
