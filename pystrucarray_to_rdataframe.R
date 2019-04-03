source("loadPythonStrucarray.R")

setwd("~/Dropbox/2018 MLS 2 Species Community/PhytonCode/MLS Model")
filename <- "./Data/parScan_fixedVar_timeScalePlot20190402_15h14.npz"

data <- load.python.stuctarray(filename)
p
library(scatterplot3d)
library(plotly)
library(latex2exp)
library(plot3D)

data$log.rel.her <- log10(data$tauHer / data$tau_H)
data$log.rel.var <- log10(data$tauVar / data$tau_H)
zdata <- data$F_mav

scatter3D(xdata, y=ydata, z=zdata, colvar=zdata,
              cex = 0.2, col = 'viridis',
              main="Fraction cooperators",
              xlab = TeX('$\\tau_{her}/\\tau_{R}$'),
              ylab = TeX('$\\tau_{var}/\\tau_{R}$'),
              zlab = "Fraction cooperators")

# p <- plot_ly(data, x = ~log.rel.her, y = ~log.rel.var, z = ~zdata,
#              marker = list(color = ~zdata, colorscale = 'viridis', showscale = TRUE)) %>%
#   add_markers() %>%
#   layout(scene = list(xaxis = list(title = ('$\\tau_{her}/\\tau_{R}$')),
#                       yaxis = list(title = ('$\\tau_{var}/\\tau_{R}$')),
#                       zaxis = list(title = 'Fraction cooperators')))
# 
# p
