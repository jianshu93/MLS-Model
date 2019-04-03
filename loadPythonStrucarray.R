load.python.stuctarray <- function(filename) {
  #function loads python sructured array and converts to r dataframe
  #load python functions
  library(reticulate)
  load.data <- import("MLS_plot_general_code", convert = FALSE)
  np <- import("numpy", convert = FALSE)
  
  #load data from python datafile
  dataset.py <- load.data$load_process_data(filename)
  data1D.py <- dataset.py['endStat_1D']
  
  #get properties of data
  fieldnames <- py_to_r(data1D.py$dtype$names)
  npoint = length(data1D.py[fieldnames[1]])
  nvar = length(fieldnames)
  
  #convert to 2D matrix in r
  data1D.r <- py_to_r(data1D.py$view(np$float64))
  data2D.r <- matrix(data = data1D.r, nrow = npoint, ncol = nvar, byrow = TRUE)
  
  #add column names
  colnames(data2D.r) <- fieldnames
  
  #convert to data frame
  data.frame.r <- data.frame(data2D.r)
  
  return(data.frame.r)
}
