# Readme 

## Author: simonvanvliet
vanvliet@zoology.ubc.ca


## The following Code is supplied:

### MODEL CODE 
#### MLS_static_fast.py
Implementation of two species microbiome Multilevel selection model

#### MLS_evolveCoop_fast.py
Implementation of single species micobiome Multilevel selection model where investment is a continuous trait

#### mls_general_code.py
Defines several helper functions used in rest of code


### Utility CODE 
#### singleRunMLS.py
Code provides explanantion of model parameters, allows user to quickly set parameters, and run model


### Figure CODE
#### MLS_figure_X.py
Run to recreate figures describes in Manuscript
Each code tries to load existing data file from Data_Paper folder and uses this to make figures
If data file cannot be found, or if parameters have changed, the model is rerun 
(warning: can take several hours to days for some of the figures!)

