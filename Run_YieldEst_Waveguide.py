# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 08:54:39 2020

@author: Mona Fuhrländer (Technische Universität Darmstadt, mona.fuhrlaender@tu-darmstadt.de)


Script for Evaluating Yield Estimation for simple dielectrical waveguide with 4 uncertain parameters
   using GPR-Hybrid Estimation or Monte Carlo
"""


from YieldEstOpt_GPR import YieldEstOpt_GPR

import numpy as np
from Eval_Waveguide import S_ana
from Est_GPR import build_sur_gpr




# UNCERTAIN DESIGN PARAMETERS
Problem = 'Waveguide'
number_uq_para = 4      # number of uncertain parameters (1,2,4)

#truncated normal distributed with [mean, std, trunc. range neg., trunc. range pos.]
distr_fill_l = [10.36,0.7,-3,3] #length of the inlay
distr_offset = [4.76,0.7,-3,3] #length of the offset
distr_eps8 = [0.58,0.3,-0.3,0.3] #impact factor for permittivity of the inlay
distr_mue8 = [0.64,0.3,-0.3,0.3] #impact factor for permeability of the inlay

distr_uq = [distr_fill_l, distr_offset, distr_eps8, distr_mue8] #uq parameters complete
distr_nominal = np.array(distr_uq).T[0] #uq parameters - only nominal mean values


# PERFORMANCE FEATURE SPECIFICATIONS
FreqRangePara = [ 6.5,  7.5,  0.1] # range parameter interval T_w with step size 0.1
# pfs1: upper bound -24 in freq. range [6.5,7.5]; 
PF_Threshold = [[-24,6.5,7.5,'ub']] # threshold for QoI (in dB)


# YIELD ESTIMATION SETTINGS
#Sample Sizes
N_mc = 2500 # MC sample size
fcalls = 10 # Initial Training Data size (per freq. point)

# Options for Yield Estimation Method:
# MC = pure Monte Carlo, GPR = pure GPR, Hybrid = GPR-Hybrid
YE_method = 'Hybrid' #'Hybrid' #'GPR' #'MC' 

# Options for Sorting Strategy for MC sample  -  ONLY IF Hybrid is chosen as YE_method
# none = no sorting strategy, EGL = sorting acc. EGL criterion, FS = sorting acc. Hybrid (=FS) criterion
Sorting_Strategy = 'none' #'none' 'EGL' 'FS' # only if Hybrid is chosen as YE_Method

# Batch Size for GPR model update  -  ONLY IF Hybrid is chosen as YE_method
# 1 = no batches / update with each critical sample point, ..>N_mc = no updates at all
Batch_Size = 1 #1 #20 #50 #N_samples+1

# Safety factor for trsuted upper and lower bounds
Safety_Factor = 2.


# MODEL INITIALIZATION
np.random.seed(23)
YEO = YieldEstOpt_GPR(Problem, S_ana, FreqRangePara, PF_Threshold, N_mc, distr_uq, YE_method, Sorting_Strategy, Batch_Size, Safety_Factor, number_uq_para)


# BUILD SURROGATE
if YE_method != 'MC':
    build_sur_gpr(YEO, fcalls, distr = 'gaussian')
    print("---surrogate model built--- \n")

    
# RESULTS FOR YIELD ESTIMATION
np.random.seed(23)
print("Estimation: Yield =", YEO.estimate_Yield(distr_nominal, YEO.Nmc, res = 'Prob'))


























    

    