# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 08:54:39 2020

@author: Mona Fuhrländer (Technische Universität Darmstadt, mona.fuhrlaender@tu-darmstadt.de)


Script for Evaluating Yield Estimation for Lowpass Filter with 6 uncertain parameters
   using GPR-Hybrid Estimation or Monte Carlo
"""


from YieldEstOpt_GPR import YieldEstOpt_GPR

import numpy as np
from Eval_Lowpass_List import eval_lowpass
from Est_GPR import build_sur_gpr




# UNCERTAIN DESIGN PARAMETERS
Problem = 'Lowpass'
number_uq_para = 6      # number of uncertain parameters (1,2,4)

#truncated normal distributed with [mean, std, trunc. range neg., trunc. range pos.]
# geometry parameters
L1 = [6.8,0.3,-3,3] 
L2 = [5.1,0.3,-3,3] 
L3 = [9.0,0.3,-3,3] 
W1 = [1.4,0.1,-0.3,0.3] 
W2 = [1.4,0.1,-0.3,0.3] 
W3 = [1.3,0.1,-0.3,0.3] 

distr_uq = [L1,L2,L3,W1,W2,W3] #uq parameters complete
distr_nominal = np.array(distr_uq).T[0]  #uq parameters - only nominal mean values


# PERFORMANCE FEATURE SPECIFICATIONS
FreqRangePara = [ 0.,  7.,  1.] # range parameter interval T_w with step size 1.0
# pfs1: lower bound -1 in freq. range [0,4]; pfs2: upper bound in freq. range [5,7]
PF_Threshold = [[-1,0,4,'lb'],[-20,5,7,'ub']] # threshold for QoI (in dB)


# YIELD ESTIMATION SETTINGS
#Sample Sizes
N_mc = 2500 # MC sample size
fcalls = 30 # Initial Training Data size (per freq. point)

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
Safety_Factor = 3.


# MODEL INITIALIZATION
np.random.seed(23)
YEO = YieldEstOpt_GPR(Problem, eval_lowpass, FreqRangePara, PF_Threshold, N_mc, distr_uq, YE_method, Sorting_Strategy, Batch_Size, Safety_Factor, number_uq_para)


# BUILD SURROGATE
if YE_method != 'MC':
    build_sur_gpr(YEO, fcalls, distr = 'gaussian')
    print("---surrogate model built--- \n")

    
# RESULTS FOR YIELD ESTIMATION
np.random.seed(23)
print("Estimation: Yield =", YEO.estimate_Yield(distr_nominal, res = 'Prob'))
    

    
