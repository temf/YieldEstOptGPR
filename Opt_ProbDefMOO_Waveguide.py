# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:53:51 2020

@author: Mona Fuhrländer (Technische Universität Darmstadt, mona.fuhrlaender@tu-darmstadt.de)

Problem definition for genetic Multi-objective for simple dielectrical waveguide with 4 uncertain parameters.
Objective functions:
- maximization of the yield
- robust minimization of the width of the waveguide
"""


import numpy as np
from pymoo.model.problem import Problem
from statistics import mean, stdev
import copy

from Eval_Waveguide import S_ana
from YieldEstOpt_GPR import YieldEstOpt_GPR
from Est_GPR import build_sur_gpr


# UNCERTAIN DESIGN PARAMETERS
Model_Problem = 'Waveguide'
number_uq_para = 4      # number of uncertain parameters (1,2,4)

#truncated normal distributed with [mean, std, trunc. range neg., trunc. range pos.]
distr_fill_l = [9.,0.7,-3,3] #length of the inlay
distr_offset = [5.,0.7,-3,3] #length of the offset
distr_eps8 = [1.,0.3,-0.3,0.3] #impact factor for permittivity of the inlay
distr_mue8 = [1.,0.3,-0.3,0.3] #impact factor for permeability of the inlay

distr_uq = [distr_fill_l, distr_offset, distr_eps8, distr_mue8] #uq parameters complete
distr_nominal = np.array(distr_uq).T[0] #uq parameters - only nominal mean values


# PERFORMANCE FEATURE SPECIFICATIONS
FreqRangePara = [ 6.5,  7.5,  0.1] # range parameter interval T_w with step size 0.1
# pfs1: upper bound -24 in freq. range [6.5,7.5]; 
PF_Threshold = [[-24,6.5,7.5,'ub']] # threshold for QoI (in dB)


# YIELD ESTIMATION SETTINGS
#Sample Sizes
N_mc = 2500 # MC sample size (only relevant for nonadaptive optimization)
fcalls = 10 # Initial Training Data size (per freq. point)

# Options for Yield Estimation Method:
# MC = pure Monte Carlo, GPR = pure GPR, Hybrid = GPR-Hybrid
YE_method = 'MC' #'Hybrid' #'GPR' #'MC' 

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
YEO = YieldEstOpt_GPR(Model_Problem, S_ana, FreqRangePara, PF_Threshold, N_mc, distr_uq, YE_method, Sorting_Strategy, Batch_Size, Safety_Factor, number_uq_para)


# BUILD SURROGATE
if YE_method != 'MC':
    build_sur_gpr(YEO, fcalls, distr = 'gaussian')
    print("---surrogate model built--- \n")


# DEFINITION OF ROBUST WIDTH FUNCTION
def width(distr):
    # width of the waveguide = distr_fill_l + 2*distr_offset
    # mean and standard deviation of the width
    mean_width = mean(YEO.sample_MOO.T[0]+2*YEO.sample_MOO.T[1])
    std_width = stdev(YEO.sample_MOO.T[0]+2*YEO.sample_MOO.T[1])
    
    # robust formulation of the width (for min) = E[width] + Std[width]
    robust_width = mean_width + std_width
    
    return robust_width
        



# =============================================================================
#   #Definition of the problem as class
#       #number of optimization varaibles: n_var
#       #objective function(s) as minimization: f1,f2
#           - and their number: n_obj
#       #constraint(s) as <0 condition: g1,g2 
#           - and their number: n_constr
#       #lower/upper bounds for optimization variable: xl,xu
# =============================================================================

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([5,3,0.5,0.5]),
                         xu=np.array([25,15,1.5,1.5]))


    def _evaluate(self, x, out, *args, **kwargs):
        
        # INITIALIZATION
        # Initialization of objective and constraint functions
        #    for loop over all individuals
        f1 = []
        f2 = []
        g1 = []
        
        # Loop over all individuals
        for i in range(len(x)):
            # SET UQ PARAMETERS
            distr_fill_l  = copy.deepcopy(YEO.input_distr[0])
            distr_fill_l[0] = x[i,0]
            distr_offset  = copy.deepcopy(YEO.input_distr[1])
            distr_offset[0] = x[i,1]
            distr_eps8  = copy.deepcopy(YEO.input_distr[2])
            distr_eps8[0] = x[i,2]
            distr_mue8  = copy.deepcopy(YEO.input_distr[3])
            distr_mue8[0] = x[i,3]
            
            distr_uq = [distr_fill_l, distr_offset, distr_eps8, distr_mue8]
            distr_nominal = np.array(distr_uq).T[0]
            
            print('\n\n',i+1,'-th individual in this generation')
            
            ### 2nd OBJECTIVE FUNCTION ###
            # f2: max(Yield) = min(-Yield)
            np.random.seed(23)
            Yield, valids = YEO.estimate_Yield(distr_nominal, YEO.Nmc)
            f2i = -Yield
            
            ### 1st OBJECTIVE FUNCTION ###
            # f1: min(robust_width)
            f1i = width(distr_uq)

            ### CONSTRAINT ###
            # g1: Yield >= Yield_min = 0.8
            Yield_min = 0.8
            g1i = Yield_min - Yield
            
            # COLLECT DATA
            f1.append(f1i)
            f2.append(f2i)
            g1.append(g1i)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1])