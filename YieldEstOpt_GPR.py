# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:17:56 2019

@author: Mona Fuhrländer (Technische Universität Darmstadt, mona.fuhrlaender@tu-darmstadt.de)


Class for Yield Estimation (and in future: Optimization)
"""


from __future__ import print_function
from Est_GPR import Estimation_GPR
from Opt_adaptNewtonMC import Opt_Newton
import numpy as np
from numpy import genfromtxt
import scipy.stats
import copy
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
np.seterr(divide = 'ignore') 


class YieldEstOpt_GPR():
    
    def __init__(self, problem, model, input_freq, threshold, N_mc, input_distr, YE_method, Sorting_Strategy, Batch_Size, Safety_Factor, number_uq_para):
        # instance of numerical model, e.g. Scattering Ana
        self.problem = problem
        self.model = model
        self.freqrange = np.arange(input_freq[0],input_freq[1]+0.01,input_freq[2])
        self.threshold=threshold
        self.Nmc = N_mc
        self.input_distr = input_distr
        self.YE_method = YE_method
        self.Sorting_Strategy = Sorting_Strategy
        self.Batch_Size = Batch_Size
        self.number_uq_para = number_uq_para
        self.Safety_Factor = Safety_Factor
        self.kernel = C(1e-1, (1e-5, 1e-1)) * RBF(1.,(1e-5, 1e5))

    
    def sample_generator(self, mean, distr, sample_size, normal = 'true'):
        # Generate sample set for the waveguide problem acc. to specific distribution
        if self.problem == 'Waveguide':
            samples = []
            if normal: # truncated normal according to pdf (for MC sample points and initial training data)
                for k in range(len(distr)):
                    mu = mean[k]; sigma = distr[k][1]; lb = mu+distr[k][2]; ub = mu+distr[k][3];
                    if lb < 0: lb = 0
                    samples_k = scipy.stats.truncnorm.rvs((lb - mu)/sigma, (ub-mu)/sigma, loc = mu, scale = sigma, size = sample_size)
                    samples.append(samples_k)
            else: # uniform within bounds (alternative option for initial training data)
                for k in range(len(distr)):
                    mu = mean[k]; lb = mu+distr[k][2]; ub = mu+distr[k][3];
                    samples_k = scipy.stats.uniform.rvs(lb,ub-lb,sample_size)
                    samples.append(samples_k)        
            samples = np.array(samples)
            self.sample_MOO = copy.deepcopy(samples)
        
        # Call sample set for the lowpass filter problem from prepared list
        elif self.problem == 'Lowpass':
            samples = genfromtxt('Lowpass_data_sample.csv', delimiter=',',skip_header=True)
            samples = samples[0:sample_size,:].T
            
            # load QoI results from list
            QoI_all = genfromtxt('Lowpass_data_QoI.csv', delimiter=',',skip_header=True)
            self.QoI = QoI_all[0:sample_size*len(self.freqrange)]
        
        return samples

  

    def estimate_Yield(self,start_uq, Nmc_est,display=1, res='all'):
        Yield, valids = Estimation_GPR(self, start_uq, Nmc_est, display)
        if res == 'all':
            return Yield, valids
        elif res == 'Prob':
            return Yield
    
    
    
    def optimize_Yield(self, start_uq,adaptive=True):
        Yield_start, Yield_opt, start_uq, p_opt, opttime_end, self.ItVec_short = Opt_Newton(self,start_uq,adaptive)
        return Yield_start, Yield_opt, start_uq, p_opt, opttime_end, self.ItVec_short
    





    
    
    