# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:17:56 2019

@author: Mona


Class for Yield Estimation (and in future: Optimization)
"""


from __future__ import print_function
from Est_GPR import Estimation_GPR
import numpy as np
import time
import scipy.stats
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
np.seterr(divide = 'ignore') 


class YieldEstOpt_GPR():
    
    def __init__(self, model, input_freq, threshold, N_mc, input_distr, YE_method, Sorting_Strategy, Batch_Size, Safety_Factor, number_uq_para):
        # instance of numerical model, e.g. Scattering Ana
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
        return samples

  

    def estimate_Yield(self,start_uq, display=1, res='all'):
        Yield, valids = Estimation_GPR(self, start_uq, display)
        if res == 'all':
            return Yield, valids
        elif res == 'Prob':
            return Yield
    





    
    
    