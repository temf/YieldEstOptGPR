# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:46:08 2019

@author: Mona

Functions for Yield Estimation using Gaussian Process Regression (GPR) and/or Monte Carlo (MC)
"""


from __future__ import print_function
from UQSetting import prove_pfs, paraUQ
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import time

      
# Build GPR Surrogate Models with given Size of Training Data Set and distribution of the Training Data Points (default = gaussian)
def build_sur_gpr(self, TrainSize, distr = 'gaussian'):
    mean = np.array(self.input_distr).T[0] # mean value of nominal parameter
    self.HFevals_forGPRmodel = TrainSize*len(self.freqrange) # total number of high fidelity evaluations for building the initial GPR models
    
    # Generate initial training data set
    if distr == 'uniform': # uniform
        training_data_initial = self.sample_generator(mean, self.input_distr, TrainSize, normal = 'false')
    elif distr == 'gaussian': # gaussian
        training_data_initial = self.sample_generator(mean, self.input_distr, TrainSize)
    else:
        print('ERROR: Please insert valid distribution for initial training data.')
    nodes = np.array(training_data_initial).T # initial training data points
    
    # Initialize surrogate models (for real and imag part separately)
    self.surrogates=[]
    self.y_imag=[[0] * TrainSize for i in range(len(self.freqrange))]
    self.y_real=[[0] * TrainSize for i in range(len(self.freqrange))]
    self.gp_imag = []; self.gp_real = []
    #...and for each frequency point separately
    for k in range(len(self.freqrange)):
        self.surrogates.append(nodes)
        self.gp_imag.append(GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5,normalize_y=True, n_restarts_optimizer=10)) 
        self.gp_real.append(GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5,normalize_y=True, n_restarts_optimizer=10)) 
    
    # Build surrogate models
    for k in range(len(self.freqrange)):
        for n in range(TrainSize):
            s_val_compl = self.model(paraUQ(self.freqrange[k],self.surrogates[k][n],self.number_uq_para)) #,False)
            s_val_imag = np.imag(s_val_compl)
            s_val_real = np.real(s_val_compl)
            self.y_imag[k][n] = (s_val_imag.ravel())
            self.y_real[k][n] = (s_val_real.ravel())
        self.gp_imag[k].fit(self.surrogates[k], self.y_imag[k])
        self.gp_real[k].fit(self.surrogates[k], self.y_real[k]) 
    
    print('Total number of HF evaluations for GPR models:',  self.HFevals_forGPRmodel )   




# Evaluate GPR Models for one MC sample point (test data point) and obtain prediction (and std=sigma) for all frequency points
def interpolate(self, test_data_point, ret_all=True):
    test_data_point=np.reshape(test_data_point,(len(test_data_point),1)).T
    sigma_range = [] # standard deviation of the GPR model in the considered MC point
    s_range=[] # prediction of the quantity of interest (here: S-parameter) in the MC point
    for k in range(len(self.freqrange)):
        s_pred_imag, sigma_imag = self.gp_imag[k].predict(test_data_point, return_std=True)
        s_pred_real, sigma_real = self.gp_real[k].predict(test_data_point, return_std=True)
        s_pred = np.atleast_2d(s_pred_real + s_pred_imag*1j)
        sigma = sigma_real + sigma_imag*1j

        sigma_range.append(sigma);
        s_range.append(s_pred);
        
    if ret_all:
        return sigma_range, s_range
    else:
        return s_range
       

# Update GPR Models with critical sample points
def update_GPR(self,HF_freq,HF_sample,HF_val,GPR_val,display = 0): 
    # If not all critical sample points should be added, setting Diff_tol>0 is necessary.
    Diff_tol = 0 
    
    # for each freq. point the according surrogate model might be updated
    for k in range(len(self.freqrange)):
        freqv = self.freqrange[k]
        idx_list_freqv = [i for i, value in enumerate(HF_freq) if value == freqv]
        # if there is no critical sample point for a certain frequency point --> no update
        if len(idx_list_freqv) < 1:
            continue
        # if there is a least one critical sample point for a certain frequency point --> update
        else:
            # collect data from the lists corresponding to the k-th freq. point.
            sample_list_freqv = np.array(HF_sample)[idx_list_freqv]
            val_list_freqv = np.array(HF_val)[idx_list_freqv]
            GPR_val_list_freqv = np.array(GPR_val)[idx_list_freqv]
            # calculate difference between GPR prediction and HF value
            Diff = abs(GPR_val_list_freqv - val_list_freqv)
            # one sample point (with largest error) is always added, therefor we set:
            Diff_val = Diff_tol+1
            # count, how many sample points are added to the training data set
            addsamp = 0
            # while the not small enough and there are still not added sample points --> update
            while Diff_val > Diff_tol and len(sample_list_freqv)>addsamp:#0:
                addsamp = addsamp+1
                # choose sample point with largest error (Diff) as new training data point for specific freq. point
                idx_new = np.argmax(Diff)
                # prepare data to be added to GPR model
                new_sample = [sample_list_freqv[idx_new]]
                new_val_imag = np.imag(val_list_freqv[idx_new])
                new_val_real = np.real(val_list_freqv[idx_new])
                
                # add data of new training data point to GPR model
                self.surrogates[k]=np.append(self.surrogates[k],new_sample, axis=0)
                self.y_imag[k]=np.append(self.y_imag[k],new_val_imag)
                self.y_real[k]=np.append(self.y_real[k],new_val_real)
                self.gp_imag[k].fit(self.surrogates[k],self.y_imag[k])
                self.gp_real[k].fit(self.surrogates[k],self.y_real[k])
                
                # Re-evaluate all the other sample points in the batch on the updated GPR Model (for specific freq. point)
                GPR_val_list_freqv = []
                for samplei in range(len(sample_list_freqv)):
                    GPR_range = interpolate(self,sample_list_freqv[samplei],False)
                    GPR_val_list_freqv.append(GPR_range[k][0][0])
                
                # ...and calculate their difference between updated GPR prediction and HF value
                Diff = abs(GPR_val_list_freqv - val_list_freqv)
                # the sample point with the largest Diff_val value might be added in the next loop (if < Diff_tol)
                Diff_val = np.max(Diff)
            
            if display == 1:
                print('GPR model for Freq ', freqv, 'GHz updated with', addsamp, 'new training data points (', len(idx_list_freqv), 'critical samples).')
            


# Estimate Yield with pure GPR, pure MC or Hybrid approach        
def Estimation_GPR(self, start_uq, display=1):#, onlyEst = 0, EstUpdate = 1): # unnötige inputs????????????!!!!!!!!!!!
    print('p:', start_uq.T,'\n')
    
    # Generate MC sample
    samples_uq = self.sample_generator(start_uq, self.input_distr, self.Nmc).T
    
    # Options for sorting the sample points according to EGL or FS (=Hybrid) criterion.
    if self.YE_method == 'Hybrid':
        if self.Sorting_Strategy == 'EGL':
            samples_uq = sort_samples_EGL(self,samples_uq, display = 0)#, n_freq_point=0)
            print('MC sample has been evaluated on GPR model and sorted according to EGL criterion.\n')
        elif self.Sorting_Strategy == 'FS':
            samples_uq = sort_samples_FS(self,samples_uq, display = 0)
            print('MC sample has been evaluated on GPR model and sorted according to FS (=Hybrid) criterion.\n')
        elif self.Sorting_Strategy == 'none':
            print('MC sample has not been sorted.\n')
        else:
            print('Set a valid sorting strategy! \n')
    

    # Initialize counting variables
    counter_valid=0   # number of valid/accepted sample points
    valids = []   # list of valid/accepted sample points
    self.hf_evals = 0 # number of high fidelity evaluations for evaluating MC sample points
    i=0   # counter for considered MC sample points
    
    # Initialize data for GPR Model update in Hybrid case
    self.approx_s_update = [] #GPR S-Parameter value for sample points which have been reevaluated on high fidelity (HF) model (critical samples)
    self.val_update = [] # reevaluated HF solution value of S-Parameter
    self.sample_update = [] # sample points which are reevaluated on HF model(critical samples)
    self.freq_update = [] # frequency values for reeveluated sample points (critical samples)
    
    # For each MC sample point check if it is valid (performance feature specifications (pfs) are fulfilled)
    while i<len(samples_uq):
        # pure GPR approach
        if self.YE_method == 'GPR':
            # Evaluate sample point on GPR model for all frequency points...
            s_range = interpolate(self,samples_uq[i], False)
            s_dB_range = 20*np.log10(np.abs(s_range))
            # ...and choose the one with highest dB value
            s_max = max(s_dB_range)
            # proof if maximal S-Parameter value fulfills performance feature specification
                # (since pfs is upper bound, no further tests are necessary to classify sample point)
            if prove_pfs(s_max,self.threshold,'dB') == 1:
                counter_valid+=1
                valids.append(samples_uq[i])

        # pure MC approach
        elif self.YE_method == 'MC':
            sample_accepted = 1
            # to save computing time, consider one frequency point after the other
            for freqv in self.freqrange:
                # Evaluate sample point on HF model
                s= self.model(paraUQ(freqv,samples_uq[i], self.number_uq_para))
                self.hf_evals = self.hf_evals+1
                # if sample point fulfills pfs for this freq. point, continue with next freq. point
                if prove_pfs(s,self.threshold,'compl') == 1:
                    continue
                # if sample point fulfills pfs NOT for this freq. point, sample point is NOT valid
                    # other freq. points are not needed to be evaluated for this sample point
                else:
                    sample_accepted = 0
                    break
            # if sample point fulfilled pfs for all freq. points --> valid
            if sample_accepted == 1:
                counter_valid+=1
                valids.append(samples_uq[i]) 

        # Hybrid approach combining GPR and high fidelity MC
        elif self.YE_method == 'Hybrid':
            sample_accepted = 1
            sample_i = samples_uq[i]
            
            # Evaluate sample point on GPR model for all frequency points
            sigma_range, s_range = interpolate(self,sample_i,True)
            s_dB_range = 20*np.log10(np.abs(s_range))

            # Sort frequency points according to S-Parameter dB value (to start with largest value,
                # which has the highest risk to fail pfs)
            sort_indices = np.argsort(np.ravel(s_dB_range))[::-1]
            s_sort = np.array(s_range)[sort_indices]
            sigma_sort = np.array(sigma_range)[sort_indices]
            freq_sort = np.array(self.freqrange)[sort_indices]
            
            # For each frequency point (starting with the one with highest failing risk)
            for j in range(len(self.freqrange)):
                s = s_sort[j]
                freq = freq_sort[j]
                
                # Define error and confidence interval using sigma from GPR
                err = sigma_sort[j]
                trusted_upper_bound = abs(s) + self.Safety_Factor*abs(err)
                if self.Safety_Factor*abs(err)<abs(s):
                    trusted_lower_bound = abs(s) - self.Safety_Factor*abs(err)
                else: 
                    trusted_lower_bound=0.0   
                    
                # if clearly not accepted --> sample point not valid
                if prove_pfs(trusted_lower_bound, self.threshold,'compl') == 0:
                    sample_accepted = 0
                    break
                # if clearly accepted --> continue with next freq. point
                elif prove_pfs(trusted_upper_bound, self.threshold,'compl') == 1:
                    continue
                # else: classify as critical and re-evaluate on HF model
                else:
                     val = self.model(paraUQ(freq,sample_i,self.number_uq_para))
                     self.hf_evals=self.hf_evals+1
                     # Save data for GPR model update
                     self.freq_update.append(freq)
                     self.sample_update.append(sample_i)
                     self.approx_s_update.append(s.flatten()[0])
                     self.val_update.append(val)
                # if HF-solution for this freq. point is accepted --> continue with next freq. point  
                if prove_pfs(val,self.threshold, 'compl') == 1:
                    continue
                # if HF-solution for this freq. point not accepted --> sample point not valid (continue with next sample point)
                else:
                    sample_accepted = 0
                    break
            
            # if sample point fulfilled pfs for all freq. points --> valid
            if sample_accepted == 1:
                counter_valid+=1
                valids.append(sample_i)
        
        # If Hybrid approach, update GPR model after each 'self.Batch_Size' HF evaluations
        if self.hf_evals % self.Batch_Size == 0 and len(self.sample_update)>0 and self.YE_method == 'Hybrid':
            #start = time.time()
            update_GPR(self,self.freq_update,self.sample_update,self.val_update,self.approx_s_update)
            #ende = time.time()
            #print('{:5.3f}s'.format(ende-start))
            
            # reset data for GPR model update
            self.freq_update = []
            self.sample_update = []
            self.val_update = []
            self.approx_s_update = []
            print('GPR models updated after', self.Nmc-len(samples_uq)+i+1, 'MC sample points and ', self.hf_evals, 'critical MC sample points.')
            
            # After updating the GPR model --> Evaluate all remaining sample points on GPR model and sort them according to chosen criterion
            if self.Sorting_Strategy != 'none':
                # list of remaining sample points
                samples_uq = samples_uq[i+1:len(samples_uq)]
                # evaluating and sorting
                if self.Sorting_Strategy == 'EGL':
                    samples_uq = sort_samples_EGL(self,samples_uq, display = 0)#, n_freq_point=0)
                    print('Remaining MC sample has been evaluated on GPR model and sorted according to EGL criterion.')
                    print('Remaining sample points:', len(samples_uq),'\n')
                elif self.Sorting_Strategy == 'FS':
                    samples_uq = sort_samples_FS(self,samples_uq, display = 0)
                    print('Remaining MC sample has been evaluated on GPR model and sorted according to FS criterion.')
                    print('Remaining sample points:', len(samples_uq),'\n')
                else:
                    print('Set a valid sorting method')
                # continue with first sample point of the list of remaining sample points
                i=0
            else:
                # continue with the next sample point in the original list, if no sorting has taken place
                i=i+1
        
        else:
            # continue with next sample point from the list, if no updating has taken place
            i=i+1
        
    # Caluculate Yield and display results and effort (#HF Evaluations)    
    Yield = float(counter_valid)/self.Nmc
    
    if display: 
        print("\n Number of MC sample points: ", self.Nmc,  ", Yield is: ", Yield)
        if self.YE_method != 'MC':
            print('HF evaluations for critical MC sample points:', self.hf_evals)
            print('HF evaluations to build GPR models:', self.HFevals_forGPRmodel)
            print('Total number of HF evaluations:', self.hf_evals+self.HFevals_forGPRmodel,'\n \n')
        else:
            print('HF evaluations for MC sample points:', self.hf_evals,'\n \n')
    
    return Yield, valids



# Evaluate sample points on GPR model and sort them to Hybrid / FS criterion
def sort_samples_EGL(self,samples_list,display=1):#, n_freq_point=0):
    # Initialize lists
    self.Sabs_matrix = [];
    self.sigma_matrix = [];
    self.SC_matrix = [];
    self.SC_vector = [];
    
    N_samples = len(samples_list)
    for i in range(N_samples):
        # Evaluate QoI (s) and std (sigma) on GPR models for all freq. points
        sigma_range, s_range = interpolate(self,samples_list[i],True)
        sdB_range = 20*np.log10(np.abs(s_range))
        s_range = np.array(s_range).flatten().tolist()
        sigma_range = np.array(sigma_range).flatten().tolist()
        # Collect data
        self.Sabs_matrix.append(s_range)
        self.sigma_matrix.append(sigma_range)
        # Calculate EGL sorting criterion
        J = np.abs(self.threshold - sdB_range) / np.abs(sigma_range)
        self.SC_matrix.append(J.tolist())
        # Save lowest criterion value for each sample point (over freq. points)
        self.SC_vector.append(np.min(J))

    # Sort sample points (and all the corresponding data), starting with the lowest value of the sorting criterion J
    SC_sorted_indices = np.argsort(np.ravel(self.SC_vector))#[::-1]
    samples_sorted = np.array(samples_list)[SC_sorted_indices]
    self.Sabs_matrix = np.array(self.Sabs_matrix)[SC_sorted_indices]
    self.sigma_matrix = np.array(self.sigma_matrix)[SC_sorted_indices]
    self.SC_matrix = np.array(self.SC_matrix)[SC_sorted_indices]
    self.SC_vector = np.array(self.SC_vector)[SC_sorted_indices]
  
    if display == 1:
        return samples_sorted, self.Sabs_matrix, self.sigma_matrix
    else:
        return samples_sorted


# Evaluate sample points on GPR model and sort them to Hybrid / FS criterion
def sort_samples_FS(self,samples_list, display=1):
    # Initialize lists
    self.Sabs_matrix = [];
    self.sigma_matrix = [];
    self.SC_matrix = [];
    self.SC_vector = [];
    
    N_samples = len(samples_list)
    for i in range(N_samples):
        # Evaluate QoI (s) and std (sigma) on GPR models for all freq. points
        sigma_range, s_range = interpolate(self,samples_list[i],True)
        s_range = np.array(s_range).flatten().tolist()
        sigma_range = np.array(sigma_range).flatten().tolist()
        # Collect data
        self.Sabs_matrix.append(s_range)
        self.sigma_matrix.append(sigma_range)
        # Calculate difference between threshold and lower (L) / upper (U) bound
        L = self.threshold-20*np.log10(np.abs(np.abs(s_range)-self.Safety_Factor*np.abs(sigma_range)))
        U = 20*np.log10(np.abs(np.abs(s_range)+self.Safety_Factor*np.abs(sigma_range))) - self.threshold
        # Calculate Sorting criterion
        LU = L*U
        self.SC_matrix.append(LU.tolist())
        # Save highest criterion value for each sample point (over freq. points)
        self.SC_vector.append(np.max(LU))

    # Sort sample points (and all the corresponding data), starting with the highest value of the sorting criterion LU
    SC_sorted_indices = np.argsort(np.ravel(self.SC_vector))[::-1]
    samples_sorted = np.array(samples_list)[SC_sorted_indices]
    self.Sabs_matrix = np.array(self.Sabs_matrix)[SC_sorted_indices]
    self.sigma_matrix = np.array(self.sigma_matrix)[SC_sorted_indices]
    self.SC_matrix = np.array(self.SC_matrix)[SC_sorted_indices]
    self.SC_vector = np.array(self.SC_vector)[SC_sorted_indices]
  
    if display == 1:
        return samples_sorted, self.Sabs_matrix, self.sigma_matrix
    else:
        return samples_sorted
























