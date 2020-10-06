# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:46:08 2019

@author: Mona Fuhrländer (Technische Universität Darmstadt, mona.fuhrlaender@tu-darmstadt.de)

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
    if self.problem == 'Waveguide':
        if distr == 'uniform': # uniform
            training_data_initial = self.sample_generator(mean, self.input_distr, TrainSize, normal = 'false')
        elif distr == 'gaussian': # gaussian
            training_data_initial = self.sample_generator(mean, self.input_distr, TrainSize)
        else:
            print('ERROR: Please insert valid distribution for initial training data.')
        #nodes = np.array(training_data_initial).T # initial training data points
    
    elif self.problem == 'Lowpass':
        self.tdp = 2500 # = Nmc if Nmc+TrainSize<length of sample_list (10000) number of sample points to be skipped (because they are MC sample points)
        training_data_initial = self.sample_generator(mean, self.input_distr, self.tdp+TrainSize)
        # skip the first self.tdp sample points (because they are MC sample points)
        training_data_initial = np.array(training_data_initial).T[self.tdp:self.tdp+TrainSize].T
    
    nodes = np.array(training_data_initial).T # initial training data points
    
    # Initialize surrogate models (for real and imag part separately)
    self.surrogates=[]
    self.y_imag=[[0] * TrainSize for i in range(len(self.freqrange))]
    self.y_real=[[0] * TrainSize for i in range(len(self.freqrange))]
    self.gp_imag = []; self.gp_real = []
    Sreal_forAllTD = []; Simag_forAllTD = []
    #...and for each frequency point separately
    for k in range(len(self.freqrange)):
        self.surrogates.append(nodes)
        self.gp_imag.append(GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5,normalize_y=True, n_restarts_optimizer=10)) 
        self.gp_real.append(GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5,normalize_y=True, n_restarts_optimizer=10)) 
    
    # Build surrogate models
    # For the waveguide
    if self.problem == 'Waveguide':
        for k in range(len(self.freqrange)):
            for n in range(TrainSize):
                s_val_real, s_val_imag = self.model(self,paraUQ(self,self.freqrange[k],self.surrogates[k][n],n))
                self.y_imag[k][n] = (s_val_imag.ravel())
                self.y_real[k][n] = (s_val_real.ravel())
            self.gp_imag[k].fit(self.surrogates[k], self.y_imag[k])
            self.gp_real[k].fit(self.surrogates[k], self.y_real[k]) 

    # For the Lowpass Filter
    elif self.problem == 'Lowpass':
        for n in range(TrainSize):
            freqSreal, freqSimag = self.model(self,paraUQ(self,self.freqrange[k],self.surrogates[k][n],n+self.tdp))
            Sreal = np.transpose(freqSreal)[1]
            Simag = np.transpose(freqSimag)[1]
            Sreal_forAllTD.append(Sreal)
            Simag_forAllTD.append(Simag)
    
        for k in range(len(self.freqrange)):
            for n in range(TrainSize):
                s_val_real = Sreal_forAllTD[n][k]
                self.y_real[k][n] = (s_val_real.ravel())
                s_val_imag = Simag_forAllTD[n][k]
                self.y_imag[k][n] = (s_val_imag.ravel())
            self.gp_real[k].fit(self.surrogates[k],self.y_real[k])
            self.gp_imag[k].fit(self.surrogates[k],self.y_imag[k])
    
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
#    print('len',len(HF_freq), len(HF_sample),len(HF_val),len(GPR_val))
#    print('HF_val',HF_val)
    GPR_val = np.array(GPR_val).flatten().tolist()
#    print('GPR_val',GPR_val)
    
    # for each freq. point the according surrogate model might be updated
    for k in range(len(self.freqrange)):
        freqv = self.freqrange[k]
        idx_list_freqv = [i for i, value in enumerate(HF_freq) if value == freqv]
#        print('idx_list',len(idx_list_freqv),idx_list_freqv)
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
#            print('Diff',len(Diff),Diff)
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
                    
                    if self.problem == 'Lowpass':
                        GPR_range = np.array(GPR_range).flatten()
                        #print('GPR range', GPR_range, 'GPR_val', GPR_range[k][0][0])
                        GPR_val_list_freqv.append(GPR_range[k])
                    elif self.problem == 'Waveguide':
                        GPR_val_list_freqv.append(GPR_range[k][0][0])
                    
                
                # ...and calculate their difference between updated GPR prediction and HF value
                Diff = abs(GPR_val_list_freqv - val_list_freqv)
                # the sample point with the largest Diff_val value might be added in the next loop (if < Diff_tol)
                Diff_val = np.max(Diff)
            
            if display == 1:
                print('GPR model for Freq ', freqv, 'GHz updated with', addsamp, 'new training data points (', len(idx_list_freqv), 'critical samples).')
            


# Estimate Yield with pure GPR, pure MC or Hybrid approach        
def Estimation_GPR(self, start_uq, display=1):
    print('p:', start_uq.T,'\n')
    
    # Generate MC sample
    samples_uq = self.sample_generator(start_uq, self.input_distr, self.Nmc).T
    
    # Option for sorting the sample points according to EGL or FS (=Hybrid) criterion.
    if self.YE_method == 'Hybrid':
        if self.Sorting_Strategy == 'EGL':
            samples_uq = sort_samples_EGL(self,samples_uq, display = 0)
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
    count_update = 0 # count number of additional training data points until the next update
    self.approx_s_update = [] #GPR S-Parameter value for sample points which have been reevaluated on high fidelity (HF) model (critical samples)
    self.val_update = [] # reevaluated HF solution value of S-Parameter
    self.sample_update = [] # sample points which are reevaluated on HF model(critical samples)
    self.freq_update = [] # frequency values for reeveluated sample points (critical samples)
    
    # For each MC sample point check if it is valid (performance feature specifications (pfs) are fulfilled)
    while i<len(samples_uq):
        # pure GPR approach
        if self.YE_method == 'GPR':
            sample_accepted = 1
            # Evaluate sample point on GPR model for all frequency points...
            s_range = interpolate(self,samples_uq[i], False)
            for j in range(len(self.freqrange)):
                freqv = self.freqrange[j]
                sv = s_range[j]
                # if sample point fulfills pfs for this freq. point, continue with next freq. point
                if prove_pfs(self,sv,'compl',freqv) == 1:
                    continue
                # if sample point fulfills pfs NOT for this freq. point, sample point is NOT valid
                else:
                    sample_accepted = 0
                    break
            # if sample point fulfilled pfs for all freq. points --> valid
            if sample_accepted == 1:
                counter_valid+=1
                valids.append(samples_uq[i]) 


        # pure MC approach
        elif self.YE_method == 'MC':
            sample_accepted = 1
            # For the Lowpass Filter - the whole freq. range is solved simutaneously
            if self.problem == 'Lowpass':
                s_real_range, s_imag_range = self.model(self,paraUQ(self,0,samples_uq[i], i))
                s_real_range = np.transpose(s_real_range)[1]
                s_imag_range = np.transpose(s_imag_range)[1]
                s_range = s_real_range + 1j * s_imag_range
                self.hf_evals = self.hf_evals+1
            
            # to save computing time, consider one frequency point after the other
            for j in range(len(self.freqrange)):
                freqv = self.freqrange[j]
                #for freqv in self.freqrange:
                if self.problem == 'Waveguide':
                    # Evaluate sample point on original (HF) model
                    s_real, s_imag= self.model(self,paraUQ(self,freqv,samples_uq[i], i))
                    s = s_real + 1j * s_imag
                    self.hf_evals = self.hf_evals+1
                elif self.problem == 'Lowpass':
                    # load value for this sample point at considered freq. point
                    s = s_range[j]
                    
                # if sample point fulfills pfs for this freq. point, continue with next freq. point
                if prove_pfs(self,s,'compl',freqv) == 1:
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
            
            # If there is only one pfs, start with the frequency point with the highest failing risk
            if len(self.threshold) == 1:
                if self.threshold[0][3] == 'ub':
                    # Sort frequency points according to S-Parameter dB value (to start with largest value,
                        # which has the highest risk to fail pfs)
                    sort_indices = np.argsort(np.ravel(s_dB_range))[::-1]
                    s_sort = np.array(s_range)[sort_indices]
                    sigma_sort = np.array(sigma_range)[sort_indices]
                    freq_sort = np.array(self.freqrange)[sort_indices]
                elif self.threshold[0][3] == 'lb':
                    # Sort frequency points according to S-Parameter dB value (to start with smallest value,
                        # which has the highest risk to fail pfs)
                    sort_indices = np.argsort(np.ravel(s_dB_range))
                    s_sort = np.array(s_range)[sort_indices]
                    sigma_sort = np.array(sigma_range)[sort_indices]
                    freq_sort = np.array(self.freqrange)[sort_indices]
            # If there are several pfs, keep the order of the frequency points
            else:
                s_sort = np.array(s_range)
                sigma_sort = np.array(sigma_range)
                freq_sort = np.array(self.freqrange)
                    
            
            # For each frequency point (if app.: starting with the one with highest failing risk)
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
                
                
                # Check bound specification for frequency point
                bound_spec = 'none'
                for ts in range(len(self.threshold)):
                    interval = [self.threshold[ts][1],self.threshold[ts][2]]
                    if freq >= interval[0] and freq <= interval[1]:
                        if bound_spec == 'none':
                            bound_spec = self.threshold[ts][3]
                        elif bound_spec != self.threshold[ts][3]:
                            bound_spec = 'both'
                            break
                    
                # if freq.point has lower bound
                if bound_spec == 'lb':
                    if prove_pfs(self,trusted_upper_bound, 'compl',freq) == 0:
                        sample_accepted = 0
                        break
                    elif prove_pfs(self,trusted_lower_bound, 'compl',freq) == 1:
                        continue
                    
                # if freq. point has upper bound
                elif bound_spec == 'ub':
                    if prove_pfs(self,trusted_lower_bound, 'compl',freq) == 0:
                        sample_accepted = 0
                        break
                    elif prove_pfs(self,trusted_upper_bound, 'compl',freq) == 1:
                        continue
                
                # if freq. point has lower and upper bound
                elif bound_spec == 'both':
                    if prove_pfs(self,trusted_upper_bound, 'compl',freq) == 0:
                        sample_accepted = 0
                        break
                    elif prove_pfs(self,trusted_lower_bound, 'compl',freq) == 0:
                        sample_accepted = 0
                        break
                    elif prove_pfs(self,trusted_lower_bound, 'compl',freq) == 1 and prove_pfs(self,trusted_upper_bound, 'compl',freq) == 1:
                        continue
                
                # if freq. point has no bound
                elif bound_spec == 'none':
                    continue
                
                # If sample point is critical...
                # ...evaluate sample point on HF model for whole freq. range if Lowpass Filter
                if self.problem == 'Lowpass':
                    # Evaluate critical sample point
                    fSreal, fSimag = self.model(self,[i,samples_uq[i]],dB=False)    
                    self.hf_evals=self.hf_evals+1
                    count_update = count_update+1
                    
                    # Save data for GPR model update
                    for j2 in range(len(self.freqrange)):
                        self.freq_update.append(self.freqrange[j2])
                        self.sample_update.append(samples_uq[i])
                        self.approx_s_update.append(s_range[j2])
                        self.val_update.append(np.transpose(fSreal)[1][j2]+1j*np.transpose(fSimag)[1][j2])
                    
                    # Check if HF solution fulfills pfs
                    Scompl = np.transpose(fSreal)[1]+1j*np.transpose(fSimag)[1]
                    for j3 in range(len(self.freqrange)):
                        if prove_pfs(self,Scompl[j3],'compl',self.freqrange[j3]) == 1:
                            continue
                        else:
                            sample_accepted = 0
                            break
                    # Continue with next sample point (after a model update if app.)
                    break

                # ...evaluate sample point on HF model for specific freq. point if Waveguide
                elif self.problem == 'Waveguide':
                    # Evaluate critical sample point
                    val_real, val_imag = self.model(self,paraUQ(self,freq,sample_i,i))
                    val = val_real + 1j * val_imag
                    self.hf_evals=self.hf_evals+1
                    count_update = count_update+1
                    # Save data for GPR model update
                    self.freq_update.append(freq)
                    self.sample_update.append(sample_i)
                    self.approx_s_update.append(s.flatten()[0])
                    self.val_update.append(val)
                    # if HF-solution for this freq. point is accepted --> continue with next freq. point  
                    if prove_pfs(self, val, 'compl', freq) == 1:
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
        if (count_update >= self.Batch_Size or self.hf_evals % self.Batch_Size == 0) and len(self.sample_update)>0 and self.YE_method == 'Hybrid':
        #if self.hf_evals % self.Batch_Size == 0 and len(self.sample_update)>0 and self.YE_method == 'Hybrid':
            #start = time.time()
            #print('update start')
            update_GPR(self,self.freq_update,self.sample_update,self.val_update,self.approx_s_update)
            #ende = time.time()
            #print('update stop','{:5.3f}s'.format(ende-start))
            
            # reset data for GPR model update
            count_update = 0
            self.freq_update = []
            self.sample_update = []
            self.val_update = []
            self.approx_s_update = []
            print('GPR models updated after', self.Nmc-len(samples_uq)+i+1, 'MC sample points and ', self.hf_evals, 'critical MC sample points.')
            
            # After updating the GPR model --> Evaluate all remaining sample points on GPR model and sort them according to chosen criterion
            if self.Sorting_Strategy != 'none':
                # list of remaining sample points
                samples_uq = samples_uq[i+1:len(samples_uq)]
                # ... and their QoI data if Lowpass Filter
                if self.problem == 'Lowpass':
                    self.QoI = self.QoI[(i+1)*len(self.freqrange):len(self.QoI)]
                # evaluating and sorting
                if self.Sorting_Strategy == 'EGL':
                    #start = time.time()
                    #print('sorting start')
                    samples_uq = sort_samples_EGL(self,samples_uq, display = 0)#, n_freq_point=0)
                    #ende = time.time()
                    #print('sorting stop','{:5.3f}s'.format(ende-start))
                    print('Remaining MC sample has been evaluated on GPR model and sorted according to EGL criterion.')
                    print('Remaining sample points:', len(samples_uq),'\n')
                elif self.Sorting_Strategy == 'FS':
                    #start = time.time()
                    #print('sorting start')
                    samples_uq = sort_samples_FS(self,samples_uq, display = 0)
                    #ende = time.time()
                    #print('sorting stop','{:5.3f}s'.format(ende-start))
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
        # Calculate EGL sorting criterion...
        J = []
        # ...for each freq. point...
        for j in range(len(self.freqrange)):
            Jj = []
            freqv = self.freqrange[j]
            # ...and each pfs...
            for ii in range(len(self.threshold)):
                pfs = self.threshold[ii]
                if freqv >= pfs[1] and freqv <= pfs[2]:
                    bound = pfs[0]
                    Jji = np.abs(bound - sdB_range[j]) / np.abs(sigma_range[j])
                else:
                    Jji = np.inf
                Jj.append(Jji)
            # ...and choose smallest value
            J.append(np.min(Jj))       
        # Save sorting criteria values in a matrix
        self.SC_matrix.append(J)
        # Save lowest criterion value for each sample point (over freq. points) in a vector
        self.SC_vector.append(np.min(J))
    
    # Sort sample points (and all the corresponding data), starting with the lowest value of the sorting criterion J
    SC_sorted_indices = np.argsort(np.ravel(self.SC_vector))#[::-1]
    samples_sorted = np.array(samples_list)[SC_sorted_indices]
    self.Sabs_matrix = np.array(self.Sabs_matrix)[SC_sorted_indices]
    self.sigma_matrix = np.array(self.sigma_matrix)[SC_sorted_indices]
    self.SC_matrix = np.array(self.SC_matrix)[SC_sorted_indices]
    self.SC_vector = np.array(self.SC_vector)[SC_sorted_indices]
    
    # If data comes from a list (e.g. Lowpass list) update / sort also the QoI list
    if self.problem == 'Lowpass':
        Indices_QoI = []
        for i in range(len(SC_sorted_indices)):
            for l in range(len(self.freqrange)):
                i_QoI = SC_sorted_indices[i]*len(self.freqrange) + l
                Indices_QoI.append(i_QoI)
        self.QoI = np.array(self.QoI)[Indices_QoI]
  
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
        # Calculate FS / Hybrid sorting criterion...
        LU = []
        # ...for each freq. point...
        for j in range(len(self.freqrange)):
            LUj = []
            freqv = self.freqrange[j]
            # ...and each pfs...
            for ii in range(len(self.threshold)):
                pfs = self.threshold[ii]
                if freqv >= pfs[1] and freqv <= pfs[2]:
                    bound = pfs[0]
                    # Calculate difference between threshold (bound) and lower (L) / upper (U) bound of the safety puffer
                    Lji = bound-20*np.log10(np.abs(np.abs(s_range[j])-self.Safety_Factor*np.abs(sigma_range[j])))
                    Uji = 20*np.log10(np.abs(np.abs(s_range[j])+self.Safety_Factor*np.abs(sigma_range[j]))) - bound
                    LUji = Lji*Uji
                else:
                    LUji = -1*np.inf
                LUj.append(LUji)
            # ...and choose highest value
            LU.append(np.max(LUj))
        
        # Save sorting criteria values in a matrix
        self.SC_matrix.append(LU)
        # Save highest criterion value for each sample point (over freq. points) in a vector
        self.SC_vector.append(np.max(LU))

    # Sort sample points (and all the corresponding data), starting with the highest value of the sorting criterion LU
    SC_sorted_indices = np.argsort(np.ravel(self.SC_vector))[::-1]
    samples_sorted = np.array(samples_list)[SC_sorted_indices]
    self.Sabs_matrix = np.array(self.Sabs_matrix)[SC_sorted_indices]
    self.sigma_matrix = np.array(self.sigma_matrix)[SC_sorted_indices]
    self.SC_matrix = np.array(self.SC_matrix)[SC_sorted_indices]
    self.SC_vector = np.array(self.SC_vector)[SC_sorted_indices]
    
    # If data comes from a list (e.g. Lowpass list) update / sort also the QoI list
    if self.problem == 'Lowpass':
        Indices_QoI = []
        for i in range(len(SC_sorted_indices)):
            for l in range(len(self.freqrange)):
                i_QoI = SC_sorted_indices[i]*len(self.freqrange) + l
                Indices_QoI.append(i_QoI)
        self.QoI = np.array(self.QoI)[Indices_QoI]
    
    if display == 1:
        return samples_sorted, self.Sabs_matrix, self.sigma_matrix
    else:
        return samples_sorted
























