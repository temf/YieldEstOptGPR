# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:57:04 2019

@author: Mona Fuhrländer (Technische Universität Darmstadt, mona.fuhrlaender@tu-darmstadt.de)

Adaptive Newton-MC method for Yield Optimization (single-objective)
"""

import numpy as np
import numpy.linalg as LA
import time


def Opt_Newton(self, start_uq, adaptive=True):
    # time measure 
    opttime_start = time.time()
    
    # Initialize for adaptive process
    if adaptive:
        Nmc_opt = 100 # initial MC sample size
    else:
        Nmc_opt = self.Nmc # MC sample size for nonadaptive optimization is set in main file (Run...)
    errmc = 0.01 # maximal MC error (std of yield estimator)
    fac_err = 0.5   #1.0  # constant factor for sample size increase  
                    #for Yield Estimation with SC: 1.0; For Yield Estimation with GPR: 0.5
    
    # Initialize counting variables
    self.hf_evals_opt = 0 # number of high fidelity evaluations
    self.N_yes = 0 # number of yield estimations
    self.N_negYGH = 0 # number of Gradient and Hessian evaluations
    It = 0  # number of Newton iterations
    
    # Initialize UQ settings
    Std_vec = np.array((np.matrix(self.input_distr).T)[1])[0]  # vector of standard deviations for design parameters
    Cov = np.matrix(np.diag(Std_vec)) # covariance matrix 
    p_k = np.matrix(start_uq).T # mean value of initial design
    np.random.seed(23)
    
    # Yild Estimation for starting point
    Yield_k, valids = self.estimate_Yield(p_k, Nmc_opt, display=0, res='all') # evaluate yield
    self.hf_evals_opt += self.hf_evals # count high fidelity evaluations
    self.N_yes += 1 # count yield evaluations
    Yield_k, Grad_k, Hess_k = negYieldGradHess(self,p_k,Cov, Nmc_opt, Yield_k, valids) # evaluate Gradient and Hessian
    Yield_start = -Yield_k

    # Save history
    self.history = [[It, -Yield_k, self.hf_evals, Nmc_opt]]

    # Initialize parameters for Newton method
    beta = 0.5; #in (0,1)
    gamma = 0.01; #in (0,1)
    alpha1 = 1e-6; #>0
    alpha2 = 1e-6; #in (0,1)
    po = 0.1; #>0
    TolDet = 0.01;
    TolGrad = 0.001;
    change = 0;

    # Until stopping criterion Grad =~ 0 not reached
    while LA.norm(Grad_k) > TolGrad:
        
        # SEARCH DIRECTION
        print('\n\n ########### \n')
        It+=1; print(It, "-th iteration")
        print("Sample Size in this iteration:",Nmc_opt)
        # Search direction if Hess not invertible or angle condition not fulfilled: gradient step
        s_k = -Grad_k 
        # Check regularity of Hessian 
        if abs(LA.det(Hess_k)) > TolDet:
            # Newton step
            d_k = -np.matmul(LA.inv(Hess_k), Grad_k)
            # angle condition
            if -np.matmul(Grad_k.T, d_k) >= min(alpha1, alpha2 * LA.norm(d_k)**po) * LA.norm(d_k)**2:
                # Search direction fixed: Newton step
                s_k = d_k 

        # ARMIJO STEPSIZE
        left = 2; right = 1; expon = 0;
        while left > right:
            sigma_k = beta**expon
            # check if improvement smaller than standard deviation: stop and check for MC sample size increase
            if LA.norm(sigma_k * s_k) < 1e-3 * LA.norm(Std_vec):
                change = 0
                break
            # check if to many Armijo backsteps have been made: stop and check for MC sample size increase
            elif expon > 3:
                change = 2
                break
            # else: run Armijo loop
            else:
                print("\n Recalculation for step size")
                change = 1
                # formulate test design point
                p_test = p_k + sigma_k * s_k
                # evaluate yield, gradient and Hessian for test point
                np.random.seed(23)
                Yield, valids_test = self.estimate_Yield(p_test, Nmc_opt, display=0, res='all') # evaluate yield
                self.hf_evals_opt += self.hf_evals # count high fidelity evaluations
                self.N_yes += 1 # count yield evaluations
                Yield, Grad, Hess = negYieldGradHess(self,p_test,Cov, Nmc_opt, Yield, valids_test) # evaluate Gradient and Hessian
                # prepare for Armijo check
                left = Yield - Yield_k
                right = sigma_k * gamma * np.matmul(Grad_k.T,s_k)
                expon+=1

            
        # if adaptive sample size increase:
        if adaptive:
            # ERROR INDICATOR FOR YIELD ESTIMATION
            err = np.sqrt((np.abs(Yield)*(1-np.abs(Yield)))/Nmc_opt)

            # CHANGES (in Y) AND ADJUSTMENT (of Nmc)
            # if there is no improvement anymore...
            if change == 0 or change == 2:
                # ... check if defined MC accuracy is reached. If not:
                if err > fac_err * errmc:
                    hf_sum = self.hf_evals
                    Yield_k_big = Yield
                    while err > fac_err * errmc and np.abs(np.abs(Yield_k_big) - np.abs(Yield)) < 0.01:
                        print("\n Additional calculation for sample size increase")
                        # add MC sample points
                        Nmc_add = 100
                        Nmc_opt = Nmc_opt + Nmc_add
                        # evaluate yield for additional MC sample points
                        Yield_k_add, valids_add = self.estimate_Yield(p_k, Nmc_add, display=0, res='all') # evaluate yield
                        self.hf_evals_opt += self.hf_evals # count high fidelity evaluations
                        self.N_yes += 1 # count yield evaluations
                        # merge 'old' yield and additional yield --> obtain new (big) yield
                        valids_big = valids + valids_add; valids = valids_big
                        In_big = len(valids_big)
                        Yield_k_big = float(In_big)/Nmc_opt
                        # calculate MC error for new (big) yield
                        err = np.sqrt((np.abs(Yield_k_big)*(1-np.abs(Yield_k_big)))/Nmc_opt)
                        # count all high fielity evaluations in this sample size increase procedure
                        hf_sum = hf_sum + self.hf_evals
                    # accept new (big) yield and calculate corresponding gradient and Hessian
                    Yield_k = Yield_k_big;
                    Yield_k, Grad_k, Hess_k = negYieldGradHess(self,p_k, Cov, Nmc_opt, Yield_k, valids)
                    # save history
                    self.history.append([It, -Yield_k, hf_sum, Nmc_opt])
                else:
                    # if no improvement anymore, but defined MC accurracy reached: stop optimization
                    print("\n Stop because: no improvement anymore")
                    break
            
            # if there has been improvement in this iteration: accept test point
            else:
                # update iteration results and save history
                p_k = p_test; valids = valids_test
                Yield_k = Yield; Grad_k = Grad; Hess_k = Hess;
                self.history.append([It, -Yield_k, self.hf_evals,Nmc_opt])
        
        # if fixed sample size: 
        else:
            # update iteration results and save history
            p_k = p_test; valids = valids_test
            Yield_k = Yield; Grad_k = Grad; Hess_k = Hess;
            self.history.append([It, -Yield_k, self.hf_evals,Nmc_opt])
    

    # Accept final solution as optimal solution
    p_opt = p_k; Yield_opt = -Yield_k;
    # Stop time measurement 
    opttime_end = time.time() - opttime_start 
        
    # Output
    print('\n\n *****************************')
    print('*****OPTIMIZATION RESULTS*****')
    print('initial Yield value:',Yield_start)
    print('optimal Yield value:',Yield_opt)
    print('initial design vector:',start_uq)
    print('optimal design vector:',p_opt.T[0])
    
    print('\n***Computational effort***')
    print("total number of Newton iterations:", It)
    print("total number of high fidelity evaluations:", self.hf_evals_opt)
    print("total number of yield estimations:", self.N_yes)
    print("total number Grad/Hess evaluations:", self.N_negYGH)
    print('final sample size:',Nmc_opt)
    print('running time (in sec.):',opttime_end)
    
    print('\n***History***')
    print('[iteration,Yield,#high fidelity evaluations,sample size]:',self.history)
    print('NOTE: Initial Yield has been evaluated with initial sample size, \
          optimal Yield has been evaluated with final sample size. \
          In case higher accuracy is required for the final Yield, please run an estimation for the optimal design vector.')
    
    return Yield_start, Yield_opt, start_uq, p_opt, opttime_end, self.history


# Calculate Gradient and Hessian of the yield
# since the yield shall be maximized, but the optimization method is formulated for minimization
# negYieldGradHess returns the -Yield and Grad(-Yield) and Hess(-Yield)
def negYieldGradHess(self, start_uq, Cov, Nmc, Yield, valids):
    self.N_negYGH = self.N_negYGH+1 # counts gradient and Hessian evaluations
    # mean vector of design parameter as input
    p = start_uq
    # negative yield (for minimization purpose)
    Yield = -Yield
    # mean value of valid sample points
    p_delta = np.matrix(np.mean(valids,axis=0)).T
    # standard devaition of valid sample points
    if len(valids) == 0:
        Std_delta = [0]
    else:
        Std_delta = np.std(valids,axis=0)
    # covariance matrix of valid sample points
    Cov_delta = np.matrix(np.diag(Std_delta))
    # Gradient
    Grad = Yield * np.matmul(LA.inv(Cov),(p_delta-p))
    # Hessian
    Hess = Yield * np.matmul(np.matmul(LA.inv(Cov),Cov_delta+np.matmul(p_delta-p,(p_delta-p).T)-Cov),LA.inv(Cov))
    
    return Yield, Grad, Hess

























