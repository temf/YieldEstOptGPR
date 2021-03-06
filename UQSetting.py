# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:56:13 2019

@author: Mona Fuhrländer (Technische Universität Darmstadt, mona.fuhrlaender@tu-darmstadt.de)


Util Functions for Validation of sample points
"""


import numpy as np


def prove_pfs(self,QoI,unit='compl',freqv=[]):
    # validate if performance feature specifications are fulfilled: yes=1, no=0
    if unit == 'compl':
        QoI_dB = 20*np.log10(abs(QoI))
    elif unit == 'dB':
        QoI_dB = QoI
    
    # transform each pfs into an upper bound constraint
    for i in range(len(self.threshold)):
        pfs = self.threshold[i]
        if pfs[3] == 'ub':
            bound = pfs[0]
            QoI_dB_test = QoI_dB
        elif pfs[3] == 'lb':
            bound = -pfs[0]
            QoI_dB_test = -np.array(QoI_dB)
        if freqv >= pfs[1] and freqv <= pfs[2]:
            #check
            if QoI_dB_test <= bound:
                pfs_fulfilled = 1
            else:
                pfs_fulfilled = 0
                break
        else:
            pfs_fulfilled = 1
        
        
    if pfs_fulfilled == 1:
        return 1
    else:
        return 0


     
# 1 uncertain parameters for waveguide
def paraUQ1_WG(freq, uq_input):
        #Set certain and uncertain design parameters
        params = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        #[freq, width, height, fill_l, offset, epsr, muer]
        params[0] = freq # frequency
        params[1] = 30 # waveguide width (not uncertain)
        params[2] = 0.1 #3 # waveguide height (not uncertain)
        params[3] = uq_input[0] #filling length
        params[4] = 5 #vacuum offset
        
        #Material parameters mue_r and eps_r
        w=2*np.pi*freq*1e9 # Omega
        eps8 = 1  # HF permittivity
        epss = 2.0 # static permittivity
        tau = 1 / (5.0*1e9 * 2 * np.pi) # relaxation time of medium
        epsr=eps8+(epss-eps8)/(1+1j*w*tau) # relative permittivity
        mue8 = 1  # HF permeability
        mues = 3.0 # static permeability
        tau = 1.1 * 1 / (20*1e9 * 2 * np.pi) # relaxation time of medium
        muer=mue8+(mues-mue8)/(1+1j*w*tau)  # relative permeability
        
        params[5] = epsr
        params[6] = muer
        
        return params


# 2 uncertain parameters for waveguide
def paraUQ2_WG(freq, uq_input):
        #Set certain and uncertain design parameters
        params = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        #[freq, width, height, fill_l, offset, epsr, muer]
        params[0] = freq # frequency
        params[1] = 30 # waveguide width (not uncertain)
        params[2] = 0.1 #3 # waveguide height (not uncertain)
        params[3] = uq_input[0] #filling length
        params[4] = uq_input[1] #5 #vacuum offset
        
        #Material parameters mue_r and eps_r
        w=2*np.pi*freq*1e9 # Omega
        eps8 = 1  # HF permittivity
        epss = 2.0 # static permittivity
        tau = 1 / (5.0*1e9 * 2 * np.pi) # relaxation time of medium
        epsr=eps8+(epss-eps8)/(1+1j*w*tau) # relative permittivity
        mue8 = 1  # HF permeability
        mues = 3.0 # static permeability
        tau = 1.1 * 1 / (20*1e9 * 2 * np.pi) # relaxation time of medium
        muer=mue8+(mues-mue8)/(1+1j*w*tau)  # relative permeability
        
        params[5] = epsr
        params[6] = muer
        
        return params


# 4 uncertain parameters for waveguide
def paraUQ4_WG(freq, uq_input):
        #Set certain and uncertain design parameters
        params = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        #[freq, width, height, fill_l, offset, epsr, muer]
        params[0] = freq # frequency
        params[1] = 30 # waveguide width (not uncertain)
        params[2] = 0.1 #3 # waveguide height (not uncertain)
        params[3] = uq_input[0] #filling length
        params[4] = uq_input[1] #vacuum offset
        
        #Material parameters mue_r and eps_r
        w=2*np.pi*freq*1e9 # Omega
        eps8 = 1 + uq_input[2] # HF permittivity
        epss = 2.0 # static permittivity
        tau = 1 / (5.0*1e9 * 2 * np.pi) # relaxation time of medium
        epsr=eps8+(epss-eps8)/(1+1j*w*tau) # relative permittivity
        mue8 = 1 + uq_input[3] # HF permeability
        mues = 3.0 # static permeability
        tau = 1.1 * 1 / (20*1e9 * 2 * np.pi) # relaxation time of medium
        muer=mue8+(mues-mue8)/(1+1j*w*tau)  # relative permeability
        
        params[5] = epsr
        params[6] = muer
        
        return params


# 6 uncertain parameters for lowpass filter
def paraUQ6_LP(uq_input, i):
    params = [i,uq_input]
    
    return params


# Transformation of uncertain parameters for S-parameter calculation
def paraUQ(self, freq, uq_input, i):#,cst_list=0):
    if self.problem == 'Waveguide':
        if self.number_uq_para == 1:
            params = paraUQ1_WG(freq, uq_input)
        elif self.number_uq_para == 2:
            params = paraUQ2_WG(freq, uq_input)
        elif self.number_uq_para == 4:
            params = paraUQ4_WG(freq, uq_input)
        else:
            print("Error: Number of uncertain parameters not valid")
    elif self.problem == 'Lowpass':
        if self.number_uq_para == 6:
            params = paraUQ6_LP(uq_input, i)
        else:
            print("Error: Number of uncertain parameters not valid")
    return params
    
    
    
    
    
    
    
    
    
    
    
    
    
